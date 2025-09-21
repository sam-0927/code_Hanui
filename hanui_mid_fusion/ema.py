# ema.py
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """초기 shadow 파라미터 등록"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """EMA 업데이트"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """EMA 파라미터 적용"""
        self.backup = {name: param.data.clone()
                       for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """원래 파라미터 복원"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """EMA 상태 저장"""
        return {
            'decay': self.decay,
            'shadow': {k: v.clone() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state_dict):
        """EMA 상태 불러오기"""
        self.decay = state_dict['decay']
        for name, param in state_dict['shadow'].items():
            if name in self.shadow:
                self.shadow[name] = param.clone()


class EMAOptimizer:
    """Optimizer + EMA 업데이트 통합"""
    def __init__(self, optimizer, model, decay=0.999):
        self.optimizer = optimizer
        self.ema = EMA(model, decay)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.ema.update()
        return loss

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)

    def apply_shadow(self):
        self.ema.apply_shadow()

    def restore(self):
        self.ema.restore()
