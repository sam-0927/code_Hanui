import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import ml
from audiotools import STFTParams
from einops import rearrange
from torch.nn.utils import weight_norm

import math


# ---------------------------
# 2D Sin-Cos Positional Embedding (크기 가변, 파라미터 0)
# ---------------------------
def get_2d_sincos_pos_embed(embed_dim, h, w, device=None, dtype=None):
    def get_1d_pos_embed(n_pos):
        pos = torch.arange(n_pos, device=device, dtype=dtype).unsqueeze(1)  # (n,1)
        div = torch.exp(torch.arange(0, embed_dim//2, 2, device=device, dtype=dtype) * (-math.log(10000.0) / (embed_dim//2)))
        # 각 축당 embed_dim/2를 쓰기 위해 짝수/홀수 분리
        sin = torch.sin(pos * div)
        cos = torch.cos(pos * div)
        pe = torch.cat([sin, cos], dim=1)  # (n, embed_dim/2)
        # embed_dim/2를 맞추기 위해 부족하면 pad
        if pe.shape[1] < embed_dim // 2:
            pad = torch.zeros(n_pos, embed_dim//2 - pe.shape[1], device=device, dtype=dtype)
            pe = torch.cat([pe, pad], dim=1)
        return pe  # (n, embed_dim/2)

    assert embed_dim % 2 == 0, "embed_dim은 짝수여야 함"
    pe_h = get_1d_pos_embed(h)  # (h, embed_dim/2)
    pe_w = get_1d_pos_embed(w)  # (w, embed_dim/2)
    pe_h = pe_h.unsqueeze(1).expand(h, w, -1)  # (h,w,embed_dim/2)
    pe_w = pe_w.unsqueeze(0).expand(h, w, -1)  # (h,w,embed_dim/2)
    pe = torch.cat([pe_h, pe_w], dim=-1)       # (h,w,embed_dim)
    pe = pe.reshape(1, h*w, embed_dim)         # (1,N,embed_dim)
    return pe  # no grad


# ---------------------------
# Small ViT (vit_b_16 스타일 patch-embed + Transformer)
# ---------------------------
class SmallViT(nn.Module):
    def __init__(
        self,
        in_chans=64,
        embed_dim=64,
        depth=4,
        num_heads=2,
        patch_size=8,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        drop_path=0.0,
        return_logits=False,  # 병렬 상위에서 head를 붙일 거라 False 추천
        num_classes=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.return_logits = return_logits

        # vit_b_16과 같은 방식: Conv2d(K=patch, S=patch)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=(patch_size, patch_size),
                              stride=(patch_size, patch_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,   # (B, S, E)
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 간단 DropPath 대용 (depthwise는 생략)
        self.attn_drop = nn.Dropout(attn_dropout)

        # 파라미터 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _resize_to_multiple(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        # H, W가 patch_size 배수가 아니면 padding
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        return x
    
    def forward(self, x, x_recon):
        x = torch.cat((x, x_recon), dim=1)  # ch축으로 concat.
        
        # x: (B, 32, H, W) (H,W 다 달라도 OK)
        x = self._resize_to_multiple(x)               # (B,C=64,H',W')
        x = self.proj(x)                              # (B,E,H/P,W/P)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)              # (B, N, E)  N=Hp*Wp

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)        # (B,1,E)

        # 크기 가변 pos-embed (파라미터 0)
        pos = get_2d_sincos_pos_embed(self.embed_dim, Hp, Wp, device=x.device, dtype=x.dtype)
        pos = pos.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + torch.cat([torch.zeros_like(cls), pos], dim=1)
        x = self.transformer(x)                       # (B, N+1, E)
        cls_out = self.cls_ln(x[:, 0])                # (B,E)
        
        return self.head(cls_out)                     # (B,E) or (B,2) if return_logits=True


# ---------------------------
# 75개 SmallViT 병렬 + 최종 2-class
# ---------------------------
class Discriminator_vit(nn.Module):
    def __init__(
        self,
        in_chans=64,
        patch_size=8,
        embed_dim=64,
        depth=4,
        num_heads=2,
        mlp_ratio=4.0,
        num_models=75,
        num_classes=2,
        dropout=0.0,
    ):
        super().__init__()
        self.num_models = num_models
        self.branches = nn.ModuleList([
            SmallViT(
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                return_logits=False,   # 임베딩만 뽑아서 위에서 합침
                num_classes=num_classes
            ) for _ in range(num_models)
        ])

    # def forward(self, x_list):
    #     # x_list: 길이 75, 각 (B,32,H_i,W_i)
    #     assert isinstance(x_list, (list, tuple)) and len(x_list) == self.num_models, \
    #         f"expected list of {self.num_models} tensors"

    #     feats = []
    #     for br, x in zip(self.branches, x_list):
    #         feats.append(br(x))   # (B, embed_dim)

    #     summed_logit_mrd = sum(feats)
    #     return summed_logit_mrd


# ---------------------------
# 사용 예시 & 파라미터 수 확인
# ---------------------------
if __name__ == "__main__":
    disc = Discriminator_vit()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
        
