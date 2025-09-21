import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import ml
from audiotools import STFTParams
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))

class MPD(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.bands = BANDS
        ch = 32   # 32,128,512,1024,1024
        # self.convs = WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1))
        self.convs = nn.ModuleList(
            [
                WNConv2d(ch*2, 2, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(ch*2*4, 2, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(ch*2*4*4, 2, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(ch*2*4*4*2, 2, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(ch*2*4*4*2, 2, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        # self.convs = nn.ModuleList(
        #     [
        #         WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
        #         WNConv2d(ch*4, ch*4, (3, 3), (1, 1), padding=(1, 1)),
        #         WNConv2d(ch*4*4, ch*4*4, (3, 3), (1, 1), padding=(1, 1)),
        #         WNConv2d(ch*4*4*4, ch*4*4*4, (3, 3), (1, 1), padding=(1, 1)),
        #         WNConv2d(ch*4*4*4, ch*4*4*4, (3, 3), (1, 1), padding=(1, 1)),
        #     ]
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        
    def forward(self, x, x_recon): # [5, B, 32, 13, 102] = [각 layer, B,c,f,t]
        
        logit_list = []    
        # for x_fmap, layer in zip(x, self.convs):# 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
        for i in range(len(x)):
            x_fmap = torch.cat((x[i].detach(), x_recon[i].detach()), dim=1) # ch축으로 concat.
            x_fmap = self.convs[i](x_fmap)
            x_fmap = self.avgpool(x_fmap)
            logit_list.append(x_fmap.squeeze(-1).squeeze(-1))
        
        return sum(logit_list)
        # for layer in self.convs:# 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
        #     x_fmap = layer(x_fmap)
        
        # x_fmap = self.avgpool(x_fmap)
        # logit = self.fc(x_fmap.squeeze(-1).squeeze(-1))
        # return logit
    
# 2048:[B,32=ch,13=t,f], f = 102~13, 154~20, 256~32, 256, 257
# 1024:[B,32=ch,25=t,f], f = 51~7, 77~10, 128~16, 128, 129
# 512: [B,32=ch,50=t,f], f = 25~4, 39~5, 64~8, 64, 65
BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
class MRD(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.bands = BANDS
        ch = 32
        self.convs = nn.ModuleList(
            [
                WNConv2d(ch*2, ch, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(ch, ch, (5, 5), (1, 1), padding=(2, 2)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        # self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(ch, 2)
        
        
    def forward(self, x, x_recon): # [B, 32, 13, 102] = [B,c,f,t] 이게 5개 (한 band에 대한 fmap)
        #x_fmap = x_recon
        #x_fmap = x - x_recon
        x_fmap = torch.cat((x, x_recon), dim=1)  # ch축으로 concat.
        
        for layer in self.convs:# 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
            x_fmap = layer(x_fmap)
        
        x_fmap = self.avgpool(x_fmap)
        logit = self.fc(x_fmap.squeeze(-1).squeeze(-1))
        return logit
        
        # logit_list = []    
        # for x_fmap, stack in zip(x, self.band_convs):# 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
        #     for layer in stack:
        #         x_fmap = layer(x_fmap)
        #     x_fmap = self.avgpool(x_fmap)
        #     logit_list.append(self.fc(x_fmap.squeeze(-1).squeeze(-1)))
        
        # return logit_list
    

class Discriminator_fmap(ml.BaseModel):
    def __init__(
        self,
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        n_layer: int = 5
    ):
        super().__init__()
        discs = []
        discs += [MRD() for _ in range(len(fft_sizes) * len(BANDS) * n_layer)]
        self.discriminators_mrd = nn.ModuleList(discs)
        self.weights = nn.Parameter(torch.ones(75))  
        '''
        self.fc = nn.Sequential(nn.Linear(75, 50),
                                nn.LeakyReLU(),
                                nn.Linear(50, 25),
                                nn.LeakyReLU(),
                                nn.Linear(25, 2))
        '''
    
    def classifier(self, logit_list):
        x = torch.stack(logit_list, dim=1)
        w = F.softmax(self.weights, dim=0)
        combined_logit = torch.sum(x * w.view(1,-1,1), dim=1)
        return combined_logit
        '''
        x = torch.cat(logit_list, dim=1)
        out = self.fc(x)
        return out
        '''
    
        # discs = []
        # discs += [MPD() for _ in range(len(periods))]
        # self.discriminators_mpd = nn.ModuleList(discs)  

    
    # def forward(self, x, x_recon):  
    #     logit_list = [d(x, x_recon) for d in self.discriminators]
    #     return logit_list
    


if __name__ == "__main__":
    disc = Discriminator_fmap()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
