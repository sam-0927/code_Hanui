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


# BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
class MRD(nn.Module):
    def __init__(
        self,
        # window_length: int,
        # hop_factor: float = 0.25,
        # sample_rate: int = 44100,
        # bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()
        # self.window_length = window_length
        # self.hop_factor = hop_factor
        # self.sample_rate = sample_rate
        # self.stft_params = STFTParams(
        #     window_length=window_length,
        #     hop_length=int(window_length * hop_factor),
        #     match_stride=True,
        # )

        # n_fft = window_length // 2 + 1
        # bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        # self.bands = bands

        ch = 64
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)
        self.freq_post = weight_norm(nn.Conv1d(128, 2, kernel_size=3, padding=1))
        

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())    # [32,1,1025,13]=[B,1,fft,frame_num] -> [...,2]=[...,[real,img]]
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")    # [32,2,13,1025]
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)# [5,32,2,13,102] = [num_band, B, (real,img), num_frame, band_freq_range_0], [0][0][0][0] 일때 size. 맨 앞이 [1]이면 [32,2,13,154].
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):# 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        # x[0]=[32,32,13,13]=[B,ch,t,f], 5개의 band에 대해 마지막 f가 다름->(13,20,32,32,33)
        x = torch.cat(x, dim=-1)    # [32,32,13,130]=[B,ch,t,f]:5개 band의 freq 영역 다 concat.
        x = self.conv_post(x)       # [32, 1,13,130]=[B,ch,t,f]: channel 1로 줄임.
        fmap.append(x)
        return fmap

    def forward_logit(self, x): # [B, 1024, 25] = [B,f,t]
        x = x.transpose(1,2).unsqueeze(1)
        for layer in self.convs:
            x = layer(x)
        
        x = self.conv_post(x)       # [32,64,25,128]->[32,1,25,128]=[B,ch,t,f]: channel 1로 줄임.
        x = x.squeeze(1)
        x = F.adaptive_avg_pool1d(x.transpose(1,2), 1)  # [32,128,1]
        x = self.freq_post(x)
        return x.squeeze()


class Discriminator_latent(ml.BaseModel):
    def __init__(
        self,
        # fft_sizes: list = [2048, 1024, 512, 256, 128],
        # sample_rate: int = 16000
        # bands: list = BANDS,
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        # discs += [MPD(p) for p in periods]
        # discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        # discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        # self.discriminators = nn.ModuleList(discs)
        self.discriminators = MRD()
        

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):  
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps
    
    def forward_logit(self, x): # [32, 1, 6400]
        # if len(x.size()) == 3 and x.size(1) != 1:
        #     x = x.permute(1,0,2)
        # x = self.preprocess(x)
        logits = self.discriminators.forward_logit(x)
        # logits = [d.forward_logit(x) for d in self.discriminators]
        return logits


if __name__ == "__main__":
    disc = Discriminator_latent()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
