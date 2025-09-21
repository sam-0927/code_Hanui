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


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
class MRD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
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

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(1, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

        self.freq_post_4096 = weight_norm(nn.Conv1d(258, 2, kernel_size=3, padding=1))
        self.freq_post_2048 = weight_norm(nn.Conv1d(130, 2, kernel_size=3, padding=1))

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())    # [32,1,1025,13]=[B,1,fft,frame_num] -> [...,2]=[...,[real,img]]
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")    # [32,2,13,1025]
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def spectrogram_mag(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x.stft()
        x_mag = x.magnitude # [32,1,1025,13]
        x_mag_log = x.magnitude.clamp(1e-5).pow(2).log10()
        x_mag = rearrange(x_mag, "b 1 f t -> b 1 t f")
        x_mag_log = rearrange(x_mag_log, "b 1 f t -> b 1 t f")
        
        # Split into bands
        x_bands_mag = [x_mag[..., b[0] : b[1]] for b in self.bands]
        x_bands_mag_log = [x_mag_log[..., b[0] : b[1]] for b in self.bands]
        return x_bands_mag, x_bands_mag_log

    def forward(self, x):
        x_bands = self.spectrogram(x)
        
        fmap = []
        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        
        x = torch.cat(x, dim=-1)    
        x = self.conv_post(x)       
        fmap.append(x)

        return fmap

    def forward_logit(self, x):
        # x_bands = self.spectrogram(x)   # [5,32,2,13,102] = [num_band, B, (real,img), num_frame, band_freq_range_0], [0][0][0][0] 일때 size. 맨 앞이 [1]이면 [32,2,13,154].
        x_bands, _ = self.spectrogram_mag(x)
        
        x = []
        for band, stack in zip(x_bands, self.band_convs):   # 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
            for layer in stack:
                band = layer(band)
            x.append(band)
        # x[0]=[32,32,13,13]=[B,ch,t,f], 5개의 band에 대해 마지막 f가 다름->(13,20,32,32,33)
        x = torch.cat(x, dim=-1)    # [32,32,13,130]=[B,ch,t,f]:5개 band의 freq 영역 다 concat.
        x = self.conv_post(x)       # [32, 1,13,130]=[B,ch,t,f]: channel 1로 줄임.
        x = x.squeeze(1)
        x = F.adaptive_avg_pool1d(x.transpose(1,2), 1)  # [32,130,1]
        if x.size(1) == 258:
            x = self.freq_post_4096(x)
        elif x.size(1) == 130:
            x = self.freq_post_2048(x)
        else:
            breakpoint()
        return x.squeeze()


class Discriminator_student(ml.BaseModel):
    def __init__(
        self,
        rates: list = [1],
        periods: list = [3, 5, 7],
        fft_sizes: list = [4096, 2048],#, 512, 256, 128],
        sample_rate: int = 16000,
        bands: list = BANDS,
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

        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)
        self.logit_fc = weight_norm(nn.Linear(4, 2, bias=False))
        self.logit_weight = nn.Parameter(torch.ones(2))

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
        if len(x.size()) == 3 and x.size(1) != 1:
            x = x.permute(1,0,2)
        x = self.preprocess(x)
        logits = [d.forward_logit(x) for d in self.discriminators]
        '''
        logits = torch.stack((logits[0], logits[1]),dim=1)
        if len(logits.size()) == 2:
            logits = logits.unsqueeze(0)
        
        weight = F.softmax(self.logit_weight, dim=0)
        logit_combine = torch.sum(logits * weight.view(1, -1, 1), dim=1)
        '''
        logits = torch.cat((logits[0], logits[1]),dim=-1)
        logit_combine = self.logit_fc(logits)
        return logit_combine


if __name__ == "__main__":
    disc = Discriminator_student()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
