import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import ml
from audiotools import STFTParams
from einops import rearrange
from torch.nn.utils import weight_norm

from torchaudio.transforms import Spectrogram
from torchaudio.prototype.transforms import ChromaScale

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
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(2, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 2, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward_logit(self, x, x_recon): # [32,1,6400]
        x = self.pad_to_period(x)   # [32,1,6402]
        x_recon = self.pad_to_period(x_recon)
        
        x = torch.cat((x, x_recon), dim=1)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period) # [32,1,3201,2], period=2 기준.
        
        for layer in self.convs:
            x = layer(x)
        # x:[32,1024,40,2] = [B,ch,t,period]
        x = self.conv_post(x)    # [B,2,40,2]
        x = F.adaptive_avg_pool2d(x, (1,1)) 
        return x.squeeze()

class MSD(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(2, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 2, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward_logit(self, x, x_recon):
        x = AudioSignal(x, self.sample_rate)
        x_recon = AudioSignal(x_recon, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x_recon.resample(self.sample_rate // self.rate)
        x = x.audio_data    # [32,1,6400]
        x_recon = x_recon.audio_data
        
        x = torch.cat((x, x_recon), dim=1)

        for l in self.convs:
            x = l(x)
        # x:[32,1024,25]
        x = self.conv_post(x)   # [32,1,25]
        x = F.adaptive_avg_pool1d(x, 1)  # [32,2,1]
        return x.squeeze()

class MRD_Ch(nn.Module):
    def __init__(
        self,
        window_length: int,
    ):
        super().__init__()

        self.window_length = window_length

        ch = 32
        self.convs = nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        # self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

        self.freq_post = weight_norm(nn.Conv1d(25, 2, kernel_size=3, padding=1))

        self.spec_transform = Spectrogram(n_fft=self.window_length)
        self.chroma_scale = ChromaScale(sample_rate=16000, n_freqs=self.window_length // 2 + 1)
    
    def chroma_spectrogram(self, x):
        spec_x = self.spec_transform(x) # [B, F=12, T]
        chroma_x = self.chroma_scale(spec_x)
        chroma_x = rearrange(chroma_x, "b 1 f t -> b 1 t f")
        return chroma_x

    def forward_logit(self, x, x_recon):
        x_chroma = self.chroma_spectrogram(x)  # [32,1,4,12] = [B, 1, num_frame, F]
        x_chroma_recon = self.chroma_spectrogram(x_recon)

        x_chroma = torch.cat((x_chroma, x_chroma_recon), dim=1)
        
        x = []
        for layer in self.convs:
            x_chroma = layer(x_chroma)
            x.append(x_chroma)
        
        x = torch.cat(x, dim=-1)    # [32,32,4,25]=[B,ch,t,f]:5개 band의 freq 영역 다 concat.
        
        x = self.conv_post(x)       # [32,1,4,25]=[B,ch,t,f]: channel 1로 줄임.
        x = x.squeeze(1)
        
        x = F.adaptive_avg_pool1d(x.transpose(1,2), 1)  # [32,130,1]
        x = self.freq_post(x)
        
        return x.squeeze()



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
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
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
        # self.freq_post_1024 = weight_norm(nn.Conv1d(66, 2, kernel_size=3, padding=1))
        # self.freq_post_512 = weight_norm(nn.Conv1d(34, 2, kernel_size=3, padding=1))
        # self.freq_post_256 = weight_norm(nn.Conv1d(18, 2, kernel_size=3, padding=1))
        # self.freq_post_128 = weight_norm(nn.Conv1d(10, 2, kernel_size=3, padding=1))

        # self.logit_fc = weight_norm(nn.Linear(4, 2, bias=False))

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

    def forward_logit(self, x, x_recon):
        x_bands, _ = self.spectrogram_mag(x)  # [5,32,1,13,102] = [num_band, B, 1, num_frame, band_freq_range_0]
        x_bands_recon, _ = self.spectrogram_mag(x_recon)
        # x_bands = self.spectrogram(x)   # [5,32,2,13,102] = [num_band, B, (real,img), num_frame, band_freq_range_0], [0][0][0][0] 일때 size. 맨 앞이 [1]이면 [32,2,13,154].
        # x_bands_recon = self.spectrogram(x_recon)
        
        x_bands = [torch.cat([x_bands[i], x_bands_recon[i]], dim=1) for i in range(len(x_bands))]
        
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
        # elif x.size(1) == 66:
        #     x = self.freq_post_1024(x)
        # elif x.size(1) == 34:
        #     x = self.freq_post_512(x)
        # elif x.size(1) == 18:
        #     x = self.freq_post_256(x)
        # elif x.size(1) == 10:
        #     x = self.freq_post_128(x)
        else:
            breakpoint()
        return x.squeeze()

    def forward_logit_log(self, x, x_recon):
        _, x_bands_log = self.spectrogram_mag(x)  # [5,32,1,13,102] = [num_band, B, 1, num_frame, band_freq_range_0]
        _, x_bands_recon_log = self.spectrogram_mag(x_recon)
        
        x_bands_log = [torch.cat([x_bands_log[i], x_bands_recon_log[i]], dim=1) for i in range(len(x_bands_log))]
        
        x = []
        for band, stack in zip(x_bands_log, self.band_convs):   # 각 band에 대해 convs를 각 각 적용.= 5x5 = 25
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
        # elif x.size(1) == 66:
        #     x = self.freq_post_1024(x)
        # elif x.size(1) == 34:
        #     x = self.freq_post_512(x)
        # elif x.size(1) == 18:
        #     x = self.freq_post_256(x)
        # elif x.size(1) == 10:
        #     x = self.freq_post_128(x)
        else:
            breakpoint()
        return x.squeeze()

    # def forward_logit_combine(self, x, x_recon):
    #     logit = self.forward_logit(x, x_recon)
    #     logit_log = self.forward_logit_log(x, x_recon)
    #     logit_ = torch.cat((logit, logit_log), dim=-1)
    #     logit_combine = self.logit_fc(logit_)
    #     return logit_combine


class Discriminator_student_2(ml.BaseModel):
    def __init__(
        self,
        rates: list = [1],
        periods: list = [3, 5, 7],
        fft_sizes: list = [4096, 2048], #1024, 512, 256, 128],
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
        # discs += [MPD(p) for p in periods]
        # discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        # discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        # discs += [MRD_Ch(f) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)
        
        self.logit_fc = weight_norm(nn.Linear(4, 2, bias=False))
        self.logit_fc_log = weight_norm(nn.Linear(4, 2, bias=False))
        self.logit_fc_chroma = weight_norm(nn.Linear(4, 2, bias=False))


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
    
    def forward_logit(self, x, x_recon): # [32, 1, 6400]
        x = self.preprocess(x)
        x_recon = self.preprocess(x_recon)
        logits = [d.forward_logit(x, x_recon) for d in self.discriminators]
        return logits

    def forward_logit_log(self, x, x_recon): # [32, 1, 6400]
        x = self.preprocess(x)
        x_recon = self.preprocess(x_recon)
        logits = [d.forward_logit_log(x, x_recon) for d in self.discriminators]
        return logits

    def forward_logit_combine(self, x, x_recon): # [32, 1, 6400]
        x = self.preprocess(x)
        x_recon = self.preprocess(x_recon)
        logits = [d.forward_logit(x, x_recon) for d in self.discriminators]
        # logits_log = [d.forward_logit_log(x, x_recon) for d in self.discriminators[2:4]]
        # logits_chroma = [d.forward_logit(x, x_recon) for d in self.discriminators[2:]]
        
        logits = torch.cat((logits[0], logits[1]),dim=-1)
        # logits_log = torch.cat((logits_log[0], logits_log[1]),dim=-1)
        # logits_chroma = torch.cat((logits_chroma[0], logits_chroma[1]),dim=-1)
        
        logit_combine = self.logit_fc(logits)
        # logit_combine_log = self.logit_fc_log(logits_log)
        # logit_combine_chroma = self.logit_fc_chroma(logits_chroma)
        
        return logit_combine


if __name__ == "__main__":
    disc = Discriminator_student_2()
    x = torch.zeros(1, 1, 44100)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
