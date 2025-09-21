# DAG-FDD용 CVaR Loss 계산 함수
def binary_search_lambda(losses, alpha, max_iters=10):
    left, right = losses.min().item(), losses.max().item()
    for _ in range(max_iters):
        mid = (left + right) / 2
        cvar = mid + (losses[losses > mid] - mid).sum() / (alpha * len(losses))
        if cvar < 0:
            right = mid
        else:
            left = mid
    return (left + right) / 2

def dag_fdd_loss(losses, alpha=0.2):
    lambda_star = binary_search_lambda(losses, alpha)
    mask = (losses > lambda_star).float()
    cvar_loss = lambda_star + (mask * (losses - lambda_star)).sum() / (alpha * len(losses))
    return cvar_loss


import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy 
def plot_melspectrogram(x):
    spectrogram = melspectrogram(x)
    # breakpoint()
    spectrogram = spectrogram.squeeze()
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


import torchaudio.transforms as T
import torch
def melspectrogram(x):
    x_mel = x.mel_spectrogram(
    n_mels=80,
    mel_fmin=0,        
    mel_fmax=8000,   
    )
    log_mel = torch.log(x_mel + 1e-9)
    return log_mel.detach().cpu().numpy()
    
    