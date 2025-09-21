import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# VGGish embedding 가져오기 (pip install torchvggish 필요)
import torch
from torchvggish import vggish

# -----------------------------
# 1. FAD 관련 함수
# -----------------------------
def mean_covar(E):
    mu = E.mean(axis=0)
    X = E - mu
    cov = (X.T @ X) / (len(E) - 1)
    return mu, cov

def frechet_distance(mu_r, cov_r, mu_g, cov_g, eps=1e-6):
    cov_r = cov_r + np.eye(cov_r.shape[0]) * eps
    cov_g = cov_g + np.eye(cov_g.shape[0]) * eps
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return diff @ diff + np.trace(cov_r + cov_g - 2 * covmean)

# -----------------------------
# 2. 오디오 → 임베딩
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vggish().to(device) 
model.eval()
model.postprocess = False  # PCA + quantization step 건너뛰기

def audio_to_embedding(path, sr=16000):
    import librosa
    import torch
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    # torchvggish는 보통 raw waveform이 아니라
    # log-mel patch (96x64) 입력을 요구해
    from torchvggish import vggish_input
    examples_batch = vggish_input.waveform_to_examples(y.cpu().numpy(), sr)

    examples_batch = torch.tensor(examples_batch).to(device).float()
    with torch.no_grad():
        emb = model(examples_batch)   # shape: (N, 128)
    return emb.cpu().numpy()

# -----------------------------
# 3. 데이터 경로
# -----------------------------
gt_dir = "../audio_data/gt"       # GT 오디오들
gen_dir = "../audio_data/recon"     # 생성 오디오들
gt_files = sorted(glob.glob(os.path.join(gt_dir, "*b*.wav")))
gen_files = sorted(glob.glob(os.path.join(gen_dir, "*b*.wav")))
assert len(gt_files) == len(gen_files)

# -----------------------------
# 4. 레퍼런스(GT 전체) 분포
# -----------------------------
gt_embeddings = []
for f in gt_files:
    gt_embeddings.append(audio_to_embedding(f))
gt_embeddings = np.vstack(gt_embeddings)
mu_r, cov_r = mean_covar(gt_embeddings)

# -----------------------------
# 5. per-clip FAD 계산
# -----------------------------
fads = []
for f in gen_files:
    emb = audio_to_embedding(f)
    mu_g, cov_g = mean_covar(emb)
    fad_val = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    fads.append(fad_val)

fads = np.array(fads)  # shape (N,)

# -----------------------------
# 6. 히스토그램 출력
# -----------------------------
plt.hist(fads, bins=200)
plt.xlabel("per-clip FAD")
plt.ylabel("Count")
plt.title("Histogram of per-clip FAD")
plt.show()
plt.savefig('bona.png')