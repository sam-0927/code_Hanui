import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# -----------------------
# 유틸: 입력 정리
# -----------------------
def ensure_list_of_clips(x):
    """
    x can be:
      - list of arrays [ (T_i,D), ... ]
      - ndarray of shape (N, T, D)
      - ndarray of shape (total_frames, D) -> treated as single concatenated set
    returns: list_of_clips, emb_dim
    """
    if isinstance(x, list):
        if len(x) == 0:
            raise ValueError("empty list provided")
        D = x[0].shape[1]
        return x, D
    x = np.asarray(x)
    if x.ndim == 3:
        # (N, T, D)
        return [x[i] for i in range(x.shape[0])], x.shape[2]
    elif x.ndim == 2:
        # (total_frames, D) -> treat as one "clip" consisting of all frames
        return [x], x.shape[1]
    else:
        raise ValueError("embeddings must be list or ndarray with ndim 2 or 3")

# -----------------------
# 통계 계산
# -----------------------
def compute_mean_cov(emb, eps=1e-6):
    """
    emb: (T, D) frames x dim
    returns: mu (D,), cov (D,D)
    If T == 1, cov returns zeros + eps*I
    """
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError("emb must be 2D (T, D)")
    T, D = emb.shape
    mu = emb.mean(axis=0)
    if T <= 1:
        cov = np.eye(D) * eps
    else:
        # rowvar=False: variables are columns (features)
        cov = np.cov(emb, rowvar=False, bias=False)  # shape (D,D)
        # numerical stabilize
        cov = cov + np.eye(D) * eps
    return mu, cov

# -----------------------
# Frechet distance (FID style)
# -----------------------
def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    mu*: (D,), cov*: (D,D)
    returns scalar distance
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)
    diff = mu1 - mu2
    # 안정화
    cov1 += np.eye(cov1.shape[0]) * eps
    cov2 += np.eye(cov2.shape[0]) * eps

    # sqrt of matrix product
    cov_prod = cov1.dot(cov2)
    covmean, info = linalg.sqrtm(cov_prod, disp=False)
    if not np.isfinite(covmean).all() or info != 0:
        # try with extra offset if numerical issues
        offset = np.eye(cov1.shape[0]) * (eps * 10)
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # make real if tiny complex residuals
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr = np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean)
    
    return float(diff.dot(diff) + tr)

# -----------------------
# 메인: per-clip FAD 계산
# -----------------------
def per_clip_fad(gt_embeddings, gen_embeddings, eps=1e-6):
    """
    gt_embeddings: list or ndarray (see ensure_list_of_clips)
    gen_embeddings: list or ndarray (same)
    returns: np.array of shape (num_gen_clips,)
    """
    gt_list, D1 = ensure_list_of_clips(gt_embeddings)
    gen_list, D2 = ensure_list_of_clips(gen_embeddings)
    if D1 != D2:
        raise ValueError(f"dim mismatch GT {D1} vs GEN {D2}")

    # GT 전체를 프레임 단위로 concat -> global stats
    all_gt_frames = np.vstack(gt_list)  # shape (sum_T, D)
    mu_gt, cov_gt = compute_mean_cov(all_gt_frames, eps=eps)

    fad_scores = []
    for i, emb in enumerate(gen_list):
        emb = np.asarray(emb)
        if emb.size == 0:
            fad_scores.append(np.nan)
            continue
        mu_g, cov_g = compute_mean_cov(emb, eps=eps)
        fad = frechet_distance(mu_g, cov_g, mu_gt, cov_gt, eps=eps)
        
        fad_scores.append(fad)

    return np.array(fad_scores)

# -----------------------
# 예시 사용법
# -----------------------
if __name__ == "__main__":
    # 예시: GT와 GEN이 npy 파일(각 클립별 (T,D))로 폴더에 있을 때 로드
    def load_npy_folder(folder):
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")])
        # print(files)
        arrs = [np.load(f) for f in files]
        return arrs

    name = "b"
    # 혹은 이미 (N,T,D) ndarray를 가지고 있다면 그걸 바로 넘기면 됨.
    gt_list = load_npy_folder("emb_data/gt_"+name)
    gen_list = load_npy_folder("emb_data/recon_"+name)
    # gt_list = load_npy_folder("emb_data/gt_s")
    # gen_list = load_npy_folder("emb_data/recon_s")
    
    # # 여기서는 데모용 랜덤 생성 (실사용 땐 위 load_npy_folder로 대체)
    # # 가정: 1024 클립, 각 클립 프레임 수는 랜덤(예: 16~64), dim=250
    # rng = np.random.RandomState(0)
    # GT_demo = [rng.randn(rng.randint(16,64), 250) for _ in range(512)]
    # GEN_demo = [rng.randn(rng.randint(16,64), 250) + 0.2 for _ in range(512)]

    fad_scores = per_clip_fad(gt_list, gen_list)
    print("FAD stats: mean {:.4f}, median {:.4f}, std {:.4f}, n={}".format(
        np.nanmean(fad_scores), np.nanmedian(fad_scores), np.nanstd(fad_scores), len(fad_scores)
    ))

    # 히스토그램
    plt.figure(figsize=(6,4))
    plt.hist(fad_scores[~np.isnan(fad_scores)], bins=200)
    plt.xlabel("per-clip FAD")
    plt.ylabel("Count")
    plt.title("Histogram of per-clip FAD")
    plt.tight_layout()
    # plt.show()
    # plt.savefig("fad_emb_spoof.png")
    plt.savefig("fad_emb_"+ name +".png")

    # 필요시 저장
    np.save("440_per_clip_fad_scores_"+ name +".npy", fad_scores)

