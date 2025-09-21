import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
arr1 = np.load('440_per_clip_fad_scores_b.npy')
arr2 = np.load('440_per_clip_fad_scores_s.npy')

# KDE 계산
kde1 = gaussian_kde(arr1, bw_method=0.15)
kde2 = gaussian_kde(arr2, bw_method=0.15)

# x축 범위 설정
x = np.linspace(min(arr1.min(), arr2.min()) - 1, max(arr1.max(), arr2.max()) + 1, 1000)

# KDE 곡선 그리기
plt.plot(x, kde1(x), alpha=0.5, color='blue', label='Bona fide')
plt.plot(x, kde2(x), alpha=0.5, color='red', label='Spoof')

# # 히스토그램 그리기
# plt.hist(arr2, bins=100, alpha=0.5, color='red', label='Spoof')
# plt.hist(arr1, bins=100, alpha=0.5, color='blue', label='Bona fide')


plt.xlabel('per-sample FAD')
plt.ylabel('Count')
plt.title('Histogram of per-sample FAD')
plt.legend()
plt.savefig("two_kde.png")
