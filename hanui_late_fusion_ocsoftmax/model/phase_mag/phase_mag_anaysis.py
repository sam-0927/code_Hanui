import numpy as np
import librosa
import soundfile as sf
from scipy.signal import stft, istft

# 파일 로드
y, sr = librosa.load("gt_1_Validation_490.wav", sr=None)

# STFT
f, t, Zxx = stft(y, fs=sr, nperseg=1024)

# 진폭과 위상 분리
magnitude = np.abs(Zxx)
phase = np.angle(Zxx)

# 1. 원래 위상으로 복원
Z_with_phase = magnitude * np.exp(1j * phase)
_, y_with_phase = istft(Z_with_phase, fs=sr)
sf.write("reconstructed_with_phase.wav", y_with_phase, sr)

# 2. 위상 무시 (0으로 설정)
Z_zero_phase = magnitude
_, y_zero_phase = istft(Z_zero_phase, fs=sr)
sf.write("reconstructed_zero_phase.wav", y_zero_phase, sr)

# 3. 무작위 위상
random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=phase.shape))
Z_random_phase = magnitude * random_phase
_, y_random_phase = istft(Z_random_phase, fs=sr)
sf.write("reconstructed_random_phase.wav", y_random_phase, sr)
