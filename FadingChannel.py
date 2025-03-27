import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: 定義模擬參數
# ---------------------------
v = 30               # 相對速度 (m/s)
fc = 2e9             # 載波頻率 (Hz)
c = 3e8              # 光速 (m/s)
lam = c / fc         # 波長 (m)
fd = v / lam         # 最大多普勒頻率 (Hz)
fs = 2000            # 取樣頻率 (Hz)，須滿足 fs >= 10*fd
T = 1                # 模擬持續時間 (秒)
Ns = int(T * fs)     # 總樣本數

# ---------------------------
# Step 2: 產生頻域複數高斯噪聲並施加赫米特對稱性
# 這裡採用 np.fft.irfft，因此只需產生 [0, fs/2] 頻段的噪聲
# ---------------------------
# 產生頻率向量 (0 到 fs/2)
f = np.fft.rfftfreq(Ns, 1/fs)

# ---------------------------
# Step 3: 定義多普勒 PSD 與濾波器
# 多普勒 PSD 為: S(f) = 1/(π·fd) / sqrt(1 - (f/fd)^2) 當 |f|<=fd，否則 S(f)=0
# 注意：在 f 接近 fd 時分母趨近 0，實際模擬中可利用遮罩 (mask) 限制 f <= fd
# ---------------------------
S = np.zeros_like(f)
mask = f < fd  # 避免 f==fd 造成除零問題
S[mask] = 1.0 / (np.pi * fd) / np.sqrt(1 - (f[mask] / fd)**2)

# 定義濾波器 H(f) = √S(f)
H = np.sqrt(S)

# ---------------------------
# Step 4: 為 I 與 Q 分支產生獨立的複數高斯噪聲並施加濾波器
# ---------------------------
# 產生噪聲序列 (利用 np.fft.irfft 輸入為實頻域數據)
W1 = (np.random.randn(len(f)) + 1j * np.random.randn(len(f))) / np.sqrt(2)
W2 = (np.random.randn(len(f)) + 1j * np.random.randn(len(f))) / np.sqrt(2)

# 施加濾波器 (保持頻域數據的對稱性由 np.fft.irfft 自動處理)
W1_filtered = W1 * H
W2_filtered = W2 * H

# ---------------------------
# Step 5: 轉換至時域 (利用 IFFT)
# ---------------------------
# np.fft.irfft 會返回實值信號，保證 h1 與 h2 為實信號
h1 = np.fft.irfft(W1_filtered, n=Ns)
h2 = np.fft.irfft(W2_filtered, n=Ns)

# ---------------------------
# Step 6: 組合 I 與 Q 分支產生瑞利衰落包絡 r(t)
# ---------------------------
r = np.sqrt(h1**2 + h2**2)

# ---------------------------
# Step 7: 分析與繪圖
# ---------------------------
t = np.arange(Ns) / fs  # 時間軸

plt.figure(figsize=(12, 10))

# (1) 時域包絡圖
plt.subplot(3, 1, 1)
plt.plot(t, r)
plt.title("瑞利衰落包絡 r(t)")
plt.xlabel("時間 (秒)")
plt.ylabel("幅度")

# (2) 頻域分析：估計 r(t) 的功率譜密度 (PSD)
# 這裡使用 FFT 並取平方後歸一化
R = np.fft.fft(r)
freq = np.fft.fftfreq(len(r), 1/fs)
psd = np.abs(R)**2 / len(r)
# 調整頻率順序方便繪圖
plt.subplot(3, 1, 2)
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(psd))
plt.title("r(t) 的估計功率譜密度 (PSD)")
plt.xlabel("頻率 (Hz)")
plt.ylabel("功率")
plt.xlim([-fd*1.5, fd*1.5])  # 聚焦於多普勒頻率附近

# (3) 自相關函數分析：計算 r(t) 的自相關
r_corr = np.correlate(r, r, mode='full')
lags = np.arange(-len(r) + 1, len(r))
r_corr = r_corr / np.max(r_corr)  # 正規化
plt.subplot(3, 1, 3)
plt.plot(lags / fs, r_corr)
plt.title("r(t) 的自相關函數")
plt.xlabel("延遲 (秒)")
plt.ylabel("自相關")
plt.xlim([-0.05, 0.05])  # 顯示小延遲區間

plt.tight_layout()
plt.show()
