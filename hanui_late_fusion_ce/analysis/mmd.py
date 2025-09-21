import torch
import matplotlib.pyplot as plt

def kernel_matrix(x, y, sigma=1.0):
    # x, y shape: [n]
    x = x.unsqueeze(1)  # [n, 1]
    y = y.unsqueeze(0)  # [1, n]
    dist = (x - y) ** 2
    K = torch.exp(-dist / (2 * sigma ** 2))  # [n, n]
    return K

# # 예시: ch=2, f=3, t=4 → flat size n=24
# x = torch.randn(2, 3, 4).flatten()
# y = torch.randn(2, 3, 4).flatten()

# K = kernel_matrix(x, y, sigma=1.0)

def draw_graph(K, title="mmd"):
    plt.figure(figsize=(6, 5))
    plt.imshow(K, cmap="viridis", origin="lower")
    plt.colorbar(label="Kernel Value")
    plt.title("Kernel Feature Map (n x n)")
    plt.xlabel("y-dim index")
    plt.ylabel("x-dim index")
    # plt.show()
    plt.savefig(title + ".png")

def draw_mmd(x, y, sigma=1.0):
    x = x.flatten()
    y = y.flatten()
    Kxx = kernel_matrix(x, x, sigma)
    Kyy = kernel_matrix(y, y, sigma)
    Kxy = kernel_matrix(x, y, sigma)
    return Kxx + Kyy - 2 * Kxy

    # mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    # return mmd

