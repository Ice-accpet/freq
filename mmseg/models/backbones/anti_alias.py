
"""
anti_alias_block.py
===================
模块功能：Edge-Aware Blur + Learnable Gabor Blur 组合而成的抗混叠模块
用途：用于卷积神经网络中的特征图下采样（如语义分割），以减少频域混叠 aliasing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 辅助函数：构造 1D 高斯核 ------------------------
def _make_1d_gauss(sigma: torch.Tensor, k: int) -> torch.Tensor:
    """
    为每个像素位置的 sigma 构造 1D 高斯核
    sigma: [N,1]，每个像素的模糊强度（标准差）
    k: 核大小（奇数）
    return: [N,k]，每个像素自己的高斯核
    """
    half = k // 2
    x = torch.arange(-half, half + 1, device=sigma.device).view(1, -1)  # 位置坐标
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))                         # 高斯函数
    g = g / g.sum(dim=1, keepdim=True)                                 # 归一化
    return g

# ---------------- Edge-Aware Blur 边缘感知模糊模块 ----------------
class EdgeAwareBlur(nn.Module):
    def __init__(
        self,
        channels: int,
        ksize: int = 5,
        stride: int = 1,
        sigma_min: float = 0.6,
        sigma_max: float = 1.2,
    ):
        """
        参数：
            channels : 输入通道数
            ksize    : 高斯核大小（建议奇数）
            stride   : 下采样步长（默认不采样）
            sigma_min/max : 模糊范围上下限
        """
        super().__init__()
        self.channels = channels
        self.k = ksize
        self.stride = stride
        self.half = ksize // 2

        # σ范围寄存器
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))

        # Sobel 边缘检测卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer("sobel_x", sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: [B,C,H,W]
        输出   : [B,C,H/stride,W/stride]，每个像素根据其边缘强度决定模糊强度
        """
        # 计算 Sobel 边缘图（逐通道）
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)
        edge = (gx ** 2 + gy ** 2).mean(1, keepdim=True)  # [B,1,H,W]
        edge = (edge - edge.amin(dim=(2, 3), keepdim=True)) / (
            edge.amax(dim=(2, 3), keepdim=True) + 1e-6
        )

        # 根据边缘强度映射 σ
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * edge
        B, _, H, W = sigma.shape
        C, k, half = self.channels, self.k, self.half
        sigma_flat = sigma.expand(-1, C, -1, -1).reshape(-1, 1)            # [B*C*H*W,1]

        # --- 生成像素级 1-D 高斯核权 ---------------------
        gauss_flat = _make_1d_gauss(sigma_flat, k)                         # [B*C*H*W,k]
        weight = gauss_flat.view(B*C, H*W, k).permute(0, 2, 1)             # [B*C,k,H*W]

        # ----------------------------------------------
        # 3-A. 横向动态卷积
        # ----------------------------------------------
        patches = F.unfold(x.view(1, B*C, H, W),
                        kernel_size=(1, k), padding=(0, half))          # [1, B*C*k, H*W]
        patches = patches.view(B*C, k, H*W)                                # [B*C,k,H*W]
        out = (patches * weight).sum(dim=1).view(B*C, 1, H, W)             # [B*C,1,H,W]

        # ----------------------------------------------
        # 3-B. 纵向动态卷积（复用同一 weight）
        # ----------------------------------------------
        patches = F.unfold(out, kernel_size=(k, 1), padding=(half, 0))
        patches = patches.view(B*C, k, H*W)                                # [B*C,k,H*W]
        out = (patches * weight).sum(dim=1).view(B, C, H, W)               # [B,C,H,W]

        # ----------------------------------------------
        # 4. 下采样
        # ----------------------------------------------
        return out


# ---------------- Learnable Gabor Blur 模块 ------------------------
class LearnableGaborBlur(nn.Module):
    def __init__(self, channels: int, ksize: int = 7, stride: int = 2):
        """
        每个通道一个 Gabor 核：σ、频率 f、方向 θ 可学习
        """
        super().__init__()
        self.channels = channels
        self.k = ksize
        self.stride = stride

        # 每通道可学习参数（对数形式保证为正）
        self.log_sigma = nn.Parameter(torch.log(torch.ones(channels) * 1.0))
        self.log_freq = nn.Parameter(torch.log(torch.ones(channels) * 0.2))
        self.theta = nn.Parameter(torch.zeros(channels))

        # 构建网格坐标
        half = ksize // 2
        gx, gy = torch.meshgrid(
            torch.arange(-half, half + 1),
            torch.arange(-half, half + 1),
            indexing="ij"
        )
        self.register_buffer("grid", torch.stack([gx, gy], dim=-1).float())  # [k,k,2]

    def _gabor_kernel(self, sigma, freq, theta):
        """
        构建单个 Gabor 卷积核
        """
        rot = torch.stack([
            torch.stack([torch.cos(theta), torch.sin(theta)]),
            torch.stack([-torch.sin(theta), torch.cos(theta)]),
        ])  # [2,2] 旋转矩阵
        xy = (self.grid @ rot.T) / sigma
        gauss = torch.exp(-(xy[..., 0]**2 + xy[..., 1]**2) / 2)
        cos = torch.cos(2 * math.pi * freq * xy[..., 0])
        kern = gauss * cos
        return kern / kern.sum()

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        kernels = []
        for c in range(C):
            σ = self.log_sigma[c].exp()
            f = self.log_freq[c].exp()
            θ = self.theta[c]
            kernels.append(self._gabor_kernel(σ, f, θ).to(device))
        weight = torch.stack(kernels).unsqueeze(1)  # [C,1,k,k]
        x = F.conv2d(x, weight, padding=self.k // 2, groups=C)
        return x
'''class GaborBlurResidual(nn.Module):
    def __init__(self, channels, ksize=7, hp_gain=0.0):
        super().__init__()
        self.k, self.hp_gain = ksize, hp_gain

        # 可学习 σ / f / θ
        self.log_sigma = nn.Parameter(torch.zeros(channels))
        self.log_freq  = nn.Parameter(torch.log(torch.ones(channels) * 0.25))
        self.theta     = nn.Parameter(torch.zeros(channels))

        half = ksize // 2
        gx, gy = torch.meshgrid(torch.arange(-half, half + 1),
                                torch.arange(-half, half + 1), indexing='ij')
        self.register_buffer('grid', torch.stack([gx, gy], -1).float())

        # depth‑wise ↓2，真正的 down‑sample 发生在这里
        self.dw_conv = nn.Conv2d(channels, channels, 3,
                                 stride=1, padding=1, groups=channels, bias=False)
        nn.init.kaiming_normal_(self.dw_conv.weight, mode='fan_out')

        # SE‑gate 控制高频残差
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid())

    # ── 生成单个 Gabor kernel ────────────────────────────
    def _gabor(self, sigma, freq, theta):
        rot = torch.stack([torch.stack([torch.cos(theta),  torch.sin(theta)]),
                           torch.stack([-torch.sin(theta), torch.cos(theta)])])
        xy  = (self.grid @ rot.T) / sigma
        ker = torch.exp(-(xy[..., 0]**2 + xy[..., 1]**2) / 2) \
              * torch.cos(2 * math.pi * freq * xy[..., 0])
        return ker / ker.sum()

    def forward(self, x):
        B, C, H, W = x.shape
        dev = x.device
        # batch‑wise组装 [C,1,k,k] 卷积核
        kernels = [self._gabor(self.log_sigma[c].exp(),
                               self.log_freq[c].exp(),
                               self.theta[c]).to(dev) for c in range(C)]
        weight = torch.stack(kernels).unsqueeze(1)     # [C,1,k,k]

        lp = F.conv2d(x, weight, padding=self.k // 2, groups=C)  # 低通
        hp = x - lp                                            # 高频残差
        gamma = self.gate(x)                                   # 通道注意力

        fused = lp + self.hp_gain * gamma * hp                 # 频率融合
        out   = self.dw_conv(fused)                            # ↓2 真正下采样
        return out'''


# ---------------- 综合模块：EA + Gabor 组合 -------------------------
class AntiAliasBlock(nn.Module):
    def __init__(self, channels: int, ksize: int = 5,gb_ks=7,hp_gain=0.0):
        """
        综合模块：先执行 Edge-Aware 模糊，再 Gabor 模糊 + 下采样
        """
        super().__init__()
        self.lp = EdgeAwareBlur(channels, ksize, stride=1)
        self.gb = LearnableGaborBlur(channels, ksize=gb_ks)

    def forward(self, x):
        return self.gb(x)      # Gabor 滤波 + stride=2 下采样
