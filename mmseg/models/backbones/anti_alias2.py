# aa_dyn_carafe.py  ──────────────────────────────────────────────────────
import math, torch, torch.nn as nn, torch.nn.functional as F

# ─────────────────── ① 先尝试导入 CUDA 版 CARAFE ───────────────────────
try:
    from mmcv.ops import carafe, normal_init, xavier_init   # mmcv-full 装好时走这里
except (ImportError, ModuleNotFoundError):
    # ───────────── ② 若 mmcv-full 不存在，就用纯 PyTorch 后备实现 ──────
    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    # -------- 完整 CARAFE fallback：支持 up / group，维度已对齐 ----------
    def carafe(x, normed_mask, kernel_size, group=1, up=1):
        """
        x           : [B, C,  H,  W]
        normed_mask : [B, k²·g·up², H, W]
        returns     : [B, C,  H·up, W·up]
        """
        b, c, h, w   = x.shape
        k2           = kernel_size * kernel_size
        pad          = kernel_size // 2
        assert normed_mask.shape[1] == k2 * group * up * up
        assert normed_mask.shape[2:] == (h, w)

        # 1) unfold
        x_pad  = F.pad(x, [pad]*4, mode='reflect')
        unfold = F.unfold(x_pad, kernel_size, stride=1).view(b, c, k2, h, w)

        # 2) 最近邻上采
        if up > 1:
            unfold = F.interpolate(unfold.flatten(1,2), scale_factor=up, mode='nearest'
                                   ).view(b, c, k2, h*up, w*up)

        # 3) group reshape & 加权求和
        unfold = unfold.view(b, group, c//group, k2, h*up, w*up)      # B,g,cg,k²,H',W'
        mask   = normed_mask.view(b, group, k2, up*up, h, w)          # B,g,k²,up²,H,W
        mask   = F.pixel_shuffle(mask, up).view(b, group, k2, h*up, w*up)
        out = (unfold * mask.unsqueeze(2)).sum(3).view(b, c, h*up, w*up)
        return out
    
def init_mask_gen_as_lp(mask_gen, k=5, group=1, window='hamming'):
        """
        将 DynamicLP.mask_gen 的卷积参数初始化成 k×k 低通核:
        - 全 0 权重                         —— 不依赖邻域内容
        - bias = log(目标核)               —— 经过 softmax 即得目标权重
        """
        # 1) 生成目标核（均匀 / Hamming / 高斯）
        if window == 'hamming':
            w1d = torch.hamming_window(k, periodic=False)      # [k]
            kernel = torch.outer(w1d, w1d)                     # [k,k]
        elif window == 'gauss':
            ax = torch.arange(-(k-1)//2, (k+1)//2, dtype=torch.float32)
            kernel = torch.exp(-(ax[:,None]**2 + ax[None,:]**2)/(2*(k/6)**2))
        else:  # uniform
            kernel = torch.ones(k, k)

        kernel = kernel / kernel.sum()                         # 归一化
        kernel = kernel.flatten()                              # k²
        log_k  = kernel.log()                                  # log(p_i)

        # 2) 对每个 group 重复 log_k 作为 bias
        bias_init = log_k.repeat(group)                        # (k²g,)

        # 3) 赋值
        with torch.no_grad():
            nn.init.constant_(mask_gen.weight, 0.)             # 权重全 0
            mask_gen.bias.copy_(bias_init)
# ────────────────────────────── Dynamic soft-mask Low-Pass ─────────────
# aa_dyn_carafe.py  ─────────────────────────────────────────────

# … try / except 导入 CARAFE 保持不变 …

# ───────────────────── Dynamic soft-mask Low-Pass ─────────────
class DynamicLP(nn.Module):
    def __init__(self, in_ch, ksize=5, group=1):
        super().__init__()
        self.ksize, self.group = ksize, group
        self.mask_gen = nn.Conv2d(in_ch, ksize*ksize*group, 3, padding=1)
        normal_init(self.mask_gen, std=0.001)

    def forward(self, x):
        mask = F.softmax(self.mask_gen(x), 1)
        # +x 做 skip，保梯度 & Identity warm‑start
        return carafe(x, mask, self.ksize, self.group, up=1) + x

# ───────────────────── Gabor  + High‑Pass residual ────────────
class GaborBlurResidual(nn.Module):
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
        return out

# ───────────────────── 集成 Block ─────────────────────────────
class AntiAliasBlockDyn(nn.Module):
    """
    Dynamic low‑pass  →  Gabor + gated HP  →  depth‑wise ↓2
    """
    def __init__(self, channels, lp_ks=5, gb_ks=7,
                 hp_gain=0.0, warm_start=True):
        super().__init__()
        self.lp = DynamicLP(channels, ksize=lp_ks)
        init_mask_gen_as_lp(self.lp.mask_gen, k=lp_ks, window='hamming')

        self.gb = GaborBlurResidual(channels, ksize=gb_ks,
                                    hp_gain=hp_gain)

        if warm_start:
            self._init_identity()

    # Identity warm‑start：低通＝均值，高通残差关掉
    def _init_identity(self):
        nn.init.constant_(self.lp.mask_gen.weight, 0.)
        conv = self.gb.gate[1]           # Conv1×1
        nn.init.constant_(conv.weight, 0.)
        nn.init.constant_(conv.bias,  -10.)    # Sigmoid(-10)≈0
        self.gb.hp_gain = 0.0

    def forward(self, x):
        return self.gb(self.lp(x))
