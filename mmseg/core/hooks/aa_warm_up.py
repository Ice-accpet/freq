# aa_warmup_hook.py  ───────────────────────────────────────────
import math
from mmcv.runner import HOOKS, Hook
from mmseg.models.backbones.anti_alias2 import AntiAliasBlockDyn   # 你的模块

@HOOKS.register_module()
class AntiAliasWarmUpHook(Hook):
    """
    在训练前若干 iter 内把
        - gb.hp_gain
        - gb.gate.bias
        - lp.mask_gen.bias
    从 0 → 目标值（余弦 / 线性）
    """
    def __init__(self,
                 total_iters: int = 5000,       # warm‑up 长度
                 final_hp_gain: float = 0.5,    # 训练后期想要的 hp_gain
                 mode: str = 'cos'):            # 'linear' or 'cos'
        self.total_iters   = total_iters
        self.final_hp_gain = final_hp_gain
        self.mode          = mode

    # 每次迭代之后调用
    def after_train_iter(self, runner):
        cur_iter = runner.iter
        if cur_iter > self.total_iters:
            return                              # warm‑up 结束，直接退出

        # 计算系数 p∈[0,1]
        if self.mode == 'linear':
            p = cur_iter / self.total_iters
        else:                                   # 余弦更平滑
            p = 0.5 - 0.5 * math.cos(math.pi * cur_iter / self.total_iters)

        model = runner.model
        for m in model.modules():
            if isinstance(m, AntiAliasBlockDyn):
                # ① hp_gain
                m.gb.hp_gain = self.final_hp_gain * p

                conv = m.gb.gate[1]                 # ← 改这里
                conv.bias.data.fill_(-10 * (1 - p))
                # ② gate.bias  :  从 ‑10 → 0   (Sigmoid: 0 → 0.5)

                # ③ lp.mask_gen.bias:  从 0 → 原值
                m.lp.mask_gen.bias.data.mul_(p)
