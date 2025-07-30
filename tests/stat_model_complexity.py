#!/usr/bin/env python
# stat_my_freqfusion.py
# 依赖：mmcv-full>=2.0  mmengine  mmsegmentation>=1.0  ptflops

import os
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS
from ptflops import get_model_complexity_info

from mmseg.models import build_segmentor
from mmcv import Config
# --------------------------------------------------
# ★ 1. 你的文件路径（已写死） ★
# --------------------------------------------------
CFG_FILE  = '/root/shared-nvme/ac/FreqFusion-main/FreqFusion-main/SegNeXt/local_configs/segnext/tiny/segnext.tiny.freqfusion.512x512.ade.160k.py'
CKPT_FILE = '/root/shared-nvme/ac/FreqFusion-main/FreqFusion-main/SegNeXt/work_dirs/segnext.tiny.freqfusion.512x512.ade.160k/freq_best_43.4.pth'
INPUT_SIZE = (3, 512, 512)   # C, H, W
device = torch.device('cuda:0') 

cfg = Config.fromfile(CFG_FILE)

# 旧版 cfg 里有 `pretrained` 字段时置空，避免下载
if cfg.model.get('pretrained', None):
    cfg.model.pretrained = None
if cfg.model.get('backbone', {}).get('pretrained', None):
    cfg.model.backbone.pretrained = None

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = torch.load(CKPT_FILE, map_location='cpu')
model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
assert hasattr(model, 'forward_dummy'), '模型无 forward_dummy() 方法'
model.forward = model.forward_dummy 
model = model.to(device).eval() 

# --------------------------------------------------
# 3. 统计 Params & FLOPs
# --------------------------------------------------
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model,
        input_res=INPUT_SIZE,   
        as_strings=False, # (3,512,512)
        print_per_layer_stat=False,
        verbose=False
    )

flops = macs * 2                    # 1 MAC = 2 FLOPs

print('=' * 70)
print(f'Config : {os.path.basename(CFG_FILE)}')
print(f'Weights: {os.path.basename(CKPT_FILE)}')
print(f'Input  : {INPUT_SIZE}')
print(f'Params : {params / 1e6:.2f} M')
print(f'FLOPs  : {flops  / 1e9:.2f} G')
print('=' * 70)
