import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn as nn
from .baseline_net import baseNet
from compare.cacft_net import CACFTNet
from compare.lite_hcnet import LiteHCNetWrapper
from compare.lssan import LSSAN  # [新增] 导入 LSSAN

class Wrapper5Dto4D(nn.Module):
    """通用降维拦截器：将主干传入的 5D 张量剥离为 4D 供对比网络使用"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        return self.model(x)

_MODEL_REGISTRY = {
    'baseline': baseNet,
    'cacft': CACFTNet,
    'lite_hcnet': LiteHCNetWrapper,
    'lssan': LSSAN,  # [新增] 挂载模型
}

def build_model(model_name, in_channels, num_classes, patch_size=7):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")
        
    model_cls = _MODEL_REGISTRY[model_name]
    
    if model_name == 'baseline':
        return model_cls(in_channels, num_classes)
    elif model_name == 'cacft':
        return Wrapper5Dto4D(model_cls(in_channels, num_classes))
    elif model_name == 'lite_hcnet':
        return model_cls(in_channels, num_classes, patch_size)
    elif model_name == 'lssan':
        # [新增] LSSAN 也是 4D 网络，注入参数并套上 5D 拦截器
        return Wrapper5Dto4D(model_cls(in_channels, num_classes))
    else:
        return model_cls(in_channels, num_classes, patch_size)