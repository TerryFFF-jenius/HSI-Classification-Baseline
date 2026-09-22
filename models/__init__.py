import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn as nn
from .baseline_net import baseNet
from compare.cacft_net import CACFTNet
from compare.lite_hcnet import LiteHCNetWrapper
from compare.lssan import LSSAN
from compare.msdan import MSDAN
from compare.simpoolformer import SimPoolFormer
from compare.gscvit import GSCViTWrapper, GSCViTTSSRWrapper  # [新增]
from compare.spectralformer import SpectralFormerWrapper  # [新增]
from compare.ssftt import SSFTTWrapper  # [新增]

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
    'lssan': LSSAN,
    'msdan': MSDAN,
    'simpoolformer': SimPoolFormer,
    'gscvit': GSCViTWrapper,  # [新增]
    'gscvit_tssr': GSCViTTSSRWrapper,
    'spectralformer': SpectralFormerWrapper,  # [新增]
    'ssftt': SSFTTWrapper,  # [新增]
}

def build_model(model_name, in_channels, num_classes, patch_size=7,
                spectral_groups=8, route_strength=0.5,
                route_temperature=1.0):
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
        return Wrapper5Dto4D(model_cls(in_channels, num_classes))
    elif model_name == 'msdan':
        return model_cls(in_channels, num_classes, patch_size)
    elif model_name == 'simpoolformer':
        return Wrapper5Dto4D(model_cls(in_channels, num_classes, patch_size))
    elif model_name == 'gscvit':
        return model_cls(in_channels, num_classes, patch_size)
    elif model_name == 'gscvit_tssr':
        return model_cls(in_channels, num_classes, patch_size,
                         spectral_groups=spectral_groups,
                         route_strength=route_strength,
                         route_temperature=route_temperature)
    elif model_name == 'spectralformer':      # [新增]
        return model_cls(in_channels, num_classes, patch_size)
    elif model_name == 'ssftt':               # [新增]
        return model_cls(in_channels, num_classes, patch_size)
    else:
        return model_cls(in_channels, num_classes, patch_size)
