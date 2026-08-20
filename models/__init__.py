import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from .baseline_net import baseNet
from compare.cacft_net import CACFTNet
from compare.lite_hcnet import LiteHCNetWrapper

_MODEL_REGISTRY = {
    'baseline': baseNet,
    'cacft': CACFTNet,
    'lite_hcnet': LiteHCNetWrapper,
}

def build_model(model_name, in_channels, num_classes, patch_size=7):
    """
    工厂函数：支持异构参数透传。
    patch_size 对 baseline/cacft 无意义，但由调用方统一传入。
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    
    model_cls = _MODEL_REGISTRY[model_name]
    
    # 根据模型类签名动态选择参数
    if model_name == 'baseline':
        return model_cls(in_channels, num_classes)
    elif model_name == 'cacft':
        return model_cls(in_channels, num_classes)
    elif model_name == 'lite_hcnet':
        return model_cls(in_channels, num_classes, patch_size)
    else:
        # 兜底：尝试全参数透传
        return model_cls(in_channels, num_classes, patch_size)