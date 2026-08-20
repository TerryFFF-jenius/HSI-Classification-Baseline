import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 下面是你原本的代码
# from .baseline_net import baseNet ...
from .baseline_net import baseNet
from compare.cacft_net import CACFTNet

_MODEL_REGISTRY = {
    'baseline': baseNet,
    'cacft': CACFTNet,
}

def build_model(model_name, in_channels, num_classes):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")
    return _MODEL_REGISTRY[model_name](in_channels, num_classes)