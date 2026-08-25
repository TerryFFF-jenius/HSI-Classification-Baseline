import torch
from functools import partial  # [新增] 加上这一行
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


# ==================== 原始 GSCViT 组件（学术逻辑零修改）====================

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class SpectralCalibration(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GSC(nn.Module):
    def __init__(self, dim_in, dim_out, padding=1, num_groups=8):
        super().__init__()
        self.gpwc = nn.Conv2d(dim_in, dim_out, groups=num_groups, kernel_size=1)
        self.gc = nn.Conv2d(dim_out, dim_out, kernel_size=3, groups=num_groups, padding=padding, stride=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.gc(self.gpwc(x))))


class GSSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0., group_spatial_size=3):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.group_spatial_size = group_spatial_size
        inner_dim = dim_head * heads

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.group_tokens = nn.Parameter(torch.randn(dim))
        self.group_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h=heads),
        )
        self.group_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, height, width, heads, gss = x.shape[0], *x.shape[-2:], self.heads, self.group_spatial_size
        assert (height % gss) == 0 and (width % gss) == 0, \
            f'height {height} and width {width} must be divisible by group spatial size {gss}'
        num_groups = (height // gss) * (width // gss)

        x = rearrange(x, 'b c (h g1) (w g2) -> (b h w) c (g1 g2)', g1=gss, g2=gss)
        w = repeat(self.group_tokens, 'c -> b c 1', b=x.shape[0])
        x = torch.cat((w, x), dim=-1)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))
        q = q * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        group_tokens, grouped_fmaps = out[:, :, 0], out[:, :, 1:]

        if num_groups == 1:
            fmap = rearrange(grouped_fmaps, '(b x y) h (g1 g2) d -> b (h d) (x g1) (y g2)',
                             x=height // gss, y=width // gss, g1=gss, g2=gss)
            return self.to_out(fmap)

        group_tokens = rearrange(group_tokens, '(b x y) h d -> b h (x y) d',
                                 x=height // gss, y=width // gss)
        grouped_fmaps = rearrange(grouped_fmaps, '(b x y) h n d -> b h (x y) n d',
                                  x=height // gss, y=width // gss)
        w_q, w_k = self.group_tokens_to_qk(group_tokens).chunk(2, dim=-1)
        w_q = w_q * self.scale
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)
        w_attn = self.group_attend(w_dots)
        aggregated_grouped_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, grouped_fmaps)

        fmap = rearrange(aggregated_grouped_fmap, 'b h (x y) (g1 g2) d -> b (h d) (x g1) (y g2)',
                         x=height // gss, y=width // gss, g1=gss, g2=gss)
        return self.to_out(fmap)


class Transformer(nn.Module):
    def __init__(self, dim, depth, dim_head=16, heads=8, dropout=0.,
                 norm_output=True, groupsize=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PreNorm(dim, GSSA(dim, group_spatial_size=groupsize, heads=heads,
                                  dim_head=dim_head, dropout=dropout))
            )
        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return self.norm(x)


class GSCViT(nn.Module):
    def __init__(self, *, num_classes, depth, heads, group_spatial_size,
                 channels=200, dropout=0.1, padding, dims=(256, 128, 64, 32),
                 num_groups=[16, 16, 16]):
        super().__init__()
        num_stages = len(depth)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        hyperparams_per_stage = [heads]
        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))
        
        self.sc = SpectralCalibration(channels, dims[0])
        self.bn_1 = nn.BatchNorm2d(dims[0])
        self.relu_1 = nn.ReLU(inplace=True)

        self.layers_trans = nn.ModuleList([])
        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_heads, p, num_group) in enumerate(
                zip(dim_pairs, depth, *hyperparams_per_stage, padding, num_groups)):
            is_last = ind == (num_stages - 1)
            self.layers_trans.append(nn.ModuleList([
                GSC(layer_dim_in, layer_dim, p, num_group),
                Transformer(dim=int(layer_dim), depth=layer_depth, heads=layer_heads,
                            groupsize=group_spatial_size[ind], dropout=dropout,
                            norm_output=not is_last),
                nn.BatchNorm2d(layer_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer_dim, layer_dim, 1)
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        # 原生 5D 兼容: (B, 1, C, H, W) → (B, C, H, W)
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.sc(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        for peg, transformer, bn, relu, pw in self.layers_trans:
            x = peg(x)
            y = x
            x = transformer(x)
            x = pw(x) + y
            x = bn(x)
            x = relu(x)
        return self.mlp_head(x)


# ==================== 统一接口包装类（新增）====================

class GSCViTWrapper(nn.Module):
    """
    GSCViT 的物理级拦截包装类。
    处理 patch_size 与 group_spatial_size 的兼容性，维护学术等价性。
    """
    def __init__(self, in_channels, num_classes, patch_size):
        super().__init__()
        
        # [关键修复] 保持原设计的 group_spatial_size=3，不阉割组间注意力
        # 若 patch_size 不是 3 的倍数，前置 AdaptiveAvgPool2d 进行适配
        self.needs_resize = (patch_size % 3 != 0)
        if self.needs_resize:
            # 计算最近的 3 的倍数（向上取整）
            self.target_size = ((patch_size + 2) // 3) * 3
            self.resize = nn.AdaptiveAvgPool2d((self.target_size, self.target_size))
        else:
            self.target_size = patch_size
            self.resize = nn.Identity()
        
        # 计算 stages: target_size 经过 3 层 GSC 后保持 spatial 不变（stride=1, padding=1）
        # 但 SpectralCalibration 不改变 spatial
        # 因此 group_spatial_size 各 stage 均为 3
        num_stages = 3  # 与 dims 长度匹配
        group_spatial_size = [3] * num_stages
        
        self.net = GSCViT(
            num_classes=num_classes,
            channels=in_channels,
            depth=(1, 1, 1),
            heads=(1, 1, 1),
            group_spatial_size=group_spatial_size,
            dropout=0.1,
            padding=[1, 1, 1],
            dims=(256, 128, 64),
            num_groups=[16, 16, 16]
        )

    def forward(self, x):
        # 5D → 4D
        if x.dim() == 5:
            x = x.squeeze(1)
        # 若需要，调整空间尺寸为 3 的倍数
        x = self.resize(x)
        return self.net(x)