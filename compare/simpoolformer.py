import torch
import torch.nn as nn

# [架构修复] 剔除危险的 einops 依赖，使用原生张量算子替代
class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class CreatePatches(nn.Module):
    def __init__(self, channels, embed_dim, patch_size):
        super().__init__()
        self.patch = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.patch(x).flatten(2).transpose(1, 2)

class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.eps = 1e-6  # [架构修复] 解除张量设备绑定，避免 CPU/GPU 同步崩溃

    def prepare_input(self, x):
        gap_cls = x.mean(-2).unsqueeze(1) 
        return gap_cls, x

    def forward(self, x):
        gap_cls, x = self.prepare_input(x)
        q, k, v = gap_cls, self.norm_patches(x), self.norm_patches(x)
        B, Nq, dq = q.shape
        _, Nk, dk = k.shape
        _, Nv, dv = v.shape

        qq = self.wq(q).reshape(B, Nq, self.num_heads, dq // self.num_heads).permute(0, 2, 1, 3)
        kk = self.wk(k).reshape(B, Nk, self.num_heads, dk // self.num_heads).permute(0, 2, 1, 3)
        vv = v.reshape(B, Nv, self.num_heads, dv // self.num_heads).permute(0, 2, 1, 3)

        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ vv).transpose(1, 2).reshape(B, Nq, dq)
        
        # [架构修复] 移除原代码错误的 squeeze()，保持 (B, 1, d) 进行安全广播
        return x

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
    def forward(self, x):
        return x * self.alpha + self.beta

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):
    def __init__(self, dim, num_patch, mlp_dim, dropout=0.0, init_values=1e-4):
        super().__init__()
        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            TransposeLayer(1, 2),
            nn.Linear(num_patch, num_patch),
            TransposeLayer(1, 2),
        )
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x

class ResMLP(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim):
        super().__init__()
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.mlp_blocks = nn.ModuleList([MLPblock(dim, self.num_patch, mlp_dim) for _ in range(depth)])
        self.affine = Aff(dim)
        self.mlp_head = nn.Sequential(nn.Linear(dim, mlp_dim))
    def forward(self, x):
        x = self.to_patch_embedding(x).flatten(2).transpose(1, 2)
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)
        x = self.affine(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.simpool = SimPool(embed_dim, num_heads=1)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.pre_norm(x)
        # [架构修复] 去除 [0] 错误索引，避免破坏整个 Batch 的独立性
        x = x + self.simpool(x_norm)
        x = x + self.MLP(self.norm(x))
        return x

class SimPoolFormer(nn.Module):
    """规范化入口：对接你工程管道的输入参数"""
    def __init__(self, in_channels, num_classes, img_size):
        super().__init__()
        
        # 强制将内部 token 划分力度设为 1，完美吸收不同数据集传来的任意 patch_size (如 7, 9)
        inner_patch_size = 1
        embed_dim = 256
        hidden_dim = 128
        num_heads = 4
        num_layers = 4
        dropout = 0.0
        depth = 4
        mlp_dim = 256

        num_patches = (img_size // inner_patch_size) ** 2
        self.patches = CreatePatches(channels=in_channels, embed_dim=embed_dim, patch_size=inner_patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.attn_layers = nn.ModuleList([
            AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.resmlp = ResMLP(in_channels, embed_dim, num_classes, inner_patch_size, img_size, depth, mlp_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x1 = self.resmlp(x)
        x = self.patches(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        x = x + x1
        return self.head(x)