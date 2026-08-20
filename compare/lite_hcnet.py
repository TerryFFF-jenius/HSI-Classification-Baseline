import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ==================== 原始 Lite-HCNet 组件（学术逻辑零修改）====================

def bsm(n, d):
    a = [[0] * n for _ in range(n)]
    p = 0
    q = n - 1
    w = int((n + 1) / 2)
    t = 0
    while p < d:
        for i in range(p, q):
            a[p][i] = t
        for i in range(p, q):
            a[i][q] = t
        for i in range(q, p, -1):
            a[q][i] = t
        for i in range(q, p, -1):
            a[i][p] = t
        p += 1
        q -= 1
    while p == d or p > d and p < q:
        for i in range(p, q):
            a[p][i] = 1
        for i in range(p, q):
            a[i][q] = 1
        for i in range(q, p, -1):
            a[q][i] = 1
        for i in range(q, p, -1):
            a[i][p] = 1
        a[w - 1][w - 1] = 1
        p += 1
        q -= 1
    return np.array(a, dtype=np.float32)


class ScaleMaskModule(nn.Module):
    """尺度掩膜：预计算 bsm 矩阵为 buffer，消除每轮 CPU→GPU 同步"""
    def __init__(self, d, patch_size):
        super(ScaleMaskModule, self).__init__()
        self.d = d
        # 预计算 mask 并注册为 persistent buffer
        mask = bsm(patch_size, d)  # (patch_size, patch_size)
        self.register_buffer('mask', torch.from_numpy(mask))

    def forward(self, x):
        # x: (B, C, H, W)，其中 H=W=patch_size
        p, o = x.shape[0], x.shape[1]
        # 广播扩展: (patch_size, patch_size) → (P, O, patch_size, patch_size)
        out = self.mask.unsqueeze(0).unsqueeze(0).expand(p, o, -1, -1)
        return x * out


class NCAM3D(nn.Module):
    def __init__(self, c, patch_size):
        super(NCAM3D, self).__init__()
        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c, 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size - 1) // 2 - 1, patch_size)
        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21),
                                padding=(0, (kernel_size_21 - 1) // 2), dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21),
                                  padding=(0, (kernel_size_21 - 1) // 2), dilation=1)

    def forward(self, x):
        out = x
        out_1 = out.shape[1]  # 保存原始通道维度（如 ax）
        out_2 = out.shape[2]  # 保存原始光谱维度（如 C）
        # 降维至 4D 供 2D 池化/卷积处理: (B, ax, C, H, W) → (B, ax*C, H, W)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])

        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x).reshape(out_x.shape[0], -1)
        out_x2 = torch.flip(out_x1, dims=[1])  # 倒序，替代 reversed()
        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])
        out_xx = torch.cat([out_x1, out_x2], dim=2)

        out1 = self.avg_pool(out).reshape(out.shape[0], -1)
        out2 = torch.flip(out1, dims=[1])
        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])
        outx = torch.cat([out1, out2], dim=2)

        at1 = torch.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * \
              torch.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)
        at = torch.sigmoid((at1 - 0.2) * 2)
        out = out * at
        # 还原 5D: (B, ax*C, H, W) → (B, ax, C, H, W)
        out = out.reshape(out.shape[0], out_1, out_2, out.shape[2], out.shape[3])
        return out


class NCAM2D(nn.Module):
    def __init__(self, c, patch_size):
        super(NCAM2D, self).__init__()
        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c, 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size - 1) // 2 - 1, patch_size)
        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21),
                                padding=(0, (kernel_size_21 - 1) // 2), dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21),
                                  padding=(0, (kernel_size_21 - 1) // 2), dilation=1)

    def forward(self, x):
        out = x
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x).reshape(out_x.shape[0], -1)
        out_x2 = torch.flip(out_x1, dims=[1])
        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])
        out_xx = torch.cat([out_x1, out_x2], dim=2)

        out1 = self.avg_pool(out).reshape(out.shape[0], -1)
        out2 = torch.flip(out1, dims=[1])
        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])
        outx = torch.cat([out1, out2], dim=2)

        at1 = torch.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * \
              torch.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)
        at = torch.sigmoid((at1 - 0.2) * 2)
        out = out * at
        return out


class LE_DSC3D(nn.Module):
    def __init__(self, nin, nout, kernel_size_c, kernel_size_h, kernel_size_w,
                 pca_components, patch_size, padding=True):
        super(LE_DSC3D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM3D(self.nin * pca_components, patch_size)
        self.at2 = NCAM3D(self.nout * pca_components, patch_size)

        if padding:
            self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, 1, kernel_size_w),
                                       padding=((kernel_size_c - 1) // 2, 0, (kernel_size_w - 1) // 2), groups=nin)
            self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, kernel_size_h, 1),
                                        padding=((kernel_size_c - 1) // 2, (kernel_size_h - 1) // 2, 0), groups=nin)
            self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1, kernel_size_h, kernel_size_w),
                                        padding=(0, (kernel_size_h - 1) // 2, (kernel_size_w - 1) // 2), groups=nin)
        else:
            self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, 1, kernel_size_w), groups=nin)
            self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, kernel_size_h, 1), groups=nin)
            self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1, kernel_size_h, kernel_size_w), groups=nin)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        out1 = self.depthwise(x)
        out2 = self.depthwise1(x)
        out3 = self.depthwise2(x)
        out = out1 + out2 + out3
        out = self.at1(out)
        out = self.pointwise(out)
        out = self.at2(out)
        return out


class LE_DSC2D(nn.Module):
    def __init__(self, nin, nout, kernel_size_h, kernel_size_w, patch_size, padding=True):
        super(LE_DSC2D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM2D(self.nin, patch_size)
        self.at2 = NCAM2D(self.nout, patch_size)

        if padding:
            self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h, 1),
                                       padding=((kernel_size_h - 1) // 2, 0), groups=nin)
            self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1, kernel_size_w),
                                        padding=(0, (kernel_size_w - 1) // 2), groups=nin)
        else:
            self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h, 1), groups=nin)
            self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1, kernel_size_w), groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out1 = self.depthwise(x)
        out2 = self.depthwise1(x)
        out = out1 + out2
        out = self.at1(out)
        out = self.pointwise(out)
        out = self.at2(out)
        return out


class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class LE_HCL(nn.Module):
    def __init__(self, ax, aa, c, pca_components, patch_size):
        super(LE_HCL, self).__init__()
        self.conv3d = nn.Sequential(
            LE_DSC3D(1, ax, c, c, c, pca_components, patch_size),
            nn.BatchNorm3d(ax),
            hswish(),
        )
        self.conv2d = nn.Sequential(
            LE_DSC2D(aa, aa // ax, c, c, patch_size),
            nn.BatchNorm2d(aa // ax),
            hswish(),
        )

    def forward(self, x):
        out = self.conv3d(x)
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.conv2d(out)
        out = out + x
        return out


class Lite_HCNet(nn.Module):
    def __init__(self, in_channels, class_num, patch_size):
        super(Lite_HCNet, self).__init__()
        e = 3
        self.unit1 = LE_HCL(e, e * in_channels, 3, in_channels, patch_size)
        self.unit2 = LE_HCL(e, e * in_channels, 7, in_channels, patch_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, class_num)

    def forward(self, x):
        out1 = self.unit1(x)
        out2 = self.unit2(x)
        out = out1 + out2
        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out


# ==================== 统一接口包装类（新增）====================

class LiteHCNetWrapper(nn.Module):
    """
    Lite-HCNet 的物理级拦截包装类。
    安全降级 5D -> 4D，适配当前数据管道。
    """
    def __init__(self, in_channels, num_classes, patch_size):
        super().__init__()
        self.net = Lite_HCNet(in_channels, num_classes, patch_size)

    def forward(self, x):
        # 拦截 5D 剥离为 4D: (B, 1, C, H, W) -> (B, C, H, W)
        if x.dim() == 5:
            x = x.squeeze(1)
        # Lite_HCNet 内部 LE_DSC3D 会自行 unsqueeze(1) 回 5D
        return self.net(x)