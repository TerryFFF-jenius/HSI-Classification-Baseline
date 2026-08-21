import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Attention(nn.Module):
    """底层物理级修复：仅针对 SE 支路的 160 维通道坍塌进行 1D 卷积重建"""
    def __init__(self, n, nin, in_channels):
        super(Attention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.GAP = nn.AdaptiveAvgPool3d(1)
        # 将 3D Conv 改为纯正的 1D ECA 跨通道交互
        self.conv = nn.Conv1d(1, 1, kernel_size=n, padding=(n-1)//2)

        self.AvgPool = nn.AvgPool3d((in_channels//2,1,1))
        self.conv1 = nn.Conv3d(nin, nin, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))

        self.fc1 = nn.Linear(nin, nin // 8, bias=False)
        self.fc2 = nn.Linear(nin // 8, nin, bias=False)

    def forward(self, x):
        n1, c, l, w, h = x.shape
        
        # 修复 SE 支路：先 GAP 保留 C 维，再用 Conv1d 交互
        se_pool = self.GAP(x).view(n1, 1, c) 
        se = self.sigmoid(self.conv(se_pool).view(n1, c, 1, 1, 1))

        # SA 与 CA 支路保持原汁原味的学术逻辑不变
        sa = self.sigmoid(self.conv1(self.AvgPool(x)))

        x1 = self.GAP(x)
        x1 = x1.reshape(n1, -1)
        f1 = self.fc1(x1)
        f2 = self.fc2(f1)

        ca = self.sigmoid(f2)
        ca = ca.reshape(n1, c, 1, 1, 1)

        w = se * sa * ca
        out = x * w
        return out

class Unit(nn.Module):
    def __init__(self, n):
        super(Unit, self).__init__()

        # 3D层
        self.bn1 = nn.BatchNorm3d(64)
        self.Conv3d_1 = nn.Conv3d(64, 32, kernel_size=(1, n, n), stride=1,padding=(0,(n-1)//2,(n-1)//2))
        self.bn2 = nn.BatchNorm3d(32)
        self.Conv3d_2 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1,padding=((n-1)//2,0,0))

        self.bn3 = nn.BatchNorm3d(96)
        self.Conv3d_3 = nn.Conv3d(96, 32, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))
        self.bn4 = nn.BatchNorm3d(32)
        self.Conv3d_4 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1, padding=((n - 1) // 2, 0, 0))

        self.bn5 = nn.BatchNorm3d(128)
        self.Conv3d_5 = nn.Conv3d(128, 32, kernel_size=(1, n, n), stride=1, padding=(0, (n - 1) // 2, (n - 1) // 2))
        self.bn6 = nn.BatchNorm3d(32)
        self.Conv3d_6 = nn.Conv3d(32, 32, kernel_size=(n, 1, 1), stride=1, padding=((n - 1) // 2, 0, 0))

    def forward(self, x):
        out1 = self.Conv3d_1(F.relu(self.bn1(x)))
        x1 = self.Conv3d_2(F.relu(self.bn2(out1)))
        out1 = torch.cat([x1, x], dim=1)

        out2 = self.Conv3d_3(F.relu(self.bn3(out1)))
        x2 = self.Conv3d_4(F.relu(self.bn4(out2)))
        out2 = torch.cat([x2, x, x1], dim=1)

        out3 = self.Conv3d_5(F.relu(self.bn5(out2)))
        x3 = self.Conv3d_6(F.relu(self.bn6(out3)))
        out3 = torch.cat([x3, x, x1,x2], dim=1)


        return out3

class baseNet(nn.Module):
    def __init__(self, in_channels, class_num):
        super(baseNet, self).__init__()

        self.conv3d = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(2,1,1), padding=(1,1,1),)
        )
        self.bneck_1 = Unit(3)
        self.at1 = Attention(3, 160, in_channels)

        self.bneck_2 = Unit(5)
        self.at2 = Attention(5, 160, in_channels)

        self.bneck_3 = Unit(7)
        self.at3 = Attention(7, 160, in_channels)

        # 动态推导经过第一层 3D 卷积后的真实深度，彻底击碎奇偶数硬编码
        actual_depth = (in_channels - 1) // 2 + 1 
        self.conv3d_2 = nn.Sequential(
            nn.BatchNorm3d(160),
            nn.ReLU(inplace=True),
            nn.Conv3d(160, 256, kernel_size=(actual_depth, 1, 1), stride=1)
        )
        self.conv3d_3 = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 64, kernel_size=(256, 1, 1), stride=1, )
        )

        self.MaxPool = nn.MaxPool3d((1, 7, 7))
        self.conv3d_4 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1,padding=(0,1,1))
        )

        self.GAP = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(64, class_num)


    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x = self.conv3d(x)

        x1 = self.bneck_1(x)
        x1 = self.at1(x1)

        x2 = self.bneck_2(x)
        x2 = self.at2(x2)

        x3 = self.bneck_3(x)
        x3 = self.at3(x3)

        out = x1+x2+x3

        out = self.conv3d_2(out).permute(0, 2, 1, 3, 4)
        out = self.conv3d_3(out)
        out = self.MaxPool(out)

        out = self.conv3d_4(out)
        out = self.GAP(out)

        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out

if __name__ == '__main__':
    model = baseNet(12, 9)
    x = torch.randn(2, 12, 7, 7)
    y = model(x)
    print(y.shape)  # 应输出 torch.Size([2, 9])