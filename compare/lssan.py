import torch
import torch.nn as nn
import torch.nn.functional as F

# LSSAN
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)  
        return x * y.expand_as(x) 

class SPA(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SPA, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  

        proj_key = x.view(m_batchsize, -1, width * height).permute(0, 2, 1).permute(1, 0, 2)  
        # 提取中心像素特征
        y = proj_key[[(width * height)//2+1]].permute(1, 0, 2)
        y = y.permute(0, 2, 1)

        energy = torch.cosine_similarity(proj_query.unsqueeze(2), y.unsqueeze(1), dim=3)
        attention = self.sigmoid(energy)
        proj_value = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  

        out = proj_value * attention
        out = out.reshape(m_batchsize, C, width, height)
        return out

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class LSSAN(nn.Module):
    def __init__(self, in_channels, class_num):
        super(LSSAN, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=1),
            hswish(),
        )
        self.bneck_1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=2, groups=128),
            SE_Block(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.bneck_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1, groups=64),
            SPA(64),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.bneck_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1, groups=64),
            SE_Block(64),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.bneck_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=1, groups=32),
            SE_Block(32),
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.SE = SE_Block(32)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.bneck_5 = nn.Sequential(
            hswish(),
            nn.Linear(32, class_num),
        )

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.bneck_1(out)
        out = self.bneck_2(out)
        out = self.bneck_3(out) + out
        out = self.bneck_4(out) + out
        out = self.SE(out)
        out = self.GAP(out)
        out = out.reshape(out.shape[0], -1)
        out = self.bneck_5(out)
        return out