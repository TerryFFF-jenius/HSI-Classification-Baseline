"""
Contrastive Method: Lite-HCNet
Source: Li et al., "Lite-HCNet: A Lightweight Hybrid Convolutional Network for 
        Hyperspectral Image Classification", IEEE TGRS, 2023.
This implementation is adapted from the official code for fair comparison.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from operator import truediv
import warnings
import time
from torchsummary import summary
import math
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline
import os
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, \
#                                          deprocess_image, \
#                                          preprocess_image
import matplotlib as mpl


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.savefig('./results'+data+'-'+title+'.png')
    plt.show()

def reports(test_loader, y_test, name):
    count = 0
    # 模型测试
    since = time.time()
    net.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:

            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            if count == 0:
                y_pred = outputs
                count = 1
            else:
                y_pred = np.concatenate((y_pred, outputs))
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if name == 'HU':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives']
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if name == 'GF':
        target_names = ['互花米草', '水体', '芦苇沼泽', '盐地碱蓬'
            , '盐碱滩', '裸潮滩', '稀疏潮滩',
                        '柽柳']#, 'Wheat', 'Woods'
        labels = [1, 2, 3, 4, 5, 6, 7, 8]
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif name == 'UP':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    classification = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification,labels, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


# 模型代码
# -------------------------------------------------------------------------------------------- #
def bsm(n,d):
    a = [[0]*n for x in range(n)]
    p = 0
    q = n-1
    w = (n+1)/2
    w =int(w)
    #print(w)
    #w1 = 1 / w
    #print(w1)
    t = 0
    while p < d:
        for i in range(p,q):
            a[p][i] = t


        for i in range(p,q):
            a[i][q] = t


        for i in range(q,p,-1):
            a[q][i] = t


        for i in range(q,p,-1):
            a[i][p] = t

        p += 1
        q -= 1
        #t += w1

    while p==d or p>d and p<q:
        for i in range(p,q):
            a[p][i] = 1


        for i in range(p,q):
            a[i][q] = 1


        for i in range(q,p,-1):
            a[q][i] = 1


        for i in range(q,p,-1):
            a[i][p] = 1

        a[w-1][w-1] = 1
        p += 1
        q -= 1
    return np.array(a)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ScaleMaskModule(nn.Module):#尺度掩膜
    def __init__(self,d):
        # w是空间尺度，n是光谱维度，p是批次大小
        super(ScaleMaskModule, self).__init__()

        self.d = d
    def forward(self,x):
        w = x.shape[3]
        n = x.shape[2]
        o = x.shape[1]
        p = x.shape[0]
       # print(x.shape)
        out = bsm(w,self.d)
        #print(out.shape)
        out = torch.from_numpy(out)
        out = out.repeat(p, o, 1, 1)#out.repeat(p, o,n, 1, 1)
        #print(out.shape)
        out = out.type(torch.FloatTensor)
        out = out.to(device)
        #print(x * out)
        return x * out

class NCAM3D(nn.Module):# 3D NCAM
    def __init__(self, c):
        super(NCAM3D, self).__init__()
        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c, 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size-1)//2-1)

        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2,kernel_size_21), padding=(0,(kernel_size_21 - 1) // 2), dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                               dilation=1)

    def forward(self, x):

        out =x
        #### 通道注意力
        out_1 = out.shape[1]
        out_2 = out.shape[2]
        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        ###中心像素的光谱
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x)
        out_x1 = out_x1.reshape(out_x1.shape[0], -1)
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])

        out_xx = torch.cat([out_x1, out_x2], dim=2)
        #######
        ###全局空间的光谱
        out1 = self.avg_pool(out)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = reversed(out1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])

        outx = torch.cat([out1, out2], dim=2)
        #########
        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)
        at = F.sigmoid((at1-0.2)*2)
        out = out * at
        #####
        out = out.reshape(out.shape[0], out_1, out_2, out.shape[2], out.shape[3])

        return out


class NCAM2D(nn.Module):  # 2D NCAM
    def __init__(self, c):
        super(NCAM2D, self).__init__()

        gamma = 2
        b = 3
        kernel_size_21 = int(abs((math.log(c , 2) + b) / gamma))
        kernel_size_21 = kernel_size_21 if kernel_size_21 % 2 else kernel_size_21 + 1#保证结果为奇数，当%2不为0时执行if语句，即奇数；

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ScaleMaskModule = ScaleMaskModule((patch_size - 1) // 2-1)

        self.conv1d = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                                dilation=1)
        self.conv1d1 = nn.Conv2d(1, 1, kernel_size=(2, kernel_size_21), padding=(0, (kernel_size_21 - 1) // 2),
                                 dilation=1)


    def forward(self, x):
        out = x
        #### 通道注意力

        ###中心像素的光谱
        out_x = self.ScaleMaskModule(out)
        out_x1 = self.avg_pool(out_x)
        out_x1 = out_x1.reshape(out_x1.shape[0], -1)
        out_x2 = reversed(out_x1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out_x1 = out_x1.reshape(out_x1.shape[0], 1, 1, out_x1.shape[1])
        out_x2 = out_x2.reshape(out_x2.shape[0], 1, 1, out_x2.shape[1])

        out_xx = torch.cat([out_x1, out_x2], dim=2)
        #######
        ###全局空间的光谱
        out1 = self.avg_pool(out)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = reversed(out1.permute(1, 0)).permute(1, 0)  # 原地翻转，倒序:[1,2,3]->[3,2,1]

        out1 = out1.reshape(out1.shape[0], 1, 1, out1.shape[1])
        out2 = out2.reshape(out2.shape[0], 1, 1, out2.shape[1])

        outx = torch.cat([out1, out2], dim=2)
        #########
        at1 = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2) * F.sigmoid(self.conv1d1(out_xx)).permute(0, 3, 1, 2)

        at = F.sigmoid((at1 - 0.2) * 2)
        #print(at)
        #at = F.sigmoid(self.conv1d(outx)).permute(0, 3, 1, 2)
        out = out * at

        return out

class LE_DSC3D(nn.Module):# 3D LE-DSC
    def __init__(self, nin, nout,kernel_size_c,kernel_size_h,kernel_size_w,padding=True):
        super(LE_DSC3D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM3D(self.nin*pca_components)
        self.at2 = NCAM3D(self.nout * pca_components)

        if padding == True:

         self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, 1, kernel_size_w),
                                   padding=((kernel_size_c - 1) // 2, 0, (kernel_size_w - 1) // 2), groups=nin)
         self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c, kernel_size_h, 1),
                                    padding=((kernel_size_c - 1) // 2, (kernel_size_h - 1) // 2, 0), groups=nin)
         self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1,kernel_size_h,kernel_size_w),
                                    padding=(0, (kernel_size_h - 1) // 2, (kernel_size_w - 1) // 2), groups=nin)

        else:
         self.depthwise = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c,1,kernel_size_w),  groups=nin)
         self.depthwise1 = nn.Conv3d(nin, nin, kernel_size=(kernel_size_c,kernel_size_h,1),  groups=nin)
         self.depthwise2 = nn.Conv3d(nin, nin, kernel_size=(1,kernel_size_h,kernel_size_w),  groups=nin)

        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):

        out1 = self.depthwise(x)

        out2 = self.depthwise1(x)

        out3 = self.depthwise2(x)

        out3 = out1+out2+out3 #

        out =out3
        #### 通道注意力
        out = self.at1(out)

        out = self.pointwise(out)

        #### 通道注意力
        out = self.at2(out)
        ####

        return out
###################################################################
class LE_DSC2D(nn.Module):#深度可分离混合卷积 ---- 2D数据版本
    def __init__(self, nin, nout,kernel_size_h,kernel_size_w,padding=True):
        super(LE_DSC2D, self).__init__()
        self.nout = nout
        self.nin = nin
        self.at1 = NCAM2D(self.nin)
        self.at2 = NCAM2D(self.nout)

        if padding == True:

         self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h,1), padding=((kernel_size_h - 1) // 2,0), groups=nin)
         self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1,kernel_size_w), padding=(0,(kernel_size_w - 1) // 2), groups=nin)
        else:

         self.depthwise = nn.Conv2d(nin, nin, kernel_size=(kernel_size_h,1),  groups=nin)
         self.depthwise1 = nn.Conv2d(nin, nin, kernel_size=(1,kernel_size_w),  groups=nin)

        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out1 = self.depthwise(x)

        out2 = self.depthwise1(x)

        out3 = out1+out2 #

        out =out3
        #### 通道注意力
        out = self.at1(out)

        out = self.pointwise(out)
        #### 通道注意力
        out = self.at2(out)
        ####

        return out

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
class LE_HCL(nn.Module):#输入默认为一个三维块，即三维通道数为1
    def __init__(self, ax, aa,c):#ax二维通道数，c卷积核大小，d为padding和dilation大小
        super(LE_HCL, self).__init__()
        self.conv3d = nn.Sequential(
            LE_DSC3D(1, ax, c, c, c),
            nn.BatchNorm3d(ax),
            hswish(),

        )
        self.conv2d = nn.Sequential(
            LE_DSC2D(aa, aa // ax, c, c),
            nn.BatchNorm2d(aa // ax),
            hswish(),

        )

    def forward(self, x):

        out = self.conv3d(x)

        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.conv2d(out)

        out = out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = out + x

        return out
# 主模型
class Lite_HCNet(nn.Module):
    def __init__(self):
        super(Lite_HCNet, self).__init__()
        ########
        e = 3 # LE-HCL中的e参数
        self.unit1 = LE_HCL(e, e*pca_components, 3)
        self.unit2 = LE_HCL(e, e*pca_components, 7)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(pca_components, class_num)

    def forward(self, x):

        out1 = self.unit1(x)
        out2 = self.unit2(x)

        out = out1+out2

        out = out.reshape(out.shape[0], -1, out.shape[3], out.shape[4])
        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)

        return out

# -------------------------------------------------------------------------------------------- #
# PCA降维
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # 获得 y 中的标记样本数
    # 实图像中大部分为是0，没有label，我们要取的，只是有label的部分。
    # 因此，先做个循环，看看有多少个像素有 label，然后记录在 count 里，这样 count 比以前的X.shape[0] * X.shape[1]要小很多，内存就不会爆了。
    count = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r, c] != 0:
                count = count + 1

    # split patches
    patchesData = np.zeros((count, windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((count))
    count = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] != 0:

                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[count, :, :, :] = patch
                patchesLabels[count] = y[r - margin, c - margin]
                count = count + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test



# 用于测试样本的比例
test_ratio = 0.99
train_ratio = '0.01'
# 训练周期
epochs = 50
# 批次大小
batch_size = 16

# 数据集设置

# IP数据集
# data = 'IP'
# class_num = 16

# patch_size =19 # 每个像素周围提取 patch 的尺寸
# pca_components =15 # 使用 PCA 降维，得到主成分的数量
# UP数据集
data = 'UP'
class_num = 9
X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
patch_size =7 # 每个像素周围提取 patch 的尺寸
pca_components =12 # 使用 PCA 降维，得到主成分的数量
# SA数据集
# data = 'SA'
# class_num = 16

# HU数据集
# data = 'HU'
# class_num = 15

# patch_size =7 # 每个像素周围提取 patch 的尺寸
# pca_components =17 # 使用 PCA 降维，得到主成分的数量
# YRD数据集
# data = 'YRD'
# class_num = 8

# patch_size =11 # 每个像素周围提取 patch 的尺寸
# pca_components =17 # 使用 PCA 降维，得到主成分的数量

class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len
def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    # 随机生成长度为len(a)的序列，将a和b按同样的随机顺序重新排列
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]
def split_data(pixels, labels, percent, splitdset="custom", rand_state=345):
    """
    :param pixels: len(pixels.shape) >3表示cube，小于则表示location
    :param labels: 标签
    :param percent: 训练集的比重，为整数时，表示每一类选取多少个作为训练集（splitdset="custom"时）
    :param mode: CNN模式划分训练集和测试集，GAN模式只需要训练集
    :param splitdset: 使用sklearn还是自己设计的划分方式，“sklearn”表示用sklearn，“custom”表示自己的
    :param rand_state: 保证每次的划分方式相同
    :return:
    """
    if splitdset == "sklearn":
        # train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
        return train_test_split(pixels, labels, test_size=percent, stratify=labels, random_state=rand_state)
    elif splitdset == "custom":

            # np.unique:该函数是去除数组中的重复数字，并进行排序之后输出。return_counts = true,返回个数（用于统计各个元素出现的次数）
            # pixels_number:各类标签的个数，按顺序记录在列表中。
            pixels_number = np.unique(labels, return_counts=True)[1]
            # 设置各类地物训练集的大小
            # train_set_size = np.ones(len(pixels_number)) * percent

            # train_set_size = [1, 14, 8, 2, 5, 7, 1, 5, 1, 10, 25, 6, 2, 13, 4, 1]
            train_set_size = [66, 186, 20, 30, 13, 50, 13, 36, 9]
            # 训练集总大小
            tr_size = int(sum(train_set_size))
            # 测试集总大小
            te_size = int(sum(pixels_number)) - int(sum(train_set_size))
            sizetr = np.array([tr_size] + list(pixels.shape[1:]))
            sizete = np.array([te_size] + list(pixels.shape[1:]))
            train_x = np.empty((sizetr));
            train_y = np.empty((tr_size))
            test_x = np.empty((sizete));
            test_y = np.empty((te_size))
            trcont = 0
            tecont = 0
            for cl in np.unique(labels):
                pixels_cl = pixels[labels == cl]
                labels_cl = labels[labels == cl]
                pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
                for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):

                    if cont < train_set_size[int(cl)]:
                        train_x[trcont, :, :, :] = a
                        train_y[trcont] = b
                        trcont += 1
                    else:
                        test_x[tecont, :, :, :] = a
                        test_y[tecont] = b
                        tecont += 1
            train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
            test_x, test_y = random_unison(test_x, test_y, rstate=rand_state)
            return train_x, test_x, train_y, test_y


# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Lite_HCNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [40])
from PIL import Image
# 数据矩阵转图片的函数
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
if __name__ == "__main__":
    my_color = np.array([[255, 255, 255],# 画IP的图时，要有17个，即个数要比数据集的地物类别数要多
                         [184, 40, 99],
                         [74, 77, 145],
                         [35, 102, 193],
                         [238, 110, 105],
                         [117, 249, 76],
                         [114, 251, 253],
                         [126, 196, 59],
                         [234, 65, 247],
                         [141, 79, 77],
                         [183, 40, 99],
                         [0, 39, 245],
                         [90, 196, 111],
                         [50, 140, 100],
                         [70, 140, 200],
                         [100, 150, 170]])
    my_color = my_color / 255
    cmap = mpl.colors.ListedColormap(my_color)
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')

    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = split_data(X_pca, y, test_ratio, splitdset="sklearn")
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    # 创建 trainloader 和 testloader
    trainset = TrainDS()
    testset = TestDS()
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=256, shuffle=False, num_workers=0)
    # 开始训练
    # net.train()
    total_loss = 0
    min_loss = 1000
    since = time.time()
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (
            epoch + 1, total_loss / (epoch + 1), loss.item()))
        # if epoch > epochs-10:
        #  if loss.item() < min_loss:
        #     min_loss = loss.item()
        
        # print("save model")
        scheduler.step()
    # print('Finished Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    classification, labels, confusion, oa, each_acc, aa, kappa = reports(test_loader, ytest, data)
    cm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]  # 归一化
    plot_confusion_matrix(cm, labels, 'confusion_matrix')  # 绘制混淆矩阵图，可视化

    print(classification)
    print(' Kappa accuracy: {}%'.format(kappa))
    print(' Overall accuracy: {}%'.format(oa))
    print(' Average accuracy: {}%'.format(aa))
    # 输出网络结构
    summary(net, (1, pca_components, patch_size, patch_size))

    # 输出网络参数量与计算量
    # from thop import profile
    # img1 = torch.randn(batch_size, 1, pca_components, patch_size, patch_size)
    # macs, params = profile(net, (img1.cuda(),))
    # print('FLOPs: ', 2 * macs, 'Params: ', params)


# 可视化结果生成
# load the original image
    # IP数据集
    
    # UP数据集
    X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
    y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
    # SA数据集
    
    # HU数据集
    
    # YRD数据集
    

    height = y.shape[0]
    width = y.shape[1]

    X = applyPCA(X, numComponents=pca_components)
    X = padWithZeros(X, patch_size // 2)

    # 逐像素预测类别
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if int(y[i, j]) == 0:
                continue
            else:
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                prediction = net(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')

    sio.savemat('./result/'+data+'-'+train_ratio+'.mat',mdict={'outputs': outputs})
    ypr = sio.loadmat('./result/'+data+'-'+train_ratio+'.mat')['outputs']

    plt.imshow(y, cmap=cmap)
    plt.show()

    plt.imshow(ypr,cmap=cmap)
    plt.show()