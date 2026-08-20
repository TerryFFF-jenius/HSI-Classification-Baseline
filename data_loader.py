import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import os
from osgeo import gdal
import warnings
warnings.filterwarnings("ignore")


def trans_tif(image, output_path):
    if len(image.shape) == 3:
        bands = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]
    else:
        bands = 1
        height = image.shape[0]
        width = image.shape[1]
    if 'uint8' in image.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'uint16' in image.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'float64' in image.dtype.name:
        datatype = gdal.GDT_Float64
    else:
        datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, bands, datatype)

    if len(image.shape) == 3:
        for i in range(image.shape[0]):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(image[i])
            band.SetNoDataValue(0)
    else:
        dataset.GetRasterBand(1).WriteArray(image)


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    count = 0
    for r in range(0, y.shape[0]):
        for c in range(0, y.shape[1]):
            if y[r, c] != 0:
                count = count + 1

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


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels, train_ratio, val_ratio=0.0, splitdset="sklearn", rand_state=345):
    if splitdset == "sklearn":
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < -1e-9:
            raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) > 1.0")

        if test_ratio > 1e-9:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                pixels, labels, test_size=test_ratio, stratify=labels, random_state=rand_state)
        else:
            X_trainval, y_trainval = pixels, labels
            X_test = np.empty((0, *pixels.shape[1:]), dtype=pixels.dtype)
            y_test = np.empty((0,), dtype=labels.dtype)

        if val_ratio > 1e-9 and len(X_trainval) > 0:
            adj = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=adj, stratify=y_trainval, random_state=rand_state)
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val = np.empty((0, *pixels.shape[1:]), dtype=pixels.dtype)
            y_val = np.empty((0,), dtype=labels.dtype)

        return X_train, X_val, X_test, y_train, y_val, y_test

    elif splitdset == "custom":
        raise NotImplementedError("Custom split not implemented for 3-way splits yet.")


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class ValDS(torch.utils.data.Dataset):
    def __init__(self, Xval, yval):
        self.len = Xval.shape[0]
        self.x_data = torch.FloatTensor(Xval)
        self.y_data = torch.LongTensor(yval)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def _load_dataset(args):
    """严禁覆盖用户命令行传入的参数，仅在缺失时注入默认值"""
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        default_bands, default_class, default_patch, default_pca = X.shape[2], 9, 7, 12
    elif args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        default_bands, default_class, default_patch, default_pca = X.shape[2], 15, 7, 17
    elif args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        default_bands, default_class, default_patch, default_pca = X.shape[2], 16, 19, 15
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 使用 getattr 安全注入，绝对不覆盖已存在的命令行参数
    if getattr(args, 'hsi_bands', None) is None: args.hsi_bands = default_bands
    if getattr(args, 'num_class', None) is None: args.num_class = default_class
    if getattr(args, 'patch_size', None) is None: args.patch_size = default_patch
    if getattr(args, 'PCA', None) is None: args.PCA = default_pca
    
    args.in_channels = args.PCA if args.PCA is not None else args.hsi_bands

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    return X, y


def _build_loaders_from_cubes(X_cube, y_cube, args):
    """从已切分的 cubes 构建三个 DataLoader，PCA/无PCA 逻辑统一"""
    val_ratio = getattr(args, 'val_ratio', 0.0)

    print('\n... ... create train & val & test data ... ...')
    Xtrain, Xval, Xtest, ytrain, yval, ytest = split_data(
        X_cube, y_cube, args.train_ratio, val_ratio, splitdset="sklearn")
    print('Xtrain shape: ', Xtrain.shape)
    if len(Xval) > 0:
        print('Xval shape:   ', Xval.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 确定当前使用的波段维度
    bands = args.PCA if args.PCA is not None else args.hsi_bands

    Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, bands, 1)
    Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, bands, 1)
    if len(Xval) > 0:
        Xval = Xval.reshape(-1, args.patch_size, args.patch_size, bands, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    if len(Xval) > 0:
        Xval = Xval.transpose(0, 4, 3, 1, 2)
        print('after transpose: Xval shape:   ', Xval.shape)

    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=256, shuffle=False, num_workers=0)

    if len(Xval) > 0:
        valset = ValDS(Xval, yval)
        val_loader = torch.utils.data.DataLoader(
            dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    return train_loader, val_loader, test_loader


def build_data_loader(args):
    X, y = _load_dataset(args)

    if args.is_train:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')
            X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X_pca.shape)
            print('Data cube y shape: ', y.shape)
            return _build_loaders_from_cubes(X_pca, y, args)
        else:
            print('\n... ... create data cubes ... ...')
            X, y = createImageCubes(X, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X.shape)
            print('Data cube y shape: ', y.shape)
            return _build_loaders_from_cubes(X, y, args)
    else:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')
            X = padWithZeros(X_pca, args.patch_size // 2)
        else:
            X = padWithZeros(X, args.patch_size // 2)
        return X, y


def build_data_sim_loader(args):
    """对比方法 SIM 的数据管道（patch_size 与主流程不同）"""
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 9
        args.PCA = 12
    elif args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 9
        args.PCA = 17
    elif args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 9
        args.PCA = 15
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    if args.is_train:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')
            X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X_pca.shape)
            print('Data cube y shape: ', y.shape)
            return _build_loaders_from_cubes(X_pca, y, args)
        else:
            print('\n... ... create data cubes ... ...')
            X, y = createImageCubes(X, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X.shape)
            print('Data cube y shape: ', y.shape)
            return _build_loaders_from_cubes(X, y, args)
    else:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')
            X = padWithZeros(X_pca, args.patch_size // 2)
        else:
            X = padWithZeros(X, args.patch_size // 2)
        return X, y


def build_data_cacf_loader(args):
    """对比方法 CACF 的数据管道（原文不用 PCA）"""
    if args.dataset == 'PaviaU':
        X = sio.loadmat('./oridata/PaviaU/PaviaU.mat')['paviaU']
        y = sio.loadmat('./oridata/PaviaU/PaviaU_gt.mat')['paviaU_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 9
        args.patch_size = 7
    elif args.dataset == 'Houston':
        X = sio.loadmat('./oridata/Houston/Houston.mat')['Houston']
        y = sio.loadmat('./oridata/Houston/Houston_gt.mat')['Houston_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 15
        args.patch_size = 7
    elif args.dataset == 'IP':
        X = sio.loadmat('./oridata/Indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('./oridata/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']
        args.hsi_bands = X.shape[2]
        args.num_class = 16
        args.patch_size = 7
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    if args.is_train:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')
            X_pca, y = createImageCubes(X_pca, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X_pca.shape)
            print('Data cube y shape: ', y.shape)
            return _build_loaders_from_cubes(X_pca, y, args)
        else:
            print('\n... ... create data cubes ... ...')
            X, y = createImageCubes(X, y, windowSize=args.patch_size)
            print('Data cube X shape: ', X.shape)
            print('Data cube y shape: ', y.shape)
            return _build_loaders_from_cubes(X, y, args)
    else:
        if args.PCA is not None:
            print('\n... ... PCA tranformation ... ...')
            X_pca = applyPCA(X, numComponents=args.PCA)
            print('Data shape after PCA: ', X_pca.shape)
            print('\n... ... create data cubes ... ...')
            X = padWithZeros(X_pca, args.patch_size // 2)
        else:
            X = padWithZeros(X, args.patch_size // 2)
        return X, y
def build_test_loader(args):
    """为测试脚本提供不破坏 args.is_train 语义的测试集 DataLoader"""
    import copy
    tmp_args = copy.copy(args)
    tmp_args.is_train = True
    _, _, test_loader = build_data_loader(tmp_args)
    return test_loader    