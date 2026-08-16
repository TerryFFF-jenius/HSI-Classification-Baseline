import math
import os
import torch
import torch.optim
from torch import nn
from models import baseNet
from data_loader import build_data_loader, trans_tif
import numpy as np
from util.util import prepare_training
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import argparse
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def args_parser():
    project_name = 'own'
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', type=str, default='./results/')
    parser.add_argument('-checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('-project_name', type=str, default=project_name)
    parser.add_argument('-dataset', type=str, default='PaviaU',
                        choices=['PaviaU', 'Houston'])

    # learning setting
    parser.add_argument('--epochs', type=int, default=200,
                        help='end epoch for training')
                
    # model setting
    parser.add_argument('--hidden_size', type=int, default=512)

    # dataset setting
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--train_ratio', type=list, default=[3,4,5,6,7,8,2,1,3,4])
    parser.add_argument('--train_ratio', type=int, default=7,
                        help='samples for training')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='train or test')
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='/zbssd/yuyu/code/HyBenchmark/checkpoints/own/PaviaU/model_93.92.pth')
    parser.add_argument('--seed', type=int, default=300,
                        help='random seed') # 5,7 300 ; 10 200
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    parser.add_argument('--allimg', type=bool, default=False, help='allimg')

    args = parser.parse_args()
    return args

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def pred_allimg(model, device, X, y, epoch, args):
    model.eval()
    height = y.shape[0]
    width = y.shape[1]
    with torch.no_grad():
        outputs = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                if args.allimg:
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                    1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
                else:
                    if int(y[i, j]) == 0:
                        continue
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                    1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
            if i % 20 == 0:
                print('... ... row ', i, ' handling ... ...')
    if args.allimg:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/All_PRED.tif'
        trans_tif(outputs, finalmodelfile)
    else:
        finalmodelfile = args.results + args.project_name+ '/' + args.dataset+ '/Label_PRED.npy'
        y[y!=0] = 1
        outputs = outputs*y
        trans_tif(outputs, finalmodelfile)
    
def main():
    args = args_parser()
    print (args)
    model_dir_path = os.path.join(args.results, args.project_name + '/')
    log_file = os.path.join(args.results, args.project_name +'/log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.checkpoints+args.project_name+'/', exist_ok=True)
    args.log_file = log_file

    X, y = build_data_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.PCA is None:
        model = baseNet(args.hsi_bands, args.num_class).to(device)
    else:    
        model = baseNet(args.PCA, args.num_class).to(device)
    
    model.load_state_dict(torch.load(args.checkpointsmodelfile, weights_only=True))
    pred_allimg(model, device, X, y, args.epochs, args)

def main_test():
    pass



if __name__ == '__main__':
     main()
