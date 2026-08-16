import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 强行绕过 OpenMP 冲突
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
    # parser.add_argument('--lr', type=float, default=2e-4,
    #                     help='learning rate')
    parser.add_argument('--lr_scheduler', default='poly', type=str) #cosinewarm poly
    parser.add_argument('--lr_start', default=3e-4, type=int)
    parser.add_argument('--lr_decay', default=0.95, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.005,
                    help='weight decay (default: 0.001)')
    parser.add_argument('--lr_min', default=2e-6, type=int)
    parser.add_argument('--T_0', default=20, type=int)
    parser.add_argument('--T_mult', default=2, type=int)
    parser.add_argument('--optim', default='adamw', type=str)

    # SGD
    parser.add_argument('--momentum', default=0.98, type=float)
    # Adam & AdamW
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--num', default=0, type=int)

    # dataset setting
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--train_ratio', type=list, default=[3,4,5,6,7,8,2,1,3,4])
    parser.add_argument('--train_ratio', type=float, default=0.01,
                        help='samples for training')
    # parser.add_argument('--train_ratio', type=float, default=0.8,
    #                     help='samples for training')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='train or test')
    parser.add_argument('--is_outimg', type=bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='./checkpoints/own/own.pth')
    parser.add_argument('--seed', type=int, default=300,
                        help='random seed') # 5,7 300 ; 10 200
    parser.add_argument('--PCA', type=int, default=None, help='PCA')

    args = parser.parse_args()
    return args

def custom_repr(self):
    # 新格式：在原有输出前添加形状信息 {Tensor:(shape)}
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

# 保存原始 __repr__ 方法
original_repr = torch.Tensor.__repr__  
# 用自定义方法覆盖 PyTorch 张量的 __repr__
torch.Tensor.__repr__ = custom_repr  
# 假设原张量输出为：

# PYTHON
# tensor([[1., 2.], [3., 4.]])
# 修改后会变成：

# PYTHON
# {Tensor:(2, 2)} tensor([[1., 2.], [3., 4.]])

def calc_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss =  criterion(outputs, labels)
    return loss

def train(model, device, train_loader, optimizer, epoch, args):
    model.train()
    total_loss = 0
    for i, (inputs_1, labels) in enumerate(train_loader):
        inputs_1 = inputs_1.to(device)
        labels = labels.to(device)
        if args.PCA is not None:
            inputs_1 = inputs_1.view(-1, args.PCA, args.patch_size, args.patch_size)
        else:
            inputs_1 = inputs_1.view(-1, args.hsi_bands, args.patch_size, args.patch_size)

        optimizer.zero_grad()

        # outputs, fc, f_hsi_s, f_sar_s= model(inputs_1, inputs_2)
        outputs = model(inputs_1)
        loss = calc_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(' epoch %d' % (epoch))
    print(' [loss avg: %.4f]' %( total_loss/(len(train_loader))))
    print(' [current loss: %.4f]' %(loss.item()))
    content = ' epoch %d' % (epoch) + ' [loss avg: %.4f]' %( total_loss/(len(train_loader))) + ' [current loss: %.4f]' %(loss.item())
    with open(args.log_file, 'a') as appender:
        appender.write(content + '\n')

def val(model, device, test_loader, epoch, args):
    model.eval()
    count = 0
    with torch.no_grad():
        for inputs_1, labels in test_loader:
            inputs_1 = inputs_1.to(device)
            labels = labels.to(device)

            if args.PCA is not None:
                inputs_1 = inputs_1.view(-1, args.PCA, args.patch_size, args.patch_size)
            else:
                inputs_1 = inputs_1.view(-1, args.hsi_bands, args.patch_size, args.patch_size)
            outputs= model(inputs_1)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                test_labels = labels.cpu().numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                test_labels = np.concatenate((test_labels, labels.cpu().numpy()))
            # 计算 OA
    a = 0
    for c in range(len(y_pred_test)):
        if test_labels[c]==y_pred_test[c]:
            a = a + 1
    oa = a/len(y_pred_test)*100
    
    # 计算 AA
    num_classes = args.num_class  # 类别数
    class_correct = np.zeros(num_classes)  # 每个类别预测正确的样本数
    class_total = np.zeros(num_classes)  # 每个类别的总样本数
    
    for i in range(len(test_labels)):
        label = test_labels[i]
        class_total[label] += 1
        if y_pred_test[i] == label:
            class_correct[label] += 1
    
    class_accuracy = class_correct / class_total  # 每个类别的精度
    aa = np.mean(class_accuracy) * 100  # 平均精度、

    # 计算 Kappa系数
    total_samples = len(test_labels)
    true_count = np.zeros(num_classes)  # 每个类别的真实样本数
    pred_count = np.zeros(num_classes)  # 每个类别的预测样本数
    
    # 计算每个类别的真实样本数和预测样本数
    for i in range(total_samples):
        true_count[test_labels[i]] += 1
        pred_count[y_pred_test[i]] += 1
    
    # 计算期望一致性pe
    pe = 0
    for i in range(num_classes):
        pe += (true_count[i]/total_samples) * (pred_count[i]/total_samples)
    
    # 计算kappa系数
    po = a / total_samples  # 观察一致性（即准确率）
    kappa = (po - pe) / (1 - pe)
    kappa_percentage = kappa * 100  # 转为百分比形式
    

    data = {
        "val": [f"Class {i}" for i in range(len(class_accuracy))],
        "Acc": [f"{acc:.2%}" for acc in class_accuracy],
    }
    df = pd.DataFrame(data)
    print(tabulate(df, headers='keys', tablefmt='grid'))
    print (' [The verification OA is: %.2f]' %(oa))
    print (' [The verification AA is: %.2f]' %(aa))
    print (' [The verification Kappa is: %.2f]' %(kappa_percentage))
    with open(args.log_file, 'a') as appender:
        appender.write('\n')
        appender.write('########################### Verification ###########################' + '\n')
        appender.write(' epoch: %d' % (epoch) + ' [The verification OA is: %.2f]' %(oa) + ' [The verification AA is: %.2f]' %(aa) +
                       ' [The verification Kappa is: %.2f]' %(kappa_percentage) + '\n')
        appender.write('\n')
    return oa

def main():
    args = args_parser()
    print (args)
    model_dir_path = os.path.join(args.results, args.project_name + '/', args.dataset+'/')
    log_file = os.path.join(args.results, args.project_name+'/', args.dataset+'/log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.checkpoints+args.project_name+'/'+args.dataset+'/', exist_ok=True)
    args.log_file = log_file

    train_loader, val_loader = build_data_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.PCA is None:
        model = baseNet(args.hsi_bands, args.num_class).to(device)
    else:    
        model = baseNet(args.PCA, args.num_class).to(device)

    optimizer, lr_scheduler = prepare_training(args, model)
            
    best_acc = 0
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch, args)
        lr_scheduler.step()
        if (epoch+1)%2 == 0:
            acc = val(model, device, val_loader, epoch, args)
            if acc >= best_acc:
                  best_acc = acc
                  print("save model")
                  checkpointsmodelfile = os.path.join(args.checkpoints, args.project_name, args.dataset, 'model_%.2f.pth' % best_acc)
                  torch.save(model.state_dict(), checkpointsmodelfile)

def main_test():
    pass



if __name__ == '__main__':
     main()
