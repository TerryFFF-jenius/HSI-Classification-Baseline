import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim
from torch import nn
from models import baseNet
from data_loader import build_data_loader
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


def str2bool(v):
    """自定义布尔参数解析器，彻底阻断 argparse 的 type=bool 陷阱"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    project_name = 'own'
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='./results/')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--project_name', type=str, default=project_name)
    parser.add_argument('--dataset', type=str, default='PaviaU', choices=['PaviaU', 'Houston', 'IP', 'LongKou', 'HanChuan', 'HongHu'], help='Dataset name')

    # learning setting
    parser.add_argument('--epochs', type=int, default=200,
                        help='end epoch for training')
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.01)
    parser.add_argument('--lr_start', default=3e-4, type=float)
    parser.add_argument('--lr_decay', default=0.95, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay (default: 0.001)')
    parser.add_argument('--lr_min', default=2e-6, type=float)
    parser.add_argument('--T_0', default=20, type=int)
    parser.add_argument('--T_mult', default=2, type=int)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--epoch_cycle', type=int, default=50,
                        help='epoch cycle for cosine scheduler')

    # SGD
    parser.add_argument('--momentum', default=0.98, type=float)
    # Adam & AdamW
    parser.add_argument('--betas', default=(0.9, 0.999), type=float, nargs=2)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--num', default=0, type=int)

    # dataset setting
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.01,
                        help='samples for training')
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='train or test')
    parser.add_argument('--is_outimg', type=str2bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str, default='./checkpoints/own/own.pth')
    parser.add_argument('--seed', type=int, default=300,
                        help='random seed')
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    parser.add_argument('--exp_id', type=str, default='baseline_01', help='experiment id for output isolation')
    # === 新增：注册动态维度参数（接收终端输入，若无则设为 None 交由 data_loader 填充） ===
    parser.add_argument('--num_class', type=int, default=None)
    parser.add_argument('--hsi_bands', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)
    
    # === 新增：注册模型路由参数 ===
    parser.add_argument('--model_name', type=str, default='baseline', choices=['baseline', 'cacft', 'lite_hcnet', 'lssan', 'msdan', 'simpoolformer'], help='Model routing')
    parser.add_argument('--band_patches', type=int, default=1, help='CACFTNet param')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='CACFTNet param')
    args = parser.parse_args()
    return args


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def calc_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


def train(model, device, train_loader, optimizer, epoch, args):
    model.train()
    total_loss = 0
    for i, (inputs_1, labels) in enumerate(train_loader):
        inputs_1 = inputs_1.to(device)
        labels = labels.to(device)
                # 数据管道已保证 5D 格式 (B, 1, C, H, W)，各网络内部自行处理维度
        pass
        optimizer.zero_grad()
        outputs = model(inputs_1)
        loss = calc_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(' epoch %d' % (epoch))
    print(' [loss avg: %.4f]' % (total_loss / (len(train_loader))))
    print(' [current loss: %.4f]' % (loss.item()))
    content = ' epoch %d' % (epoch) + ' [loss avg: %.4f]' % (total_loss / (len(train_loader))) + ' [current loss: %.4f]' % (loss.item())
    with open(args.log_file, 'a') as appender:
        appender.write(content + '\n')


def val(model, device, loader, epoch, args, mode='Verification'):
    """验证/测试通用评估函数。传入 val_loader 即做验证，传入 test_loader 即做测试。"""
    model.eval()
    count = 0
    with torch.no_grad():
        for inputs_1, labels in loader:
            inputs_1 = inputs_1.to(device)
            labels = labels.to(device)

                        # 数据管道已保证 5D 格式，直接送入模型
            pass
            outputs = model(inputs_1)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred = outputs
                y_true = labels.cpu().numpy()
                count = 1
            else:
                y_pred = np.concatenate((y_pred, outputs))
                y_true = np.concatenate((y_true, labels.cpu().numpy()))

    # 计算 OA
    a = 0
    for c in range(len(y_pred)):
        if y_true[c] == y_pred[c]:
            a = a + 1
    oa = a / len(y_pred) * 100

    # 计算 AA
    num_classes = args.num_class
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for i in range(len(y_true)):
        label = y_true[i]
        class_total[label] += 1
        if y_pred[i] == label:
            class_correct[label] += 1

    class_accuracy = class_correct / class_total
    aa = np.mean(class_accuracy) * 100

    # 计算 Kappa
    total_samples = len(y_true)
    true_count = np.zeros(num_classes)
    pred_count = np.zeros(num_classes)

    for i in range(total_samples):
        true_count[y_true[i]] += 1
        pred_count[y_pred[i]] += 1

    pe = 0
    for i in range(num_classes):
        pe += (true_count[i] / total_samples) * (pred_count[i] / total_samples)

    po = a / total_samples
    kappa = (po - pe) / (1 - pe)
    kappa_percentage = kappa * 100

    data = {
        "val": [f"Class {i}" for i in range(len(class_accuracy))],
        "Acc": [f"{acc:.2%}" for acc in class_accuracy],
    }
    df = pd.DataFrame(data)
    print(tabulate(df, headers='keys', tablefmt='grid'))
    print(' [The verification OA is: %.2f]' % (oa))
    print(' [The verification AA is: %.2f]' % (aa))
    print(' [The verification Kappa is: %.2f]' % (kappa_percentage))
    with open(args.log_file, 'a') as appender:
        appender.write('\n')
        appender.write('########################### ' + mode + ' ###########################' + '\n')
        appender.write(' epoch: %d' % (epoch) + ' [The verification OA is: %.2f]' % (oa) + ' [The verification AA is: %.2f]' % (aa) +
                       ' [The verification Kappa is: %.2f]' % (kappa_percentage) + '\n')
        appender.write('\n')
    return oa


def main():
    args = args_parser()
    print(args)
    
    exp_dir = f"exp_{args.exp_id}"
    model_dir_path = os.path.join(args.results, args.project_name, args.dataset, exp_dir)
    ckpt_dir = os.path.join(args.checkpoints, args.project_name, args.dataset, exp_dir)
    
    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    args.log_file = os.path.join(model_dir_path, 'log.txt')
    args.ckpt_dir = ckpt_dir

    train_loader, val_loader, test_loader = build_data_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from models import build_model
    model = build_model(args.model_name, args.in_channels, args.num_class, args.patch_size).to(device)

    optimizer, lr_scheduler = prepare_training(args, model)

    best_acc = 0
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch, args)
        lr_scheduler.step()
        if val_loader is not None and (epoch + 1) % 2 == 0:
            acc = val(model, device, val_loader, epoch, args)
            if acc >= best_acc:
                best_acc = acc
                print(f"save model at epoch {epoch}")
                checkpointsmodelfile = os.path.join(args.ckpt_dir, 'model_%.2f.pth' % best_acc)
                torch.save(model.state_dict(), checkpointsmodelfile)
                
                # [新增] 将最优 epoch 物理持久化，防宕机丢失
                with open(os.path.join(args.ckpt_dir, 'best_epoch.txt'), 'w') as f:
                    f.write(f"best_epoch: {epoch}\nbest_oa: {best_acc}\n")


if __name__ == '__main__':
    main()
