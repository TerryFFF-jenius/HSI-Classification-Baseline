import math
import os
import glob
import json
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim
from torch import nn
from models import baseNet
from data_loader import build_data_loader, build_test_loader
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch.backends.cudnn as cudnn
import torch.backends.cuda

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def str2bool(v):
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

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.01)
    parser.add_argument('--val_ratio', type=float, default=0.01)
    parser.add_argument('--is_train', type=str2bool, default=False)
    parser.add_argument('--is_outimg', type=str2bool, default=False)
    parser.add_argument('--modelfile', type=str, default='./checkpoints/own/PaviaU/model_17.52.pth')
    parser.add_argument('--seed', type=int, default=300)
    
    # 彻底解除硬编码，交由 DataLoader 动态注入
    parser.add_argument('--num_class', type=int, default=None)
    parser.add_argument('--hsi_bands', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)
    parser.add_argument('--PCA', type=int, default=None)
    
    parser.add_argument('--model_name', type=str, default='baseline', choices=['baseline', 'cacft', 'lite_hcnet', 'lssan', 'msdan', 'simpoolformer'], help='Model routing')
    parser.add_argument('--exp_id', type=str, default='baseline_01', help='experiment id for output isolation')
    args = parser.parse_args()
    return args


def test(model, device, test_loader, args):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                inputs_1 = batch[0]
                labels = batch[1]
            else:
                inputs_1 = batch
                labels = batch
            
            if isinstance(inputs_1, np.ndarray):
                inputs_1 = torch.from_numpy(inputs_1)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            
            inputs_1 = inputs_1.float().to(device)
            labels = labels.long().to(device)

            # 数据管道已保证 5D 格式 (B, 1, C, H, W)，直接送入模型
            outputs = model(inputs_1)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            
            if count == 0:
                y_pred_test = outputs
                test_labels = labels.cpu().numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                test_labels = np.concatenate((test_labels, labels.cpu().numpy()))

    a = 0
    for c in range(len(y_pred_test)):
        if test_labels[c] == y_pred_test[c]:
            a = a + 1
    oa = a / len(y_pred_test) * 100

    num_classes = args.num_class
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for i in range(len(test_labels)):
        label = test_labels[i]
        class_total[label] += 1
        if y_pred_test[i] == label:
            class_correct[label] += 1

    class_accuracy = class_correct / class_total
    aa = np.mean(class_accuracy) * 100

    total_samples = len(test_labels)
    true_count = np.zeros(num_classes)
    pred_count = np.zeros(num_classes)

    for i in range(total_samples):
        true_count[test_labels[i]] += 1
        pred_count[y_pred_test[i]] += 1

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
    print(' [The test OA is: %.2f]' % (oa))
    print(' [The test AA is: %.2f]' % (aa))
    print(' [The test Kappa is: %.2f]' % (kappa_percentage))
    
    with open(args.log_file, 'a') as appender:
        appender.write('\n')
        appender.write('########################### Test ###########################' + '\n')
        appender.write(' [The test OA is: %.2f]' % (oa) + ' [The test AA is: %.2f]' % (aa) +
                       ' [The test Kappa is: %.2f]' % (kappa_percentage) + '\n')
        appender.write('\n')

    ckpt_dir = os.path.join(args.checkpoints, args.project_name, args.dataset, f"exp_{args.exp_id}")
    os.makedirs(ckpt_dir, exist_ok=True)
    result_dict = {
        "dataset": args.dataset,
        "model": args.model_name,
        "exp_id": args.exp_id,
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa_percentage),
        "class_acc": {f"Class_{i+1}": float(acc) for i, acc in enumerate(class_accuracy)}
    }
    with open(os.path.join(ckpt_dir, 'test_result.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
    print("Metrics successfully dumped to JSON.")
    return oa


def main():
    args = args_parser()
    
    exp_dir = f"exp_{args.exp_id}"
    model_dir_path = os.path.join(args.results, args.project_name, args.dataset, exp_dir)
    
    os.makedirs(model_dir_path, exist_ok=True)
    
    args.log_file = os.path.join(model_dir_path, 'log.txt')

    # ==========================================
    # 核心管线劫持：强行写入 True 开启训练集的构建逻辑
    # 彻底阻断 data_loader 抛出残缺的一维矩阵
    # ==========================================
    test_loader = build_test_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from models import build_model
    model = build_model(args.model_name, args.in_channels, args.num_class, args.patch_size).to(device)
    
    # 自动寻址：去实验目录下捞取精度最高的 .pth
    if not getattr(args, 'modelfile', None) or not os.path.exists(args.modelfile):
        ckpt_dir = os.path.join(args.checkpoints, args.project_name, args.dataset, f"exp_{args.exp_id}")
        pth_files = glob.glob(os.path.join(ckpt_dir, 'model_*.pth'))
        if pth_files:
            # 解析文件名中的 OA 值进行绝对降序排列，防误抓
            pth_files.sort(key=lambda x: float(os.path.basename(x).replace('model_', '').replace('.pth', '')), reverse=True)
            args.modelfile = pth_files[0]  # [关键修正] 必须赋值给 modelfile
            print(f"[Auto-resolve] 成功加载最高精度权重: {args.modelfile}")
        else:
            raise FileNotFoundError(f"未找到权重文件: {ckpt_dir}")
            
    # [架构修复] 这两行必须退格到和 if 判断平齐，确保任何情况下都能被执行
    model.load_state_dict(torch.load(args.modelfile, weights_only=True))
    test(model, device, test_loader, args)


if __name__ == '__main__':
    main()