import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from models import baseNet
from data_loader import build_data_loader, trans_tif
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
    parser.add_argument('--dataset', type=str, default='PaviaU',
                        choices=['PaviaU', 'Houston'])

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.01,
                        help='samples for training')
    parser.add_argument('--val_ratio', type=float, default=0.01)
    parser.add_argument('--is_train', type=str2bool, default=False,
                        help='train or test')
    parser.add_argument('--is_outimg', type=str2bool, default=False,
                        help='output all image or not')
    parser.add_argument('--checkpointsmodelfile', type=str,
                        default='./checkpoints/own/PaviaU/model_93.92.pth')
    parser.add_argument('--seed', type=int, default=300,
                        help='random seed')
    parser.add_argument('--PCA', type=int, default=None, help='PCA')
    parser.add_argument('--allimg', type=str2bool, default=False, help='allimg')
    parser.add_argument('--exp_id', type=str, default='baseline_01', help='experiment id for output isolation')
    # === 新增：注册动态维度参数（接收终端输入，若无则设为 None 交由 data_loader 填充） ===
    parser.add_argument('--num_class', type=int, default=None)
    parser.add_argument('--hsi_bands', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)
    
    # === 新增：注册模型路由参数 ===
    parser.add_argument('--model_name', type=str, default='baseline', choices=['baseline', 'cacft', 'lite_hcnet'], help='Model routing')
    parser.add_argument('--band_patches', type=int, default=1, help='CACFTNet param')
    parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='CACFTNet param')
    args = parser.parse_args()
    return args


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def pred_allimg(model, device, X, y, args):
    """逐像素预测整张图像，输出分类结果图。
    注意：epoch 参数已移除，预测与训练轮次无关。"""
    model.eval()
    height = y.shape[0]
    width = y.shape[1]
    with torch.no_grad():
        outputs = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                if args.allimg:
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    image_patch = image_patch.reshape(
                        1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
                    X_test_image = torch.FloatTensor(
                        image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
                else:
                    if int(y[i, j]) == 0:
                        continue
                    image_patch = X[i:i + args.patch_size, j:j + args.patch_size, :]
                    image_patch = image_patch.reshape(
                        1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
                    X_test_image = torch.FloatTensor(
                        image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                    prediction = model(X_test_image)
                    prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                    outputs[i][j] = prediction + 1
            if i % 20 == 0:
                print('... ... row ', i, ' handling ... ...')

    exp_dir = f"exp_{args.exp_id}"
    out_dir = os.path.join(args.results, args.project_name, args.dataset, exp_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    if args.allimg:
        finalmodelfile = os.path.join(out_dir, 'All_PRED.tif')
        trans_tif(outputs, finalmodelfile)
    else:
        finalmodelfile = os.path.join(out_dir, 'Label_PRED.tif')
        y_mask = y.copy()
        y_mask[y_mask != 0] = 1
        outputs = outputs * y_mask
        trans_tif(outputs, finalmodelfile)


def main():
    args = args_parser()
    print(args)
    model_dir_path = os.path.join(args.results, args.project_name)
    log_file = os.path.join(args.results, args.project_name, 'log.txt')

    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints, args.project_name), exist_ok=True)
    args.log_file = log_file

    X, y = build_data_loader(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from models import build_model
    model = build_model(args.model_name, args.in_channels, args.num_class, args.patch_size).to(device)

    model.load_state_dict(torch.load(args.checkpointsmodelfile, weights_only=True))
    pred_allimg(model, device, X, y, args)


if __name__ == '__main__':
    main()