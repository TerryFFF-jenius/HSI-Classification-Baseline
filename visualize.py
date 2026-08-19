import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser("Visualization")
    parser.add_argument('--dataset', type=str, default='PaviaU', help='dataset name')
    parser.add_argument('--exp_id', type=str, default='baseline_01', help='experiment identifier')
    parser.add_argument('--project_name', type=str, default='own', help='project name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    
    exp_dir = f"exp_{args.exp_id}"
    tif_path = f"./results/{args.project_name}/{args.dataset}/{exp_dir}/Label_PRED.tif"
    output_path = f"./results/{args.project_name}/{args.dataset}/{exp_dir}/Label_PRED_visual.png"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise FileNotFoundError(f"找不到 TIF 文件: {tif_path}")
    
    img_matrix = dataset.GetRasterBand(1).ReadAsArray()
    
    plt.figure(figsize=(6, 10))
    plt.imshow(img_matrix, cmap='jet')
    plt.colorbar(label='Class ID')
    plt.title("PaviaU Classification Map")
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"伪彩色图已成功生成: {output_path}")
    plt.show()