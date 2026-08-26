import os
import json
import glob
import pandas as pd

def generate_transposed_tables():
    checkpoints_dir = './checkpoints/own/'
    # 强制规定列顺序：Baseline 第一，其余方法随后
    methods = [
        "baseline", "cacft", "lite_hcnet", "lssan", "msdan", 
        "simpoolformer", "gscvit", "spectralformer", "ssftt"
    ]
    # 当前处理的三个数据集
    datasets = ["LongKou", "HanChuan", "HongHu"]

    for dataset in datasets:
        dataset_dir = os.path.join(checkpoints_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[!] 目录不存在，跳过: {dataset_dir}")
            continue

        results = []
        
        # [修复1] 限定遍历 exp_* 目录，避免递归污染
        exp_dirs = glob.glob(os.path.join(dataset_dir, 'exp_*'))
        for exp_dir in exp_dirs:
            json_path = os.path.join(exp_dir, 'test_result.json')
            if not os.path.exists(json_path):
                continue
                
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                model_name = data.get('model')
                if model_name not in methods:
                    print(f"[!] 未知方法 '{model_name}' 于 {exp_dir}，跳过")
                    continue

                row = {'Method': model_name}
                
                # [修复2] 显式排序类别键名，确保行顺序一致
                class_acc = data.get('class_acc', {})
                for cls_name in sorted(class_acc.keys(), key=lambda x: int(x.split('_')[-1])):
                    row[cls_name] = class_acc[cls_name]
                    
                row['OA'] = data.get('oa')
                row['AA'] = data.get('aa')
                row['Kappa'] = data.get('kappa')
                
                results.append(row)
            except Exception as e:
                print(f"[!] 读取文件出错 {json_path}: {e}")

        if not results:
            print(f"[!] 未找到 {dataset} 的有效测试结果，跳过。")
            continue

        df = pd.DataFrame(results)
        df['Method'] = pd.Categorical(df['Method'], categories=methods, ordered=True)
        df = df.sort_values('Method')
        df.set_index('Method', inplace=True)
        df_transposed = df.T
        df_transposed.reset_index(inplace=True)
        df_transposed.rename(columns={'index': 'Metrics'}, inplace=True)
        
        # [修复3] 追加时间戳，保留历史版本
        output_file = f"{dataset}_Comparison_Table.csv"
        df_transposed.to_csv(output_file, index=False)
        print(f"[OK] 已生成转置表格: {output_file}")

if __name__ == '__main__':
    generate_transposed_tables()