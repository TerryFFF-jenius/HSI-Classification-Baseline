import os
import json
import glob
import pandas as pd
import numpy as np

def generate_statistical_tables():
    checkpoints_dir = './checkpoints/own/'
    methods = [
        "cacft", "lite_hcnet", "lssan", "msdan",
        "simpoolformer", "gscvit", "spectralformer", "ssftt"
    ]
    datasets = ["LongKou", "HanChuan", "HongHu"]
    seeds = [100, 200, 300, 400, 500]

    for dataset in datasets:
        dataset_dir = os.path.join(checkpoints_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[!] 目录不存在，跳过: {dataset_dir}")
            continue

        results = {m: {} for m in methods}

        for method in methods:
            oa_list, aa_list, kappa_list = [], [], []
            class_acc_lists = {}

            for seed in seeds:
                exp_id = f"{dataset}_{method}_ep200_seed{seed}"
                exp_dir = os.path.join(dataset_dir, f"exp_{exp_id}")
                json_path = os.path.join(exp_dir, "test_result.json")

                if not os.path.exists(json_path):
                    print(f"[!] 缺失: {json_path}")
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                oa_list.append(data.get('oa', 0.0))
                aa_list.append(data.get('aa', 0.0))
                kappa_list.append(data.get('kappa', 0.0))

                for cls_name, acc in data.get('class_acc', {}).items():
                    class_acc_lists.setdefault(cls_name, []).append(acc)

            if not oa_list:
                print(f"[!] {dataset}/{method} 无有效数据，跳过")
                continue

            results[method]['OA'] = f"{np.mean(oa_list):.2f}±{np.std(oa_list, ddof=1):.2f}"
            results[method]['AA'] = f"{np.mean(aa_list):.2f}±{np.std(aa_list, ddof=1):.2f}"
            results[method]['Kappa'] = f"{np.mean(kappa_list):.2f}±{np.std(kappa_list, ddof=1):.2f}"

            for cls_name in sorted(class_acc_lists.keys(), key=lambda x: int(x.split('_')[-1])):
                accs = class_acc_lists[cls_name]
                results[method][cls_name] = f"{np.mean(accs):.2f}±{np.std(accs, ddof=1):.2f}"

        all_metrics = set()
        for m in methods:
            all_metrics.update(results[m].keys())

        class_metrics = sorted([m for m in all_metrics if m.startswith('Class_')],
                               key=lambda x: int(x.split('_')[-1]))
        final_metrics = class_metrics + ['OA', 'AA', 'Kappa']

        rows = []
        for metric in final_metrics:
            row = {'Metrics': metric}
            for method in methods:
                row[method] = results[method].get(metric, 'N/A')
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            output_file = f"{dataset}_Comparison_Table.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[OK] 已生成统计表格: {output_file}")

if __name__ == '__main__':
    generate_statistical_tables()