import os
import json
import pandas as pd

def generate_table(dataset):
    methods = ["baseline", "cacft", "lite_hcnet", "lssan", "msdan", "simpoolformer"]
    rows = []
    
    for method in methods:
        ckpt_dir = f"./checkpoints/own/{dataset}/exp_{dataset}_{method}_ep200"
        json_path = os.path.join(ckpt_dir, "test_result.json")
        epoch_path = os.path.join(ckpt_dir, "best_epoch.txt")
        
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        best_epoch = "N/A"
        if os.path.exists(epoch_path):
            with open(epoch_path, 'r') as f:
                best_epoch = f.readline().split(": ")[1].strip()
                
        row = {
            "Method": method,
            "OA(%)": f"{data['oa']:.2f}",
            "AA(%)": f"{data['aa']:.2f}",
            "Kappa": f"{data['kappa']:.2f}",
            "Best Epoch": best_epoch,
        }
        for cls, acc in data["class_acc"].items():
            row[cls] = f"{acc:.2f}"
            
        rows.append(row)
        
    if rows:
        df = pd.DataFrame(rows)
        cols = ["Method", "OA(%)", "AA(%)", "Kappa", "Best Epoch"] + [c for c in df.columns if "Class" in c]
        df = df[cols]
        output_path = f"{dataset}_Comparison_Table.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 成功生成表格: {output_path}")

if __name__ == '__main__':
    for d in ["LongKou", "HanChuan", "HongHu"]:
        generate_table(d)