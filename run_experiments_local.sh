#!/bin/bash

# 本地显卡保护策略：一次只放开一个数据集。跑完 LongKou 后，再改成 HanChuan 继续跑
DATASETS=("LongKou") 
METHODS=("baseline" "cacft" "lite_hcnet" "lssan" "msdan" "simpoolformer")

EPOCHS=200
SEED=300
BATCH_SIZE=16

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        EXP_ID="${dataset}_${method}_ep${EPOCHS}"
        
        echo "========================================================"
        echo ">>> [TRAINING START] Dataset: $dataset | Method: $method"
        python main_train.py --dataset "$dataset" --model_name "$method" --exp_id "$EXP_ID" --seed "$SEED" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"
        
        echo ">>> [TESTING START] Auto-evaluating..."
        python main_test.py --dataset "$dataset" --model_name "$method" --exp_id "$EXP_ID"
            
        echo "<<< [DONE] Dataset: $dataset | Method: $method"
        
        # 强制休眠 60 秒，释放显存，给 GPU 降温
        echo "Cooling down local hardware for 60 seconds..."
        sleep 60
    done
done