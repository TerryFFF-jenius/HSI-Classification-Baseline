#!/bin/bash

METHODS=("cacft" "lite_hcnet" "lssan" "msdan" "simpoolformer" "gscvit" "spectralformer" "ssftt")
DATASETS=("LongKou" "HanChuan" "HongHu")
SEEDS=(100 200 300 400 500)

EPOCHS=200
BATCH_SIZE=16

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP_ID="${dataset}_${method}_ep${EPOCHS}_seed${seed}"
            
            echo "========================================================"
            echo ">>> [TRAINING] Dataset: $dataset | Method: $method | Seed: $seed"
            python main_train.py \
                --dataset "$dataset" \
                --model_name "$method" \
                --exp_id "$EXP_ID" \
                --seed "$seed" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE"
            
            echo ">>> [TESTING] Auto-evaluating..."
            python main_test.py \
                --dataset "$dataset" \
                --model_name "$method" \
                --exp_id "$EXP_ID" \
                --seed "$seed"
                
            echo "<<< [DONE] Dataset: $dataset | Method: $method | Seed: $seed"
            sleep 10
        done
    done
done