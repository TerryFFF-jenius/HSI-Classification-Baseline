#!/usr/bin/env bash
set -u

# Formal TSSR component ablations.
# Each run is identified by variant, dataset, route configuration, seed, and
# epoch count. Re-running this script skips runs with an existing test_result.

PROJECT_NAME="own"
MODEL_NAME="gscvit_tssr"
EPOCHS=200
BATCH_SIZE=16
TRAIN_RATIO=0.01
VAL_RATIO=0.01
SPECTRAL_GROUPS=8
ROUTE_STRENGTH=0.5
ROUTE_TEMPERATURE=1.0
SEEDS=(100 200 300 400 500)
DATASETS=(HanChuan HongHu LongKou)
VARIANTS=(no_spectral no_spatial)

LOG_DIR="cloud_logs/ablation"
STATUS_FILE="${LOG_DIR}/ablation_status.tsv"
mkdir -p "${LOG_DIR}"

if [ ! -f "${STATUS_FILE}" ]; then
    printf 'variant\tdataset\tseed\tstatus\texp_id\n' > "${STATUS_FILE}"
fi

run_one() {
    local variant="$1"
    local dataset="$2"
    local seed="$3"
    local exp_id="ablation_${variant}_${dataset}_g8_s050_t10_seed${seed}_ep200"
    local run_log="${LOG_DIR}/${exp_id}.log"
    local result_file="checkpoints/${PROJECT_NAME}/${dataset}/exp_${exp_id}/test_result.json"

    if [ -f "${result_file}" ]; then
        echo "[SKIP] ${exp_id} (test_result.json exists)"
        printf '%s\t%s\t%s\tSKIP\t%s\n' "${variant}" "${dataset}" "${seed}" "${exp_id}" >> "${STATUS_FILE}"
        return 0
    fi

    echo "[START] ${exp_id}"
    (
        echo "[RunConfig] variant=${variant} dataset=${dataset} seed=${seed} epochs=${EPOCHS} spectral_groups=${SPECTRAL_GROUPS} route_strength=${ROUTE_STRENGTH} route_temperature=${ROUTE_TEMPERATURE}"
        date -Is
        echo "[TRAIN]"
        if ! python main_train.py \
            --dataset "${dataset}" \
            --model_name "${MODEL_NAME}" \
            --router_variant "${variant}" \
            --spectral_groups "${SPECTRAL_GROUPS}" \
            --route_strength "${ROUTE_STRENGTH}" \
            --route_temperature "${ROUTE_TEMPERATURE}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --train_ratio "${TRAIN_RATIO}" \
            --val_ratio "${VAL_RATIO}" \
            --seed "${seed}" \
            --exp_id "${exp_id}" \
            --project_name "${PROJECT_NAME}"; then
            echo "[FAIL] training"
            exit 1
        fi

        echo "[TEST]"
        if ! python main_test.py \
            --dataset "${dataset}" \
            --model_name "${MODEL_NAME}" \
            --router_variant "${variant}" \
            --spectral_groups "${SPECTRAL_GROUPS}" \
            --route_strength "${ROUTE_STRENGTH}" \
            --route_temperature "${ROUTE_TEMPERATURE}" \
            --batch_size "${BATCH_SIZE}" \
            --train_ratio "${TRAIN_RATIO}" \
            --val_ratio "${VAL_RATIO}" \
            --seed "${seed}" \
            --exp_id "${exp_id}" \
            --project_name "${PROJECT_NAME}"; then
            echo "[FAIL] testing"
            exit 1
        fi
        date -Is
    ) > "${run_log}" 2>&1

    local status=$?
    if [ "${status}" -eq 0 ] && [ -f "${result_file}" ]; then
        echo "[DONE] ${exp_id}"
        printf '%s\t%s\t%s\tDONE\t%s\n' "${variant}" "${dataset}" "${seed}" "${exp_id}" >> "${STATUS_FILE}"
    else
        echo "[FAIL] ${exp_id}; see ${run_log}"
        printf '%s\t%s\t%s\tFAIL\t%s\n' "${variant}" "${dataset}" "${seed}" "${exp_id}" >> "${STATUS_FILE}"
    fi
}

for variant in "${VARIANTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_one "${variant}" "${dataset}" "${seed}"
        done
    done
done

echo "========== ABLATION BATCH SUMMARY =========="
column -t -s $'\t' "${STATUS_FILE}" 2>/dev/null || cat "${STATUS_FILE}"
