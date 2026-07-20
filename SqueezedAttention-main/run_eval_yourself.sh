#!/bin/bash

# =========================================================
# CHỈ CHẠY PRED VÀ EVAL (BỎ QUA CLUSTERING)
# =========================================================

REPO="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/SqueezedAttention-main"
export PYTHONPATH="${REPO}/transformers/src:${REPO}:$PYTHONPATH"
PYTHON="/home/mtahackathon/anaconda3/envs/sq310/bin/python"


export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PATH_TO_CLUSTERS="${REPO}/Clusters-GPU/" 
PERC_CLUSTERS="5"
PERCENTILES=("0.70" "0.80" "0.90")
MODEL_NAME="LLaMA-2-7B-32K" 
DATASETS=("qasper")

cd ${REPO}/LongBench

# ─── XÓA FILE CŨ ĐỂ TRÁNH LỖI CHẤM ĐIỂM TÍCH LŨY ─────────────
echo "Đang dọn dẹp các file kết quả cũ bị lỗi..."
rm -rf pred/${MODEL_NAME}*

# ─── BƯỚC 2 & 3: CHẠY BASELINE ────────────────────────────────
echo "========================================================="
echo ">>> CHẠY BASELINE (All KV)"
echo "========================================================="
for DATASET in "${DATASETS[@]}"; do
    $PYTHON pred.py --model $MODEL_NAME --task $DATASET
    $PYTHON eval.py --model $MODEL_NAME
done

# ─── BƯỚC 2 & 3: CHẠY SQUEEZED ATTENTION ──────────────────────
echo "========================================================="
echo ">>> CHẠY SQUEEZED ATTENTION"
echo "========================================================="
for DATASET in "${DATASETS[@]}"; do
    for PERCENTILE in "${PERCENTILES[@]}"; do
        echo ">>> Pruning: ${PERCENTILE}"
        $PYTHON pred.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters $PATH_TO_CLUSTERS \
            --obs_window 100 \
            --task $DATASET

        $PYTHON eval.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

# ─── BƯỚC 4: BẢNG KẾT QUẢ TỔNG HỢP ────────────────────────────
$PYTHON << 'PYEOF'
import json, os, glob

model = "LLaMA-2-7B-32K"
perc_clusters = 5
percentiles = [0.70, 0.80, 0.90]

def load_result(path):
    for p in [path, path.replace('PERC0.7/', 'PERC0.70/'), path.replace('PERC0.8/', 'PERC0.80/'), path.replace('PERC0.9/', 'PERC0.90/')]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return None

print("\n" + "="*62)
print(f"{'Config':<15} {'Budget':<8} {'NQA':<10} {'Qspr':<10} {'MFQA':<10}")
print("="*62)

r = load_result(f"pred/{model}_baseline/result.json")
if r: print(f"{'All KV':<15} {'1.000':<8} {str(r.get('narrativeqa', 'N/A')):<10} {str(r.get('qasper', 'N/A')):<10} {str(r.get('multifieldqa_en', 'N/A')):<10}")
else: print(f"{'All KV':<15} {'1.000':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("-"*62)

for p in percentiles:
    r = load_result(f"pred/{model}_PC{perc_clusters}_PERC{p}/result.json")
    label = f"Sq-{int(p*100)}%"
    if r: print(f"{label:<15} {'~0.058':<8} {str(r.get('narrativeqa', 'N/A')):<10} {str(r.get('qasper', 'N/A')):<10} {str(r.get('multifieldqa_en', 'N/A')):<10}")
    else: print(f"{label:<15} {'~0.058':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
print("="*62)
PYEOF