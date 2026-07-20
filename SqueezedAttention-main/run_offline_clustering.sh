#!/bin/bash

# ==============================================================================
# PIPELINE SQUEEZED ATTENTION & BASELINE (QASPER + NARRATIVEQA)
# ==============================================================================

REPO="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/SqueezedAttention-main"
VA_PROJECT="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/value_aware_squeezed_project"
export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="/home/mtahackathon/anaconda3/envs/sq310/bin/python"

PATH_TO_CLUSTERS="${REPO}/Clusters-GPU/" 
PERC_CLUSTERS=5
PERCENTILES=("0.70" "0.80" "0.90")
MODEL_NAME="LLaMA-2-7B-32K" 
# --- THÊM NARRATIVEQA VÀO ĐÂY ---
DATASETS=("qasper" "narrativeqa")
# DATASETS=("multifieldqa_en")

cd $REPO

# ─── BƯỚC 1: OFFLINE CLUSTERING ───────────────────────────────────────────────
echo "========================================================="
echo ">>> BƯỚC 1: Chạy Clustering cho các dataset: ${DATASETS[*]}"
echo "========================================================="

cd $REPO
for DATASET in "${DATASETS[@]}"; do
    CLUSTERS_DIR="${PATH_TO_CLUSTERS}${DATASET}/"
    mkdir -p "$CLUSTERS_DIR"
    
    # Check: Nếu thư mục không rỗng (đã có file .pt), thì bỏ qua bước Clustering
    if [ "$(ls -A $CLUSTERS_DIR)" ]; then
        echo "✅ [SKIP] Đã tìm thấy file cụm cho ${DATASET}, bỏ qua Bước 1."
    else
        echo "🔄 [RUN] Không tìm thấy cụm, bắt đầu Clustering cho: ${DATASET}..."
        $PYTHON offline_clustering.py $MODEL_NAME \
            --dataset $DATASET \
            --percent_clusters $PERC_CLUSTERS \
            --observation_window 100 \
            --output_path "$CLUSTERS_DIR"
        
        if [ $? -ne 0 ]; then
            echo "❌ [ERROR] Clustering thất bại!"
            exit 1
        fi
    fi
done

cd ${REPO}/LongBench

# ─── BƯỚC 2 & 3: CHẠY PRED & EVAL ─────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo ">>> BƯỚC 2 & 3: Xử lý dataset: ${DATASET}"
    echo "========================================================="

    # 1. Chạy Baseline
    echo "🚀 Chạy Baseline..."
    $PYTHON pred.py --model $MODEL_NAME --task $DATASET
    $PYTHON eval.py --model $MODEL_NAME 

    # 2. Chạy Squeezed Attention
    for PERCENTILE in "${PERCENTILES[@]}"; do
        echo "🚀 Chạy Squeezed Attention (Pruning: ${PERCENTILE})..."
        $PYTHON pred.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters $PATH_TO_CLUSTERS \
            --obs_window 100 \
            --task $DATASET

        # 4. Chạy Eval (BỎ --task $DATASET)
        $PYTHON eval.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

# ─── BƯỚC 4: BẢNG KẾT QUẢ TỔNG HỢP ───────────────────────────────────────────
echo "========================================================="
echo ">>> BẢNG KẾT QUẢ CUỐI CÙNG"
echo "========================================================="

$PYTHON << 'PYEOF'
import json, os
model = "LLaMA-2-7B-32K"
datasets = ["qasper", "narrativeqa"]
percentiles = [0.70, 0.80, 0.90]

print(f"{'Config':<15} {'Budget':<8} " + " ".join([f"{d[:4].upper():<10}" for d in datasets]))
print("="*60)

# Hàm lấy điểm
def get_score(d_name, conf_dir):
    p = f"pred/{conf_dir}/{d_name}.json" # eval.py thường lưu kết quả ở file json cùng tên task
    if os.path.exists(p):
        with open(p) as f: return json.load(f).get("score", "N/A")
    return "N/A"

# Baseline
base_scores = [get_score(d, f"{model}_baseline") for d in datasets]
print(f"{'All KV':<15} {'1.000':<8} " + " ".join([f"{s:<10}" for s in base_scores]))

# Squeezed
for p in percentiles:
    conf = f"{model}_PC2.5_PERC{p}"
    sq_scores = [get_score(d, conf) for d in datasets]
    budget = (2.5 / 100.0) + (1.0 - p)
    print(f"{'Sq-'+str(int(p*100))+'%':<15} {budget:<8.3f} " + " ".join([f"{s:<10}" for s in sq_scores]))
print("="*60)
PYEOF