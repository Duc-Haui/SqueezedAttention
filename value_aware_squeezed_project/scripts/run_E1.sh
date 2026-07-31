#!/bin/bash

# ==============================================================================
# PIPELINE ABLATION: E1 (Chỉ Joint K-V Clustering)
# ==============================================================================
# Pipeline 4 bước:
#   1. Offline Clustering VALUE-AWARE → Clusters-E1/
#   2. Pred (Baseline full KV + E1)
#   3. Eval cả 2
#   4. Bảng kết quả: Baseline vs E1
# ==============================================================================

set -e  # Dừng ngay nếu có lỗi

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
REPO="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/SqueezedAttention-main"
VA_PROJECT="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/value_aware_squeezed_project"
export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="python"

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
MODEL_NAME="LLaMA-2-7B-32K"
DATASETS=("qasper")
PERC_CLUSTERS=5
PERCENTILES=("0.70")
OBS_WINDOW=100

# Value-Aware params
ALPHA=1.0       # Trọng số K trong joint K-V clustering
BETA=0.05       # Trọng số V (E1: Có bật Joint K-V)
GAMMA=0.5         # Hệ số boost variance khi tính threshold (E1: Tắt Variance Boost)

# Output path cho clusters E1
PATH_TO_CLUSTERS_VA="${REPO}/Clusters-E1/"

echo "========================================================="
echo "  PIPELINE ABLATION: E1 (Joint K-V Clustering)"
echo "  Model:      $MODEL_NAME"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Centroids:  ${PERC_CLUSTERS}%"
echo "  Pruning:    ${PERCENTILES[*]}"
echo "  VA params:  α=$ALPHA β=$BETA γ=$GAMMA"
echo "  Clusters:   $PATH_TO_CLUSTERS_VA"
echo "========================================================="

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: OFFLINE CLUSTERING VALUE-AWARE
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================="
echo ">>> BƯỚC 1: Clustering E1 cho: ${DATASETS[*]}"
echo "========================================================="

cd "$REPO"
for DATASET in "${DATASETS[@]}"; do
    CLUSTERS_DIR="${PATH_TO_CLUSTERS_VA}${DATASET}/"
    mkdir -p "$CLUSTERS_DIR"

    if [ "$(ls -A "$CLUSTERS_DIR" 2>/dev/null)" ]; then
        echo "✅ [SKIP] Đã tìm thấy cluster E1 cho ${DATASET}, bỏ qua."
    else
        echo "🔄 [RUN] Clustering E1 cho: ${DATASET}..."
        $PYTHON "${VA_PROJECT}/patches/offline_clustering_value_aware.py" "$MODEL_NAME" \
            --dataset "$DATASET" \
            --percent_clusters $PERC_CLUSTERS \
            --observation_window $OBS_WINDOW \
            --alpha $ALPHA \
            --beta $BETA \
            --gamma $GAMMA \
            --output_path "$CLUSTERS_DIR" \
            --device 0 2>&1 | tee "${CLUSTERS_DIR}/clustering.log"

        if [ $? -ne 0 ]; then
            echo "❌ [ERROR] Clustering E1 thất bại cho ${DATASET}!"
            exit 1
        fi
        echo "✅ [DONE] Clustering E1 xong cho ${DATASET}"
    fi
done

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 & 3: PRED + EVAL (Baseline full KV + E1)
# ═════════════════════════════════════════════════════════════════════════════
cd "${REPO}/LongBench"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================="
    echo ">>> BƯỚC 2 & 3: Xử lý dataset: ${DATASET}"
    echo "========================================================="

    # ─── (A) Baseline: Full KV (TẠM TẮT ĐỂ TIẾT KIỆM THỜI GIAN TUNING) ───
    # BASELINE_FILE="pred/${MODEL_NAME}_baseline/${DATASET}.jsonl"
    # if [ -f "$BASELINE_FILE" ]; then
    #     echo "✅ [SKIP] Baseline đã có: ${BASELINE_FILE}"
    # else
    #     echo "🚀 Chạy Baseline (Full KV)..."
    #     PRED_DIR=pred $PYTHON pred.py --model $MODEL_NAME --task $DATASET
    #     PRED_DIR=pred $PYTHON eval.py --model $MODEL_NAME
    # fi

    # ─── (B) E1 (lưu vào pred_E1/) ───
    for PERCENTILE in "${PERCENTILES[@]}"; do
        VA_RESULT="pred_E1/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${PERCENTILE}/${DATASET}.jsonl"

        if [ -f "$VA_RESULT" ]; then
            echo "✅ [SKIP] E1 (${PERCENTILE}) đã có: ${VA_RESULT}"
            continue
        fi

        echo "🚀 Chạy E1 (Pruning: ${PERCENTILE})..."
        PRED_DIR=pred_E1 $PYTHON pred.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters "$PATH_TO_CLUSTERS_VA" \
            --obs_window $OBS_WINDOW \
            --task $DATASET \
            --load_8bit

        PRED_DIR=pred_E1 $PYTHON eval.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: BẢNG KẾT QUẢ SO SÁNH
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================="
echo ">>> BẢNG KẾT QUẢ: BASELINE vs E1"
echo "========================================================="

$PYTHON <<PYEOF
import json, os
model = "$MODEL_NAME"
datasets = ["qasper", "narrativeqa"]
percentiles = [0.70, 0.80, 0.90]
pc = $PERC_CLUSTERS

print(f"{'Config':<25} {'Budget':<8} " + " ".join([f"{d[:4].upper():<10}" for d in datasets]))
print("=" * 60)

def get_score(base_dir, d_name, conf_dir):
    """Đọc điểm từ result.json hoặc {dataset}.json"""
    r = f"{base_dir}/{conf_dir}/result.json"
    if os.path.exists(r):
        with open(r) as f:
            data = json.load(f)
            return data.get(d_name, data.get("score", "N/A"))
    p = f"{base_dir}/{conf_dir}/{d_name}.json"
    if os.path.exists(p):
        with open(p) as f: return json.load(f).get("score", "N/A")
    return "N/A"

# ─── Baseline (Full KV) ───
base_scores = [get_score("pred", d, f"{model}_baseline") for d in datasets]
print(f"{'Baseline (Full KV)':<25} {'1.000':<8} " + " ".join([f"{s:<10}" for s in base_scores]))
print("-" * 60)

# ─── E1 ───
for p in percentiles:
    conf = f"{model}_PC{pc}_PERC{p}"
    sq_scores = [get_score("pred_E1", d, conf) for d in datasets]
    budget = (pc / 100.0) + (1.0 - p)
    print(f"{'E1-'+str(int(p*100))+'%':<25} {budget:<8.3f} " + " ".join([f"{s:<10}" for s in sq_scores]))
print("=" * 60)

# ─── So sánh E1 vs Baseline ───
print()
print("So sánh E1 vs Baseline:")
for p in percentiles:
    conf = f"{model}_PC{pc}_PERC{p}"
    deltas = []
    for d in datasets:
        base = get_score("pred", d, f"{model}_baseline")
        va   = get_score("pred_E1", d, conf)
        if isinstance(base, (int, float)) and isinstance(va, (int, float)):
            deltas.append(f"{va - base:+.2f}")
        else:
            deltas.append("N/A")
    budget = (pc / 100.0) + (1.0 - p)
    print(f"  {'Δ E1-'+str(int(p*100))+'%':<23} {budget:<8.3f} " + " ".join([f"{d:<10}" for d in deltas]))
PYEOF

echo ""
echo "========================================================="
echo "  PIPELINE E1 HOÀN THÀNH"
echo "  Clusters E1:  $PATH_TO_CLUSTERS_VA"
echo "  Pred gốc:     ${REPO}/LongBench/pred/"
echo "  Pred E1:      ${REPO}/LongBench/pred_E1/"
echo "========================================================="
