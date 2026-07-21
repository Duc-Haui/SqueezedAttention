#!/bin/bash

# ==============================================================================
# PIPELINE VALUE-AWARE SQUEEZED ATTENTION + BASELINE
# (QASPER + NARRATIVEQA)
# ==============================================================================
# Pipeline 4 bước:
#   1. Offline Clustering VALUE-AWARE → Clusters-VA/ (Joint K-V + variance)
#   2. Pred (Baseline full KV + VA-Squeezed)
#   3. Eval cả 2
#   4. Bảng kết quả: Baseline vs VA-Squeezed
# ==============================================================================

set -e  # Dừng ngay nếu có lỗi

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
REPO="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/SqueezedAttention-main"
VA_PROJECT="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/value_aware_squeezed_project"
export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="/home/mtahackathon/anaconda3/envs/sq310/bin/python"

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
MODEL_NAME="LLaMA-2-7B-32K"
DATASETS=("narrativeqa")
PERC_CLUSTERS=5
PERCENTILES=("0.70" "0.80" "0.90")
OBS_WINDOW=100

# Value-Aware params
ALPHA=1.0       # Trọng số K trong joint K-V clustering
BETA=0.2        # Trọng số V (0 = tắt value-aware, về baseline gốc)
GAMMA=0.05       # Hệ số boost variance khi tính threshold

# Output path cho clusters VA
PATH_TO_CLUSTERS_VA="${REPO}/Clusters-VA/"

echo "========================================================="
echo "  PIPELINE VALUE-AWARE SQUEEZED ATTENTION"
echo "  Model:      $MODEL_NAME"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Centroids:  ${PERC_CLUSTERS}%"
echo "  Pruning:    ${PERCENTILES[*]}"
echo "  VA params:  α=$ALPHA β=$BETA γ=$GAMMA"
echo "  Clusters:   $PATH_TO_CLUSTERS_VA"
echo "========================================================="

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: OFFLINE CLUSTERING VALUE-AWARE (Joint K-V + variance)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================="
echo ">>> BƯỚC 1: Clustering VALUE-AWARE cho: ${DATASETS[*]}"
echo "========================================================="

cd "$REPO"
for DATASET in "${DATASETS[@]}"; do
    CLUSTERS_DIR="${PATH_TO_CLUSTERS_VA}${DATASET}/"
    mkdir -p "$CLUSTERS_DIR"

    if [ "$(ls -A "$CLUSTERS_DIR" 2>/dev/null)" ]; then
        echo "✅ [SKIP] Đã tìm thấy cluster VA cho ${DATASET}, bỏ qua."
    else
        echo "🔄 [RUN] Clustering VALUE-AWARE cho: ${DATASET}..."
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
            echo "❌ [ERROR] Clustering VA thất bại cho ${DATASET}!"
            exit 1
        fi
        echo "✅ [DONE] Clustering VA xong cho ${DATASET}"
    fi
done

# ═════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 & 3: PRED + EVAL (Baseline full KV + VA-Squeezed)
# ═════════════════════════════════════════════════════════════════════════════
cd "${REPO}/LongBench"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================================="
    echo ">>> BƯỚC 2 & 3: Xử lý dataset: ${DATASET}"
    echo "========================================================="

    # ─── (A) Baseline: Full KV (lưu vào pred/) ───
    BASELINE_FILE="pred/${MODEL_NAME}_baseline/${DATASET}.jsonl"
    if [ -f "$BASELINE_FILE" ]; then
        echo "✅ [SKIP] Baseline đã có: ${BASELINE_FILE}"
    else
        echo "🚀 Chạy Baseline (Full KV)..."
        PRED_DIR=pred $PYTHON pred.py --model $MODEL_NAME --task $DATASET
        PRED_DIR=pred $PYTHON eval.py --model $MODEL_NAME
    fi

    # ─── (B) VA-Squeezed (lưu vào pred_VA/) ───
    for PERCENTILE in "${PERCENTILES[@]}"; do
        VA_RESULT="pred_VA/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${PERCENTILE}/${DATASET}.jsonl"

        if [ -f "$VA_RESULT" ]; then
            echo "✅ [SKIP] VA-Squeezed (${PERCENTILE}) đã có: ${VA_RESULT}"
            continue
        fi

        echo "🚀 Chạy VA-Squeezed (Pruning: ${PERCENTILE})..."
        PRED_DIR=pred_VA $PYTHON pred.py \
            --model $MODEL_NAME \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters "$PATH_TO_CLUSTERS_VA" \
            --obs_window $OBS_WINDOW \
            --task $DATASET

        PRED_DIR=pred_VA $PYTHON eval.py \
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
echo ">>> BẢNG KẾT QUẢ: BASELINE vs VA-SQUEEZED"
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

# ─── VA-Squeezed ───
for p in percentiles:
    conf = f"{model}_PC{pc}_PERC{p}"
    sq_scores = [get_score("pred_VA", d, conf) for d in datasets]
    budget = (pc / 100.0) + (1.0 - p)
    print(f"{'VA-Sq-'+str(int(p*100))+'%':<25} {budget:<8.3f} " + " ".join([f"{s:<10}" for s in sq_scores]))
print("=" * 60)

# ─── So sánh VA vs Baseline ───
print()
print("So sánh VA vs Baseline:")
for p in percentiles:
    conf = f"{model}_PC{pc}_PERC{p}"
    deltas = []
    for d in datasets:
        base = get_score("pred", d, f"{model}_baseline")
        va   = get_score("pred_VA", d, conf)
        if isinstance(base, (int, float)) and isinstance(va, (int, float)):
            deltas.append(f"{va - base:+.2f}")
        else:
            deltas.append("N/A")
    budget = (pc / 100.0) + (1.0 - p)
    print(f"  {'Δ VA-Sq-'+str(int(p*100))+'%':<23} {budget:<8.3f} " + " ".join([f"{d:<10}" for d in deltas]))
PYEOF

echo ""
echo "========================================================="
echo "  PIPELINE HOÀN THÀNH"
echo "  Clusters VA:  $PATH_TO_CLUSTERS_VA"
echo "  Pred gốc:     ${REPO}/LongBench/pred/"
echo "  Pred VA:      ${REPO}/LongBench/pred_VA/"
echo "========================================================="
