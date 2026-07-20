#!/bin/bash
# ============================================================================
# Offline Value-Aware Clustering Script
# (drop-in replacement cho run_offline_clustering.sh của repo gốc)
# ============================================================================
# Yêu cầu:
#   1. Đặt project value_aware_squeezed_project ở cùng cấp với repo SqueezedAttention
#   2. Hoặc copy patches/offline_clustering_value_aware.py vào root SqueezedAttention
#   3. Đảm bảo PYTHONPATH chứa value_aware package
# ============================================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Configurable
MODEL=${MODEL:-"LLaMA-2-7B-32K"}
DATASETS=${DATASETS:-"2wikimqa"}      # space-separated: "2wikimqa qasper hotpotqa"
PERC_CLUSTERS=${PERC_CLUSTERS:-5}     # % centroids
OBS_WINDOW=${OBS_WINDOW:-100}
ALPHA=${ALPHA:-1.0}                   # Trọng số K
BETA=${BETA:-0.5}                     # Trọng số V (0 = baseline)
GAMMA=${GAMMA:-0.3}                   # Hệ số boost variance
OUTPUT_BASE=${OUTPUT_BASE:-"./fixed-prompt-clusters-va"}

# Path setup: assume value_aware_squeezed_project là sibling của SqueezedAttention
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VA_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$VA_PROJECT_DIR:$PYTHONPATH"

echo "=========================================="
echo "Value-Aware Offline Clustering"
echo "Model:        $MODEL"
echo "Datasets:     $DATASETS"
echo "Centroids:    ${PERC_CLUSTERS}%"
echo "Obs window:   $OBS_WINDOW"
echo "α=$ALPHA β=$BETA γ=$GAMMA"
echo "Output base:  $OUTPUT_BASE"
echo "=========================================="

# Loop qua datasets
for DATASET in $DATASETS; do
    OUTPUT_PATH="${OUTPUT_BASE}/${DATASET}/"
    mkdir -p "$OUTPUT_PATH"

    echo ""
    echo "[$DATASET] Running offline clustering..."
    python patches/offline_clustering_value_aware.py "$MODEL" \
        --dataset "$DATASET" \
        --output_path "$OUTPUT_PATH" \
        --percent_clusters "$PERC_CLUSTERS" \
        --observation_window "$OBS_WINDOW" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --gamma "$GAMMA" \
        --device 0 2>&1 | tee "$OUTPUT_PATH/clustering.log"
done

echo ""
echo "Done. Clusters saved to $OUTPUT_BASE"
