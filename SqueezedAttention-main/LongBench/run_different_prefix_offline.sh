# Dừng script ngay lập tức nếu có bất kỳ lỗi nào xảy ra
set -e

# Đồng bộ môi trường
export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main:$PYTHONPATH"
export PYTHONPATH="/home/ubuntu/Desktop/SqueezedAttention-main/value_aware_squeezed_project:$PYTHONPATH"

# =====================================================================
# 🚀 CẤU HÌNH GPU
# Lệnh này cực kỳ quan trọng: Nó ép toàn bộ các tiến trình Python 
# phía dưới chỉ được phép nhìn thấy và chạy trên GPU đầu tiên (cuda:0).
export CUDA_VISIBLE_DEVICES=0
# =====================================================================

REPO="/home/ubuntu/Desktop/SqueezedAttention-main/SqueezedAttention-main"
PYTHON="/home/ubuntu/Desktop/venv_39/bin/python"
PATH_TO_CLUSTERS="${REPO}/fixed-prompt-clusters_v2"
PERC_CLUSTERS="5"
PERCENTILES=("0.70" "0.80" "0.90")
DATASETS=("narrativeqa" "qasper" "multifieldqa_en")

# ─── BƯỚC 1: CLUSTERING ───────────────────────────────────────
echo "========================================================="
echo ">>> BƯỚC 1: Chạy Offline Clustering cho tất cả datasets trên GPU"
echo "========================================================="

cd $REPO

for DATASET in "${DATASETS[@]}"; do
    mkdir -p "${PATH_TO_CLUSTERS}/${DATASET}"
    echo ">>> Clustering: ${DATASET}"
    
    # Thêm cờ --device 0 để chắc chắn K-means chạy trên VRAM của GPU
    $PYTHON offline_clustering.py SmolLM2-135M \
        --dataset $DATASET \
        --percent_clusters $PERC_CLUSTERS \
        --output_path "${PATH_TO_CLUSTERS}/${DATASET}" \
        --device 0 
    
    if [ $? -ne 0 ]; then
        echo "Clustering thất bại cho ${DATASET}, dừng lại."
        exit 1
    fi
    echo "Clustering xong: ${DATASET}"
done

# ─── BƯỚC 2: INFERENCE + EVAL ─────────────────────────────────
echo "========================================================="
echo ">>> BƯỚC 2: Chạy Inference và Đánh giá trên GPU"
echo "========================================================="

cd ${REPO}/LongBench

for DATASET in "${DATASETS[@]}"; do
    for PERCENTILE in "${PERCENTILES[@]}"; do
        echo "========================================================="
        echo "Task=${DATASET} | Cắt tỉa=${PERCENTILE}"
        echo "========================================================="

        # pred.py tự động dò tìm thiết bị qua thư viện torch.cuda.device_count()
        # Vì ta đã set CUDA_VISIBLE_DEVICES=0 ở trên, nó sẽ tự động nhận diện
        # và đẩy thẳng models + inputs lên bộ nhớ GPU.
        $PYTHON pred.py \
            --model SmolLM2-135M \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS \
            --path_to_clusters $PATH_TO_CLUSTERS \
            --task $DATASET

        echo "Task=${DATASET} | Cắt tỉa=${PERCENTILE}"

        # eval.py chạy trên CPU vì nó chỉ làm nhiệm vụ so sánh Text đáp án (Rất nhẹ)
        $PYTHON eval.py \
            --model SmolLM2-135M \
            --use_centroids \
            --percentile $PERCENTILE \
            --percent_clusters $PERC_CLUSTERS
    done
done

echo "========================================================="