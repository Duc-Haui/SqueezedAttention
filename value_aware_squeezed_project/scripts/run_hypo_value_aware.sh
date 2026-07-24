#!/bin/bash
# Kịch bản Auto-Tuning (Grid Search) cho Value-Aware Squeezed Attention

# ─── CẤU HÌNH MÔI TRƯỜNG ───────────────────────────────────────────────────────
REPO="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/SqueezedAttention-main"
VA_PROJECT="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/value_aware_squeezed_project"
export PYTHONPATH="${VA_PROJECT}:${REPO}/transformers/src:${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PRED_DIR="pred_test"
PYTHON="/home/mtahackathon/anaconda3/envs/sq310/bin/python"

# ─── CẤU HÌNH BÀI TOÁN ────────────────────────────────────────────────────────
MODEL_NAME="LLaMA-2-7B-32K"
DATASET="narrativeqa"
PERC_CLUSTERS=5
OBS_WINDOW=100

# Dùng 40 mẫu làm Validation Set.
# LƯU Ý: Trong file pred.py, bạn phải sửa dòng data_all = data_all[:50] thành data_all = data_all[:40] để đồng bộ.
MAX_SAMPLES=40  

# ─── DANH SÁCH THAM SỐ CẦN THỬ NGHIỆM ──────────────────────────────────────────
BETAS=(0.0 0.01 0.05 0.1)
GAMMAS=(0.0 0.01 0.05 0.1 0.15)

LOG_FILE="${VA_PROJECT}/scripts/hpo_results_${DATASET}_PC${PERC_CLUSTERS}.txt"
echo "=== KẾT QUẢ TUNE THAM SỐ CHO $DATASET (PC=$PERC_CLUSTERS) ===" > $LOG_FILE
echo "Ngày chạy: $(date)" >> $LOG_FILE
echo "----------------------------------------------------" >> $LOG_FILE

for BETA in "${BETAS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
        echo -e "\n\n=================================================="
        echo " ĐANG THỬ NGHIỆM: BETA=$BETA | GAMMA=$GAMMA"
        echo "=================================================="
        
        # 1. XÓA SẠCH CỤM CŨ (Để ép tạo lại với tham số mới)
        rm -rf "${VA_PROJECT}/scripts/Clusters-VA-test/${DATASET}/"
        
        # 2. CHẠY CLUSTERING (Tạo cụm offline với 40 mẫu)
        cd $REPO
        $PYTHON ${VA_PROJECT}/patches/offline_clustering_value_aware.py \
            $MODEL_NAME \
            --dataset $DATASET \
            --output_path "${VA_PROJECT}/scripts/Clusters-VA-test/" \
            --percent_clusters $PERC_CLUSTERS \
            --observation_window $OBS_WINDOW \
            --alpha 1.0 \
            --beta $BETA \
            --gamma $GAMMA \
            --max_samples $MAX_SAMPLES
            
        # 3. CHẠY PREDICT
        cd $REPO/LongBench
        $PYTHON pred.py \
            --model $MODEL_NAME \
            --task $DATASET \
            --use_centroids \
            --percent_clusters $PERC_CLUSTERS \
            --percentile 0.5 \
            --obs_window $OBS_WINDOW \
            --path_to_clusters "${VA_PROJECT}/scripts/Clusters-VA-test/"
            
        # 4. CHẠY EVAL (Lấy điểm)
        # Bắt buộc truyền --percentile 0.5 để khớp tên thư mục do pred.py tạo ra
        EVAL_OUTPUT=$($PYTHON eval.py --model $MODEL_NAME --use_centroids --percent_clusters $PERC_CLUSTERS --percentile 0.5)
        
        # 5. GHI KẾT QUẢ VÀO FILE LOG
        echo "BETA=$BETA | GAMMA=$GAMMA ---> $EVAL_OUTPUT" >> $LOG_FILE
        echo ">>> Hoàn thành vòng lặp. Đã lưu kết quả."
        
        # Giải phóng RAM/VRAM
        sleep 2
    done
done

echo -e "\n\n🎉 TẤT CẢ ĐÃ XONG! Hãy mở file $LOG_FILE để xem cặp tham số nào đạt điểm cao nhất."
