import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np

# Cấu hình
DATASETS = ["qasper", "narrativeqa", "multifieldqa_en"]
MODEL = "LLaMA-2-7B-32K"
PERCENTILES = ["0.7", "0.8", "0.9"]

def read_all_samples(filepath):
    samples = []
    if not os.path.exists(filepath):
        print(f"Cảnh báo: Không tìm thấy file {filepath}")
        return samples
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def extract_words(text):
    return re.findall(r'\b\w+\b', str(text).lower())

stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'this', 'that', 'it', 'are', 'was', 'be', 'as', 'or', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'have', 'has', 'had', 'been', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'from', 'but', 'not', 'no', 'if', 'then', 'than', 'so', 'am', 'being', 'about', 'which', 'who', 'whom', 'what', 'where', 'when', 'why', 'how', 'there', 'here', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such'}

def evaluate_sample(pred_text, answers):
    gt_words = set()
    for ans in answers:
        gt_words.update(extract_words(ans))
        
    words = extract_words(pred_text)
    meaningful_words = [w for w in words if w not in stopwords]
    freq = Counter(meaningful_words)
    
    # Tổng số từ lặp lại vô nghĩa (ảo giác) > 2 lần
    hallucination_score = sum(count for word, count in freq.items() if count > 2)
    
    # Số lượng từ khóa trùng khớp đáp án
    unique_words = set(words)
    match_score = sum(1 for w in unique_words if w in gt_words and w not in stopwords)
    
    return hallucination_score, match_score

def evaluate_dataset(samples):
    if not samples:
        return 0, 0
    total_hal = 0
    total_match = 0
    for sample in samples:
        hal, match = evaluate_sample(sample['pred'], sample['answers'])
        total_hal += hal
        total_match += match
    return total_hal / len(samples), total_match / len(samples)

def process_dataset(DATASET):
    print(f"\n--- ĐANG XỬ LÝ DATASET: {DATASET.upper()} ---")
    base_path = f"LongBench/pred/{MODEL}_baseline/{DATASET}.jsonl"
    base_samples = read_all_samples(base_path)

    if not base_samples:
        print(f"Lỗi: Không tìm thấy dữ liệu Baseline cho {DATASET}.")
        return

    base_hal, base_match = evaluate_dataset(base_samples)
    print(f"Baseline - Đã xử lý toàn bộ {len(base_samples)} mẫu. Avg Hal: {base_hal:.2f}, Avg Match: {base_match:.2f}")

    sq_hal_scores = []
    sq_match_scores = []
    va_hal_scores = []
    va_match_scores = []

    x_labels = [f"{int(float(p)*100)}%" for p in PERCENTILES]

    for perc in PERCENTILES:
        sq_path = f"LongBench/pred/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
        va_path = f"LongBench/pred_VA/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
        
        sq_samples = read_all_samples(sq_path)
        va_samples = read_all_samples(va_path)
        
        if sq_samples and va_samples:
            sq_h, sq_m = evaluate_dataset(sq_samples)
            va_h, va_m = evaluate_dataset(va_samples)
            print(f"Mức {perc} - Đã xử lý SQ: {len(sq_samples)} mẫu, VA: {len(va_samples)} mẫu.")
        else:
            sq_h, sq_m, va_h, va_m = 0, 0, 0, 0
            
        sq_hal_scores.append(sq_h)
        sq_match_scores.append(sq_m)
        
        va_hal_scores.append(va_h)
        va_match_scores.append(va_m)

    # --- VẼ BIỂU ĐỒ CỘT (BAR CHART) ---
    # Thay vì 1 hàng 2 cột, ta chỉ vẽ 1 biểu đồ duy nhất
    fig, ax1 = plt.subplots(figsize=(8, 6))
    #fig.suptitle(f'Global Token Generation Analysis (N={len(base_samples)}) - Dataset: {DATASET.upper()}', fontsize=16, fontweight='bold', y=1.0)
    fig.suptitle(f'Average Autoregressive Repetition (Hallucination)', fontsize=14, fontweight='bold')

    x_pos = np.arange(len(PERCENTILES))
    width = 0.35

    COLOR_SQ = '#F4A582'
    COLOR_VA = '#B2182B'

    # Biểu đồ 1: Số lượng từ Ảo giác
    rects1 = ax1.bar(x_pos - width/2, sq_hal_scores, width, label='Original Squeezed', color=COLOR_SQ, edgecolor='black')
    rects2 = ax1.bar(x_pos + width/2, va_hal_scores, width, label='VA-Squeezed', color=COLOR_VA, edgecolor='black')

    # Đường Baseline
    ax1.axhline(base_hal, color='#2166AC', linestyle='--', linewidth=2, label='Baseline')

    # ax1.set_title('Average Autoregressive Repetition (Hallucination)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('KV Cache Sparsity Rate', fontsize=12)
    ax1.set_ylabel('Avg. Hallucinated Tokens / Sample', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=12)
    
    # Chỉnh lại legend để nằm hẳn ra ngoài, tránh đè lên cột
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=1.0)
    
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_axisbelow(True)

    def autolabel_inside(ax, rects):
        # Đưa label vào bên trong cột (dùng cho Original Squeezed)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -15), textcoords="offset points", ha='center', va='top', fontweight='bold', color='black')

    def autolabel_above(ax, rects):
        # Để label phía trên cột (dùng cho VA-Squeezed)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='black')

    autolabel_inside(ax1, rects1)
    autolabel_above(ax1, rects2)

    # Đã xóa phần chữ chú thích (note_text) ở dưới biểu đồ theo ý bạn

    plt.tight_layout()
    output_file = f"hallucination_analysis_{DATASET}_left_only.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig) # Đóng figure để giải phóng bộ nhớ
    print(f"Đã xuất biểu đồ Phân tích Hallucination cho {DATASET.upper()} ra file: {output_file}")

# Chạy cho tất cả các tập dữ liệu
for dataset in DATASETS:
    process_dataset(dataset)
