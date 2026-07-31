import json
import os
import matplotlib.pyplot as plt
import textwrap
from collections import Counter
import re

# Cấu hình
DATASET = "qasper" # Có thể đổi thành narrativeqa
MODEL = "LLaMA-2-7B-32K"
PERCENTILES = ["0.7", "0.8", "0.9"]

# Màu ANSI cho Terminal
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def read_first_sample(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line.strip():
            return json.loads(line)
        return None

def extract_words(text):
    # Tách từ đơn giản, bỏ dấu câu để so sánh
    return re.findall(r'\b\w+\b', text.lower())

def analyze_and_format(text, ground_truth_words):
    words = extract_words(text)
    raw_tokens = text.split()
    
    # Đếm tần suất các từ (loại bỏ các từ nối phổ biến để tránh nhiễu)
    stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'this', 'that', 'it'}
    meaningful_words = [w for w in words if w not in stopwords]
    freq = Counter(meaningful_words)
    
    # Xác định các từ bị ảo giác/lặp lại nhiều (>2 lần)
    hallucinated_words = {w for w, count in freq.items() if count > 2}
    
    # Format text cho Terminal với màu sắc
    colored_text = []
    for token in raw_tokens:
        clean_token = re.sub(r'[^\w\s]', '', token.lower())
        if clean_token in ground_truth_words:
            colored_text.append(f"{GREEN}{BOLD}{token}{RESET}")
        elif clean_token in hallucinated_words:
            colored_text.append(f"{RED}{token}{RESET}")
        else:
            colored_text.append(token)
            
    # Thống kê top lặp từ
    top_repeats = freq.most_common(5)
    repeat_str = ", ".join([f"'{w}': {c} lần" for w, c in top_repeats if c > 2])
    
    return " ".join(colored_text), repeat_str

base_path = f"LongBench/pred/{MODEL}_baseline/{DATASET}.jsonl"
base_data = read_first_sample(base_path)

output_text_file = "sample_comparison_output.txt"
output_plot_file = "sample_comparison_chart.png"

def wrap_text(text, width=90):
    return textwrap.fill(text, width=width)

with open(output_text_file, "w", encoding="utf-8") as out_f:
    out_f.write("="*90 + "\n")
    out_f.write(f" SO SÁNH TẤT CẢ CÁC MỨC PRUNING - DATASET: {DATASET.upper()}\n")
    out_f.write("="*90 + "\n")

    if not base_data:
        print(f"❌ Lỗi: Không tìm thấy Baseline tại {base_path}")
        exit()

    ground_truth = base_data['answers'][0]
    gt_words = set(extract_words(ground_truth))
    gt_len = len(ground_truth.split())
    
    out_f.write(f"\n[CÂU TRẢ LỜI ĐÁP ÁN (GROUND TRUTH)]:\n")
    out_f.write(f"- Độ dài: {gt_len} từ\n")
    out_f.write(f"👉 {wrap_text(ground_truth)}\n\n")
    
    # In ra terminal
    print("\n" + "="*80)
    print(f"{BOLD}[ĐÁP ÁN (GROUND TRUTH)]{RESET}: {ground_truth}")
    print(f"Chú giải màu: {GREEN}Xanh (Khớp đáp án){RESET} | {RED}Đỏ (Nghi vấn Ảo giác/Lặp từ > 2 lần){RESET}")
    print("="*80)
    
    names_for_plot = ["Đáp án\n(GT)"]
    counts_for_plot = [gt_len]
    colors_for_plot = ['#808080']
    
    # --- 2. Baseline ---
    text_base = base_data['pred'].replace('\n', ' ').strip()
    len_base = len(text_base.split())
    colored_base, rep_base = analyze_and_format(text_base, gt_words)
    
    out_f.write(f"[Baseline (Full KV)]\n- Độ dài sinh ra: {len_base} từ\n")
    if rep_base: out_f.write(f"- Cảnh báo lặp từ: {rep_base}\n")
    out_f.write(f"- Nội dung:\n{wrap_text(text_base)}\n" + "-"*90 + "\n")
    
    print(f"\n{BOLD}[Baseline (Full KV)]{RESET} (Dài {len_base} từ)")
    print(wrap_text(colored_base, 100))
    if rep_base: print(f"⚠️ Lặp từ: {rep_base}")
    
    names_for_plot.append("Baseline\n(Full KV)")
    counts_for_plot.append(len_base)
    colors_for_plot.append('#2166AC') 
    
    # --- 3. Các mức Pruning (70%, 80%, 90%) ---
    for perc in PERCENTILES:
        sq_path = f"LongBench/pred/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
        va_path = f"LongBench/pred_VA/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
        
        sq_data = read_first_sample(sq_path)
        va_data = read_first_sample(va_path)
        
        # Squeezed Gốc
        if sq_data:
            text_sq = sq_data['pred'].replace('\n', ' ').strip()
            len_sq = len(text_sq.split())
            colored_sq, rep_sq = analyze_and_format(text_sq, gt_words)
            
            out_f.write(f"[Squeezed Gốc - Nén {int(float(perc)*100)}%]\n- Độ dài sinh ra: {len_sq} từ\n")
            if rep_sq: out_f.write(f"- Cảnh báo lặp từ: {rep_sq}\n")
            out_f.write(f"- Nội dung:\n{wrap_text(text_sq)}\n" + "-"*90 + "\n")
            
            print(f"\n{BOLD}[Squeezed Gốc - Nén {int(float(perc)*100)}%]{RESET} (Dài {len_sq} từ)")
            print(wrap_text(colored_sq, 100))
            if rep_sq: print(f"⚠️ Lặp từ: {rep_sq}")
            
            names_for_plot.append(f"SQ Gốc\n{int(float(perc)*100)}%")
            counts_for_plot.append(len_sq)
            colors_for_plot.append('#F4A582')
            
        # VA-Squeezed
        if va_data:
            text_va = va_data['pred'].replace('\n', ' ').strip()
            len_va = len(text_va.split())
            colored_va, rep_va = analyze_and_format(text_va, gt_words)
            
            out_f.write(f"[VA-Squeezed - Nén {int(float(perc)*100)}%]\n- Độ dài sinh ra: {len_va} từ\n")
            if rep_va: out_f.write(f"- Cảnh báo lặp từ: {rep_va}\n")
            out_f.write(f"- Nội dung:\n{wrap_text(text_va)}\n" + "-"*90 + "\n")
            
            print(f"\n{BOLD}[VA-Squeezed - Nén {int(float(perc)*100)}%]{RESET} (Dài {len_va} từ)")
            print(wrap_text(colored_va, 100))
            if rep_va: print(f"⚠️ Lặp từ: {rep_va}")
            
            names_for_plot.append(f"VA-SQ\n{int(float(perc)*100)}%")
            counts_for_plot.append(len_va)
            colors_for_plot.append('#B2182B')
            
    print("\n" + "="*80)
    print(f"✅ Đã xuất báo cáo chi tiết ra file txt: {output_text_file}")
    
    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names_for_plot, counts_for_plot, color=colors_for_plot, width=0.6)
    plt.title(f'So sánh Số lượng từ sinh ra ở TẤT CẢ các mức nén - Dataset: {DATASET.upper()}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Số lượng từ (Word count)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts_for_plot)*0.02), int(yval), ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_plot_file, dpi=300)
    print(f"✅ Đã vẽ và xuất biểu đồ ra file: {output_plot_file}")
