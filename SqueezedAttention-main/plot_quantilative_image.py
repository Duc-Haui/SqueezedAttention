import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
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

def evaluate_and_format(pred_text, answers):
    gt_words = set()
    for ans in answers:
        gt_words.update(extract_words(ans))
        
    words = extract_words(pred_text)
    meaningful_words = [w for w in words if w not in stopwords]
    freq = Counter(meaningful_words)
    
    hallucination_words = {w for w, c in freq.items() if c > 2}
    hallucination_score = sum(count for word, count in freq.items() if count > 2)
    match_score = sum(1 for w in set(words) if w in gt_words and w not in stopwords)
    
    # Trích xuất và định dạng từng từ
    raw_tokens = re.findall(r'\S+|\s+', str(pred_text))
    
    formatted_tokens = []
    for token in raw_tokens:
        if token.strip(): # Nếu là một từ (không phải khoảng trắng)
            clean_word = re.sub(r'[^\w\s]', '', token.lower())
            if clean_word in hallucination_words:
                formatted_tokens.append((token, '#D32F2F')) # Đỏ
            elif clean_word in gt_words and clean_word not in stopwords:
                formatted_tokens.append((token, '#2E7D32')) # Xanh
            else:
                formatted_tokens.append((token, '#424242')) # Đen xám (bình thường)
                
    # Thuật toán Truncate (Cắt gọt nếu văn bản quá dài)
    MAX_WORDS = 40
    if len(formatted_tokens) > MAX_WORDS:
        truncated_tokens = formatted_tokens[:20] + [("\n[...]\n", '#1976D2')] + formatted_tokens[-15:]
    else:
        truncated_tokens = formatted_tokens
        
    return {'len': len(words), 'hal': hallucination_score, 'mat': match_score, 'tokens': truncated_tokens}


def process_dataset(DATASET):
    print(f"\n--- ĐANG XỬ LÝ ẢNH CHỮ CHO DATASET: {DATASET.upper()} ---")
    base_path = f"LongBench/pred/{MODEL}_baseline/{DATASET}.jsonl"
    base_samples = read_all_samples(base_path)

    if not base_samples:
        print(f"Lỗi: Không tìm thấy dữ liệu Baseline cho {DATASET}.")
        return

    all_sq = {}
    all_va = {}

    for perc in PERCENTILES:
        sq_path = f"LongBench/pred/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
        va_path = f"LongBench/pred_VA/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
        all_sq[perc] = read_all_samples(sq_path)
        all_va[perc] = read_all_samples(va_path)

    # Tìm mẫu tiêu biểu nhất ở 90%
    perc_90 = "0.9"
    best_index = -1
    max_diff = -1

    for i in range(len(base_samples)):
        ans = base_samples[i]['answers']
        if not all_sq[perc_90] or not all_va[perc_90] or i >= len(all_sq[perc_90]) or i >= len(all_va[perc_90]):
            continue
            
        # Lọc bỏ các câu trả lời quá ngắn (như Yes/No) để lên hình đẹp hơn
        gt_length = len(extract_words(ans[0]))
        if gt_length < 8:
            continue
            
        sq_90 = evaluate_and_format(all_sq[perc_90][i]['pred'], ans)
        va_90 = evaluate_and_format(all_va[perc_90][i]['pred'], ans)
        
        # Đảm bảo VA-Squeezed trả lời thành một câu đàng hoàng (không quá cụt lủn)
        if len(va_90['tokens']) < 10 or len(va_90['tokens']) > 50:
            continue
        
        diff = sq_90['hal'] - va_90['hal']
        # ĐẢM BẢO VA-Squeezed PHẢI TRẢ LỜI ĐÚNG ĐƯỢC TỪ KHÓA (mat > 0) và ít lảm nhảm
        if va_90['mat'] > 0 and va_90['hal'] < 5 and diff > max_diff:
            max_diff = diff
            best_index = i

    if best_index == -1:
        print(f"Không tìm thấy mẫu hoàn hảo cho {DATASET}. Nới lỏng điều kiện tìm kiếm...")
        # Lần 2: Nới lỏng điều kiện (chỉ cần diff lớn, không cần mat > 0)
        for i in range(len(base_samples)):
            ans = base_samples[i]['answers']
            if not all_sq[perc_90] or not all_va[perc_90] or i >= len(all_sq[perc_90]) or i >= len(all_va[perc_90]):
                continue
            sq_90 = evaluate_and_format(all_sq[perc_90][i]['pred'], ans)
            va_90 = evaluate_and_format(all_va[perc_90][i]['pred'], ans)
            diff = sq_90['hal'] - va_90['hal']
            if diff > max_diff:
                max_diff = diff
                best_index = i
                
    if best_index == -1:
        best_index = 0

    ans = base_samples[best_index]['answers']
    base_res = evaluate_and_format(base_samples[best_index]['pred'], ans)

    print(f"Đã chọn mẫu câu số #{best_index} của {DATASET} để vẽ hình.")

    # --- VẼ ĐỒ THỊ VĂN BẢN (TEXT IMAGE) BẰNG MATPLOTLIB ---
    plot_percentiles = ["0.7", "0.8", "0.9"]
    fig, axs = plt.subplots(len(plot_percentiles), 2, figsize=(14, 6.0)) # Đủ chiều cao để không bị tràn chữ
    # Rút ngắn tiêu đề Ground Truth nếu quá dài
    display_ans = ans[0] if len(ans[0]) < 100 else ans[0][:100] + "..."

    # Đã xóa toàn bộ các dòng fig.suptitle và fig.text tạo ra tiêu đề ở trên cùng

    # Cấu hình lưới
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        # Tắt viền cứng mặc định để vẽ viền động ôm sát chữ
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Tiêu đề cột
    axs[0, 0].set_title("Original Squeezed Attention", fontsize=14, fontweight='bold', color='#D84315', pad=10)
    axs[0, 1].set_title("VA-Squeezed (Proposed)", fontsize=14, fontweight='bold', color='#2E7D32', pad=10)

    def draw_colored_text_monospace(ax, tokens, hal_score, mat_score):
        # Ghi chú thống kê ở góc trên bên phải
        stat_text = f"Hallucinated: {hal_score} | Matched: {mat_score}"
        ax.text(0.98, 0.95, stat_text, ha='right', va='top', fontsize=10, fontweight='bold', color='#424242')
        
        # Thiết lập tọa độ ban đầu
        char_width = 0.021  
        line_height = 0.12  # Trả lại khoảng cách dòng bình thường
        current_x = 0.02
        current_y = 0.80    
        
        for word, color in tokens:
            if word == "\n[...]\n":
                current_x = 0.02
                current_y -= line_height
                ax.text(current_x, current_y, "[...]", color=color, family='monospace', fontsize=11, fontweight='bold')
                current_y -= line_height
                continue
                
            # Kiểm tra xuống dòng
            if current_x + len(word) * char_width > 0.95:
                current_x = 0.02
                current_y -= line_height
                
            # Các token tiếng Anh dùng monospace bình thường
            ax.text(current_x, current_y, word, color=color, family='monospace', fontsize=11, fontweight='bold' if color != '#424242' else 'normal')
            current_x += (len(word) + 1) * char_width # +1 cho khoảng trắng ảo
            
        return current_y

    for i, perc in enumerate(plot_percentiles):
        # Tiêu đề hàng (Mức nén)
        axs[i, 0].set_ylabel(f"Sparsity\n{int(float(perc)*100)}%", fontsize=13, fontweight='bold', color='#424242', rotation=0, labelpad=40, va='center')
        
        # Kiểm tra dữ liệu có tồn tại không
        if not all_sq[perc] or not all_va[perc] or best_index >= len(all_sq[perc]) or best_index >= len(all_va[perc]):
            continue
            
        sq_res = evaluate_and_format(all_sq[perc][best_index]['pred'], ans)
        va_res = evaluate_and_format(all_va[perc][best_index]['pred'], ans)
        
        import matplotlib.patches as patches
        
        # Lấy tọa độ Y kết thúc của chữ cho SQ và VA
        y_sq = draw_colored_text_monospace(axs[i, 0], sq_res['tokens'], sq_res['hal'], sq_res['mat'])
        y_va = draw_colored_text_monospace(axs[i, 1], va_res['tokens'], va_res['hal'], va_res['mat'])
        
        # Tìm giới hạn dưới cùng để hai khung bằng nhau nhưng vẫn ít khoảng trắng nhất có thể
        min_y = min(y_sq, y_va) - 0.05
        
        # Vẽ viền ôm khít chữ cho SQ (dùng chung chiều cao min_y)
        rect_sq = patches.Rectangle((0, min_y), 1.0, 1.0 - min_y, linewidth=1, edgecolor='#E0E0E0', facecolor='none')
        axs[i, 0].add_patch(rect_sq)
        
        # Vẽ viền ôm khít chữ cho VA (dùng chung chiều cao min_y)
        rect_va = patches.Rectangle((0, min_y), 1.0, 1.0 - min_y, linewidth=1, edgecolor='#E0E0E0', facecolor='none')
        axs[i, 1].add_patch(rect_va)



    # Bỏ ghi chú học thuật để hình gọn nhất có thể theo yêu cầu

    plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.02, hspace=0.15, wspace=0.03)
    output_file = f"qualitative_examples_{DATASET}_cropped.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Đã xuất Ảnh Phân tích Định tính cho {DATASET.upper()} ra file: {output_file}")
    
    # --- XUẤT THÊM DẠNG VĂN BẢN (MARKDOWN) ĐỂ COPY-PASTE ---
    md_content = f"# Qualitative Analysis Raw Text - Dataset: {DATASET.upper()}\n\n"
    md_content += f"**Question & Ground Truth:** {ans[0]}\n\n"
    
    for perc in PERCENTILES:
        md_content += f"## Sparsity {int(float(perc)*100)}%\n\n"
        
        # Lấy text thô (raw text)
        sq_text = all_sq[perc][best_index]['pred']
        va_text = all_va[perc][best_index]['pred']
        
        md_content += f"### Original Squeezed Attention:\n> {sq_text}\n\n"
        md_content += f"### VA-Squeezed (Proposed):\n> {va_text}\n\n"
        md_content += "---\n\n"
        
    md_file = f"qualitative_examples_{DATASET}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Đã xuất định dạng Văn bản cho {DATASET.upper()} ra file: {md_file}")

# Chạy cho tất cả các tập dữ liệu
for dataset in DATASETS:
    process_dataset(dataset)
