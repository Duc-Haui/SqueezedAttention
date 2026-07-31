import json
import os
import re
from collections import Counter

# Cấu hình
DATASET = "qasper" 
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

stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'this', 'that', 'it', 'are', 'was', 'be', 'as', 'or'}

def evaluate_and_format(pred_text, answers):
    gt_words = set()
    for ans in answers:
        gt_words.update(extract_words(ans))
        
    words = extract_words(pred_text)
    meaningful_words = [w for w in words if w not in stopwords]
    freq = Counter(meaningful_words)
    
    hallucination_words = {w for w, c in freq.items() if c > 2}
    hallucination_score = sum(count for word, count in freq.items() if count > 2)
    
    unique_words = set(words)
    match_score = sum(1 for w in unique_words if w in gt_words and w not in stopwords)
    
    def colorize_token(match):
        token = match.group(0)
        token_lower = token.lower()
        if token_lower in hallucination_words:
            return f'<span style="color: red; font-weight: bold;">{token}</span>'
        elif token_lower in gt_words and token_lower not in stopwords:
            return f'<span style="color: green; font-weight: bold;">{token}</span>'
        else:
            return token

    formatted_text = re.sub(r'\b\w+\b', colorize_token, str(pred_text))
    total_tokens = len(words)
    
    return {'len': total_tokens, 'hal': hallucination_score, 'mat': match_score, 'txt': formatted_text}

# --- XỬ LÝ DỮ LIỆU ---
base_path = f"LongBench/pred/{MODEL}_baseline/{DATASET}.jsonl"
base_samples = read_all_samples(base_path)

if not base_samples:
    print("Lỗi: Không tìm thấy file dữ liệu Baseline.")
    exit()

# Tải dữ liệu cho tất cả các mức nén
all_sq_samples = {}
all_va_samples = {}

for perc in PERCENTILES:
    sq_path = f"LongBench/pred/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
    va_path = f"LongBench/pred_VA/{MODEL}_PC5_PERC{perc}/{DATASET}.jsonl"
    
    all_sq_samples[perc] = read_all_samples(sq_path)
    all_va_samples[perc] = read_all_samples(va_path)

# Tìm các mẫu có sự chênh lệch ảo giác lớn nhất ở mức nén 90%
scored_samples = []
for i in range(len(base_samples)):
    ans = base_samples[i]['answers']
    
    # Kiểm tra xem có đủ dữ liệu ở 90% không
    perc_90 = "0.9"
    if not all_sq_samples[perc_90] or not all_va_samples[perc_90] or i >= len(all_sq_samples[perc_90]) or i >= len(all_va_samples[perc_90]):
        continue
        
    sq_90 = evaluate_and_format(all_sq_samples[perc_90][i]['pred'], ans)
    va_90 = evaluate_and_format(all_va_samples[perc_90][i]['pred'], ans)
    base_res = evaluate_and_format(base_samples[i]['pred'], ans)
    
    diff = sq_90['hal'] - va_90['hal']
    
    sample_data = {
        'index': i,
        'diff': diff,
        'answers': ans[0],
        'base': base_res,
        'percs': {}
    }
    
    for perc in PERCENTILES:
        # Nếu thiếu dữ liệu ở mức nén này thì bỏ qua
        if not all_sq_samples[perc] or not all_va_samples[perc] or i >= len(all_sq_samples[perc]) or i >= len(all_va_samples[perc]):
            continue
            
        sq_res = evaluate_and_format(all_sq_samples[perc][i]['pred'], ans)
        va_res = evaluate_and_format(all_va_samples[perc][i]['pred'], ans)
        
        sample_data['percs'][perc] = {
            'sq': sq_res,
            'va': va_res
        }
        
    scored_samples.append(sample_data)

# Sắp xếp theo chênh lệch giảm dần và lấy 3 mẫu tiêu biểu
scored_samples.sort(key=lambda x: x['diff'], reverse=True)
top_samples = scored_samples[:3]

# --- XUẤT RA FILE MARKDOWN ---
md_content = """# Phân tích Định tính Đa mốc (Qualitative Examples across Sparsity Levels)

Tài liệu này trích xuất các mẫu câu sinh ra từ mô hình, thể hiện sự suy thoái dần của thuật toán **SQ Gốc** khi mức độ nén tăng lên (70% -> 90%), đồng thời chứng minh sự tráng kiện của thuật toán **VA-Squeezed** trên cùng các mức độ nén đó.

**Chú giải:**
- <span style="color: green; font-weight: bold;">Văn bản màu xanh</span>: Token trùng khớp chính xác với Đáp án (Ground Truth Match).
- <span style="color: red; font-weight: bold;">Văn bản màu đỏ</span>: Token vô nghĩa bị lặp lại nhiều lần (Hallucination Loop).

---

"""

for idx, sample in enumerate(top_samples):
    md_content += f"## Ví dụ {idx + 1} (Mẫu dữ liệu số #{sample['index']})\n\n"
    md_content += f"**Đáp án chuẩn (Ground Truth):**\n> {sample['answers']}\n\n"
    
    md_content += "| Mô hình | Mức Nén | Thống kê (Tokens) | Văn bản sinh ra (Generated Text) |\n"
    md_content += "| :--- | :---: | :--- | :--- |\n"
    
    b = sample['base']
    md_content += f"| **Baseline** | **0%** | Sinh ra: **{b['len']}**<br>Ảo giác: **{b['hal']}**<br>Khớp: **{b['mat']}** | {b['txt']} |\n"
    
    for perc in PERCENTILES:
        if perc not in sample['percs']:
            continue
            
        s = sample['percs'][perc]['sq']
        v = sample['percs'][perc]['va']
        perc_num = int(float(perc) * 100)
        
        # In dải SQ
        hal_color_sq = "red" if s['hal'] > 0 else "black"
        md_content += f"| **SQ Gốc** | **{perc_num}%** | Sinh ra: **{s['len']}**<br>Ảo giác: **<span style='color:{hal_color_sq}'>{s['hal']}</span>**<br>Khớp: **{s['mat']}** | {s['txt']} |\n"
        
        # In dải VA
        hal_color_va = "red" if v['hal'] > 0 else "black"
        md_content += f"| **VA-Squeezed** | **{perc_num}%** | Sinh ra: **{v['len']}**<br>Ảo giác: **<span style='color:{hal_color_va}'>{v['hal']}</span>**<br>Khớp: **{v['mat']}** | {v['txt']} |\n"
    
    md_content += "\n---\n\n"

output_file = "examples_for_paper.md"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"Đã xuất các ví dụ định tính (đầy đủ các mức nén) ra file: {output_file}")
