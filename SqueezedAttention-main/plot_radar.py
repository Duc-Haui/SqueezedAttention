import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Cấu hình
MODEL = "LLaMA-2-7B-32K"
PERC = "0.7"

paths = {
    "Baseline (Full KV)": f"LongBench/pred/{MODEL}_baseline/result.json",
    f"Squeezed Gốc ({int(float(PERC)*100)}%)": f"LongBench/pred/{MODEL}_PC5_PERC{PERC}/result.json",
    f"VA-Squeezed ({int(float(PERC)*100)}%)": f"LongBench/pred_VA/{MODEL}_PC5_PERC{PERC}/result.json"
}

data_scores = {}
datasets = set()

# Đọc dữ liệu
for label, path in paths.items():
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                # Có thể có nhiều dòng hoặc 1 json dictionary, lấy json hợp lệ đầu tiên
                content = json.load(f)
                data_scores[label] = content
                datasets.update(content.keys())
            except:
                pass

if not data_scores:
    print("❌ Lỗi: Không tìm thấy file result.json nào. Hãy chắc chắn bạn đã chạy lệnh eval.py.")
    exit()

datasets = list(datasets)
N = len(datasets)

if N < 3:
    print("⚠️ Radar Chart cần ít nhất 3 dataset để hiển thị đẹp. Hiện tại chỉ có:", N)
    if N == 0: exit()

# Tính toán góc cho radar chart
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Khép vòng

plt.figure(figsize=(10, 8), dpi=300)
ax = plt.subplot(111, polar=True)

# Lưới và nhãn
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], [d.upper() for d in datasets], size=10, fontweight='bold')
ax.set_rlabel_position(0)
plt.yticks(color="grey", size=8)
plt.ylim(0, max([max(scores.values()) for scores in data_scores.values()]) + 5)

# Màu sắc chuyên nghiệp
colors = {
    "Baseline (Full KV)": "#2166AC",  # Blue
    f"Squeezed Gốc ({int(float(PERC)*100)}%)": "#F4A582", # Light Red/Orange
    f"VA-Squeezed ({int(float(PERC)*100)}%)": "#B2182B"   # Dark Red
}

# Vẽ từng đường
for label, scores in data_scores.items():
    values = [scores.get(d, 0) for d in datasets]
    values += values[:1] # Khép vòng
    
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors.get(label, 'black'))
    ax.fill(angles, values, color=colors.get(label, 'black'), alpha=0.1)

plt.title("So sánh Đa tác vụ (Multi-task Benchmark) trên LongBench", size=16, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

output_file = "radar_chart.png"
plt.tight_layout()
plt.savefig(output_file)
print(f"✅ Đã xuất biểu đồ Radar thành công ra file: {output_file}")
