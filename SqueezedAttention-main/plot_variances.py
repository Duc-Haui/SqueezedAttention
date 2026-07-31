import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

# Cấu hình
DATASET = "narrativeqa"
CLUSTERS_DIR = f"Clusters-VA/{DATASET}"

# Tìm file variance đầu tiên (dataidx 0)
search_pattern = os.path.join(CLUSTERS_DIR, "normalized_variance_0_*.pt")
files = glob.glob(search_pattern)

if not files:
    print(f"❌ Lỗi: Không tìm thấy file {search_pattern}")
    print("Hãy chắc chắn bạn đã chạy Clustering VA xong.")
    exit()

var_file = files[0]
print(f"Đang đọc dữ liệu từ: {var_file}")

# Load tensor variance: shape có thể là (Layers, Heads, Clusters) hoặc được bọc trong Dict
try:
    nvar = torch.load(var_file, map_location='cpu')
except Exception as e:
    print(f"Lỗi khi load file: {e}")
    exit()

# Xử lý trường hợp file lưu dưới dạng Dictionary (từng Layer)
if isinstance(nvar, dict):
    tensor_list = [t.flatten() for t in nvar.values() if isinstance(t, torch.Tensor)]
    if tensor_list:
        nvar = torch.cat(tensor_list)
    else:
        print("Lỗi: Không tìm thấy Tensor nào trong Dictionary.")
        exit()

# Chuyển thành numpy 1D array để vẽ
nvar_flat = nvar.flatten().numpy()

# Lọc bỏ các giá trị 0 hoặc quá nhỏ để thấy rõ phần đuôi (tail)
nvar_filtered = nvar_flat[nvar_flat > 1e-4]

if len(nvar_filtered) == 0:
    print("Không có giá trị variance nào đáng kể để vẽ.")
    exit()

# --- VẼ BIỂU ĐỒ PHÂN TÁN (SCATTER) & HISTOGRAM ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Scatter Plot (Mô phỏng tác động)
# Tạo x-axis ngẫu nhiên (chỉ để dàn đều các điểm ra cho dễ nhìn)
x_random = np.random.rand(len(nvar_filtered))
ax1.scatter(x_random, nvar_filtered, alpha=0.3, color='#B2182B', s=10)
ax1.set_title(f"Phân tán Giá trị Variance của các Cụm (Clusters)", fontweight='bold')
ax1.set_ylabel("Normalized Value Variance")
ax1.set_xticks([]) # Ẩn trục X vì nó chỉ là ngẫu nhiên

# Vẽ đường GAMMA threshold mô phỏng (ví dụ những cụm có variance > mức trung bình)
mean_var = np.mean(nvar_filtered)
ax1.axhline(mean_var, color='blue', linestyle='--', label=f'Trung bình (Mean)')
ax1.axhline(mean_var + np.std(nvar_filtered), color='green', linestyle=':', label=f'Cao (Rescued by VA)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Histogram (Phân bố)
ax2.hist(nvar_filtered, bins=50, color='#2166AC', edgecolor='black', alpha=0.7)
ax2.set_title(f"Phân bố (Distribution) của Value Variance", fontweight='bold')
ax2.set_xlabel("Normalized Value Variance")
ax2.set_ylabel("Số lượng cụm (Count)")
ax2.grid(axis='y', alpha=0.3)

# Thêm ghi chú
fig.suptitle(f"Trực quan hóa Phân bố Value Variance - Dataset: {DATASET.upper()}", fontsize=16, fontweight='bold')
plt.figtext(0.5, 0.01, "Ý nghĩa: Các điểm nằm trên cao (màu đỏ) là các cụm mang nội dung phức tạp. Thuật toán VA-Squeezed ưu tiên giữ lại chúng!", ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
output_file = "variance_analysis.png"
plt.savefig(output_file, dpi=300)
print(f"✅ Đã xuất biểu đồ Variance thành công ra file: {output_file}")
