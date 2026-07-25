"""
Script tạo bảng kết quả và biểu đồ chuẩn bài báo khoa học.
Output:
  - results_table.png       : Bảng kết quả dạng ảnh (booktabs-style)
  - results_chart.png       : Biểu đồ grouped bar chart
  - results_table.tex       : Mã LaTeX bảng (copy thẳng vào paper)
  - results_summary.csv     : File CSV
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ============================================================
# TỰ ĐỘNG PHÁT HIỆN ĐƯỜNG DẪN
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "LongBench")
PRED_DIR = os.path.join(BASE, "pred")
PRED_VA_DIR = os.path.join(BASE, "pred_VA")
OUTPUT_DIR = SCRIPT_DIR

print(f"Script dir : {SCRIPT_DIR}")
print(f"Pred dir   : {PRED_DIR}  (exists: {os.path.isdir(PRED_DIR)})")
print(f"Pred_VA dir: {PRED_VA_DIR}  (exists: {os.path.isdir(PRED_VA_DIR)})")

# ============================================================
# FONT
# ============================================================
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================
def load_result(folder, subfolder):
    path = os.path.join(folder, subfolder, "result.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [OK]   {path}  ->  {data}")
        return data
    print(f"  [MISS] {path}")
    return {}

print("\n" + "=" * 60)
print("ĐANG ĐỌC DỮ LIỆU...")
print("=" * 60)

print("\n[Baseline]")
baseline = load_result(PRED_DIR, "LLaMA-2-7B-32K_baseline")

percs = ["0.7", "0.8", "0.9"]
perc_labels = ["70%", "80%", "90%"]

sq_results = {}
va_results = {}
for perc in percs:
    sub = f"LLaMA-2-7B-32K_PC5_PERC{perc}"
    print(f"\n[Squeezed - {perc}]")
    sq_results[perc] = load_result(PRED_DIR, sub)
    print(f"[Value-Aware - {perc}]")
    va_results[perc] = load_result(PRED_VA_DIR, sub)

metrics = ["narrativeqa", "qasper", "multifieldqa_en"]

def v(d, key):
    """Lấy giá trị, format 2 chữ số thập phân."""
    val = d.get(key, None)
    if val is None:
        return "—"
    return f"{float(val):.2f}"

def vf(d, key):
    val = d.get(key, None)
    return float(val) if val is not None else 0.0

# Kiểm tra
all_data = [baseline] + list(sq_results.values()) + list(va_results.values())
if sum(len(d) for d in all_data) == 0:
    print("\n[LỖI] Không đọc được dữ liệu!")
    sys.exit(1)

# ============================================================
# 1. TẠO MÃ LaTeX CHO BẢNG (copy thẳng vào Overleaf)
# ============================================================
latex_lines = []
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Performance comparison on LongBench Single-Document QA tasks (F1 score). "
                   r"\textbf{Bold} indicates the best result per compression rate.}")
latex_lines.append(r"\label{tab:results}")
latex_lines.append(r"\begin{tabular}{llccc}")
latex_lines.append(r"\toprule")
latex_lines.append(r" & & \multicolumn{3}{c}{\textbf{Single-Doc. QA}} \\")
latex_lines.append(r"\cmidrule(lr){3-5}")
latex_lines.append(r"\textbf{Method} & \textbf{Rate} & \textbf{NQA} & \textbf{Qspr} & \textbf{MFQA} \\")
latex_lines.append(r"\midrule")
latex_lines.append(f"All KV (Baseline) & 100\\% & {v(baseline,'narrativeqa')} & {v(baseline,'qasper')} & {v(baseline,'multifieldqa_en')} \\\\")
latex_lines.append(r"\midrule")

for perc, label in zip(percs, perc_labels):
    sq = sq_results[perc]
    va = va_results[perc]
    label_tex = label.replace("%", "\\%")

    # Tìm giá trị tốt nhất cho mỗi metric ở mức nén này để in đậm
    best = {}
    for m in metrics:
        sv, vv = vf(sq, m), vf(va, m)
        best[m] = "va" if vv > sv else "sq" if sv > vv else "both"

    def fmt(source, metric, val_str):
        b = best[metric]
        if b == source or b == "both":
            return r"\textbf{" + val_str + "}"
        return val_str

    latex_lines.append(
        f"Squeezed Attn & {label_tex} & "
        f"{fmt('sq','narrativeqa', v(sq,'narrativeqa'))} & "
        f"{fmt('sq','qasper', v(sq,'qasper'))} & "
        f"{fmt('sq','multifieldqa_en', v(sq,'multifieldqa_en'))} \\\\"
    )
    latex_lines.append(
        f"VA-Squeezed (Ours) & {label_tex} & "
        f"{fmt('va','narrativeqa', v(va,'narrativeqa'))} & "
        f"{fmt('va','qasper', v(va,'qasper'))} & "
        f"{fmt('va','multifieldqa_en', v(va,'multifieldqa_en'))} \\\\"
    )
    if perc != percs[-1]:
        latex_lines.append(r"\midrule")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_code = "\n".join(latex_lines)

tex_path = os.path.join(OUTPUT_DIR, "results_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_code)
print(f"\n-> LaTeX table saved: {tex_path}")
print("\n--- LaTeX Code ---")
print(latex_code)
print("--- End LaTeX ---\n")

# ============================================================
# 2. LƯU CSV
# ============================================================
csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("Method,Rate,NQA,Qspr,MFQA\n")
    f.write(f"All KV (Baseline),100%,{v(baseline,'narrativeqa')},{v(baseline,'qasper')},{v(baseline,'multifieldqa_en')}\n")
    for perc, label in zip(percs, perc_labels):
        sq, va = sq_results[perc], va_results[perc]
        f.write(f"Squeezed Attn,{label},{v(sq,'narrativeqa')},{v(sq,'qasper')},{v(sq,'multifieldqa_en')}\n")
        f.write(f"VA-Squeezed (Ours),{label},{v(va,'narrativeqa')},{v(va,'qasper')},{v(va,'multifieldqa_en')}\n")
print(f"-> CSV saved: {csv_path}")

# ============================================================
# 3. VẼ BẢNG DẠNG HÌNH ẢNH (Booktabs-style, gọn gàng)
# ============================================================
from matplotlib.patches import FancyBboxPatch

n_rows = 7  # 1 baseline + 3*(1 sq + 1 va)
row_h = 0.45
header_h = 0.5
span_h = 0.35
pad_top = 0.15
pad_bot = 0.15

total_h = pad_top + span_h + header_h + n_rows * row_h + pad_bot
fig_w = 6.0

fig_t, ax_t = plt.subplots(figsize=(fig_w, total_h))
ax_t.set_xlim(0, fig_w)
ax_t.set_ylim(0, total_h)
ax_t.axis("off")

# Vị trí các cột (tâm) - co lại sát nhau
#                    Method    Rate    NQA     Qspr    MFQA
cx = [1.35, 2.85, 3.75, 4.65, 5.55]
left_edge = 0.15
right_edge = fig_w - 0.15

# --- Spanning header "Single-Doc. QA" ---
y_span = total_h - pad_top - span_h * 0.5
ax_t.text((cx[2] + cx[4]) / 2, y_span, "Single-Doc. QA",
          ha="center", va="center", fontsize=10, fontweight="bold", fontstyle="italic")
# Cmidrule dưới spanning
y_cmi = total_h - pad_top - span_h + 0.02
ax_t.plot([cx[2] - 0.42, cx[4] + 0.42], [y_cmi, y_cmi],
          color="black", linewidth=0.7)

# --- Top rule ---
y_top = total_h - pad_top
ax_t.plot([left_edge, right_edge], [y_top, y_top],
          color="black", linewidth=1.8)

# --- Header row ---
y_hdr = total_h - pad_top - span_h - header_h * 0.5
headers = ["Method", "Rate", "NQA", "Qspr", "MFQA"]
for i, h in enumerate(headers):
    ax_t.text(cx[i], y_hdr, h, ha="center", va="center",
              fontsize=9.5, fontweight="bold")

# --- Midrule dưới header ---
y_mid = total_h - pad_top - span_h - header_h
ax_t.plot([left_edge, right_edge], [y_mid, y_mid],
          color="black", linewidth=1.8)

# --- Dữ liệu bảng ---
rows_data = []
rows_data.append({
    "method": "All KV (Baseline)", "rate": "100%",
    "nqa": v(baseline, "narrativeqa"), "qspr": v(baseline, "qasper"),
    "mfqa": v(baseline, "multifieldqa_en"),
    "is_ours": False, "sep_after": True,
})

for pi, (perc, label) in enumerate(zip(percs, perc_labels)):
    sq, va = sq_results[perc], va_results[perc]
    best = {}
    for m in metrics:
        sv, vv = vf(sq, m), vf(va, m)
        best[m] = "va" if vv > sv else ("sq" if sv > vv else "both")

    rows_data.append({
        "method": "Squeezed Attn", "rate": label,
        "nqa": v(sq, "narrativeqa"), "qspr": v(sq, "qasper"),
        "mfqa": v(sq, "multifieldqa_en"),
        "is_ours": False, "sep_after": False,
        "best": {k: (best[k] in ("sq", "both")) for k in metrics},
    })
    rows_data.append({
        "method": "VA-Squeezed (Ours)", "rate": label,
        "nqa": v(va, "narrativeqa"), "qspr": v(va, "qasper"),
        "mfqa": v(va, "multifieldqa_en"),
        "is_ours": True, "sep_after": (pi < len(percs) - 1),
        "best": {k: (best[k] in ("va", "both")) for k in metrics},
    })

y_cur = y_mid
for row in rows_data:
    yc = y_cur - row_h * 0.5  # tâm dòng

    # Highlight dòng "Ours"
    if row["is_ours"]:
        rect = FancyBboxPatch((left_edge, y_cur - row_h + 0.02),
                              right_edge - left_edge, row_h - 0.04,
                              boxstyle="square,pad=0",
                              facecolor="#EBF5FB", edgecolor="none", zorder=0)
        ax_t.add_patch(rect)

    vals = [row["method"], row["rate"], row["nqa"], row["qspr"], row["mfqa"]]
    mkeys = [None, None, "narrativeqa", "qasper", "multifieldqa_en"]

    for i, val_str in enumerate(vals):
        is_best = False
        if "best" in row and mkeys[i]:
            is_best = row["best"].get(mkeys[i], False)
        fw = "bold" if (row["is_ours"] and i == 0) or is_best else "normal"
        ax_t.text(cx[i], yc, val_str, ha="center", va="center",
                  fontsize=9, fontweight=fw)

    y_cur -= row_h

    if row["sep_after"]:
        ax_t.plot([left_edge, right_edge], [y_cur, y_cur],
                  color="#AAAAAA", linewidth=0.5)

# --- Bottom rule ---
ax_t.plot([left_edge, right_edge], [y_cur, y_cur],
          color="black", linewidth=1.8)

table_path = os.path.join(OUTPUT_DIR, "results_table.png")
fig_t.savefig(table_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
print(f"-> Table image saved: {table_path}")

# ============================================================
# 4. VẼ BIỂU ĐỒ ĐƯỜNG (Line Chart - Publication Quality)
# ============================================================
metric_short = {"narrativeqa": "NarrativeQA", "qasper": "Qasper", "multifieldqa_en": "MultifieldQA"}

x_ticks = [70, 80, 90]  # Dùng số thật để khoảng cách đều

color_sq = "#2166AC"   # Xanh đậm (ColorBrewer)
color_va = "#B2182B"   # Đỏ đậm (ColorBrewer)
color_bl = "#404040"
fill_sq = "#92C5DE"    # Xanh nhạt cho fill
fill_va = "#FDDBC7"    # Hồng nhạt cho fill

chart_paths = []
chart_pdfs = []

for idx, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    short = metric_short[metric]

    bl_val = vf(baseline, metric)
    sq_vals = [vf(sq_results[p], metric) for p in percs]
    va_vals = [vf(va_results[p], metric) for p in percs]

    # --- Vùng tô (shaded area) giữa 2 đường ---
    ax.fill_between(x_ticks, sq_vals, va_vals, alpha=0.12,
                    color=color_va, zorder=1)

    # --- Đường Baseline (ngang, nét đứt) ---
    ax.axhline(y=bl_val, color=color_bl, linestyle="--", linewidth=1.0,
               zorder=2, alpha=0.7)
    # Ghi giá trị Baseline trực tiếp lên đường
    ax.text(x_ticks[-1] + 1.2, bl_val, f"Baseline\n({bl_val:.1f})",
            va="center", ha="left", fontsize=7.5, color=color_bl,
            fontstyle="italic", fontweight="bold")

    # --- Đường Squeezed Attn ---
    ax.plot(x_ticks, sq_vals, color=color_sq, marker="o", markersize=8,
            linewidth=2.2, markeredgecolor="white", markeredgewidth=1.5,
            label="Squeezed Attn", zorder=4)

    # --- Đường Value-Aware (Ours) ---
    ax.plot(x_ticks, va_vals, color=color_va, marker="s", markersize=8,
            linewidth=2.2, markeredgecolor="white", markeredgewidth=1.5,
            label="Value-Aware (Ours)", zorder=4)

    # --- Ghi giá trị lên mỗi điểm (smart positioning) ---
    for i, (sv, vv) in enumerate(zip(sq_vals, va_vals)):
        gap = abs(vv - sv)

        # Squeezed Attn annotation
        if sv < vv:
            ax.annotate(f"{sv:.2f}", (x_ticks[i], sv),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7.5, fontweight="bold",
                        color=color_sq)
        else:
            ax.annotate(f"{sv:.2f}", (x_ticks[i], sv),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, fontweight="bold",
                        color=color_sq)

        # Value-Aware annotation
        if vv > sv:
            ax.annotate(f"{vv:.2f}", (x_ticks[i], vv),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, fontweight="bold",
                        color=color_va)
        else:
            ax.annotate(f"{vv:.2f}", (x_ticks[i], vv),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7.5, fontweight="bold",
                        color=color_va)

    # --- Trục & Style ---
    ax.set_xlabel("Compression Rate (%)", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title(short, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["70%", "80%", "90%"], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Y-axis range thông minh
    all_vals = [bl_val] + sq_vals + va_vals
    y_min_val = min(v for v in all_vals if v > 0)
    y_max_val = max(all_vals)
    span = y_max_val - y_min_val if y_max_val > y_min_val else 2.0
    ax.set_ylim(y_min_val - span * 0.30, y_max_val + span * 0.30)

    # X-axis padding để nhìn annotation baseline
    ax.set_xlim(67, 97)

    ax.grid(axis="both", linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    # Legend cho mọi biểu đồ vì đã tách riêng
    ax.legend(loc="best", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fancybox=False, fontsize=8.5,
              handlelength=2.5)

    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, f"results_chart_{metric}.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    chart_paths.append(chart_path)

    chart_pdf = os.path.join(OUTPUT_DIR, f"results_chart_{metric}.pdf")
    fig.savefig(chart_pdf, bbox_inches="tight", facecolor="white")
    chart_pdfs.append(chart_pdf)
    
    plt.close(fig)

print(f"-> Generated {len(chart_paths)} individual charts (PNG/PDF).")

# ============================================================
# 5. TẠO MÃ LaTeX BIỂU ĐỒ (pgfplots - chèn thẳng vào Overleaf)
# ============================================================
metric_tex = {"narrativeqa": "NarrativeQA", "qasper": "Qasper", "multifieldqa_en": "MultifieldQA"}

pgf_paths = []

for idx, metric in enumerate(metrics):
    pgf_lines = []
    pgf_lines.append(r"% === Yêu cầu package trong preamble ===")
    pgf_lines.append(r"% \usepackage{pgfplots}")
    pgf_lines.append(r"% \usepackage{tikz}")
    pgf_lines.append(r"% \pgfplotsset{compat=1.18}")
    pgf_lines.append("")
    pgf_lines.append(r"\begin{figure}[t]")
    pgf_lines.append(r"\centering")
    pgf_lines.append(r"\begin{tikzpicture}")
    pgf_lines.append(r"\begin{axis}[")
    pgf_lines.append(r"    width=6cm,")
    pgf_lines.append(r"    height=5.5cm,")
    pgf_lines.append(r"    title={" + metric_tex[metric] + r"},")
    pgf_lines.append(r"    ylabel={F1 Score},")
    pgf_lines.append(r"    xlabel={Compression Rate},")
    pgf_lines.append(r"    enlarge x limits=0.1,")
    pgf_lines.append(r"    ylabel style={font=\small},")
    pgf_lines.append(r"    xlabel style={font=\small},")
    pgf_lines.append(r"    tick label style={font=\footnotesize},")
    pgf_lines.append(r"    legend style={font=\footnotesize, at={(0.98,0.98)}, anchor=north east},")
    pgf_lines.append(r"    grid=major,")
    pgf_lines.append(r"    grid style={dashed, gray!30},")
    pgf_lines.append(r"    symbolic x coords={70\%, 80\%, 90\%},")
    pgf_lines.append(r"    xtick=data,")
    
    all_vals_m = [vf(baseline, metric)] + [vf(sq_results[p], metric) for p in percs] + [vf(va_results[p], metric) for p in percs]
    y_max_m = max(all_vals_m) if max(all_vals_m) > 0 else 1.0
    y_min_m = min(v for v in all_vals_m if v > 0)
    span = y_max_m - y_min_m if y_max_m > y_min_m else 2.0
    
    pgf_lines.append(f"    ymin={max(0, y_min_m - span * 0.3):.1f},")
    pgf_lines.append(f"    ymax={y_max_m + span * 0.3:.1f},")
    pgf_lines.append(r"]")

    # Baseline (dashed line)
    pgf_lines.append(r"\addplot[gray, thick, dashed, forget plot, domain=0:2, samples=2] coordinates {")
    pgf_lines.append(f"    (70\%, {vf(baseline, metric):.2f})")
    pgf_lines.append(f"    (90\%, {vf(baseline, metric):.2f})")
    pgf_lines.append(r"};")
    pgf_lines.append(r"\node[gray, font=\tiny, anchor=south west] at (axis cs:90\%, " + f"{vf(baseline, metric):.2f}" + r") {Baseline};")

    # Squeezed Attn bars
    pgf_lines.append(r"\addplot[color=blue!70!black, mark=*, thick, nodes near coords, every node near coord/.append style={font=\tiny, anchor=south}] coordinates {")
    for perc, label in zip(percs, perc_labels):
        lab_tex = label.replace("%", r"\%")
        pgf_lines.append(f"    ({lab_tex}, {vf(sq_results[perc], metric):.2f})")
    pgf_lines.append(r"};")

    # Value-Aware bars
    pgf_lines.append(r"\addplot[color=red!70!black, mark=square*, thick, nodes near coords, every node near coord/.append style={font=\tiny, anchor=north}] coordinates {")
    for perc, label in zip(percs, perc_labels):
        lab_tex = label.replace("%", r"\%")
        pgf_lines.append(f"    ({lab_tex}, {vf(va_results[perc], metric):.2f})")
    pgf_lines.append(r"};")

    pgf_lines.append(r"\legend{Squeezed Attn, Value-Aware (Ours)}")
    pgf_lines.append(r"\end{axis}")
    pgf_lines.append(r"\end{tikzpicture}")
    pgf_lines.append(f"\\caption{{Performance on {metric_tex[metric]}.}}")
    pgf_lines.append(f"\\label{{fig:results_{metric}}}")
    pgf_lines.append(r"\end{figure}")

    pgf_code = "\n".join(pgf_lines)
    pgf_path = os.path.join(OUTPUT_DIR, f"results_chart_{metric}.tex")
    with open(pgf_path, "w", encoding="utf-8") as f:
        f.write(pgf_code)
    pgf_paths.append(pgf_path)

print(f"-> Generated {len(pgf_paths)} individual LaTeX charts.")

# ============================================================
# TÓM TẮT OUTPUT
# ============================================================
print("\n" + "=" * 60)
print("TẤT CẢ OUTPUT FILES:")
print("=" * 60)
print(f"  1. Bảng kết quả (ảnh) : {table_path}")
print(f"  2. Bảng kết quả (LaTeX): {tex_path}")
print(f"  3. CSV summary         : {csv_path}")
print(f"  4. Biểu đồ riêng (PNG) :")
for cp in chart_paths:
    print(f"       - {os.path.basename(cp)}")
print(f"  5. Biểu đồ riêng (PDF) :")
for cp in chart_pdfs:
    print(f"       - {os.path.basename(cp)}")
print(f"  6. Biểu đồ riêng (TeX) :")
for cp in pgf_paths:
    print(f"       - {os.path.basename(cp)}")
print("=" * 60)
print("\nDone!")

