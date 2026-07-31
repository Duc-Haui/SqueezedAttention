import os
import sys

# Ép hệ thống dùng phiên bản transformers tuỳ chỉnh của SqueezedAttention để tránh lỗi ImportError
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_dir, "transformers", "src"))
sys.path.insert(0, current_dir)

import gc
import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# CẤU HÌNH
DATASET_NAME = "gov_report" # Tập có context dài nhất trong LongBench (tối đa ~32k)
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 10         # Sinh ra 10 token để đo generation latency
MODEL_PATH = "NousResearch/Llama-2-7b-chat-hf" # Thay đổi thành đường dẫn model thực tế của bạn
# CHÚ Ý: Bạn cần trỏ đúng thư mục đã chạy gom cụm (offline clustering) cho dataset này
CLUSTER_PATH = f"../value_aware_squeezed_project/Clusters-VA/{DATASET_NAME}/" 

def load_model(path, is_baseline=True, percentile=0.7):
    device = "cuda:0"
    config = LlamaConfig.from_pretrained(path)
    config._flash_attn_2_enabled = False
    
    if is_baseline:
        config._attn_implementation = "sdpa"
        config.use_centroids = False
    else:
        config._attn_implementation = "eager"
        config.use_centroids = True
        config.percentile = percentile
        config.percent_clusters = 5
        config.path_to_clusters_cosine = CLUSTER_PATH
        config.hierarchical_lookup = False
        config.percent_clusters_l2 = -1
        config.percentile_lower = 0.7
        config.obs_window = 100
        
    model = LlamaForCausalLM.from_pretrained(
        path,
        config=config,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    
    # Fix buffer device issue
    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    return model, tokenizer

def get_latency(model, input_ids, max_gen=10):
    # Warmup để tránh sai số lần chạy đầu tiên
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
            _ = model.generate(input_ids, max_new_tokens=1, use_cache=True)
            
    # Đo Prefill (Sinh token đầu tiên)
    torch.cuda.synchronize()
    start_prefill = time.time()
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
            _ = model.generate(input_ids, max_new_tokens=1, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = time.time() - start_prefill
    
    # Đo Tổng (Sinh max_gen token)
    torch.cuda.synchronize()
    start_total = time.time()
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
            _ = model.generate(input_ids, max_new_tokens=max_gen, use_cache=True)
    torch.cuda.synchronize()
    total_time = time.time() - start_total
    
    # Thời gian sinh các token tiếp theo (sau prefill)
    # Tính trung bình trên 1 token (chia cho max_gen - 1 vì token đầu tiên đã được tính vào prefill)
    gen_time = (total_time - prefill_time) / (max_gen - 1) if max_gen > 1 else (total_time - prefill_time)
    
    return prefill_time, gen_time

def main():
    print(f"Loading dataset '{DATASET_NAME}'...")
    try:
        dataset = load_dataset('THUDM/LongBench', DATASET_NAME, split='test')
    except Exception as e:
        print(f"Lỗi tải dataset: {e}")
        return
        
    # Sắp xếp theo chiều dài giảm dần và lấy NUM_SAMPLES mẫu dài nhất
    dataset_sorted = sorted(dataset, key=lambda x: x['length'], reverse=True)
    samples = dataset_sorted[:NUM_SAMPLES]
    
    # Lấy prompt format (mặc định cho LongBench)
    try:
        with open("LongBench/config/dataset2prompt.json", "r") as f:
            d2p = json.load(f)
            prompt_format = d2p.get(DATASET_NAME, "{context}\n\n{input}")
    except:
        prompt_format = "{context}\n\n{input}"

    configs = [
        {"name": "Baseline", "is_baseline": True, "percentile": 0.0},
        {"name": "70% Pruning", "is_baseline": False, "percentile": 0.3}, # 70% pruning = giữ lại 30% (0.3)
        {"name": "80% Pruning", "is_baseline": False, "percentile": 0.2}, # 80% pruning = giữ lại 20% (0.2)
    ]
    
    results = {}
    
    for conf in configs:
        print(f"\n=====================================")
        print(f"Đang chạy đo đạc cho: {conf['name']}")
        print(f"=====================================")
        try:
            print("Đang tải model vào GPU (quá trình này mất 1-2 phút tùy ổ cứng, vui lòng đợi)...")
            model, tokenizer = load_model(MODEL_PATH, conf["is_baseline"], conf["percentile"])
            print("Tải model thành công! Đang chạy đo đạc (tiến trình sẽ hiển thị ngay bên dưới)...")
        except Exception as e:
            print(f"Lỗi load model: {e}")
            print("LƯU Ý: Nếu lỗi liên quan đến file cụm (clusters), hãy đảm bảo thư mục CLUSTER_PATH tồn tại!")
            continue
            
        prefill_times = []
        gen_times = []
        
        for sample in tqdm(samples, desc=conf['name']):
            prompt = prompt_format.format(**sample)
            # Khởi tạo Input
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
            
            # Nếu bộ nhớ GPU (VRAM) không đủ, ta cắt bớt context xuống 16k hoặc 24k token
            # Ở đây tôi giới hạn tối đa 24k để chạy vừa trên các loại GPU 24GB thông thường
            if input_ids.shape[1] > 24000:
                input_ids = input_ids[:, -24000:]
                
            p_time, g_time = get_latency(model, input_ids, max_gen=MAX_NEW_TOKENS)
            prefill_times.append(p_time)
            gen_times.append(g_time)
            
            del input_ids
            gc.collect()
            torch.cuda.empty_cache()
            
        # Tính trung bình thời gian (giây)
        results[conf["name"]] = {
            "prefill": np.mean(prefill_times),
            "generation": np.mean(gen_times)
        }
        print(f"> Prefill avg: {np.mean(prefill_times):.4f}s | Generation avg: {np.mean(gen_times):.4f}s")
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Vẽ biểu đồ nếu có kết quả
    if "Baseline" in results:
        plot_results(results)

def plot_results(results):
    labels = ["Prefill Latency", "Generation Latency"]
    
    # Baseline
    base_prefill = results["Baseline"]["prefill"]
    base_gen = results["Baseline"]["generation"]
    
    # 70% Pruning (nếu chạy lỗi sẽ bỏ qua)
    p70_prefill = results.get("70% Pruning", {}).get("prefill", base_prefill)
    p70_gen = results.get("70% Pruning", {}).get("generation", base_gen)
    
    # 80% Pruning
    p80_prefill = results.get("80% Pruning", {}).get("prefill", base_prefill)
    p80_gen = results.get("80% Pruning", {}).get("generation", base_gen)
    
    # Chuẩn hóa về Baseline (Baseline = 1.0)
    norm_base = [1.0, 1.0]
    norm_p70 = [p70_prefill / base_prefill if base_prefill > 0 else 1, p70_gen / base_gen if base_gen > 0 else 1]
    norm_p80 = [p80_prefill / base_prefill if base_prefill > 0 else 1, p80_gen / base_gen if base_gen > 0 else 1]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Màu sắc
    color_base = '#9575CD'   # Tím (Baseline)
    color_p70 = '#64DD17'    # Xanh lá (70%)
    color_p80 = '#4CAF50'    # Xanh lá đậm (80%)
    
    rects1 = ax.bar(x - width, norm_base, width, label='Baseline', color=color_base)
    rects2 = ax.bar(x, norm_p70, width, label='70% Pruning', color=color_p70, alpha=0.9)
    rects3 = ax.bar(x + width, norm_p80, width, label='80% Pruning', color=color_p80, alpha=0.9)
    
    # Thêm số speedup (VD: 1.8X)
    def add_speedup(rects, norm_vals):
        for rect, norm_val in zip(rects, norm_vals):
            speedup = 1.0 / norm_val if norm_val > 0 else 1.0
            height = rect.get_height()
            ax.annotate(f'{speedup:.1f}X',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 10),  
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color='#1976D2')
                        
    add_speedup(rects2, norm_p70)
    add_speedup(rects3, norm_p80)
    
    ax.set_ylabel('Normalized Latency', fontsize=14)
    ax.set_title(f'Average Latency (Fixed Context Length)\nDataset: {DATASET_NAME} (Top 10 Longest Samples)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)
    
    # Tạo style khung nhẹ
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_file = 'plot_latency_benchmark.png'
    plt.savefig(out_file, dpi=300)
    print(f"\n[THÀNH CÔNG] Đã lưu biểu đồ thành công: {out_file}")

if __name__ == "__main__":
    main()
