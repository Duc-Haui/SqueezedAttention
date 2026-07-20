#!/bin/bash

# ==========================================
# DEBUG PIPELINE - 2 SAMPLES, 1 DATASET
# ==========================================

REPO="/home/mtahackathon/Desktop/DucDang/SqueezedAttention/SqueezedAttention-main"
PYTHON="/home/mtahackathon/anaconda3/envs/sq310/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:$PYTHONPATH"
export HF_HUB_OFFLINE=1

# --- CẤU HÌNH DEBUG ---
DEBUG_DATASET="qasper"
DEBUG_PERCENTILE="0.70"
DEBUG_N_SAMPLES=2
PERC_CLUSTERS="5"
MODEL_NAME="LLaMA-2-7B-32K"
PATH_TO_CLUSTERS="${REPO}/debug-clusters/"

echo "========================================================="
echo "DEBUG: dataset=${DEBUG_DATASET}, samples=${DEBUG_N_SAMPLES}"
echo "========================================================="

rm -rf "${PATH_TO_CLUSTERS}"
mkdir -p "${PATH_TO_CLUSTERS}${DEBUG_DATASET}"

# ─── DEBUG 1: CLUSTERING (8-BIT) ──────────────────────────────
echo ""
echo ">>> [1/3] Debug Clustering (${DEBUG_N_SAMPLES} samples)..."
cd $REPO

$PYTHON << PYEOF
import sys, os, json, torch
sys.path.insert(0, '${REPO}')
os.chdir('${REPO}')

from utils.model_parse import parse_model, get_layers
from squeezedattention.clustering import run_clustering, run_global_threshold
from squeezedattention.utils import truncate_fn
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

print("Imports OK")

model2path = json.load(open("LongBench/config/model2path.json"))
model2maxlen = json.load(open("LongBench/config/model2maxlen.json"))
model_path = model2path["${MODEL_NAME}"]
max_length = model2maxlen["${MODEL_NAME}"]
DEV = torch.device("cuda:0")
dataset_name = "${DEBUG_DATASET}"
output_path = "${PATH_TO_CLUSTERS}${DEBUG_DATASET}"
N = ${DEBUG_N_SAMPLES}

print(">>> Load model 8-bit (Outlier Protection)...")
# CẤU HÌNH 8-BIT TỐI ƯU CHO QUADRO
bnb = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,          
    llm_int8_has_fp16_weight=False   
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
config = LlamaConfig.from_pretrained(model_path)
config.return_qkv_states = True
config._flash_attn_2_enabled = False
config._attn_implementation = "sdpa"
model = LlamaForCausalLM.from_pretrained(model_path, config=config,
    quantization_config=bnb, device_map={"": 0})
model.eval()
print("Model loaded OK")

dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json"))
data = load_dataset('THUDM/LongBench', dataset_name, split='test')
prompt_format = dataset2prompt[dataset_name]
prompt_only_format = dataset2prompt[dataset_name + '_prompt']
data_all = list(data)[:N]
print(f"Dataset: {N} samples")

shared_prefix_length = {}
for i, d in enumerate(data_all):
    prompt = prompt_format.format(**d)
    prompt_only = prompt_only_format.format(**d)
    prompt, spl = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset_name, DEV)
    shared_prefix_length[i] = spl
    assert spl > 0

layers = get_layers(model, parse_model(model))
all_q, all_k, all_v = [], [], []
dataidx = 0

def hook_fn(module, inp, out):
    _, qkv, _ = out
    q, k, v = qkv
    sp = shared_prefix_length[dataidx]
    all_q.append(q[:,:,:sp].cpu())
    all_k.append(k[:,:,:sp].cpu())
    all_v.append(v[:,:,:sp].cpu())

for layer in layers:
    layer.self_attn.register_forward_hook(hook_fn)

os.makedirs(output_path, exist_ok=True)
for dataidx, d in enumerate(tqdm(data_all, desc="Clustering")):
    all_q.clear(); all_k.clear(); all_v.clear()

    prompt = prompt_format.format(**d)
    prompt_only = prompt_only_format.format(**d)
    prompt, _ = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset_name, DEV)
    input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)
    print(f"  [{dataidx}] tokens={len(input_ids[0])}, sp_len={shared_prefix_length[dataidx]}")

    with torch.no_grad():
        model.generate(input_ids, do_sample=False, max_new_tokens=1,
                       use_cache=False, output_attentions=False)

    sp_len = shared_prefix_length[dataidx]
    num_centroids = max(1, int(0.05 * (sp_len - 100)))
    print(f"  [{dataidx}] num_centroids={num_centroids}")

    ct_dict, cl_dict = run_clustering(all_k, num_centroids, observation_window=100, device=DEV)
    gt_dict = run_global_threshold(all_k, all_q, ct_dict, cl_dict, num_centroids,
                                   observation_window=100, device=DEV)

    for k_idx in ct_dict: ct_dict[k_idx] = ct_dict[k_idx].cpu().half()
    for k_idx in cl_dict: cl_dict[k_idx] = cl_dict[k_idx].cpu().to(torch.int32)

    torch.save(ct_dict, f'{output_path}/centroids_tensor_dict_{dataidx}_{num_centroids}.pt')
    torch.save(cl_dict, f'{output_path}/centroids_labels_dict_{dataidx}_{num_centroids}.pt')
    torch.save(gt_dict, f'{output_path}/global_threshold_{dataidx}_{num_centroids}.pt')
    print(f"  [{dataidx}] Saved OK")

print(f"Clustering xong! Files: {os.listdir(output_path)}")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ [1/3] CLUSTERING THẤT BẠI"
    exit 1
fi
echo "[1/3] Clustering OK"

# ─── DEBUG 2: PRED ────────────────────────────────────────────
echo ""
echo ">>> [2/3] Debug Pred..."
cd ${REPO}/LongBench

echo "Cluster files:"
ls "${PATH_TO_CLUSTERS}${DEBUG_DATASET}/" | head -6

$PYTHON pred.py \
    --model $MODEL_NAME \
    --use_centroids \
    --percentile $DEBUG_PERCENTILE \
    --percent_clusters $PERC_CLUSTERS \
    --path_to_clusters "${PATH_TO_CLUSTERS}" \
    --task $DEBUG_DATASET

if [ $? -ne 0 ]; then
    echo "[2/3] PRED THẤT BẠI"
    exit 1
fi

PRED_FILE="pred/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${DEBUG_PERCENTILE}/${DEBUG_DATASET}.jsonl"
if [ ! -f "$PRED_FILE" ]; then
    echo "[2/3] KHÔNG CÓ FILE: $PRED_FILE"
    exit 1
fi

echo " [2/3] Pred OK"
echo "Pred output (1 dòng đầu, 200 ký tự):"
head -1 "$PRED_FILE" | cut -c1-200

# Kiểm tra lặp từ
head -1 "$PRED_FILE" | $PYTHON -c "
import json, sys
data = json.loads(sys.stdin.read())
pred = data['pred']
words = pred.split()
if words:
    max_repeat = max(words.count(w) for w in set(words))
    if max_repeat > 20:
        print(f'WARNING: Lặp từ! max_repeat={max_repeat}, pred={pred[:100]}')
    else:
        print(f'OK: Output bình thường. pred={pred[:100]}')
else:
    print('WARNING: pred rỗng!')
"

# ─── DEBUG 3: EVAL ────────────────────────────────────────────
echo ""
echo ">>> [3/3] Debug Eval..."

$PYTHON eval.py \
    --model $MODEL_NAME \
    --use_centroids \
    --percentile $DEBUG_PERCENTILE \
    --percent_clusters $PERC_CLUSTERS

if [ $? -ne 0 ]; then
    echo "[3/3] EVAL THẤT BẠI"
    exit 1
fi

echo "[3/3] Eval OK"
echo "Kết quả:"
cat "pred/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${DEBUG_PERCENTILE}/result.json"

# ─── TỔNG KẾT ─────────────────────────────────────────────────
echo ""
echo "========================================================="
echo "DEBUG HOÀN THÀNH — Pipeline không lỗi!"
echo "========================================================="

read -p "Xóa file debug? (y/n): " choice
if [ "$choice" = "y" ]; then
    rm -rf "${PATH_TO_CLUSTERS}"
    rm -f "pred/${MODEL_NAME}_PC${PERC_CLUSTERS}_PERC${DEBUG_PERCENTILE}/${DEBUG_DATASET}.jsonl"
    echo "Đã xóa file debug"
fi