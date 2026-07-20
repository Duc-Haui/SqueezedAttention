import os
import warnings

# use this for now to filter out torch warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer , LlamaForCausalLM, LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import gc
import pickle
import textwrap
import sys
import re
from squeezedattention.utils import build_chat, truncate_fn

def post_process_pred(pred, dataset):
    """Clean up model output to remove trailing garbage and hallucinated Q&A pairs."""
    # 1. Strip trailing whitespace/newlines
    pred = pred.strip()
    
    # 2. Cut off at hallucinated "Question:" or "Answer:" patterns
    #    These appear when the model starts generating new Q&A pairs after the real answer
    for stop_pattern in ["\nQuestion:", "\nQuestion ", "\n\nQuestion", "\nQ:"]:
        idx = pred.find(stop_pattern)
        if idx > 0:  # only cut if there's content before the pattern
            pred = pred[:idx].strip()
    
    # 3. For QA datasets, also cut at repeated answer patterns
    qa_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", 
                   "2wikimqa", "musique", "triviaqa", "dureader", "arxivqa", "lsht"]
    if dataset in qa_datasets:
        # Cut at "Answer:" that appears after the first line (hallucinated follow-up)
        lines = pred.split("\n")
        cleaned_lines = []
        for i, line in enumerate(lines):
            # If we see "Answer:" after the first substantive content, stop
            if i > 0 and re.match(r'^\s*Answer\s*:', line):
                break
            cleaned_lines.append(line)
        pred = "\n".join(cleaned_lines).strip()
    
    # 4. Remove excessive trailing newlines (more than 2 consecutive)
    pred = re.sub(r'\n{3,}', '\n\n', pred)
    
    return pred

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--path_to_clusters", type=str, default="/tmp")
    parser.add_argument("--use_centroids", action="store_true")
    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--percentile_lower", type=float, default=0.7)
    parser.add_argument("--obs_window", type=int, default=100)
    parser.add_argument("--task", type=str, default=None)
    return parser.parse_args(args)

def get_pred(data, max_length, max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params):
    # Khởi tạo mô hình một lần duy nhất
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, config_params)

    # iterate over longbench dataset
    for json_obj in tqdm(data):
        different_prefix_index = json_obj.pop('different_prefix_index')
        prompt_raw = prompt_format.format(**json_obj)
        prompt_noquery_raw = prompt_only_format.format(**json_obj)

        # perform truncation
        prompt, truncated_shared_prefix_length = truncate_fn(prompt_raw, prompt_noquery_raw, tokenizer, max_length, dataset, device)
        model.model.shared_prefix_length = truncated_shared_prefix_length
        model.model.different_prefix_index = different_prefix_index

        # encode input
        input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_data.input_ids.shape[-1]

        # ─── VÁ LỖ HỔNG 1: DỌN RÁC TRƯỚC KHI GENERATE ───
        del prompt_raw, prompt_noquery_raw, prompt
        gc.collect()
        torch.cuda.empty_cache()
        # ────────────────────────────────────────────────

        with torch.no_grad():
            if dataset == "samsum": 
                output = model.generate(
                    **input_data,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    use_cache=True
                )[0]
            else:
                output = model.generate(
                    **input_data,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=[tokenizer.eos_token_id],
                    use_cache=True
                )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process_pred(pred, dataset)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

        # ─── VÁ LỖ HỔNG 1: XÓA BIẾN TẠM SAU KHI GENERATE XONG ───
        del input_data, output
        gc.collect()
        torch.cuda.empty_cache()
        # ────────────────────────────────────────────────────────

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, config_params):
    if "LLaMA-2-7B-32K" in model_name or "LWM" in model_name or "longchat" in model_name or "TinyLlama" in model_name:

        config = LlamaConfig.from_pretrained(path)
        config._flash_attn_2_enabled = False

        config._attn_implementation = "eager" if config_params["use_centroids"] else "sdpa"

        config.path_to_clusters_cosine = config_params["path_to_clusters_cosine"]
        config.use_centroids           = config_params["use_centroids"]
        config.hierarchical_lookup     = config_params["hierarchical_lookup"]
        config.percent_clusters        = config_params["percent_clusters"]
        config.percent_clusters_l2     = config_params["percent_clusters_l2"]
        config.percentile              = config_params["percentile"]
        config.percentile_lower        = config_params["percentile_lower"]
        config.obs_window              = config_params["obs_window"]

        print(f"Load {model_name} FP16 (Auto Device Map) | attn={config._attn_implementation}")

        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={
                0: "8GiB",      # Giữ lại 4GB VRAM cho KV Cache
                "cpu": "20GiB"  # Dùng RAM máy tính thoải mái
            },
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    else:
        raise NotImplementedError(f"Model {model_name} chưa được hỗ trợ.")

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    


    args = parse_args()
    
    # ─── VÁ LỖ HỔNG 2: ÉP CHẠY SINGLE THREAD ───
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    # ───────────────────────────────────────────

    model2path = json.load(open("config/model2path.json", "r"))
    # Chú ý: Đảm bảo file model2maxlen_pred.json có chứa "LLaMA-2-7B-32K": 7000
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    model_name = args.model
    max_length = model2maxlen[model_name]

    if args.task is not None:
        datasets = [args.task]
    else:
        if args.e:
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        else:
            datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                        "lcc", "repobench-p"]

    config_params = {}
    config_params['use_centroids'] = args.use_centroids
    config_params['hierarchical_lookup'] = args.hierarchical_lookup
    config_params['percent_clusters'] = args.percent_clusters
    config_params['percent_clusters_l2'] = args.percent_clusters_l2
    config_params['percentile'] = args.percentile
    config_params['percentile_lower'] = args.percentile_lower
    config_params['obs_window'] = args.obs_window

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # Cho phép đổi thư mục output qua biến môi trường PRED_DIR (mặc định: pred)
    base_dir = "pred_e" if args.e else "pred"
    PRED_DIR = os.environ.get("PRED_DIR", base_dir)
    if not os.path.exists(PRED_DIR): os.makedirs(PRED_DIR)

    for dataset in datasets:
        print('\n=========================================')
        print(f'dataset: {dataset}')
        print('=========================================')

        config_params['path_to_clusters_cosine'] = os.path.join(args.path_to_clusters, dataset) + os.sep
        
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')

        if not args.use_centroids:
            savepath = f"{PRED_DIR}/{model_name}_baseline"
        else:
            if args.hierarchical_lookup:
                savepath = f"{PRED_DIR}/{model_name}_PC1_{args.percent_clusters}_PERC1_{args.percentile}_PC2_{args.percent_clusters_l2}_PERC2_{args.percentile_lower}_lookup"
            else:
                savepath = f"{PRED_DIR}/{model_name}_PC{args.percent_clusters}_PERC{args.percentile}"

        if not os.path.exists(savepath): os.makedirs(savepath)
        out_path = savepath + f"/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        prompt_only_format = dataset2prompt[dataset + '_prompt']
        max_gen = dataset2maxlen[dataset]
        
        data_all = [data_sample for data_sample in data]
        
        # GIỚI HẠN CHẠY 50 MẪU ĐẦU TIÊN ĐỂ TEST NHANH (BỎ DÒNG NÀY NẾU MUỐN CHẠY FULL)
   
        
        for i in range(len(data_all)):
            data_all[i]['different_prefix_index'] = i

        # ─── GỌI HÀM TRỰC TIẾP, BỎ ĐA TIẾN TRÌNH ───
        get_pred(data_all, max_length, max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params)
        # ───────────────────────────────────────────