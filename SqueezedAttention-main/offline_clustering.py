import time
import os
import torch
import torch.nn as nn
import argparse
from utils.modelutils import *
from utils.datautils import *
from utils.model_parse import (
    parse_model,
    get_layers,
)
from tqdm import tqdm
import pickle
import numpy as np
import math
import sys
import textwrap
import shutil
import json
from squeezedattention.clustering import run_clustering, run_global_threshold
from squeezedattention.utils import build_chat, truncate_fn
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from transformers import BitsAndBytesConfig
from value_aware.clustering import run_value_aware_clustering, normalize_value_variance
from value_aware.threshold import run_value_aware_global_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load (LLaMA-2-7B-32K, TinyLlama-1.1B-Chat, etc.)")
    parser.add_argument('--value_aware', action='store_true',
                    help='Enable value-aware retrieval')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument(
        '--output_path', type=str, default='output/'
    )

    parser.add_argument(
        '--dataset', type=str, default='trec', choices=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                                                        "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                                                        "lcc", "repobench-p"]
    )

    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument('--observation_window', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)



    args = parser.parse_args()
    DEV = torch.device(f"cuda:{args.device}")

    # get maxlen and model path
    model2path = json.load(open("LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("LongBench/config/model2maxlen.json", "r"))
    model_path = model2path[args.model]
    max_length = model2maxlen[args.model]
    
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = LlamaConfig.from_pretrained(model_path)
    config.return_qkv_states = True
    config._flash_attn_2_enabled = False
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        # quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": DEV},
        trust_remote_code=True
    )
    model.eval()
    # Note: Do NOT call model.to(DEV) with bitsandbytes — device_map already handles placement

    # get model layers
    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    # load longbench dataset
    from datasets import load_dataset
    dataset = args.dataset
    dataset_name_prompt = dataset + '_prompt'
    data = load_dataset('THUDM/LongBench', dataset, split='test')

    # define prompt format
    import json
    dataset2prompt = json.load(open("LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("LongBench/config/dataset2maxlen.json", "r"))

    # load prompt format, and use first example in dataset as fixed context
    prompt_format = dataset2prompt[dataset]
    prompt_only_format = dataset2prompt[dataset_name_prompt]
    data_all = [data_sample for data_sample in data]

    # different prefix profiling offline (also need to account for truncation)
    shared_prefix_length = {}
    for i in range(len(data_all)):
        prompt = prompt_format.format(**data_all[i])
        prompt_only = prompt_only_format.format(**data_all[i])

        # perform truncation and get truncated shared prefix length
        prompt, truncated_shared_prefix_length = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, DEV)
        shared_prefix_length[i] = truncated_shared_prefix_length
        assert (truncated_shared_prefix_length > 0) # else, truncated part of input context as well

    # add hooks to profile attn scores
    all_queries_layers = []
    all_keys_layers = []
    all_values_layers = []
    
    # def get_attention_scores(module, inp, out):
    #     _, qkv, _ = out
    #     queries, keys, values = qkv
    #     sp_len = shared_prefix_length[dataidx]
    #     queries = queries[:,:,:sp_len]
    #     keys = keys[:,:,:sp_len]
    #     values = values[:,:,:sp_len]
    #     all_queries_layers.append(queries)
    #     all_keys_layers.append(keys)
    #     all_values_layers.append(values)
    
    
    def get_attention_scores(module, inp, out):
        _, qkv, _ = out
        queries, keys, values = qkv
        sp_len = shared_prefix_length[dataidx]

        # [SỬA Ở ĐÂY] "Đá" dữ liệu sang CPU ngay lập tức
        queries = queries[:,:,:sp_len].cpu()
        keys = keys[:,:,:sp_len].cpu()
        values = values[:,:,:sp_len].cpu()

        all_queries_layers.append(queries)
        all_keys_layers.append(keys)
        all_values_layers.append(values)
    
    
    # Attach the hook to each attention layer
    for layer in layers:
        layer.self_attn.register_forward_hook(get_attention_scores)

    # load dataset format
    for dataidx, d in enumerate(tqdm(data)):
        all_queries_layers = []
        all_keys_layers = []
        all_values_layers = []

        prompt = prompt_format.format(**d)
        prompt_only = prompt_only_format.format(**d)

        # get truncated input prompt
        prompt, _ = truncate_fn(prompt, prompt_only, tokenizer, max_length, dataset, DEV)
        
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids.to(DEV)

        print(f"dataidx: {dataidx} | length of input_ids: {len(input_ids[0])}")
        print(f"dataidx: {dataidx} | shared_prefix_length: {shared_prefix_length[dataidx]}")

        # run generation (only 1 token to extract QKV via hooks)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=1,
                use_cache=False,  # disable KV cache to save VRAM on 16GB Quadro
                output_attentions=False,
            )

        # free VRAM immediately after hook capture
        del generated_ids, input_ids
        torch.cuda.empty_cache()

        # write out data
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        # determine num_centroids based on context length
        sp_len = shared_prefix_length[dataidx]
        percentage = ((args.percent_clusters * 1.0) / 100.0)
        num_centroids = int(percentage * (sp_len - args.observation_window))
        percentage_l2 = ((args.percent_clusters_l2 * 1.0) / 100.0)
        num_centroids_l2 = int(percentage_l2 * (sp_len - args.observation_window))
        print(num_centroids)
        if num_centroids < 1:
            num_centroids = 1
        if args.hierarchical_lookup:
            assert (num_centroids_l2 >= 1)

        # code check memory and report KV budget
        tokens_kept = args.observation_window + num_centroids
        kv_budget_percent = (tokens_kept / sp_len) * 100
        
        print("\n" + "="*50)
        print(f" BÁO CÁO NHANH - MẪU DỮ LIỆU {dataidx}")
        print(f"  - Tổng chiều dài ngữ cảnh gốc : {sp_len} tokens")
        print(f"  - Số tokens giữ lại (KV Cache) : {tokens_kept} tokens")
        print(f"  --> KV Budget sử dụng         : {kv_budget_percent:.2f}% (Nén được {100 - kv_budget_percent:.2f}%)")
        
        import psutil
        ram_used_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        
        torch.cuda.empty_cache() 
        vram_allocated_gb = torch.cuda.memory_allocated(DEV) / (1024 ** 3)
        vram_peak_gb = torch.cuda.max_memory_allocated(DEV) / (1024 ** 3)
        
        print(f" TIÊU THỤ BỘ NHỚ:")
        print(f"  - System RAM (CPU) đang dùng   : {ram_used_gb:.2f} GB")
        print(f"  - GPU VRAM đang dùng (Hiện tại): {vram_allocated_gb:.2f} GB")
        print(f"  - GPU VRAM Đỉnh (Peak)         : {vram_peak_gb:.2f} GB")
        print("="*50 + "\n")
        
        
        # hierarchical
        if args.hierarchical_lookup:
            centroids_tensor_dict_l2, centroids_labels_dict_l2 = run_clustering(all_keys_layers,
                                                                                num_centroids_l2,
                                                                                observation_window=args.observation_window,
                                                                                device=torch.device('cpu'))
            centroids_tensor_dict_l1, centroids_labels_dict_l1 = run_clustering(centroids_tensor_dict_l2,
                                                                                num_centroids,
                                                                                observation_window=0,
                                                                                device=torch.device('cpu'))

            # update centroid_labels to convert L1 -> L2 mapping to be L1 -> keys for evaluation code
            num_lyrs = len(all_keys_layers)
            for i in range(num_lyrs):
                label_dict_l1 = centroids_labels_dict_l1[i]
                label_dict_l2 = centroids_labels_dict_l2[i]
                gathered_tensor = torch.gather(label_dict_l1, -1, label_dict_l2)
                centroids_labels_dict_l1[i] = gathered_tensor

            # run global threshold
            global_threshold_dict_l1 = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict_l1, centroids_labels_dict_l1, num_centroids,
                observation_window=args.observation_window,  device=torch.device('cpu')
            )

            # run global threshold (hierarchical lookup) using L2 denominator
            global_threshold_dict_l2 = run_global_threshold(
                all_keys_layers, all_queries_layers, centroids_tensor_dict_l2, centroids_labels_dict_l2, num_centroids_l2,
                observation_window=args.observation_window,  device=torch.device('cpu')
            )

            # save centroids tensor, labels, global threshold
            os.makedirs(args.output_path, exist_ok=True)
            for k,v in centroids_tensor_dict_l1.items():
                centroids_tensor_dict_l1[k] = centroids_tensor_dict_l1[k].cpu()
            for k,v in centroids_labels_dict_l1.items():
                centroids_labels_dict_l1[k] = centroids_labels_dict_l1[k].cpu()
            for k,v in centroids_tensor_dict_l2.items():
                centroids_tensor_dict_l2[k] = centroids_tensor_dict_l2[k].cpu()
            for k,v in centroids_labels_dict_l2.items():
                centroids_labels_dict_l2[k] = centroids_labels_dict_l2[k].cpu()

            torch.save(centroids_tensor_dict_l1, f'{args.output_path}/hierarchical_lookup_tensor_dict_L1_{dataidx}_{num_centroids}.pt')
            torch.save(centroids_labels_dict_l1, f'{args.output_path}/hierarchical_lookup_labels_dict_L1_{dataidx}_{num_centroids}.pt')
            torch.save(centroids_tensor_dict_l2, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{num_centroids_l2}.pt')
            torch.save(centroids_labels_dict_l2, f'{args.output_path}/centroids_labels_dict_{dataidx}_{num_centroids_l2}.pt')
            torch.save(global_threshold_dict_l1, f'{args.output_path}/hierarchical_global_threshold_L1_{dataidx}_{num_centroids}.pt')
            torch.save(global_threshold_dict_l2, f'{args.output_path}/global_threshold_{dataidx}_{num_centroids_l2}.pt')

        else:
            # # compute centroids
            # centroids_tensor_dict, centroids_labels_dict = run_clustering(all_keys_layers,
            #                                                               num_centroids,
            #                                                               observation_window=args.observation_window,
            #                                                               device=torch.device('cpu'))

            # # run global threshold
            # global_threshold_dict = run_global_threshold(
            #     all_keys_layers, all_queries_layers, centroids_tensor_dict, centroids_labels_dict, num_centroids,
            #     observation_window=args.observation_window, device=torch.device('cpu')
            # )
            
            if args.value_aware:
                kc_dict, vc_dict, lbl_dict, vvar_dict, nvar_dict = run_value_aware_clustering(
                    all_keys_layers, all_values_layers,
                    num_clusters=num_centroids,
                    observation_window=args.observation_window,
                    alpha=args.alpha, beta=args.beta,
                    device=torch.device('cpu'),  # CPU — tránh OOM trên Quadro 5000 16GB
                )
                centroids_tensor_dict, centroids_labels_dict = kc_dict, lbl_dict
                
                # Save thêm variance để dùng online
                torch.save(nvar_dict, f'{args.output_path}/normalized_variance_{dataidx}_{num_centroids}.pt')
                torch.save(vc_dict, f'{args.output_path}/value_centroids_{dataidx}_{num_centroids}.pt')
                
                # Threshold value-aware
                global_threshold_dict = run_value_aware_global_threshold(
                    keys_layers=all_keys_layers,
                    queries_layers=all_queries_layers,
                    key_centroids_dict=kc_dict,
                    labels_dict=lbl_dict,
                    normalized_variance_dict=nvar_dict,
                    num_clusters=num_centroids,
                    observation_window=args.observation_window,
                    gamma=args.gamma,
                    device=torch.device('cpu'),  # CPU — tránh OOM trên Quadro 5000 16GB
                )
            else:
                # Code gốc, không thay
                centroids_tensor_dict, centroids_labels_dict = run_clustering(
                    all_keys_layers, num_centroids,
                    observation_window=args.observation_window, device=torch.device('cpu'),  # CPU
                )
                global_threshold_dict = run_global_threshold(
                    all_keys_layers, all_queries_layers,
                    centroids_tensor_dict, centroids_labels_dict, num_centroids,
                    observation_window=args.observation_window, device=torch.device('cpu'),  # CPU
                )
            
            # save centroids tensor, labels, global threshold
            os.makedirs(args.output_path, exist_ok=True)
            for k,v in centroids_tensor_dict.items():
                centroids_tensor_dict[k] = centroids_tensor_dict[k].cpu()
            for k,v in centroids_labels_dict.items():
                centroids_labels_dict[k] = centroids_labels_dict[k].cpu()

            torch.save(centroids_tensor_dict, f'{args.output_path}/centroids_tensor_dict_{dataidx}_{num_centroids}.pt')
            torch.save(centroids_labels_dict, f'{args.output_path}/centroids_labels_dict_{dataidx}_{num_centroids}.pt')
            torch.save(global_threshold_dict, f'{args.output_path}/global_threshold_{dataidx}_{num_centroids}.pt')

        # free up memory by deleting all qkv from lists
        num_layers = len(all_keys_layers)
        for i in range(num_layers):
            del all_queries_layers[0]
            del all_keys_layers[0]
            del all_values_layers[0]
        torch.cuda.empty_cache()