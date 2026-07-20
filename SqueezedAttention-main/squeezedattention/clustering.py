import torch
import torch.nn.functional as F
import math
import time
import numpy as np

# Sử dụng sức mạnh của GPU thông qua cuML và CuPy
try:
    from cuml.cluster import KMeans
    import cupy as cp
    from torch.utils.dlpack import to_dlpack
    from cupy import fromDlpack
except ImportError:
    raise ImportError("Vui lòng đảm bảo môi trường có cài đặt 'cuml' và 'cupy' để chạy trên GPU.")



def run_clustering(tdict, num_clusters, observation_window=100, print_log=False, device=None):
    # TÁCH BIỆT: Tính toán LUÔN trên GPU, Lưu trữ theo yêu cầu (CPU)
    calc_device = "cuda:0" 
    save_device = device if device is not None else "cpu"

    centroids_tensor_dict = {}
    centroids_labels_dict = {}

    num_heads = tdict[0].shape[-3]
    num_lyrs = len(tdict)

    t1 = time.time()
    for layer_num in range(num_lyrs):
        if print_log:
            print('layer: ', layer_num)

        keys = tdict[layer_num].squeeze(0).float()
        
        K = num_clusters
        assert(len(keys.shape) == 3)

        if observation_window > 0:
            keys = keys[:,:-observation_window,:]
        num_heads = keys.shape[0]

        cluster_labels_list = []
        cluster_centers_list = []

        for H in range(num_heads):
            # 1. ÉP TENSOR LÊN GPU ĐỂ TÍNH TOÁN
            head_data_gpu = keys[H].to(calc_device)
            data_normalized = F.normalize(head_data_gpu, p=2, dim=-1)

            dlpack_tensor = to_dlpack(data_normalized)
            data_cp = fromDlpack(dlpack_tensor)

            # Chạy KMeans siêu tốc trên GPU
            kmeans = KMeans(
                n_clusters=K,
                max_iter=150,
                init='k-means++',
                verbose=0,
                random_state=0
            )
            
            kmeans.fit(data_cp)
            cluster_labels = kmeans.labels_

            # Kéo labels từ CuPy về Pytorch Tensor (trên GPU)
            dlpack_labels = cluster_labels.toDlpack()
            labels_gpu = torch.utils.dlpack.from_dlpack(dlpack_labels)

            cluster_centers = []
            for i in range(K):
                mask = labels_gpu == i
                cluster_keys = head_data_gpu[mask]
                if len(cluster_keys) > 0:
                    centroid = torch.mean(cluster_keys, dim=0)
                else:
                    centroid = torch.zeros(head_data_gpu.shape[1], dtype=head_data_gpu.dtype, device=calc_device)
                cluster_centers.append(centroid)
            cluster_centers = torch.stack(cluster_centers, dim=0)

            # 2. RÚT KẾT QUẢ VỀ CPU ĐỂ TRÁNH TRÀN VRAM 16GB
            cluster_labels_list.append(labels_gpu.to(save_device))
            cluster_centers_list.append(cluster_centers.to(save_device))
            
            # Xóa sạch biến tạm trên GPU
            del head_data_gpu, data_normalized, dlpack_tensor, data_cp, labels_gpu, cluster_centers
            torch.cuda.empty_cache()

        a = torch.stack(cluster_centers_list, dim=0).unsqueeze(0)
        b = torch.stack(cluster_labels_list, dim=0).unsqueeze(0).to(torch.int64)

        centroids_tensor_dict[layer_num] = a
        centroids_labels_dict[layer_num] = b

    return centroids_tensor_dict, centroids_labels_dict


def run_global_threshold(key_dict, query_dict, centroids_tensor_dict, centroids_labels_dict, num_clusters, observation_window=100, print_log=False, device=None):
    calc_device = "cuda:0"
    save_device = device if device is not None else "cpu"

    shared_prefix_length = query_dict[0].shape[-2]
    num_lyrs = len(query_dict)
    K = num_clusters

    attn_score_centroid_list = []
    for layer_num in range(num_lyrs):
        if print_log:
            print('layer: ', layer_num)

        # Đẩy lên GPU để nhân ma trận
        centroids_tensor = centroids_tensor_dict[layer_num].squeeze(0).to(calc_device)
        centroids_labels = centroids_labels_dict[layer_num].squeeze(0).to(calc_device)

        keys = key_dict[layer_num].squeeze(0).to(calc_device)
        queries = query_dict[layer_num].squeeze(0).to(calc_device)
        keys_shared_prefix = keys[:, :-observation_window, :]

        queries_obs_window = queries[:, -observation_window:, :].float()
        
        attn_scores_centroids = torch.matmul(queries_obs_window, centroids_tensor.transpose(1, 2)) / math.sqrt(keys.shape[-1])

        shape = (keys_shared_prefix.shape[0], keys_shared_prefix.shape[1], observation_window)
        scores = torch.zeros(shape, device=calc_device)

        for k in range(K):
            label_mask = centroids_labels == k
            current_attn_scores_centroids = attn_scores_centroids[:,:,k].unsqueeze(-2)
            scores = scores + label_mask.unsqueeze(-1) * current_attn_scores_centroids

        num_keys_per_cluster = torch.zeros((keys_shared_prefix.shape[0], K), device=calc_device)
        for k in range(K):
            label_mask = centroids_labels == k
            num_keys_per_cluster[:,k] = torch.sum(label_mask, dim=-1)

        attn_scores_centroids_est_exp = torch.exp(attn_scores_centroids)
        num_keys_per_cluster = num_keys_per_cluster.unsqueeze(-2)
        denom_est_tmp = num_keys_per_cluster * attn_scores_centroids_est_exp
        denom_est = torch.sum(denom_est_tmp, dim=-1) 

        scores_scaled_sm = torch.exp(scores) / denom_est.unsqueeze(-2)
        scored_scaled_sm_sum = torch.mean(scores_scaled_sm, dim=-1, dtype=torch.float32)
        
        # Kéo kết quả về CPU
        attn_score_centroid_list.append(scored_scaled_sm_sum.to(save_device))

        del centroids_tensor, centroids_labels, keys, queries, keys_shared_prefix
        del queries_obs_window, attn_scores_centroids, scores, num_keys_per_cluster
        torch.cuda.empty_cache()

    full_centroid_scores = torch.stack(attn_score_centroid_list, dim=0)

    qlist = [0.5, 0.7, 0.8, 0.9]
    q = torch.tensor(qlist, device="cpu")

    full_centroid_scores_cpu = full_centroid_scores.cpu().numpy()
    quantile_result = np.quantile(full_centroid_scores_cpu, q.numpy())
    thresholds = torch.tensor(quantile_result)

    tdict = {}
    for i, q_idx in enumerate(qlist):
        tdict[q_idx] = thresholds[i].item()

    tdict['shared_prefix_length'] = shared_prefix_length
    tdict['observation_window'] = observation_window

    return tdict