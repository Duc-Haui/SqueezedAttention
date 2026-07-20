"""
Value-Aware Retrieval extension for Squeezed Attention.

Cải tiến phương pháp Squeezed Attention (Hooper et al., 2024) bằng cách:
1. Joint K-V clustering thay vì chỉ K
2. Tính per-cluster value variance
3. Boost retrieval score theo variance để tránh bỏ sót cluster có values đa dạng

Public API:
    - run_value_aware_clustering: Drop-in thay thế run_clustering của repo gốc
    - run_value_aware_global_threshold: Drop-in thay thế run_global_threshold
    - value_aware_kmeans: K-means lõi joint K-V
    - normalize_value_variance: Chuẩn hóa variance về [0,1]
    - squeezed_attention_forward: Reference implementation cho 1 attention forward
"""

__version__ = "0.1.0"

from .clustering import (
    run_value_aware_clustering,
    value_aware_kmeans,
    normalize_value_variance,
)
from .threshold import (
    run_value_aware_global_threshold,
    calibrate_threshold,
)
from .retrieval import (
    compute_base_scores,
    value_aware_retrieve,
    keys_mask_from_clusters,
    squeezed_attention_forward,
    baseline_full_attention,
    key_only_attention_forward,
)

__all__ = [
    "run_value_aware_clustering",
    "value_aware_kmeans",
    "normalize_value_variance",
    "run_value_aware_global_threshold",
    "calibrate_threshold",
    "compute_base_scores",
    "value_aware_retrieve",
    "keys_mask_from_clusters",
    "squeezed_attention_forward",
    "baseline_full_attention",
    "key_only_attention_forward",
]
