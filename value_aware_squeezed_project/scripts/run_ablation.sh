#!/bin/bash
# ============================================================================
# Ablation Study (đa setting để lập Table giống Table 6 trong paper)
# ============================================================================
set -e
cd "$(dirname "$0")/.."
mkdir -p results

echo "Running ablation across multiple sparsity levels..."

for SPARSITY in 0.7 0.8 0.85 0.9; do
    echo ""
    echo "=== Sparsity = $SPARSITY ==="
    python benchmarks/benchmark_ablation.py \
        --num_seeds 3 \
        --sparsity $SPARSITY \
        --diversity 1.0 \
        --output "results/ablation_s${SPARSITY}.json" \
        2>&1 | tee "results/ablation_s${SPARSITY}.log"
done

echo ""
echo "=== Diversity sweep (sparsity=0.85) ==="
python benchmarks/synthetic_benchmark.py --experiment diversity \
    --gamma 0.3 --beta 0.5 --sparsity 0.85 \
    | tee results/diversity_sweep.log

echo ""
echo "=== Gamma sweep ==="
python benchmarks/synthetic_benchmark.py --experiment gamma \
    | tee results/gamma_sweep.log

echo ""
echo "=== Beta sweep ==="
python benchmarks/synthetic_benchmark.py --experiment beta \
    | tee results/beta_sweep.log

echo ""
echo "Done. Tất cả ablations ở results/ablation_*.json"
