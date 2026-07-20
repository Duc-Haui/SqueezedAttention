#!/bin/bash
# ============================================================================
# Pareto Sweep - nhiều sparsity để vẽ Pareto front (CosSim vs Budget)
# Tương đương Figure 5 trong paper
# ============================================================================
set -e
cd "$(dirname "$0")/.."
mkdir -p results figs

MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}

echo "Running Pareto sweep on $MODEL..."

PARETO_JSONS=""
for SPARSITY in 0.7 0.8 0.85 0.9 0.95; do
    OUT="results/accuracy_s${SPARSITY}.json"
    PARETO_JSONS="$PARETO_JSONS $OUT"

    echo ""
    echo "=== Sparsity = $SPARSITY ==="
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        python benchmarks/benchmark_accuracy.py \
            --model "$MODEL" \
            --max_context 4096 --num_queries 10 \
            --sparsity $SPARSITY --gamma 0.3 --beta 0.5 \
            --output "$OUT"
    else
        echo "Không có GPU - synthetic mode:"
        python -c "
import sys, json
sys.path.insert(0, '.')
from benchmarks.synthetic_benchmark import run_one_setting
r = run_one_setting(diversity=1.0, gamma=0.3, beta=0.5, K=32, target_sparsity=$SPARSITY)
out = {'config': {'sparsity': $SPARSITY}, 'results': {k: {'avg_cos_sim': v['cos'], 'avg_kv_budget': v['budget'], 'avg_mse': v['mse']} for k,v in r.items()}}
with open('$OUT', 'w') as f: json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
"
    fi
done

echo ""
echo "Plotting Pareto front..."
python benchmarks/plot_comparison.py \
    --pareto_jsons $PARETO_JSONS \
    --output_dir figs/

echo ""
echo "Done. Pareto plot ở figs/pareto.png"
