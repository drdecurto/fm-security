#!/bin/bash
#
# Run all tabular-anchor experiments for the OT/ICS benchmarking package.
#
# This script drives the scriptable side of the repository (RandomForest,
# TabPFN, TabICL). The LLM-side experiments are reproduced via the Jupyter
# notebooks under notebooks/experiments/.
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh
#
# Prerequisites:
#   - pip install -r requirements.txt
#   - Either Kaggle credentials configured (default backend), or datasets
#     placed manually under data/raw/{swat,hai,wustl_iiot}/
#

set -e

RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " OT/ICS Foundation-Model Benchmark"
echo " Tabular-anchor experiments"
echo "============================================"
echo ""
echo "  Results directory: $RESULTS_DIR"
echo ""

# ─── Experiment 1: Baseline cross-method comparison ─────────────────────
echo "[EXP 1] Baseline Cross-Method Comparison"
echo "--------------------------------------------"

for DATASET in swat hai wustl; do
    echo "  → Dataset: $DATASET"
    python -m src.experiments.run_baseline \
        --dataset "$DATASET" \
        --models all \
        --n_runs 5 \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/log_baseline_${DATASET}.txt"
    echo ""
done

# ─── Experiment 2: Few-shot scaling ─────────────────────────────────────
echo "[EXP 2] Few-Shot Scaling"
echo "--------------------------------------------"

for DATASET in swat hai; do
    echo "  → Dataset: $DATASET"
    python -m src.experiments.run_few_shot \
        --dataset "$DATASET" \
        --shots 5 10 25 50 100 500 \
        --n_runs 10 \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/log_fewshot_${DATASET}.txt"
    echo ""
done

# ─── Experiment 3: Rare-attack class detection (multi-class) ────────────
echo "[EXP 3] Rare-Attack Class Detection"
echo "--------------------------------------------"

echo "  → Dataset: wustl (5-class taxonomy)"
python -m src.experiments.run_rare_attack \
    --dataset wustl \
    --threshold 50 \
    --n_runs 5 \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/log_rare_wustl.txt"
echo ""

# ─── Experiment 4: Cross-domain transfer ────────────────────────────────
echo "[EXP 4] Cross-Domain Transfer"
echo "--------------------------------------------"

echo "  → SWaT → HAI"
python -m src.experiments.run_cross_domain \
    --source swat --target hai \
    --n_runs 5 \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/log_crossdomain_swat_hai.txt"

echo "  → HAI → SWaT"
python -m src.experiments.run_cross_domain \
    --source hai --target swat \
    --n_runs 5 \
    --output_dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/log_crossdomain_hai_swat.txt"

echo ""
echo "============================================"
echo " All tabular experiments complete."
echo " Results in: $RESULTS_DIR/"
echo "============================================"
echo ""
echo " For the LLM-side experiments, run the"
echo " notebooks under notebooks/experiments/."
echo "============================================"
