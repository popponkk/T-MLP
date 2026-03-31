#!/usr/bin/env bash
set -euo pipefail

cd ~/ProjectCodes/T-MLP

DATASET="${1:?usage: bash run_all_models_one_dataset.sh <dataset>}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU0_MODELS=(
  mlp
  dcnv2
  autoint
  tmlp
  moe_tmlp
  moe_tmlp_no_diversity
  tmlp-sr-pee
  pr_tmlp_residual
  pr_tmlp_residual_no_balance
  pr_tmlp_pure_experts
  hre_tmlp_no_balance
  hre_tmlp_no_alpha
  hre_tmlp_dense_routing
  agr_tmlp_no_err_head
  agr_tmlp_no_res_reg
  adr_tmlp
)

GPU1_MODELS=(
  node
  ft-transformer
  excel-former
  baseline_tmlp
  moe_tmlp_no_sparse
  tmlp-sr
  tmlp-sr-lgr
  pr_tmlp_residual_no_sparse
  pr_tmlp_residual_no_sep
  hre_tmlp
  hre_tmlp_no_div
  hre_tmlp_no_global_residual
  agr_tmlp
  agr_tmlp_no_gate_reg
  agr_tmlp_gate_from_h_only
  sga_tmlp_lite
)

CPU_MODELS=(
  xgboost
  catboost
  lightgbm
)

for model in "${GPU0_MODELS[@]}"; do
  nohup "$PYTHON_BIN" main.py --model "$model" --dataset "$DATASET" --device cuda --gpu 0 --batch_size 32 --lr 1e-5 > "logs/${model}-${DATASET}-gpu.log" 2>&1 &
done

for model in "${GPU1_MODELS[@]}"; do
  nohup "$PYTHON_BIN" main.py --model "$model" --dataset "$DATASET" --device cuda --gpu 1 --batch_size 32 --lr 1e-5 > "logs/${model}-${DATASET}-gpu.log" 2>&1 &
done

for model in "${CPU_MODELS[@]}"; do
  nohup "$PYTHON_BIN" main.py --model "$model" --dataset "$DATASET" --device cpu > "logs/${model}-${DATASET}.log" 2>&1 &
done

echo "[OK] submitted all models for dataset: $DATASET"