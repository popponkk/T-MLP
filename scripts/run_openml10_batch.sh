#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_openml10_batch.sh <batch_id>"
  exit 1
fi

BATCH_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/openml10_batches}"

mkdir -p "$LOG_DIR"

dataset_has_cat_features() {
  local dataset="$1"
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
info = Path("data/datasets") / "${dataset}" / "info.json"
payload = json.loads(info.read_text(encoding="utf-8"))
print(1 if payload.get("n_cat_features", 0) > 0 else 0)
PY
}

should_skip_run() {
  local model="$1"
  local dataset="$2"
  local has_cat
  has_cat="$(dataset_has_cat_features "$dataset")"

  if [[ "$has_cat" == "1" && ( "$model" == "excel-former" || "$model" == "excel_cgr_lite" ) ]]; then
    echo "skip: ${model} does not support categorical features"
    return 0
  fi

  if [[ "${SKIP_LARGE_TREE_MODELS:-1}" == "1" && "$dataset" == "openml_42728_airlines_depdelay_10m" ]]; then
    case "$model" in
      lightgbm|xgboost|catboost|lgbm_cgr_hybrid)
        echo "skip: ${dataset} is too large for default tree-model batch settings"
        return 0
        ;;
    esac
  fi

  return 1
}

DATASETS=(
  openml_422_topo_2_1
  openml_541_socmob
  openml_42563_house_prices_nominal
  openml_42571_allstate_claims_severity
  openml_42705_yolanda
  openml_42724_onlinenewspopularity
  openml_42726_abalone
  openml_42727_colleges
  openml_42728_airlines_depdelay_10m
  openml_42729_nyc_taxi_green_dec_2016
)

MODELS=()
case "$BATCH_ID" in
  1)
    MODELS=(
      autoint
      baseline_tmlp
      catboost
      dcnv2
      excel-former
      ft-transformer
      lightgbm
      mlp
      node
      tmlp
      xgboost
    )
    ;;
  2)
    MODELS=(
      agpl_cgr_tmlp
      apar_cgr_tmlp
      cgr_tmlp
      cgr_tmlp_stage2
      cgr_tmlp_v2
      cgr_tmlp_v3
      dpg_cgr_tmlp
      ggpl_cgr_tmlp
      iggpl_cgr_tmlp
      qcal_cgr_tmlp
      rgc_cgr_tmlp
    )
    ;;
  3)
    MODELS=(
      scgr_tmlp
      sgg_cgr_tmlp
      nr_cgr_tmlp
      excel_cgr_lite
      ggpl_tabm_cgr
      lgbm_cgr_hybrid
      adr_tmlp
      agr_tmlp
      agr_tmlp_gate_from_h_only
      agr_tmlp_no_err_head
      agr_tmlp_no_gate_reg
    )
    ;;
  4)
    MODELS=(
      agr_tmlp_no_res_reg
      agr_tmlp_rex2_guarded_lite
      agr_tmlp_rex2_lite
      agr_tmlp_softgate
      agr_tmlp_switch_hardlite
      agr_tmlp_switch_lite
      agr_tmlp_switch_lite_td
      agsi_tmlp
      sr_agr_tmlp_lite
      hlr_tmlp
      lar_tmlp
      hre_tmlp
    )
    ;;
  5)
    MODELS=(
      hre_tmlp_dense_routing
      hre_tmlp_no_alpha
      hre_tmlp_no_balance
      hre_tmlp_no_div
      hre_tmlp_no_global_residual
      moe_tmlp
      moe_tmlp_no_diversity
      moe_tmlp_no_sparse
      pr_tmlp_pure_experts
      pr_tmlp_residual
      pr_tmlp_residual_no_balance
      pr_tmlp_residual_no_sep
    )
    ;;
  6)
    MODELS=(
      pr_tmlp_residual_no_sparse
      sga_tmlp_lite
      sga_tmlp_lite_no_beta_reg
      sga_tmlp_lite_no_corr_reg
      sga_tmlp_lite_no_featmix
      sga_tmlp_lite_topk12
      sga_tmlp_lite_topk4
      tmlp-sr
      tmlp-sr-lgr
      tmlp-sr-pee
      ggtm_tmlp
    )
    ;;
  *)
    echo "Invalid batch_id: $BATCH_ID"
    exit 1
    ;;
esac

echo "=================================================="
echo "[START] batch=${BATCH_ID} gpu=${GPU}"
echo "=================================================="

for dataset in "${DATASETS[@]}"; do
  echo "--------------------------------------------------"
  echo "[DATASET] ${dataset}"
  echo "--------------------------------------------------"

  for model in "${MODELS[@]}"; do
    log_file="${LOG_DIR}/${model}-${dataset}.log"
    echo "[RUN] model=${model} dataset=${dataset} gpu=${GPU}"

    if skip_reason="$(should_skip_run "$model" "$dataset")"; then
      echo "[SKIP] model=${model} dataset=${dataset} ${skip_reason}"
      continue
    fi

    "$PYTHON_BIN" main.py \
      --model "$model" \
      --dataset "$dataset" \
      --device "$DEVICE" \
      --gpu "$GPU" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      > "$log_file" 2>&1 || {
        echo "[FAIL] model=${model} dataset=${dataset}, see ${log_file}"
      }
  done
done

echo "=================================================="
echo "[DONE] batch=${BATCH_ID} gpu=${GPU}"
echo "=================================================="
