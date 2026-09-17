#!/usr/bin/env bash
set -euo pipefail

# Serial GPU-0 launcher. It does not start automatically when added to the repo.
mode="${1:-ablation}"
ablations=(full no_channel no_graph no_graph_no_channel linear linear_no_channel linear_no_graph linear_no_graph_no_channel)
datasets=(hpcg2 hpgmg3 ramspeed mix_with_five_datasets161 raiderstream stream cachesweep)
seeds=(0 1 2)

python - <<'PY'
import sys
import torch
print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to silently run on CPU.")
PY
python scripts/validate_ggpl_gtm_datasets.py
mkdir -p logs

run_one() {
  local model="$1" ablation="$2" dataset="$3" seed="$4"
  local suffix="_seed${seed}" log
  if [[ -n "${ablation}" ]]; then
    log="logs/${model}_${ablation}_${dataset}_seed${seed}.log"
    {
      echo "model=${model} ablation=${ablation} dataset=${dataset} seed=${seed} gpu=0"
      echo "result_dir=results/${model}${suffix}/${ablation}/${dataset}"
      python main.py --model "${model}" --ablation "${ablation}" --dataset "${dataset}" \
        --device cuda --gpu 0 --batch_size 32 --lr 1e-5 --seed "${seed}" \
        --output_suffix "${suffix}"
    } >"${log}" 2>&1
  else
    log="logs/${model}_${dataset}_seed${seed}.log"
    {
      echo "model=${model} dataset=${dataset} seed=${seed} gpu=0"
      echo "result_dir=results/${model}${suffix}/${dataset}"
      python main.py --model "${model}" --dataset "${dataset}" --device cuda --gpu 0 \
        --batch_size 32 --lr 1e-5 --seed "${seed}" --output_suffix "${suffix}"
    } >"${log}" 2>&1
  fi
}

case "${mode}" in
  complete)
    for dataset in "${datasets[@]}"; do for seed in "${seeds[@]}"; do
      run_one ggpl_gtm '' "${dataset}" "${seed}"
    done; done
    ;;
  ablation)
    for ablation in "${ablations[@]}"; do
      for dataset in "${datasets[@]}"; do for seed in "${seeds[@]}"; do
        run_one ggpl_gtm_ablation "${ablation}" "${dataset}" "${seed}"
      done; done
    done
    ;;
  *)
    echo "Usage: bash scripts/run_ggpl_gtm_experiments.sh [complete|ablation]" >&2
    exit 2
    ;;
esac
