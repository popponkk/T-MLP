#!/usr/bin/env bash
set -euo pipefail

# Serial GPU-0 launcher; it intentionally runs only this independent-nn.Linear group.
models=(
  ggpl_dynonly_ablation_independent_nnlinear
  ggpl_dynonly_ablation_independent_nnlinear_no_channel
  ggpl_dynonly_ablation_independent_nnlinear_no_graph
  ggpl_dynonly_ablation_independent_nnlinear_no_graph_no_channel
)
datasets=(hpcg2 hpgmg3 ramspeed mix_with_five_datasets161 raiderstream stream cachesweep)
seeds=(0 1 2)

python - <<'PY'
import sys
import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to silently run this experiment on CPU.")
PY
python scripts/validate_dynonly_ablation_datasets.py
mkdir -p logs

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
      log="logs/${model}_${dataset}_seed${seed}.log"
      {
        echo "model=${model} dataset=${dataset} split_seed=${seed} gpu=0"
        echo "tokenizer=independent_nnlinear readout=cls config=configs/default/${model}.yaml"
        echo "result_dir=results/${model}_seed${seed}/${dataset}"
        python main.py --model "${model}" --dataset "${dataset}" \
          --device cuda --gpu 0 --batch_size 32 --lr 1e-5 \
          --seed "${seed}" --output_suffix "_seed${seed}"
      } >"${log}" 2>&1
    done
  done
done
