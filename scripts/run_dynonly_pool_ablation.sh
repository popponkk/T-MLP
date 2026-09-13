#!/usr/bin/env bash
set -euo pipefail

# Two workers run in parallel; each worker keeps its assigned GPU serial.
models=(
  ggpl_dynonly_pool_no_channel
  ggpl_dynonly_pool_linear_tokenizer
  ggpl_dynonly_pool_no_graph
  ggpl_dynonly_pool_linear_tokenizer_no_channel
  ggpl_dynonly_pool_no_graph_no_channel
  ggpl_dynonly_pool_linear_tokenizer_no_graph
  ggpl_dynonly_pool_linear_tokenizer_no_graph_no_channel
  ggpl_dynonly_pool
)
datasets=(hpcg2 hpgmg3 ramspeed mix_with_five_datasets161 raiderstream stream cachesweep)
seeds=(0 1 2)

mkdir -p logs
python scripts/validate_dynonly_pool_datasets.py
run_worker() {
  local gpu="$1"
  local parity="$2"
  local task=0
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for seed in "${seeds[@]}"; do
        if (( task % 2 == parity )); then
          local log="logs/${model}_${dataset}_seed${seed}.log"
          {
            echo "model=${model} dataset=${dataset} split_seed=${seed} gpu=${gpu}"
            echo "readout=numerical_mean config=configs/default/${model}.yaml"
            echo "result_dir=results/${model}_seed${seed}/${dataset}"
            python main.py --model "${model}" --dataset "${dataset}" \
              --device cuda --gpu "${gpu}" --batch_size 32 --lr 1e-5 \
              --seed "${seed}" --output_suffix "_seed${seed}"
          } >"${log}" 2>&1
        fi
        ((task += 1))
      done
    done
  done
}

run_worker 0 0 &
run_worker 1 1 &
wait
