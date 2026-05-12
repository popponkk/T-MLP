#!/usr/bin/env bash
set -euo pipefail

# Enabled GFG reference run.
python main.py \
  --model ggpl_tmlp_gfg \
  --dataset california \
  --device cuda \
  --gpu 0 \
  --batch_size 32 \
  --lr 1e-5 \
  --feat_gate xgb_dropout \
  --pruning mlp+sgu+layer

# Disabled GFG control run.
python main.py \
  --model ggpl_tmlp_gfg \
  --dataset california \
  --device cuda \
  --gpu 0 \
  --batch_size 32 \
  --lr 1e-5 \
  --feat_gate none \
  --pruning none
