# Formal GGPL-GTM Delivery

The active formal entries are `ggpl_gtm` and `ggpl_gtm_ablation` only.

`ggpl_gtm` is fixed to GGPLTokenizer + dynamic-only graph + channel mixing + CLS readout.
`ggpl_gtm_ablation` accepts: `full`, `no_channel`, `no_graph`,
`no_graph_no_channel`, `linear`, `linear_no_channel`, `linear_no_graph`, and
`linear_no_graph_no_channel`. `linear` always means the independent
`nn.ModuleList[nn.Linear(1, d_token)]` tokenizer.

## Removed Active Sources

The following active source/config/script families are removed by this migration:

- `models/ggpl_dynonly_ablation*.py` and matching `configs/default/` files;
- `models/ggpl_dynonly_pool*.py` and matching `configs/default/` files;
- `models/ggpl_tmlp_graph_slimtok_dynonly.py` and its default config;
- the old `check_`, `run_`, and `summarize_` scripts dedicated to those families.

Historical `results/`, logs, checkpoints, and comparison CSV files are not
deleted or rewritten. Their names are not aliases for the new formal records.

## Commands

```bash
python main.py --model ggpl_gtm --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python main.py --model ggpl_gtm_ablation --ablation full --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python main.py --model ggpl_gtm_ablation --ablation linear --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python scripts/check_ggpl_gtm_formal.py
bash scripts/run_ggpl_gtm_experiments.sh ablation
python scripts/summarize_ggpl_gtm_experiments.py --include-complete
```

`ggpl_gtm_ablation/full` is a structural control for `ggpl_gtm`, not a second
independent method. `scripts/compare_results.py` lists saved runs but has no
model-ranking feature that would count the pair twice.
