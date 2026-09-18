# Formal GGPL-GTM Delivery

The centralized formal entries are `ggpl_gtm` and `ggpl_gtm_ablation`.
Eight fixed-name derived entries are also provided for independent launch and
result directories: `ggpl_gtm_ablation_full`, `ggpl_gtm_ablation_no_channel`,
`ggpl_gtm_ablation_no_graph`, `ggpl_gtm_ablation_no_graph_no_channel`,
`ggpl_gtm_ablation_linear`, `ggpl_gtm_ablation_linear_no_channel`,
`ggpl_gtm_ablation_linear_no_graph`, and
`ggpl_gtm_ablation_linear_no_graph_no_channel`.

`ggpl_gtm` is fixed to GGPLTokenizer + dynamic-only graph + channel mixing + CLS readout.
`ggpl_gtm_ablation` accepts: `full`, `no_channel`, `no_graph`,
`no_graph_no_channel`, `linear`, `linear_no_channel`, `linear_no_graph`, and
`linear_no_graph_no_channel`. The four `linear*` choices now use one shared
`nn.Linear(1, d_token)` for all numeric features. They are saved under
`results/ggpl_gtm_ablation_shared_linear[/_seed<seed>]/<ablation>/<dataset>`
so they cannot overwrite historical independent-linear runs.

Historical independent-`nn.Linear` results remain traceable through the
separate legacy entries `ggpl_dynonly_ablation_independent_nnlinear` and its
three graph/channel variants. Those results are not re-labelled as shared
linear. Sharing changes both parameter count and feature-identity information,
so a shared-linear comparison is not evidence about every independent linear
representation.

## Removed Active Sources

The following active source/config/script families are removed by this migration:

- `models/ggpl_dynonly_pool*.py` and matching `configs/default/` files;
- `models/ggpl_tmlp_graph_slimtok_dynonly.py` and its default config;
- the old `check_`, `run_`, and `summarize_` scripts dedicated to those families.

Historical `results/`, logs, checkpoints, and comparison CSV files are not
deleted or rewritten. Their names are not aliases for the new shared-linear
formal records.

## Commands

```bash
python main.py --model ggpl_gtm --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python main.py --model ggpl_gtm_ablation --ablation full --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python main.py --model ggpl_gtm_ablation --ablation linear --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python main.py --model ggpl_gtm_ablation_linear --dataset hpcg2 --device cuda --gpu 0 --batch_size 32 --lr 1e-5
python scripts/check_ggpl_gtm_formal.py
bash scripts/run_ggpl_gtm_experiments.sh ablation
python scripts/summarize_ggpl_gtm_experiments.py --include-complete
```

`ggpl_gtm_ablation/full` is a structural control for `ggpl_gtm`, not a second
independent method. `scripts/compare_results.py` lists saved runs but has no
model-ranking feature that would count the pair twice.
