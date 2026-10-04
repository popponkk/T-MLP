# Formal HingeMix Ablation View, tau=16

The historical raw run directory is `results/ggpl_gtm_ablation_final_tau16`.
It contains 315 tasks, including framework-equivalence and Channel-removal
controls. It is retained unchanged.

The formal HingeMix display view exports only four verified mappings from that
raw manifest: `full` -> HingeMix, `no_graph` -> HingeMix-no_graph,
`shared_linear` -> HingeMix-linear, and `shared_linear_no_graph` ->
HingeMix-linear_no_graph. All use K=8, r=16 and tau=16 where Graph exists,
d=1024, L=1, lr=1e-5, batch size 32, data seed 42, and seeds 0 through 4.

```bash
python scripts/summarize_hingemix_ablation_final.py \
  --root results/ggpl_gtm_ablation_final_tau16 \
  --output results/compare/hingemix_ablation_final_tau16
```

The new runner `scripts/run_hingemix_ablation_final.py` is reserved for a
future isolated formal rerun. It schedules only these four methods (140 tasks)
and writes to `results/hingemix_ablation_final_tau16`; it never writes to the
historical raw directory.