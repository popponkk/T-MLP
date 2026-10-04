# HingeMix Formal Entries

`hingemix` is the formal single-projection model: GGPLTokenizer, dynamic Graph Token Mixing, Channel Mixing, and CLS readout. Its default configuration is K=8, r=16, tau=16, d=1024, L=1, learning rate 1e-5, and batch size 32.

`hingemix_ablation` is the centralized ablation entry. It accepts all eight implementation modes for traceability, but the formal result view contains only: `full` (HingeMix), `no_graph` (HingeMix-no_graph), `linear` (HingeMix-linear), and `linear_no_graph` (HingeMix-linear_no_graph). Channel Mixing is retained in all four formal rows. `linear` uses the current shared `nn.Linear(1, d_token)` tokenizer; it is not the historical independent-linear implementation.

The legacy registrations `ggpl_gtm` and `ggpl_gtm_ablation` import the same implementations and retain their original configuration files and result paths. They are compatibility aliases, not additional formal methods. Historical independent-linear, pooling, double-projection, no-channel, and full-ablation records are preserved but not included in formal HingeMix tables.

## Commands

```bash
python main.py --model hingemix --dataset hpcg2 --device cuda --gpu 0
python main.py --model hingemix_ablation --ablation linear --dataset hpcg2 --device cuda --gpu 0
python scripts/check_hingemix_formal.py
python scripts/run_hingemix_ablation_final.py --dry-run
python scripts/summarize_hingemix_ablation_final.py
```

The formal ablation runner schedules four models x seven datasets x five seeds (140 tasks) in `results/hingemix_ablation_final_tau16`. It never scans or rewrites the legacy `results/ggpl_gtm_ablation_final_tau16` directory.