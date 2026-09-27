# Final GGPL-GTM Ablation, tau=16

All tasks use `d_token=1024`, `n_layers=1`, `num_breakpoints=8`,
`graph_dynamic_rank=16`, `graph_temperature=16.0`, `lr=1e-5`, batch size 32,
`data_seed=42`, and training seeds 0 through 4. The final experiment is
isolated under `results/ggpl_gtm_ablation_final_tau16`.

There are nine experiment entries and 315 tasks: `full`, `full_ablation`,
`no_channel`, `no_graph`, `no_graph_no_channel`, `shared_linear`,
`shared_linear_no_channel`, `shared_linear_no_graph`, and
`shared_linear_no_graph_no_channel`.

`full` uses the public `ggpl_gtm` model. `full_ablation` is the centralized
ablation framework's structural-equivalence control. `shared_linear*` uses
one shared `nn.Linear(1, d_token)` for every numeric feature; it does not fit
or load GGPL breakpoints. For no-graph rows, r and tau are recorded as not
applicable because the graph branch is not instantiated.

The scheduler's command-line `--lr 1e-5` overrides the historical base YAML
learning rate of `1e-4`; the materialized `final_config.yaml` and `task.json`
record the actual value for each attempt.
