# GGPL-GTM Five-Seed Parameter Supplement

The original `results/ggpl_gtm_parameter` experiment is read-only.  The
supplement runner writes only to `results/ggpl_gtm_parameter_supplement` and
therefore cannot overwrite its 399 completed runs.

## Task Groups

| Group | Configurations | Seeds | Expected tasks |
| --- | ---: | ---: | ---: |
| `legacy-seeds` | Existing 19 saved configurations | 3, 4 | 266 |
| `extras` | Six configurations below | 0, 1, 2, 3, 4 | 210 |
| `all` | Both groups | As above | 476 |

The six extra configurations use the baseline values `K=8`, `r=16`,
`tau=1`, `d=1024`, `L=1` except for the listed fields:

| Config ID | K | r | tau | d | L |
| --- | ---: | ---: | ---: | ---: | ---: |
| `extra_tau8` | 8 | 16 | 8 | 1024 | 1 |
| `extra_tau16` | 8 | 16 | 16 | 1024 | 1 |
| `extra_r4_tau2` | 8 | 4 | 2 | 1024 | 1 |
| `extra_r4_tau4` | 8 | 4 | 4 | 1024 | 1 |
| `extra_r64_tau2` | 8 | 64 | 2 | 1024 | 1 |
| `extra_r64_tau4` | 8 | 64 | 4 | 1024 | 1 |

## Commands

Run all commands from the repository root.  Start with dry-run:

```bash
python -u scripts/run_ggpl_gtm_params_supplement.py --dry-run
```

Run only seed 3 and 4 for the original configurations:

```bash
python -u scripts/run_ggpl_gtm_params_supplement.py \
  --group legacy-seeds --gpus 0 1 --resume
```

Run only the six extra configurations:

```bash
python -u scripts/run_ggpl_gtm_params_supplement.py \
  --group extras --gpus 0 1 --resume
```

Recommended unified command. It skips fingerprint-verified completions and
runs only remaining tasks from both groups:

```bash
python -u scripts/run_ggpl_gtm_params_supplement.py \
  --group all --gpus 0 1 --resume --retry-failed
```

Each GPU runs at most one child `main.py` process. A failed attempt is kept in
`attempts/attempt_NNN`; retries use a new attempt directory.

## Merge Results

```bash
python scripts/summarize_ggpl_gtm_params_5seed.py \
  --legacy-root results/ggpl_gtm_parameter \
  --supplement-root results/ggpl_gtm_parameter_supplement \
  --output results/compare/ggpl_gtm_parameter_5seed
```

The merged output contains `supplement_runs.csv`, `all_runs.csv`,
`summary_5seed.csv`, `parameter_curves_5seed.csv`, `missing_runs.csv`, and
`integrity.json`. The original `results/compare/ggpl_gtm_parameter` files are
not modified.
