# HingeMix Parameter-Study Export

The completed parameter study remains in the immutable historical roots
`results/ggpl_gtm_parameter` and `results/ggpl_gtm_parameter_supplement`.
`hingemix` is the formal display name for the verified single-projection
implementation. New export files are written under `results/compare/hingemix_parameter_5seed`; raw manifests, metrics, checkpoints, and fingerprints are not rewritten.

The study has 25 configurations, seven datasets, and five training seeds. Its historical scan values, including tau=1 baseline rows, remain historical records and are distinct from the new HingeMix default tau=16.

```bash
python scripts/summarize_hingemix_params_5seed.py \
  --legacy-root results/ggpl_gtm_parameter \
  --supplement-root results/ggpl_gtm_parameter_supplement \
  --output results/compare/hingemix_parameter_5seed
```

The export contains `supplement_runs.csv`, `all_runs.csv`, `summary_5seed.csv`, `parameter_curves_5seed.csv`, `missing_runs.csv`, and `integrity.json`. Each row adds `model=hingemix` and `source_model=ggpl_gtm` for traceability.