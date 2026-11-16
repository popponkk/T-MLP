# HingeMix Configuration Extrapolation

This experiment predicts samples from a higher, unseen value of one real run
configuration parameter. It is not a label-range split and it must not infer a
configuration value from NPY row order.

## Required metadata

Create a plan from `configs/extrapolation/hingemix_extrapolation_example.yaml`.
The metadata CSV or JSON must provide one unique row per saved `idx_*.npy`
sample ID and must include the configured parameter plus every `group_fields`
column. `group_fields` must describe a full configuration; `run_id` alone is
not sufficient. Set `source.raw_input_confirmed: true` only after confirming
that the source arrays were saved before any whole-dataset preprocessing.

The first inspection does not write a dataset view:

```bash
python scripts/prepare_hingemix_extrapolation.py \
  --plan configs/extrapolation/<your_plan>.yaml --inspect
```

After reviewing `results/hingemix_extrapolation/manifests/<experiment>/`,
materialize the isolated train/validation/test view:

```bash
python scripts/prepare_hingemix_extrapolation.py \
  --plan configs/extrapolation/<your_plan>.yaml
```

The derived dataset contains raw split arrays only. The normal data processor
therefore fits feature transformations and target statistics on the derived
training split. The runner additionally uses `--strict-train-preprocessing`,
isolated preprocessing caches, isolated breakpoint caches, no test evaluation
during fitting, and final best-checkpoint evaluation only.

## Running

Inspect tasks without training:

```bash
python -u scripts/run_hingemix_extrapolation.py \
  --plan configs/extrapolation/<your_plan>.yaml --gpus 0 --dry-run
```

One HingeMix smoke task:

```bash
python -u scripts/run_hingemix_extrapolation.py \
  --plan configs/extrapolation/<your_plan>.yaml \
  --models hingemix --seeds 0 --gpus 0
```

The plan defaults to HingeMix, LightGBM, and MLP with training seeds 0--4.
Each selected GPU runs at most one task. Completed matching tasks are skipped;
to create a new attempt after a failure use `--retry-failed`.

```bash
python -u scripts/run_hingemix_extrapolation.py \
  --plan configs/extrapolation/<your_plan>.yaml --gpus 0 --resume
```

Summarize verified tasks and generate the per-configuration RMSE curve:

```bash
python scripts/summarize_hingemix_extrapolation.py \
  --plan configs/extrapolation/<your_plan>.yaml
```

Outputs are separated under `results/hingemix_extrapolation/` and
`results/hingemix_extrapolation_summary/`. They never reuse a normal-split
model, preprocessing cache, or HingeMix breakpoint cache.
