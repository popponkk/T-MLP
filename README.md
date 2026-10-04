# tmlp

\[KDD 2024] Team up GBDTs and DNNs: Advancing Efficient and Effective Tabular Prediction with Tree-hybrid MLPs

## Prepare a regression CSV

For a pure numerical regression CSV, keep all feature columns first and put the continuous target in the last column.

Convert the CSV into the dataset directory format used by this project:

```bash
python scripts/csv_to_regression_dataset.py --csv /path/to/data.csv --name my-regression-dataset
```

Then run training:

```bash
python main.py --model mlp --dataset my-regression-dataset
```

## Formal HingeMix

The formal single-projection model is registered as `hingemix`. Its default
configuration is K=8, r=16, tau=16, d=1024, L=1, learning rate 1e-5, and
batch size 32.

```bash
python main.py --model hingemix --dataset hpcg2 --device cuda --gpu 0
```

Use `hingemix_ablation` for the centralized ablation entry. Formal result
views contain `full`, `no_graph`, `linear`, and `linear_no_graph`; the latter
two use the shared numeric `nn.Linear(1, d_token)` tokenizer.
