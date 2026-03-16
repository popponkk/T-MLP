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
