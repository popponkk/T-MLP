import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved prediction metrics across models."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name under results/<model>/<dataset>/prediction.json",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory. Default: results",
    )
    parser.add_argument(
        "--sort-by",
        default="rmse",
        choices=["rmse", "mae", "r2", "loss", "time", "model"],
        help="Metric used for sorting. Default: rmse",
    )
    return parser.parse_args()


def load_prediction(prediction_file: Path):
    with open(prediction_file, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_row(model_name: str, payload: dict):
    metrics = payload.get("metrics") or {}
    return {
        "model": model_name,
        "rmse": metrics.get("rmse", payload.get("metric") if payload.get("metric_name") == "rmse" else None),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "loss": payload.get("loss"),
        "time": payload.get("time"),
    }


def sort_rows(rows, sort_by: str):
    if sort_by == "model":
        return sorted(rows, key=lambda row: row["model"])
    reverse = sort_by == "r2"
    missing = float("-inf") if reverse else float("inf")
    return sorted(
        rows,
        key=lambda row: missing if row.get(sort_by) is None else row[sort_by],
        reverse=reverse,
    )


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows):
    headers = ["model", "rmse", "mae", "r2", "loss", "time"]
    widths = {
        header: max(len(header), max((len(format_value(row[header])) for row in rows), default=0))
        for header in headers
    }
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(format_value(row[header]).ljust(widths[header]) for header in headers))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        prediction_file = model_dir / args.dataset / "prediction.json"
        if not prediction_file.exists():
            continue
        payload = load_prediction(prediction_file)
        rows.append(normalize_row(model_dir.name, payload))

    if not rows:
        raise FileNotFoundError(
            f"No prediction.json files found for dataset '{args.dataset}' under {results_dir}"
        )

    rows = sort_rows(rows, args.sort_by)
    print_table(rows)


if __name__ == "__main__":
    main()
