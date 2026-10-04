"""Read-only audit for formal HingeMix ablations and baseline result records.

The script never trains, changes configurations, or mutates existing results.
It writes an audit view below ``audit/hingemix_final_audit`` by default.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path


DATASETS = (
    "hpcg2", "hpgmg3", "ramspeed", "mix_with_five_datasets161",
    "raiderstream", "stream", "cachesweep",
)
BASELINES = {
    "LightGBM": "lightgbm", "XGBoost": "xgboost", "CatBoost": "catboost",
    "ExcelFormer": "excel-former", "FT-Transformer": "ft-transformer",
    "DCNv2": "dcnv2", "NODE": "node", "MLP": "mlp",
    "AutoInt": "autoint", "TabM": "tabm",
}
ABLATIONS = {
    "full": ("ggpl", True, "A", "HingeMix"),
    "no_graph": ("ggpl", False, "B", "HingeMix-no_graph"),
    "linear": ("shared_linear", True, "A", "HingeMix-linear"),
    "linear_no_graph": ("shared_linear", False, "B", "HingeMix-linear_no_graph"),
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prediction_records(results_root: Path, model: str, dataset: str):
    """Return valid, distinct seed records; paths remain evidence in the CSV."""
    records = []
    if not results_root.exists():
        return records
    pattern = re.compile(r"(?:^|[_/\\])seed[_-]?(\d+)(?:$|[_/\\])")
    for path in results_root.glob(f"**/{dataset}/prediction.json"):
        parts = "/".join(path.parts).lower()
        if model.lower() not in parts:
            continue
        payload = read_json(path) or {}
        rmse = (payload.get("metrics") or {}).get("rmse")
        if not isinstance(rmse, (int, float)) or not math.isfinite(rmse):
            continue
        match = pattern.search(parts + "/")
        seed = int(match.group(1)) if match else None
        records.append({"seed": seed, "rmse": rmse, "path": str(path)})
    by_seed = {}
    for item in records:
        # Do not turn duplicate exports of one seed into repetitions.
        if item["seed"] is not None and item["seed"] not in by_seed:
            by_seed[item["seed"]] = item
    return list(by_seed.values())


def baseline_rows(repo: Path, results_root: Path):
    rows, reruns = [], []
    for display, model in BASELINES.items():
        config = repo / "configs" / "default" / f"{model}.yaml"
        for dataset in DATASETS:
            records = prediction_records(results_root, model, dataset)
            seeds = sorted(item["seed"] for item in records if item["seed"] is not None)
            if not records:
                status = "待核查，证据不足"
                reason = "本地未发现可验证 prediction.json；无法核对历史划分、尺度和有效种子。"
            elif len(seeds) < 5:
                status = "可部分复用，缺少种子"
                reason = "找到有效 RMSE，但不足 5 个可识别独立训练种子；仍未核对历史划分。"
            else:
                status = "待核查，证据不足"
                reason = "找到至少 5 个有效 RMSE；仍缺少可核对的历史划分/预处理证据。"
            row = {
                "model": display, "registration": model, "dataset": dataset,
                "result_paths": ";".join(item["path"] for item in records),
                "valid_seeds": ";".join(map(str, seeds)),
                "missing_or_invalid_seeds": ";".join(map(str, sorted(set(range(5)) - set(seeds)))),
                "data_version_split_check": "无法核查" if not records else "待核对原始缓存、索引或 manifest",
                "rmse_scale_check": "无法核查" if not records else "待核对 prediction.json 与 y_std",
                "config_source": str(config) if config.exists() else "未找到默认配置",
                "status": status, "reason": reason, "next_step": "保留原记录；定位原始 manifest、训练日志和划分索引后再决定。",
            }
            rows.append(row)
    return rows, reruns


def ablation_rows():
    rows = []
    for name, (tokenizer, graph, category, display) in ABLATIONS.items():
        cut = "否" if graph else "是：CLS 是样本无关可学习 token，Channel Mixing 逐 token 处理。"
        rows.append({
            "ablation": name, "display_name": display, "tokenizer_type": tokenizer,
            "use_graph": graph, "use_channel": True, "readout": "CLS (x[:, 0])",
            "input_to_output_category": category, "path_cut": cut,
            "source_evidence": (
                "models/hingemix_ablation.py:18-26,53-64,116-137,211-218; "
                "models/ggpl_tmlp.py:119-150"
            ),
            "runtime_validation": "未执行：当前工作区无 Python 运行时、无 checkpoint/预处理数据。",
        })
    return rows


def report(results_root: Path):
    return f"""# HingeMix 正式消融与基线审查

## 结论

- **已证实**：`full` 与 `linear` 存在数值输入到 CLS 预测的路径。Graph 分支以 `A @ H` 在 token 维混合；CLS 行可聚合数值 token。
- **已证实**：`no_graph` 与 `linear_no_graph` 属于 **B：路径被切断**。Tokenizer 的 CLS 是扩展的可学习常量；移除 Graph 后，Channel Mixing 在每个 token 内独立处理，最终仅读取 `x[:, 0]`。因此数值 token 不能影响 CLS。
- **无法核查**：结果根目录 `{results_root}` 在当前工作区不存在；无法读取 checkpoint、`extra_tau16` 的 35 条记录、正式消融 315 条记录、10 个基线的日志/manifest/预测文件，也不能核对 DCNv2 的异常 RMSE。

## 源码证据

- GGPL CLS 由 `cls_token` 扩展而来，数值 token 独立计算：`models/ggpl_tmlp.py:119-150`。
- 共享 Linear 同样拼接样本无关 CLS：`models/hingemix_ablation.py:39-64`。
- Graph 的 softmax 关系聚合为 `attention @ h`：`models/hingemix.py:53-66`；消融版：`models/hingemix_ablation.py:116-128`。
- Channel 分支只对同一 token 的最后一维做 LayerNorm/Linear/GELU/Linear，不发生 token 维乘法或归约：`models/hingemix.py:68-74`、`models/hingemix_ablation.py:129-137`。
- 所有变体读出 `x[:, 0]`：`models/hingemix.py:133-140`、`models/hingemix_ablation.py:211-218`。

## 解释与处理

`no_graph` 与 `linear_no_graph` 应标注为“信息通路移除对照”，不能据其 RMSE 单独推断 Graph 优于所有替代聚合。若以后采用数值 token 平均池化修正它们，读出设计也会同时变化；需要额外的同读出 full/linear 对照，不能与本轮 CLS 结果混合。

RMSE 代码在 `utils/metrics.py:14-29,71-77`：当 `y_std` 非空时，RMSE 和 MAE 乘回目标标准差。历史基线是否也采用同一尺度，目前无记录可证实。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="audit/hingemix_final_audit")
    args = parser.parse_args()
    repo, output = Path(__file__).resolve().parents[1], Path(args.output)
    baselines, pending = baseline_rows(repo, Path(args.results_root))
    write_csv(output / "ablation_checks.csv", ablation_rows(), list(ablation_rows()[0]))
    fields = list(baselines[0])
    write_csv(output / "baseline_audit.csv", baselines, fields)
    write_csv(output / "pending_runs.csv", pending, ["model", "dataset", "reason", "action"])
    (output / "audit_report.md").write_text(report(Path(args.results_root)), encoding="utf-8")
    print(f"Wrote audit files to {output}")


if __name__ == "__main__":
    main()
