# HingeMix 正式消融与基线审查

## 结论

- **已证实**：`full` 与 `linear` 经 Graph Token Mixing 存在数值 token 至 CLS 的跨 token 信息路径。
- **已证实**：`no_graph` 与 `linear_no_graph` 属于 **B：路径被切断**。CLS 是样本无关的可学习参数；关闭 Graph 后，Channel Mixing 逐 token 处理，最终只读取 CLS，数值 token 无法影响输出。
- **无法核查**：当前工作区没有 `results/`、训练数据、checkpoint、manifest、训练日志或 Python 运行时；无法审计历史 extra_tau16、315 条消融记录及 10 基线运行记录。

## 源码证据

- GGPL CLS 与数值 tokens：`models/ggpl_tmlp.py:119-150`。
- 共享 Linear CLS 与数值 tokens：`models/hingemix_ablation.py:39-64`。
- Graph token 聚合：`models/hingemix.py:53-66`、`models/hingemix_ablation.py:116-128`。
- 逐 token Channel Mixing：`models/hingemix.py:68-74`、`models/hingemix_ablation.py:129-137`。
- CLS 读出：`models/hingemix.py:133-140`、`models/hingemix_ablation.py:211-218`。

## 解释

no_graph 与 linear_no_graph 应只作为“信息通路移除对照”。若将来通过数值 token 均值池化修正它们，读出设计亦改变，必须额外运行同读出的 full/linear 对照。

RMSE 在 `utils/metrics.py:14-29,71-77` 中按 `y_std` 反缩放。历史基线是否同尺度没有本地记录可证实。
