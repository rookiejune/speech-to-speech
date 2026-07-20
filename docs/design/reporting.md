# reporting

提供训练入口和 codec oracle 共用的纯汇总函数，不持有 logger、文件或模型状态。

## 对外能力

`window_summary(values, window=20)` 对一维数值序列生成可直接写入 JSON 的窗口摘要。
`window` 的调用契约是正整数；实际窗口大小为 `min(window, len(values))`。

空序列只返回：

```python
{"steps": 0}
```

非空序列返回：

```python
{
    "steps": len(values),
    "window": size,
    "first": values[0],
    "last": values[-1],
    "first_mean": mean(values[:size]),
    "last_mean": mean(values[-size:]),
    "last_to_first": last_mean / first_mean,
}
```

当 `first_mean == 0` 时，`last_to_first=None`，避免用无穷值进入 JSON。函数不删除或替换
NaN/Inf；监督指标的 finite 约束由产生这些数值的训练或实验入口负责。

## 边界

- `reporting` 只计算稳定 mapping，不决定指标名称、输出路径、序列采集或日志后端。
- codec oracle 与 overfit summary 复用同一函数，不能分别实现不同的窗口/比例语义。
- `first`/`last` 是单点值，`last_to_first` 是首尾窗口均值之比，调用方不能把两者混为单步比值。
