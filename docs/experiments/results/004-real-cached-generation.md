# 004 Real Cached Generation Result

## 环境

- 日期：2026-07-13。
- 机器：121，NVIDIA A100-PCIE-40GB，物理 GPU 0。
- 数据：WMT19/LongCat train split 首条 S2ST request；prompt 25 tokens，source
  acoustic prompt 27 frames。
- 模型：`Qwen/Qwen3-0.6B` 与 LongCat `16k_4codebooks`；diagnostic 将
  `acoustic_gate` 设为 1。

## 结果

真实 cached/full-recompute greedy parity 未通过：

- 两条路径第 0 step logits 完全一致，首 token 都是 `157912`。
- 第 1 step cached/full logits 最大绝对差为 `6.81`，第二个 token 分别为
  `156917` 与 `155494`（bf16 eager）。
- cached 调用长度为 `25, 1`，仅首步注入 acoustic prompt；full-recompute 为
  `25, 26`，两步都重新注入 acoustic prompt，调用形态符合 003 契约。
- 两条路径仍都产出 `(2, 1024)` acoustic features 与 `(1, 1920)` finite waveform，
  但因 semantic condition 已分叉，不具备结果可比性。

## 排查

- Flash Attention 2 与 eager 都复现，排除 attention backend。
- cached step 使用 bool mask、long mask、无 mask和显式 `position_ids/cache_position`
  得到完全相同的错误结果，排除 mask 和 position 推导。
- 完整序列 `use_cache=True/False` 的所有 hidden 与 logits 逐元素相同，问题只发生在
  增量 past-cache 路径。
- float32 eager 仍复现；cached/full 的第 0 个 hidden state 已相差 `0.03124772`，
  随后在 28 层中逐步放大，排除单纯 bf16 舍入。

根因位于 acoustic prompt embedding：`merge_by_positions()` 对无 acoustic 的 token
输出 0，但带 bias 的 linear acoustic adapter 会把所有这些位置变成非零。full-recompute
每步重新运行 acoustic adapter，因此新生成 token 会额外收到 adapter bias；cached 路径后续
不再注入 acoustic prompt。观察到的首层差接近 `1 / sqrt(1024) = 0.03125`，与 linear
bias 初始化范围一致。

## 状态

004 验收失败，真实 cached generation 待修复后重跑。当前结果支持两个处理边界：

1. 只在实验 wrapper 中禁用 acoustic adapter bias，并记录偏离正式模型配置。
2. 在 model 层把 adapted acoustic feature 重新 mask 到实际 source acoustic token 位置，
   使训练、full-recompute 与 cached generation 使用同一注入语义。
