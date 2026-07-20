# 004 Real Cached Generation Result

## 环境

- 日期：2026-07-13。
- 机器：121，NVIDIA A100-PCIE-40GB，物理 GPU 0。
- 数据：WMT19/LongCat train split 首条 S2ST request；prompt 25 tokens，source
  acoustic prompt 27 frames。
- 模型：`Qwen/Qwen3-0.6B` 与 LongCat `16k_4codebooks`；diagnostic 将
  `acoustic_prompt_gate` 设为 1。

## 初次结果

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

## 修复

选择直接修复 model 边界：根据有效 acoustic frame positions 构造 token mask，在
`acoustic_prompt_adapter` 之后把未占用 token 重新置零。这样 linear bias 只在实际 source
acoustic token 位置生效，训练、full-recompute 与 cached generation 使用同一注入语义。

新增 contract test 使用 weight 为 0、bias 为 `0.25` 的 linear adapter，验证只有 source
acoustic prompt 占用位置为 `0.25`，前后未占用 token 都严格为 0。

## 复验

修复后使用正式 bf16 + Flash Attention 2 配置重跑：

- cached/full greedy tokens 完全一致，均为 `[152417, 153381]`。
- cached 输入长度为 `25, 1`，acoustic prompt 注入为 `True, False`，past 状态为
  `False, True`。
- full-recompute 输入长度为 `25, 26`，两步都重新注入 acoustic prompt且不携带 past。
- 两边 acoustic features 均为 `(2, 1024)`，waveform 均为 `(1, 1920)`，全部 finite。
- 第 1 step logits 最大绝对差 `0.09375`；acoustic features 最大绝对差 `0.125`；
  waveform 最大绝对差 `0.002497`。这些 bf16 cache/full 数值差没有改变 greedy token。
- cached/full peak allocated CUDA memory 分别为 `4,843,997,184` 与
  `5,017,998,848` bytes。

cached 首次计时 `0.979s`，full-recompute `0.414s`；前者包含首次 flow/decode lazy
warmup，本实验不据此比较性能。

## 状态

004 真实 cached generation/decode 验收通过。标准 padded variable-length batch 仍未验证。

变长 batch generation 的后续验收见
[008 Real Batch Generation Benchmark](008-real-batch-generation-benchmark.md)。

本地回归：`30 passed, 16 subtests passed`；改动文件 Ruff、`py_compile`、shell syntax
与 `git diff --check` 通过。
