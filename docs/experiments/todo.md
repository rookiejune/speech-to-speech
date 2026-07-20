# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 真实资源验收

- 用真实 Qwen checkpoint 验收中英双向 `TextRetentionLogger`，确认训练前 wrapper 与 backbone 的文本输出一致，并观察语音训练期间的 NLL 漂移。
- 在 Python 3.9 / PyTorch 2.8 环境用官方 LongCat checkpoint 验收
  `LongCat.from_pretrained()`、短音频 encode/decode 和一步 acoustic
  forward/backward/optimizer step；本地 synthetic checkpoint 只覆盖 loader 契约。
- 用真实 100k LongCat BPE 与 Qwen checkpoint 对比优化前后的 model 初始化耗时、单步训练
  峰值显存和 cached generation 吞吐，确认分块 embedding、按模态稀疏监督 logits、RVQ valid
  frame packing 与 batched codec decode 的生产收益。
- 在真实训练 checkpoint 上复验 bfloat16 变长 batch generation，报告 batch/逐请求 token
  agreement rate 与 top-1 logit margin；随机 audio head 的逐 token parity 不作为生产门槛
  （[008 result, lines 41-49](results/008-real-batch-generation-benchmark.md#L41-L49)）。
- 按模态 token CE 改动后，重新运行真实 TTS/S2ST fixed-sample flow 与 RVQ 至少 2 steps；确认
  text/audio 局部词表 loss、backward、generation 和 waveform decode 均 finite，再决定是否重跑
  100-step overfit 趋势。
- 在相同 LongCat prepared data、model、optimizer 和训练预算下完成 codec/random audio
  embedding initialization 对照；005 的 2-step smoke 只验证闭环，2000-step 长跑只有 codec
  initialization，尚不支持优劣结论
  （[005 result, lines 15-18](results/005-acoustic-oracle-codec-screening.md#L15-L18)，
  [lines 78-96](results/005-acoustic-oracle-codec-screening.md#L78-L96)）。
- 在本轮 runtime 显式注入、输入约束、任务权重与 device 改动后，用两张 GPU 分别重新运行
  LongCat oracle 与 UniCodec fixed-sample wrapper 至少 2 steps，验收静态 `ddp`、多任务
  `find_unused_parameters=True`、跨 rank total loss 和 per-rank runtime device。

## 其他工程欠账

- 正式多任务 DDP 使用 `find_unused_parameters=True`；路径稳定后评估冻结策略或 DDP 优化。
