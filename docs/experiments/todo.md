# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 真实资源验收

- 用真实 Qwen checkpoint 验收中英双向 `TextRetentionLogger`，确认训练前 wrapper 与 backbone 的文本输出一致，并观察语音训练期间的 NLL 漂移。
- 在 Python 3.9 / PyTorch 2.8 环境用官方 LongCat checkpoint 验收
  `LongCat.from_pretrained()`、短音频 encode/decode 和一步 acoustic
  forward/backward/optimizer step；本地 synthetic checkpoint 只覆盖 loader 契约。
- 现有 100k LongCat BPE 在 010 的 1000 条临时数据上把全部 source/target 都压成单个 audio
  token，不能进入正式联合训练。完整 train split 就绪后，显式限制最大 token span 重新训练
  BPE，并先在 held-out split 验收压缩分布；随后再与 native token + Qwen checkpoint 对比
  model 初始化耗时、单步峰值显存和 cached generation 吞吐。首轮联合训练按
  [011 schedule](schedules/011-qwen-rvq-staged-joint-training.md) 使用 native token。
- 在真实训练 checkpoint 上复验 bfloat16 变长 batch generation，报告 batch/逐请求 token
  agreement rate 与 top-1 logit margin；随机 audio head 的逐 token parity 不作为生产门槛
  （[008 result, lines 41-49](results/008-real-batch-generation-benchmark.md#L41-L49)）。
- 011 P0 的真实 Qwen/native/RVQ TTS 与 S2ST 已完成 2-step forward/backward/optimizer 和
  teacher-forced waveform decode。registered `nn.Module` backbone 的 generation 能力 false
  negative 已在本地修复并补回归测试；现在原样重跑到两条 `generation.json`、`metrics.json`
  finite 且退出码为 0（[012 schedule](schedules/012-generation-capability-contract-rerun.md)，
  [011 result](results/011-qwen-rvq-staged-joint-training.md)）。
- 完成 011 的其余 P0：Flow TTS/S2ST 2-step 合同复验、两卡 DDP 与 resume、32-sample RVQ
  100-step、1k pilot，以及 010 checkpoint 严格 import 后丢弃；P0 全部门槛通过前不进入 A。
- 在相同 LongCat prepared data、model、optimizer 和训练预算下完成 codec/random audio
  embedding initialization 对照；当前代码尚无可比较结果，完整对照前不支持初始化优劣结论。
- 用真实 LongCat Flow/RVQ 各重跑单卡与两卡 2-step oracle，确认轻量 checkpoint 无 backbone key、
  state dict 可按白名单严格导入联合模型，并记录相对 010 的 checkpoint 大小与峰值显存变化。
- 在本轮 runtime 显式注入、输入约束、任务权重与 device 改动后，用两张 GPU 重新运行 UniCodec
  fixed-sample wrapper 至少 2 steps，验收多任务 `find_unused_parameters=True`、跨 rank total loss
  和 per-rank runtime device。LongCat Flow/RVQ oracle 静态 DDP 已完成
  （[010 result](results/010-codec-oracle-flow-rvq-smoke.md)）。

## 其他工程欠账

- 正式多任务 DDP 使用 `find_unused_parameters=True`；路径稳定后评估冻结策略或 DDP 优化。
- 联合 token/Flow/RVQ 的动态 batch FLOPs provider 已接入 overfit opt-in，但当前仅支持全量训练且
  不支持 REPA。正式 staged joint entry 启用 performance 前，需要为分阶段冻结定义按组件区分的
  forward/backward multiplier，补齐 REPA 口径，并在正式入口完成无 `TaskSampleLogger`/`GradLogger` 的
  performance composition 与 DDP/LBA 验收。
