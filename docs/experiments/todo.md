# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；本文只维护阶段状态、待办和待验证项，完成项及时删除。

## 阶段状态

- P0（冻结契约）：设计、代码与纯 contract tests 已对齐。
- P1-core：native audio tokenizer、semantic embedding、acoustic prompt、semantic CE、flow objective 和 waveform decode 已按冻结契约对齐。
- P1-closure：8 个任务的 fake raw sample、collator、forward/backward、optimizer step 与 waveform decode 闭环已完成。
- P1-real：121 上真实 WMT19/LongCat/Qwen3 TTS 与 S2ST 训练闭环及短 S2ST generation/decode smoke 已完成，结果见 `results/001-real-resource-smoke.md`。
- P1-overfit：固定真实 sample 的 TTS/S2ST semantic 与 flow objective 均完成 100-step
  overfit 验收，结果见 `results/002-single-batch-overfit.md`。
- P2：FiLM-conditioned acoustic DiT、Flow condition/objective 与在线 WavLM-base layer-9
  到 DiT layer-4 的可选 REPA 已接入并有 contract tests，121 真实训练与全部 callback
  smoke 已完成，结果见 `results/006-real-callback-repa-smoke.md`；8 层 Qwen RVQ decoder、
  离散 acoustic objective、Lightning 训练组合和 generation service 已接入并有 contract tests，
  真实 TTS/S2ST 100-step 训练、packed-code sampling 和 waveform decode 已验收，
  训练后 TTS/S2ST semantic-to-waveform generation 也已闭环，结果见
  `results/009-real-rvq-overfit-generation.md`。
- P3：独立 request/result、变长 batch KV cache、逐行 stop state、在线 acoustic
  condition/frame mask、一次性 generation/decode、SampleLogger 复用及真实
  Qwen3/LongCat cached smoke 已完成；真实变长 batch 4 在 float32 下完成与逐请求
  的 token parity 及 waveform decode，短生成吞吐为逐请求的 1.78x，结果见
  `results/008-real-batch-generation-benchmark.md`。
- 测试现状：93 个纯本地测试覆盖 audio tokenizer、P0 数据/ID 契约、
  模块所有权、HF backbone 加载与 vocabulary 边界、condition 对齐、全任务 semantic/flow
  objective 路由、有效监督位置 logits、fake P1 closure、cached modality generation、设备侧
  span、张量化 acoustic merge、callback RNG、DDP config composition、stage resume、codec
  oracle、text retention 与 Python 3.9 entry contract；无 CUDA 的本地环境跳过 1 个 CUDA RNG 用例。

## 真实资源验收

- 长时间完整训练使用 TensorBoard 记录监督曲线。
- 用真实 Qwen checkpoint 验收中英双向 `TextRetentionLogger`，确认训练前 wrapper 与 backbone 的文本输出一致，并观察语音训练期间的 NLL 漂移。
- 在 Python 3.9 / PyTorch 2.8 环境用官方 LongCat checkpoint 验收
  `LongCat.from_pretrained()`、短音频 encode/decode 和一步 acoustic
  forward/backward/optimizer step；本地 synthetic checkpoint 只覆盖 loader 契约。
- 用真实 100k LongCat BPE 与 Qwen checkpoint 对比优化前后的 model 初始化耗时、单步训练
  峰值显存和 cached generation 吞吐，确认分块 embedding 与稀疏监督 logits 的生产收益。
- 在本轮冻结、strategy 与 device 改动后，用两张 GPU 分别重新运行 LongCat oracle 与 UniCodec
  fixed-sample wrapper 至少 2 steps，验收静态 `ddp`、多任务 `find_unused_parameters=True` 和
  per-rank runtime device。

## 其他工程欠账

- 正式多任务 DDP 使用 `find_unused_parameters=True`；路径稳定后评估冻结策略或 DDP 优化。
