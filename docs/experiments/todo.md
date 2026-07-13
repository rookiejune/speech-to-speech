# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；本文只维护阶段状态、待办和待验证项，完成项及时删除。

## 阶段状态

- P0（冻结契约）：设计、代码与纯 contract tests 已对齐。
- P1-core：native audio tokenizer、semantic embedding、acoustic prompt、semantic CE、flow objective 和 waveform decode 已按冻结契约对齐。
- P1-closure：8 个任务的 fake raw sample、collator、forward/backward、optimizer step 与 waveform decode 闭环已完成。
- P1-real：121 上真实 WMT19/LongCat/Qwen3 TTS 与 S2ST 训练闭环及短 S2ST generation/decode smoke 已完成，结果见 `results/001-real-resource-smoke.md`。
- P1-overfit：真实单 batch TTS/S2ST 的 semantic 与 flow objective 均可优化，结果见 `results/002-single-batch-overfit.md`。
- P2：FiLM-conditioned acoustic DiT、Flow condition/objective 与在线 WavLM-base layer-9
  到 DiT layer-4 的可选 REPA 已接入并有 contract tests；RVQ decoder/model
  原语已存在，离散 acoustic objective、Lightning 训练组合和 generation service 未完成。
- P3：独立 request/result、单样本 KV cache、在线 acoustic condition、一次性 generation/decode、SampleLogger 复用及真实 Qwen3/LongCat cached smoke 已完成；标准变长 batch generation 未完成。
- 测试现状：50 个纯本地测试和 16 个任务子测试覆盖 audio tokenizer、P0 数据/ID 契约、
  模块所有权、HF backbone 加载与 vocabulary 边界、condition 对齐、全任务 semantic/flow
  objective 路由、fake P1 closure、cached generation、张量化 acoustic merge、stage resume、
  codec oracle 与 text retention contract。

## Generation

- padding + attention mask + 每行独立 EOS/EOA；frame mask 贯穿 acoustic sampling 和 decode。

## 真实资源验收

- 长时间完整训练使用 TensorBoard 记录监督曲线。
- 用真实 Qwen checkpoint 验收中英双向 `TextRetentionLogger`，确认训练前 wrapper 与 backbone 的文本输出一致，并观察语音训练期间的 NLL 漂移。
- 按 `schedules/005-acoustic-oracle-codec-screening.md` 完成 LongCat acoustic flow oracle
  与 UniCodec unified-token oracle 的 codec/random initialization 500-step single-batch
  overfit；121 smoke 已完成，结果见 `results/005-acoustic-oracle-codec-screening.md`。
- 在相同数据、初始化和训练预算下对比 DiT flow 与 DiT flow + REPA，验证 REPA 对 flow
  objective、重建质量和音频指标的增益后再确定默认权重。

## 其他工程欠账

- `loss/causal_lm.py`：实现 RVQ 离散 acoustic objective 后再暴露正式入口。
- 第一版 DDP 使用 `find_unused_parameters=True`；路径稳定后评估冻结策略或 DDP 优化。
