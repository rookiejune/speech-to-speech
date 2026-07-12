# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；本文只维护阶段状态、待办和待验证项，完成项及时删除。

## 阶段状态

- P0（冻结契约）：设计、代码与纯 contract tests 已对齐。
- P1-core：native audio tokenizer、semantic embedding、acoustic prompt、semantic CE、flow objective 和 waveform decode 已按冻结契约对齐。
- P1-closure：7 个任务的 fake raw sample、collator、forward/backward、optimizer step 与 waveform decode 闭环已完成。
- P2：Flow/RVQ 相关实现已存在；flow condition/objective 已有 contract tests，RVQ objective contract tests 未完成。
- P3：当前只有过渡性的逐行 generation、teacher-forcing decode 和 sample logging；KV cache、独立推理入口及标准变长 batch generation 未完成。
- 测试现状：20 个纯本地测试和 14 个任务子测试覆盖 audio tokenizer、P0 数据/ID 契约、模块所有权、HF backbone 加载与 vocabulary 边界、condition 对齐、全任务 semantic/flow objective 路由与 fake P1 closure。

## Generation

- 实现 KV cache：首步编码完整多模态 prompt，后续只输入新 token。
- source acoustic prompt 通过 cache 在整个生成过程中持续生效。
- text/audio generation 使用各自 allowed IDs；audio 不生成 BOA，EOA 只作为 stop token。
- semantic token 采样时在线收集预测 hidden，并按 BPE span 展开 acoustic frame condition。
- 建立独立真实推理入口，不通过 `ModelBatch.acoustic_labels is None` 判断 generation。
- 同一次生成结果同时提供 token、acoustic output 和 waveform，SampleLogger 不重复随机生成。
- padding + attention mask + 每行独立 EOS/EOA；frame mask 贯穿 acoustic sampling 和 decode。
- KV cache 与非 cache greedy 路径输出一致。

## 真实资源验收

- 使用 `wmt19_tts_codec(config.codec)` 完成一个真实 batch 的 forward、backward 和 optimizer step。
- 完成 S2ST semantic/acoustic generation 和 waveform decode。
- 长时间完整训练使用 TensorBoard 记录监督曲线。
- 将 smoke、generation 和 decode 结果记录到 `docs/experiments/results/`。

## 其他工程欠账

- 单 batch overfit diagnostic。
- `Speech.language` 接入 `Language` 枚举。
- `loss/causal_lm.py`：实现 RVQ 离散 acoustic objective 后再暴露正式入口。
- 第一版 DDP 使用 `find_unused_parameters=True`；路径稳定后评估冻结策略或 DDP 优化。
