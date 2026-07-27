# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 014 LongCat stable stage 1

剩余正式验收：

- 修正 stage 1 dtype 边界：frozen Qwen backbone 保持 BF16，随机初始化且可训练的 semantic
  speech interface 与 acoustic decoder 保持 FP32 参数存储，forward 仍使用 mixed-precision
  autocast。该因果 A/B 不从旧 BF16 checkpoint 恢复；用同 seed、同 split 从头跑 500-step A/B，对比
  step 0/100/200/300/400/500 dev CE、各 codebook top-1 和实际发生变化的参数比例。
- A/B 同时核对原始 dev `4265` frames 与 validation 有效 `4264` frames 的差异来自哪一个
  causal/truncation mask；两组实验必须使用同一有效位口径。
- 只有 FP32-storage A/B 明确优于当前 BF16 基线，且 top-1 优于无条件 codebook 众数基线，
  才继续更长 pilot 并执行 teacher-forced decode；当前停止从 step 2000 机械延长到 5k。
- 跑 native-token stable stage 1 长跑，保留 TensorBoard 监督曲线、周期 checkpoint
  和可恢复的最新 checkpoint；当前 1k pilot 只验证数据分片、两卡 DDP 和
  resume 执行契约，不支持质量或收敛结论。
- native stage 1 达标后，再在完整 train split 上重训 speech BPE 并做 shadow ablation；旧 100k
  BPE collapse artifact 不复用。
- 只有 BPE 在 held-out 分布、decode finite、dev CE 和吞吐/显存上通过 A/B，才允许进入后续 stage。

## 其他工程欠账

- stage 1 companion manifest 需要绑定 checkpoint、tokenizer/native-or-BPE、semantic vocab、
  RVQ codebook、decoder config、state-dict prefix whitelist 和 split fingerprint。
- 后续进入 Qwen partial/full joint 前，再补齐 stage-specific FLOPs provider、validation generation、
  sample archive checkpoint 与 long-run TensorBoard 监督曲线。
