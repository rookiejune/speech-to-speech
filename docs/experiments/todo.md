# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 014 LongCat stable stage 1

剩余正式验收：

- 当前 FP32 step 1000 仍未达到 CE gate；只有后续明确修改训练目标并重新达到 gate 后，才执行
  teacher-forced decode，并检查各 codebook top-1 是否稳定优于无条件众数。两项都通过后才允许
  进入更长 pilot。
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
