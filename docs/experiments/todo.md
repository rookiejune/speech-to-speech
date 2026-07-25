# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 014 LongCat stable stage 1

P0 已在 debug-migrated copy 上通过，证据见
[014 result](results/014-longcat-stable-stage1.md)。后续代码已允许缺失 duration metadata 时从
codec frame count 和 runtime frame rate 推导音频秒数，不再要求正式 root 先补写
`AudioMeta.DURATION`；正式 stable data root 仍需完成 fingerprint、split 与分布验收。

- 验证正式 stable data root 在无 duration metadata 时可直接 parse、LBA 和训练；禁止把真实音频静默
  计为 0。
- 固化正式 stable data root、split manifest、LongCat view fingerprint 和 native token/RVQ
  分布；禁止依赖 `/tmp` 或 debug copy 进入 stage 1 长跑。
- 建立 800/100/100 pilot split，并记录 parse/span error、semantic token、source/target frame、
  RVQ codebook 和文本长度统计。
- 跑 native-token stable stage 1：32-sample fixed overfit、1k pilot、两卡 DDP 2-step 与 resume。
- native stage 1 达标后，再在完整 train split 上重训 speech BPE 并做 shadow ablation；旧 100k
  BPE collapse artifact 不复用。
- 只有 BPE 在 held-out 分布、decode finite、dev CE 和吞吐/显存上通过 A/B，才允许进入后续 stage。

## 其他工程欠账

- 正式多任务 DDP 初始仍使用 `find_unused_parameters=True`；native stable path 固化后再评估静态
  DDP 或冻结策略优化。
- stage 1 companion manifest 需要绑定 checkpoint、tokenizer/native-or-BPE、semantic vocab、
  RVQ codebook、decoder config、state-dict prefix whitelist 和 split fingerprint。
- 后续进入 Qwen partial/full joint 前，再补齐 stage-specific FLOPs provider、validation generation、
  sample archive checkpoint 与 long-run TensorBoard 监督曲线。
