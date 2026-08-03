# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 014 LongCat native stage 1（blocked）

- 当前 FP32 step 1000 仍未达到 CE gate；不继续追加 native-token stable stage 1 长跑预算。
- 只有后续明确修改训练目标并重新达到 CE gate 后，才执行 teacher-forced decode，并检查各
  codebook top-1 是否稳定优于无条件众数。
- native stage 1 达标前，不重训 full train split speech BPE，也不复用旧 100k BPE collapse artifact。
- 只有 BPE 在 held-out 分布、decode finite、dev CE 和吞吐/显存上通过 A/B，才允许进入后续 stage。

## 016 Stable Codec stage 1

- id: s2s-016-stable-stage1-longrun
- state: ready
- entry: `jobs/015/01_stable_codec_stage1.sh`
- num_gpus: 1
- gpu: 1xA100-40GB
- min_vram_gb_per_gpu: 40GB-class until measured lower
- preferred_hosts: 121
- monitor: TensorBoard fixed samples + checkpoint/resume + nvidia-smi
- ready_gate: 真实单步 smoke 已验证 Stable Codec、无 BPE、ASR/TTS fixed samples 和 finite loss。
- task: 启动正式 1,000,000-step 长跑，保留每 10,000 steps 的 TensorBoard fixed samples、周期 checkpoint 与最新恢复点；单步 smoke 不支持质量或收敛结论。

## 017 Instruction template ablation

- id: s2s-017-instruction-template-ablation
- state: ready
- entry: 沿用当前 stage 1 入口
- num_gpus: 1
- gpu: probe
- min_vram_gb_per_gpu: probe
- preferred_hosts: 与当前 stage 1 相同
- monitor: 主任务 CE + fixed-sample 生成；文本任务加 retention probe
- ready_gate: 默认已是 per-task `index=0`；本项通常可跳过。
- task: 不做 `null` vs fixed 消融；不做指令语言或次要 AR/MASKED 文案消融。

## 其他工程欠账

- Interleave 方案暂缓，不作为当前 active todo；先用 PEFT LoRA 加固定 `TextRetentionLogger`
  baseline 验证参数高效适配与文本保真度，再决定是否实现 Interleave。
- stage 1 companion manifest 需要绑定 checkpoint、tokenizer/native-or-BPE、semantic vocab、
  RVQ codebook、decoder config、state-dict prefix whitelist 和 split fingerprint。
- 后续进入 Qwen partial/full joint 前，再补齐 stage-specific FLOPs provider、sample archive
  checkpoint 与 long-run TensorBoard 监督曲线；真实长跑质量仍需按结果文档验收。
