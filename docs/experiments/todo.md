# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 当前 staged-joint 前置条件

- SAC 的 conditioner、acoustic generator 预训练和 artifact 导出在仓库外的
  `semantic-acoustic-codec` 完成；本仓库只消费导出的 frame-aligned
  `AcousticGeneratorArtifact`。
- Stage 0-3 的 Flow/RVQ 正式训练开始前，需用
  `SPEECH_TO_SPEECH_SAC_ARTIFACT=<external-artifact>` 完成 composition smoke；wrapper 将其传给
  `model.acoustic.init_artifact`。同时记录 route、decoder、frame layout、backend metadata 和
  artifact 版本。
- 当前课程入口是 `jobs/011/03_staged_joint_train.sh`，顺序为 Stage 0 TTS + MT、Stage 1
  ASR + TTS + MT、Stage 2 分解任务、Stage 3 直接 S2ST 主目标。
- 实现并验证跨 stage 的 weight-only handoff：Stage 0 到 Stage 1 先 merge/export PEFT LoRA，所有
  参数策略切换都只加载模型权重并重建 optimizer；`train.ckpt_path` 仅用于同 stage 断点续训。

## 其他工程欠账

- Interleave 方案暂缓，不作为当前 active todo；先用 PEFT LoRA 加固定 `TextRetentionLogger`
  baseline 验证参数高效适配与文本保真度，再决定是否实现 Interleave。
- staged-joint companion manifest 需要绑定 checkpoint、tokenizer/native-or-BPE、semantic vocab、
  外部 SAC artifact、RVQ codebook、decoder config、state-dict prefix whitelist 和 split fingerprint。
- 后续进入 Qwen partial/full joint 前，再补齐 stage-specific FLOPs provider、sample archive
  checkpoint 与 long-run TensorBoard 监督曲线；真实长跑质量仍需按结果文档验收。
