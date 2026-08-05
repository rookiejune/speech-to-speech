# TODO

设计契约见 `docs/model-design.md` 与 `docs/design/`；已验证结论及支撑结果见
`docs/experiments/conclusion.md`。本文只维护未完成的验证和工程欠账，完成项及时删除。

## 当前 staged-joint 前置条件

- generator plugin 的 conditioner、acoustic generator 预训练和 artifact 导出在仓库外的
  `semantic-acoustic-generator` 完成；本仓库只消费导出的 frame-aligned
  `AcousticGeneratorArtifact`。
- Stage 0-3 的 Flow/RVQ 正式训练开始前，需用
  `SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT=<external-artifact>` 完成 composition smoke；wrapper 将其传给
  `model.acoustic.init_artifact`。同时记录 route、decoder、frame layout、backend metadata 和
  artifact 版本。
- 当前课程入口是 `jobs/011/03_staged_joint_train.sh`，顺序为 Stage 0 TTS + MT、Stage 1
  ASR + TTS + MT、Stage 2 分解任务、Stage 3 直接 S2ST 主目标。
- 实现并验证跨 stage 的 weight-only handoff：Stage 0 到 Stage 1 先 merge/export PEFT LoRA，所有
  参数策略切换都只加载模型权重并重建 optimizer；`train.ckpt_path` 仅用于同 stage 断点续训。

## 其他工程欠账

- 监控 BiCodec + Qwen3-0.6B Stage 0 的 10k stability / learning-curve pilot：
  - id: `s2s-022-bicodec-qwen06b-10k`
  - state: `running`
  - entry: `scripts/train.py`（Stage 0、`runtime=bicodec`、4-rank DDP、10k steps、cost
    batching 4800 frames / max 8 samples、TTS/MT workers 0）
  - num_gpus: `4`
  - gpu: `4xA100-40GB`
  - min_vram_gb_per_gpu: `32`（新 run 至 step 229 的早期观测峰值约 22.2GB，保留后续长样本余量）
  - preferred_hosts: `121`
  - host: `121`
  - cuda_visible_devices: `0,1,2,3`
  - started_at: `2026-08-05 01:56:23 +08:00`
  - estimated_hours: `about 4.1 from launch at the step 19-219 rate`
  - monitor: `TensorBoard + train log + 200ms nvidia-smi CSV + anydataset epoch-plan debug；检查
    step 220、step 1000、首个 iterator rollover、step 7000 和最终 10k`
  - ready_gate: 隔离 revision 已记录；anydataset 52 项与 speech-to-speech 20 项定向测试通过；
    split manifest 保留 39,998 条并排除两条超过 4,800 frames 的样本；首轮四 rank epoch plan
    count 已一次同步并裁齐到 1,453；新 run 已到 step 229，跨过旧 step 149 stall，step 19-139
    有效 token 吞吐较旧 cost-aware all-workers-0 提高约 12.5%。旧 fixed batch 2 run 在 step
    6907 因超长样本 OOM，不能作为完成的 10k stability run。
  - output_root: `<pami201-backed train root>/speech-to-speech/021-bicodec-qwen06b-cost-epoch/<run>`
  - task: 继续检查 step 1000、iterator rollover、step 7000 和 10k；当前没有 held-out TTS
    split，不将本 pilot 记作质量长跑通过。
- Interleave 方案暂缓，不作为当前 active todo；先用 PEFT LoRA 加固定 `TextRetentionLogger`
  baseline 验证参数高效适配与文本保真度，再决定是否实现 Interleave。
- staged-joint companion manifest 需要绑定 checkpoint、tokenizer/native-or-BPE、semantic vocab、
  外部 generator artifact、RVQ codebook、decoder config、state-dict prefix whitelist 和 split fingerprint。
- 后续进入 Qwen partial/full joint 前，再补齐 stage-specific FLOPs provider、sample archive
  checkpoint 与 long-run TensorBoard 监督曲线；真实长跑质量仍需按结果文档验收。
