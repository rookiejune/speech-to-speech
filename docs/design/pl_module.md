# pl_module、generation 与 callback

Lightning 训练集成、独立推理 service 和日志边界。生成契约见
[总览 §6](../model-design.md)。

## pl_module

`SpeechToSpeechModule[ModelT]` 是薄 Lightning wrapper：

- 构造时通过 `Objective[ModelT]` 保留 model/objective 类型配对。
- `training_step()` 调用 objective，跨 rank 归约并记录一次 total loss，同时保留分项到 backward
  完成。
- `configure_optimizers()` 委托 anytrain optimizer preset。
- `generate()` / `evaluate_text()` 只负责切换 eval mode、调用 generation 包并恢复原 mode。

`pl_module` 不实现 task 状态机、decode、文本 NLL、对齐或 loss；包级 API 只导出
`Config` 与 `SpeechToSpeechModule`。

## generation

`speech_to_speech.generation` 独立于 Lightning，公开：

- `Request(prompt_ids, task, acoustic_prompt)`：无 target、无 batch padding 的真实推理输入。
- `AcousticPrompt(codes, token_positions)`：source codec-local acoustic codes 及其 prompt token
  位置；不携带 target。
- `Result(response_ids, audio)`：裁掉 stop token 的响应与可选 `AudioOutput`。
- `generate_responses()`：按 target modality 和 acoustic prompt signature 分组、batch generation、逐行
  stop、顺序恢复与 decode。
- `generation.batch.requests_from_batch()`：仅供 teacher-forcing 日志把完整 `ModelBatch`
  转为 request；基础 generation 包不依赖训练 batch 或 Lightning。
- `decode_generated_audio()` / `decode_generated_codes()`：audio token + acoustic
  feature/code 到 waveform。
- `evaluate_text()`：greedy text generation 与 reference NLL。

unified-token codec 返回 `AudioOutput(features=None, ...)`，直接 decode semantic codes；flow
与 RVQ 都向 service 返回 codec acoustic features，上层不按 objective 分叉。

## callback

- `WorldSizeContract`：fit start 校验实际 world size。
- `StageSwitcher`：按 `epoch_milestones` 调用 datamodule 的 `set_task_weights()`，并从
  `current_epoch` 恢复当前 stage。
- `OutputsLogger`：按 task 展开 `LossItem`，不读取 model head。
- `GradLogger` / `GradNormLogger`：记录指定分项或全局梯度范数。
- `FlowMatchingLogger`：显式接收 flow runtime，不向下读取 model runtime。
- `SampleLogger`：只在 global zero 读取 datamodule 的公开 `train_samples()`/`collator`，一次
  generation 结果复用 token、features 与 waveform。
- `TextRetentionLogger`：记录 text probe generation、reference NLL 与相对基线漂移。

Sample/evaluation callback 在隔离 RNG context 内运行，不改变后续训练的 CPU 或当前 CUDA
random state。

## 边界

- `AcousticPrompt` 只允许用于 audio-source task，非法组合在 service 入口报错。
- service 在 padding 前校验原始 request：prompt 是非空一维 runtime global ID；acoustic codes
  是与 codec codebook 数量和范围一致的非空二维整数 Tensor；frame positions 是对齐、非负且
  指向 prompt 内 codec audio token 的一维整数 Tensor。无 acoustic codebook 的 codec 不接受
  `AcousticPrompt`，`-1` 只由 service 作为 batch padding 引入。
- acoustic model 把生成时已有的每行有效 frame count 返回给 service；service 不重新展开 token
  span，并把 token 数和 frame 数相同的行合并为一次 codec decode。变长 prompt 仍使用左
  padding，EOS/EOA 与结果顺序逐行跟踪。
- callback 只依赖 `Outputs`/`LossItem`、datamodule 与 pl_module 公共能力。
- total loss 只由 LightningModule 以 `sync_dist=True` 记录一次，分项 logger 不重复记录。
