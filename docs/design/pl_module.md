# pl_module 与 callback

Lightning 训练集成和日志边界。独立推理契约见 [generation](generation.md)。

## pl_module

`SpeechToSpeechModule[ModelT]` 是薄 Lightning wrapper：

- 构造时通过 `Objective[ModelT]` 保留 model/objective 类型配对。
- `training_step()` 调用 objective，跨 rank 归约并记录一次 total loss，同时保留分项到 backward
  完成。
- `current_loss_outputs()`：只在当前 training step 的 backward 完成前返回仍连接计算图的
  `Outputs`，供 `GradLogger` 计算指定分项梯度；其他时机显式报错。
- `configure_optimizers()` 委托 anytrain optimizer preset。
- `generate()` / `evaluate_text()` 只负责切换 eval mode、调用 generation 包并恢复原 mode。

`pl_module` 不实现 task 状态机、decode、文本 NLL、对齐或 loss；包级 API 只导出
`Config` 与 `SpeechToSpeechModule`。

## callback

`speech_to_speech.callback` 只导出 `StageConfig` 与 `StageSwitcher`；以下日志 callback 从
`speech_to_speech.callback.logging` 导入：

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

- `SpeechToSpeechModule.generate()` / `evaluate_text()` 不持有跨调用 generation cache，也不修改
  request/result；validation、batching 和 decode 仍由 generation service 负责。
- callback 只依赖 `Outputs`/`LossItem`、datamodule 与 pl_module 公共能力；`GradLogger` 额外要求
  LightningModule 实现 `current_loss_outputs()` 生命周期契约。
- total loss 只由 LightningModule 以 `sync_dist=True` 记录一次，分项 logger 不重复记录。
