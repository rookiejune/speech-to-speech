# pl_module 与 callback

Lightning 训练循环、生成路径和日志。生成契约的权威定义见 [总览 §6](../model-design.md)。

## pl_module 对外能力

- `SpeechToSpeech`（LightningModule）：
  - `training_step()`：调用 `Loss.forward(batch, model)`，返回的 mapping 直接满足 Lightning 契约；loss outputs 保留到 backward 结束，供 `GradLogger` 读取分项梯度。
  - generation service：接收独立的真实推理输入，按 task 组织 text/audio 状态机、allowed tokens、变长裁剪、acoustic sampling 和 decode；不通过 `ModelBatch.acoustic_labels is None` 判断推理模式。
  - teacher-forcing evaluation：消费完整 `ModelBatch`，condition 接口传 token 自身位置 `p`。
  - `configure_optimizers()`：委托 anytrain 的 optimizer preset。
- `decode_generated_audio()`：`semantic ids [B, T, K_semantic]` + `acoustic features [B, T, D]` → waveform，只要求 frame 轴对齐。FM 模型直接提供 features，RVQ 模型提供 codes 由 codec dequantize。

## callback 对外能力

- `StageSwitcher`：按 epoch milestone 切换 datamodule 的任务权重策略。
- `logging.OutputsLogger`：只消费 `Outputs` 中的 `LossItem`，按 task 聚合记录，不依赖模型内部 head；不同 rank 的 task 列表可能不同，不做同步。
- `logging.GradLogger`：对指定参数比较两个 loss 分项的梯度范数。
- `logging.FlowMatchingLogger`:记录 flow time sampler 配置和训练采样时间。
- `logging.SampleLogger`：定期对固定样本生成；token、acoustic output 与 waveform 必须复用同一次生成结果。

## 边界

- semantic generation 使用 KV cache；首步注入多模态 prompt，后续只输入新 token。
- 生成路径当前按行循环调用单样本路径，标准批量自回归是目标契约（总览 §6），欠账见 todo。
- pl_module 不实现对齐或 loss 逻辑，只组合 model 与 loss 的公开接口。
- callback 只依赖 `Outputs`/`LossItem` 和 datamodule 公开入口，不触碰模型内部结构。
