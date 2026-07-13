# pl_module 与 callback

Lightning 训练循环、生成路径和日志。生成契约的权威定义见 [总览 §6](../model-design.md)。

## pl_module 对外能力

- `SpeechToSpeech`（LightningModule）：
  - `training_step()`：调用 `Loss.forward(batch, model)`，返回的 mapping 直接满足 Lightning 契约；loss outputs 保留到 backward 结束，供 `GradLogger` 读取分项梯度。
  - generation service：接收独立的真实推理输入，按 task 组织 text/audio 状态机、allowed tokens、变长裁剪、acoustic sampling 和 decode；不通过 `ModelBatch.acoustic_labels is None` 判断推理模式。
  - `evaluate_text()`：对固定的 text probes 执行 greedy generation 和 reference teacher-forced NLL，恢复调用前的 module mode，并向 callback 返回结构化结果。
  - teacher-forcing evaluation：消费完整 `ModelBatch`，condition 接口传 token 自身位置 `p`。
  - `configure_optimizers()`：委托 anytrain 的 optimizer preset。
- `generation.Request`：无 padding 的 semantic prompt、task 和可选 `AcousticPrompt`；
  `AcousticPrompt` 把 source acoustic IDs/positions 组织成不可拆分的结构，不携带 target labels。
- `generation.Result`：裁掉 BOA/EOA 后的 token 和可选 `AudioOutput`；`AudioOutput` 将
  acoustic features/waveform 组成不可拆分的结构，三个输出来自同一次生成。
- `generation.requests_from_batch()`：仅供 teacher-forcing 样本日志使用的 adapter，不是真实推理入口。
- `decode_generated_audio()`：`semantic ids [B, T, K_semantic]` 与 acoustic features 或
  codes → waveform，只要求 frame 轴对齐。helper 已支持 codes dequantize，但当前正式
  generation service 只组合 flow model。

## callback 对外能力

- `StageSwitcher`：按 epoch milestone 切换 datamodule 的任务权重策略；fit start 根据
  `current_epoch` 恢复对应阶段，不依赖 callback 私有状态续训。
- `logging.OutputsLogger`：只消费 `Outputs` 中的 `LossItem`，按 task 聚合记录，不依赖模型内部 head；不同 rank 的 task 列表可能不同，不做同步。
- `logging.GradLogger`：对指定参数比较两个 loss 分项的梯度范数。
- `logging.GradNormLogger`：记录 module 全部有效参数梯度的全局 L2 norm。
- `logging.FlowMatchingLogger`：构造时显式接收 flow runtime，记录其 time sampler 配置和
  训练采样时间，不向下读取 Lightning module 的 model/runtime。
- `logging.SampleLogger`：按 `every_n_steps` 在 global zero 对固定样本生成；token、
  acoustic output 与 waveform 必须复用同一次生成结果；它通过 `trainer.datamodule` 的公开
  `train_samples()`/`collator` 能力准备输入，其他 rank 不准备样本或执行推理。
- `logging.TextRetentionLogger`：只在 global zero 上于 fit 开始和固定 step 对用户提供的纯文本
  probes 做 greedy generation，并记录 reference teacher-forced NLL、相对初始 NLL 变化和
  解码文本；不接入训练 loss。

```python
TextRetentionLogger(
    {
        "zh_en": {
            "instruction": "Translate into English: 昨晚的暴雨导致三趟列车晚点。",
            "reference": "Last night's heavy rain delayed three trains.",
        },
        "en_zh": {
            "instruction": "Translate into Chinese: The museum reopens next Tuesday.",
            "reference": "博物馆将于下周二重新开放。",
        },
    },
    every_n_steps=1_000,
)
```

## 边界

- semantic generation 使用 KV cache；首步注入多模态 prompt，后续只输入新 token。
- generation service 只允许 audio-source task 携带 `AcousticPrompt`；text-source 或无 source
  的 task 携带该结构时在入口显式报错。
- total loss 由 `SpeechToSpeech.training_step()` 记录一次；`OutputsLogger` 只负责按
  task/objective 展开 `LossItem`，不重复记录 total loss。
- generation service 当前按 request 调用单样本路径，标准批量自回归是目标契约（总览 §6），欠账见 todo。
- pl_module 不实现对齐或 loss 逻辑，只组合 model 与 loss 的公开接口。
- callback 只依赖 `Outputs`/`LossItem`、datamodule 和 pl_module 公开入口，不触碰模型内部结构。
