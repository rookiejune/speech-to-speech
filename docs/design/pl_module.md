# pl_module 与 callback

Lightning 训练集成和日志边界。独立推理契约见 [generation](generation.md)。

## pl_module

`SpeechToSpeechModule[ModelT]` 是薄 Lightning wrapper：

- 构造时通过 `Objective[ModelT]` 保留 model/objective 类型配对。
- `training_step()` 接收单个 `ModelBatch` 或多个 homogeneous 子 batch；联合 batch 会逐个调用
  objective，并按有效 token/frame 聚合分项 loss。它跨 rank 归约并记录一次 total loss，同时保留
  分项到 backward 完成。
- `current_loss_outputs()`：只在当前 training step 的 backward 完成前返回仍连接计算图的
  `Outputs`，供 `GradLogger` 计算指定分项梯度；其他时机显式报错。
- `configure_optimizers()` 委托 anytrain optimizer preset。
- `generate()` / `evaluate_text()` 只负责切换 eval mode、调用 generation 包并恢复原 mode。

`pl_module.composition` 负责组装 `model + objective + SpeechToSpeechModule` 的 token/Flow/RVQ
组合，入口只选择组合并传入已解析配置；该模块通过窄 Protocol 消费 acoustic config，不反向依赖
`scripts._config`。

`pl_module` 不实现 task 状态机、decode、文本 NLL、对齐或 loss；包级 API 只导出
`Config` 与 `SpeechToSpeechModule`，composition 通过显式子模块导入。

## callback

`speech_to_speech.callback` 只导出 `StageConfig` 与 `StageSwitcher`；以下日志 callback 从
`speech_to_speech.callback.logging` 导入：

- `StageSwitcher`：按 `epoch_milestones` 切换 task 权重、loader 权重和参数冻结，并从
  `current_epoch` 恢复当前 stage。
- `OutputsLogger`：按 task 展开 `LossItem`，不读取 model head。
- `GradLogger` / `GradNormLogger`：记录指定分项或全局梯度范数。
- `FlowMatchingLogger`：显式接收 flow runtime，不向下读取 model runtime。
- `LossSummary`：收集训练输出里的 total loss 与分项 `LossItem`，只在训练结束后生成窗口摘要。
- `AcousticEvaluation`：对 fixed-sample acoustic model 使用本地 generator seeds 采样，记录 feature、
  waveform 与 STFT 距离；纯评估函数位于 `generation.evaluation`，不留在脚本私有模块。
- `TaskSampleLogger`：只在 global zero 读取 datamodule 的公开 `train_samples()`/`collator`，
  按真实 task 记录 source/reference/generated metadata，并复用一次 generation 的 token、features
  与 waveform；它不运行额外神经网络评估器，也不重复计算 loss。
- `TextRetentionLogger`：记录 text probe generation、reference NLL 与相对基线漂移。

Task sample/evaluation callback 在隔离 RNG context 内运行，不改变后续训练的 CPU 或当前 CUDA
random state。

## performance

联合训练显式启用 performance 时，必须同时设置 `callbacks.task_sample.enabled=false`；入口显式拒绝
performance 与 task sample logging 同时启用。`TaskSampleLogger` 在 `on_train_batch_start` 只由 rank zero
执行 generation，DDP 的其他 rank 会在后续同步点等待，因此改变 callback 顺序也不能可靠地把这段
额外工作从各 rank 的 step time 中排除。

满足该前提后，`scripts/overfit.py` 使用 `speech_to_speech.performance.TrainingFlops` 组装
`anytrain.PerformanceCallback`。provider 按实际 module、batch 和 objective 输出分析 token、Flow 或
RVQ 路径的动态训练 FLOPs；入口把 performance callback 放在 callback 列表首位，并省略
`GradLogger` 与 `GradNormLogger`。前者的额外 `autograd.grad`、后者重复的全局梯度 norm 如果与
MFU 同时运行，会增加实测 step time，但没有对应的模型训练 FLOPs，因而会扭曲指标口径。DDP 默认
在每个 batch timer 前 barrier，使上一 batch 仅 rank zero 执行的日志与评估不会泄漏到下一步计时；
可通过 `callbacks.performance.sync_distributed` 显式关闭。

当前 provider 的支持边界是标准 Qwen3 FlashAttention 2 backbone、标准 adapter/Flow/RVQ decoder
和全量训练。它校验 objective/model 配对及实际输出分支；REPA、分阶段冻结、替换后的模块或无法
识别的结构会明确报错，不用不完整公式继续记录 MFU。

估算口径统计 Linear 与 attention matrix multiplication，并按 forward 的两倍估算 backward；lookup、
scatter、normalization、activation、loss 和冻结 codec feature extraction 不计入模型 FLOPs，但对应
耗时仍在实际 step time 中。

生产统计不从单个 `example_input_array` 推导固定 FLOPs；该字段只提供一个 forward 示例，供 summary、
tracing 或 graph logging 使用，也不直接使用 `lightning.fabric.utilities.throughput.measure_flops()`。
实际 batch 的有效序列/帧长度、padded shape、objective 分支和各 rank 的 LBA 工作量都可能不同，
FlashAttention 或其他 custom op 也可能不在通用算子计数覆盖范围内；生产 provider 因此使用动态
解析计数，DDP 聚合与 step timing 由 anytrain 负责。Lightning 的 `measure_flops()` 只用于测试或
校准受支持的基础算子公式，不能替代该生产口径。

## 边界

- `SpeechToSpeechModule.generate()` / `evaluate_text()` 不持有跨调用 generation cache，也不修改
  request/result；validation、batching 和 decode 仍由 generation service 负责。
- callback 只依赖 `Outputs`/`LossItem`、datamodule 与 pl_module 公共能力；`GradLogger` 额外要求
  LightningModule 实现 `current_loss_outputs()` 生命周期契约。
- `TrainingFlops` 负责解释 speech-to-speech 的模型、batch 与 objective；`PerformanceCallback` 只负责
  optimizer-step 聚合、计时、硬件峰值推断和 MFU 记录，不内置任务 batch schema。
- overfit performance composition 不包含 `TaskSampleLogger`、`GradLogger` 或 `GradNormLogger`；前者由
  配置显式关闭，后两者由入口自动省略。
- total loss 只由 LightningModule 以 `sync_dist=True` 记录一次，分项 logger 不重复记录。
