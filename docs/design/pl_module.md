# pl_module 与 callback

Lightning 训练集成和日志边界。独立推理契约见 [generation](generation.md)。

## pl_module

`SpeechToSpeechModule[ModelT]` 是薄 Lightning wrapper：

- 构造时通过 `Objective[ModelT]` 保留 model/objective 类型配对。
- `training_step()` 每次只接收一个 homogeneous `ModelBatch`，调用一次 objective 并记录该
  microbatch 的 total loss；多 loader 比例由 dataloader 的 accumulation window 决定，Lightning
  在 `accumulate_grad_batches` 个 microbatch 后执行 optimizer step。module 不接收或归约联合 batch
  tuple，并把当前分项保留到对应 backward 完成。
- `validation_step()` 复用同一 materialize 路径，通过 `Objective.validation()` 做 teacher-forcing
  dev 评估，并把 loss 模块提供的 `evaluator.weighted.Metric` 交给
  `anytrain.lightning.validation.log()`；它不解析 objective 名、RVQ detail key 或有效单位。Lightning
  在 epoch 结束时同步各 rank 的加权和与计数，因此不同 batch/rank 的长度差异不会退化为 batch mean
  或 rank mean。
- 可选 `batch_materializer` 只处理显式 raw waveform fallback：当 datamodule pair 或 single path
  返回 `RawSpeechBatch` 时，materializer 在当前训练 device 上对 task 实际消费且缺少 codes 的
  audio item 调用 codec encode，并在 objective 前转成标准 `ModelBatch`。没有 materializer 时，
  `training_step()` 只接受当前已 materialize 的单个 `ModelBatch`。Lightning device transfer 通过
  `ModelBatch.to(device)` 显式重建 `ModelBatch`，保留 frozen structured audio context 的不可变
  契约；`RawSpeechBatch` 在 materialize 前整体留在 CPU，避免同一 fallback batch 的 prepared
  codes 与现场 codec 产物落在不同 device。
- `current_loss_outputs()`：只在当前 training step 的 backward 完成前返回仍连接计算图的
  `Outputs`，供 `GradLogger` 计算指定分项梯度；其他时机显式报错。
- `configure_optimizers()` 委托 anytrain optimizer preset。
- `generate()` / `evaluate_text()` 只负责切换 eval mode、调用 generation 包并恢复原 mode。

`pl_module.composition.build()` 根据 `acoustic.type` 统一分发，校验 runtime 是否提供所选
acoustic composition 需要的独立 side channel，并组装
`model + objective + SpeechToSpeechModule` 的 token/Flow/RVQ 组合；`token()`、`flow()`、`rvq()`
分别封闭具体构造；返回值同时携带实际 `AcousticType`，入口不再重复解析或校验 composition。
该模块通过窄 Protocol 消费 acoustic config，不反向依赖 scripts 入口 schema。当
`acoustic.init_artifact` 非空时，composition 负责加载 SAC
`AcousticGeneratorArtifact`，校验 route、frame layout 与 backend metadata，并把已加载对象传给 model；
model 构造器不接收路径或执行文件 I/O。

`pl_module` 不实现 task 状态机、decode、文本 NLL、对齐或 loss；包级 API 只导出
`Config` 与 `SpeechToSpeechModule`，composition 通过显式子模块导入。

## callback

`speech_to_speech.callback` 导出 OOM 诊断、on-device codec materializer 与 train-batch interval helper；
以下日志 callback 从 `speech_to_speech.callback.logging` 导入：

- `OOMDiagnostics`：正式 train 与 overfit 入口默认启用。它在 train/validation batch 开始时只缓存
  `ModelBatch` / `RawSpeechBatch` 的 JSON-safe shape、dtype、device、task 和 role 元数据，不保留
  batch 或 tensor 引用；在 backward、post-backward 和 optimizer 边界更新 phase。捕获
  `torch.OutOfMemoryError` 后，每个 rank 独立向 stderr 写入一行带 epoch、global step、rank、输入摘要和
  CUDA allocated/reserved/peak bytes 的 JSON，不执行 distributed collective、`empty_cache()` 或异常恢复，
  原异常类型和 traceback 继续传播。CUDA memory 统计本身失败时把统计错误写入同一报告，不替换原 OOM。
  performance 启用时仍保持 callback 列表第一位，OOM diagnostics 紧随其后并先于领域日志 callback。
  `TaskSampleLogger` 的 fixed-sample generation 会覆盖外层 train batch 摘要，记录实际 request 的逐行
  prompt/audio context shape、padding 后 shape、generation budget 和 cache 设置；`AcousticEvaluation` 使用
  自身固定 `ModelBatch` 的摘要；text retention 在实际 autoregressive generation 与 reference-NLL forward
  边界分别附加 token shape。batch 正常结束后清空摘要，避免后续 fetch 或 transfer 错误误报上一批输入。
  Lightning 在 batch-start hook 前完成 device transfer，因此 transfer 自身的 OOM 不属于该 callback 的覆盖范围。
- `OutputsLogger`：只提供 S2S objective 到 task label 的适配，按 label 聚合 `LossItem` 的通用日志逻辑来自
  `anytrain.lightning.LossItemLoggerCallback`。
- `GradLogger` / `GradNormLogger`：只接入 S2S `TrainInterval`；指定分项或全局梯度范数的通用日志逻辑来自
  `anytrain.lightning`。
- `OnDeviceCodecMaterializer`：训练时 wav->codes 的显式 fallback。正式数据仍应提前 materialize
  codec codes；该 fallback 只把 pair/single 共用的 `RawSpeechBatch` 规范化为 `ModelBatch`，不改变
  objective 或 task loss contract。它把 waveform 转成 FP32，在关闭当前 device autocast 的上下文
  中执行 codec，避免 `bf16-mixed` Trainer 把 codec 预处理算子降精度；BiCodec structured view 与
  frame-code view 按数据表示分派，不按重叠 capability 猜测。
- `FlowMatchingLogger`：显式接收 flow runtime，不向下读取 model runtime；time histogram 和 bucketed
  loss 日志来自 `anytrain.lightning.LossTimeBucketLoggerCallback`。
- `LossSummary`：只注入 S2S objective 顺序；训练输出 total loss 与分项 `LossItem` 窗口摘要来自
  `anytrain.lightning.LossSummaryCallback`。
- `anytrain.lightning.validation.History`：正式训练入口启用 validation 时挂载该通用 callback；它在
  每次 validation 结束后收集 `val/*` 标量，区分 fit 前 sanity run 和 optimizer-step interval run，
  并把可恢复的 report 写入 `metrics.json`。本项目不重复实现 history state 或 scalar 校验。
- `AcousticEvaluation`：对 fixed-sample acoustic model 使用本地 generator seeds 采样，记录 feature、
  waveform 与 STFT 距离；纯评估函数位于 `generation.evaluation`，不留在脚本私有模块。
- `TaskSampleLogger`：只在 global zero 读取 datamodule 的公开 fixed-sample/diagnostic API，正式训练
  panel 显式绑定 `train|validation + loader + task + indices`，因此 mixed-task loader 不依赖训练
  collator 的 task allocation。dev panel 复用正式 validation 的独立数据源；通过 module 的
  `materialize_batch()` 获得标准 batch，并用 `ModelBatch.row()` 保持 raw sample、
  generation request 与 teacher-forcing reference 逐行对齐。pair 数据严格读取 source/target role，
  single 数据严格读取 `Role.DEFAULT`，不靠缺项 fallback 猜测形态。audio-source task 记录可播放的
  source waveform；所有 panel 按真实 task 记录 source/target/generated。TTS acoustic batch 还记录
  target waveform 与 teacher-forced `reference_generation`，并复用一次自回归 generation 的
  token、features 与 waveform。Stable Codec 的 full-code 路径没有 acoustic teacher-forcing，
  因此只记录 codec 重建的 target 与自回归 generated。callback 在隔离 RNG context 内应用固定 seed，
  使不同 step 的生成可比较。它还记录 generation 长度/截断、规范化字符错误率与 exact match，
  以及 waveform duration ratio、finite、RMS、peak、silence/clipping ratio；这些指标只使用已有
  text/waveform，不加载 ASR、MOS、speaker encoder 或其他评估模型，也不替代语义质量验收。每个
  callback 的 checkpoint state key 包含 split、loader、task、seed、indices 和 cadence，使 panel
  实例可独立恢复。纯 text MT loader 支持 train panels，并记录 source/target/generated text 与
  CER/exact-match；MT validation panel 当前被拒绝，因为 validation 数据源契约仍是 speech-only。
- `TextRetentionLogger`：正式 staged train 默认启用一条固定 T2TT probe；fit 开始时记录 greedy text
  generation、reference NLL 基线，后续按 `every_n_steps` 或 `every_audio_seconds` 记录 generation、
  NLL 与相对基线漂移；checkpoint resume 保留最初 baseline，并在新 fit 开始时直接记录相对该 baseline
  的当前漂移。probe 名、instruction、reference 与 generation budget 都来自严格配置；它只在 global
  zero 执行，不把 probe batch 混入训练数据或 loss。

上述 callback 需要 logger experiment 时统一通过 `anytrain.lightning.experiment` 获取 text、scalar、
audio 或 histogram 能力；本项目只负责 rank、cadence、tag 和领域数据转换，不再维护重复的 logger
Protocol/helper。

`TrainInterval` 是项目内 callback cadence 接口。`every_n_steps` 按 optimizer step 触发，同一个
accumulation window 内即使多个 microbatch 共享 `global_step` 也只能运行一次；
配置 `every_audio_seconds` 时改为按 `ModelBatch.audio_seconds` 累计全局 processed audio seconds
触发，DDP 下通过 strategy sum 聚合各 rank，gradient accumulation 下每个 train batch 都计入。
`anytrain` 不接收这个契约，因为它不拥有下游 batch schema 或 `AudioMeta.DURATION` 语义。

Task sample/evaluation callback 在隔离 RNG context 内运行，不改变后续训练的 CPU 或当前 CUDA
random state；固定 seed 只服务于同一样本跨 step 比较。

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
和全量训练。它校验 objective/model 配对及实际输出分支；PEFT LoRA、REPA、分阶段冻结、替换后的
模块或无法识别的结构会明确报错，不用不完整公式继续记录 MFU。

估算口径统计 Linear 与 attention matrix multiplication，并按 forward 的两倍估算 backward；lookup、
scatter、normalization、activation、loss 和冻结 codec feature extraction 不计入模型 FLOPs，但对应
耗时仍在实际 step time 中。

生产统计不从单个 `example_input_array` 推导固定 FLOPs；该字段只提供一个 forward 示例，供 summary、
tracing 或 graph logging 使用，也不直接使用 `lightning.fabric.utilities.throughput.measure_flops()`。
实际 batch 的有效序列/帧长度、padded shape、objective 分支和各 rank 的 dataloader 工作量都可能不同，
FlashAttention 或其他 custom op 也可能不在通用算子计数覆盖范围内；生产 provider 因此使用动态
解析计数，DDP 聚合与 step timing 由 anytrain 负责。Lightning 的 `measure_flops()` 只用于测试或
校准受支持的基础算子公式，不能替代该生产口径。

## 边界

- `SpeechToSpeechModule.generate()` / `evaluate_text()` 不持有跨调用 generation cache，也不修改
  request/result；通用 validation/batching 由 generation service 负责，audio decode 由其选择的
  capability strategy 负责。
- callback 只依赖 `Outputs`/`LossItem`、datamodule 与 pl_module 公共能力；`GradLogger` 额外要求
  LightningModule 实现 `current_loss_outputs()` 生命周期契约。
- `OutputsLogger` 按当前 `ModelBatch` 中 objective 的 `LossItem` 行粒度记录 task 指标。token loss
  覆盖当前 batch 的有效 token；RVQ、Flow 和 REPA 只覆盖带 acoustic target 的样本。若 loss 行数与
  该 objective 对应 task 行数不一致，callback 直接报错，不把不匹配的 task mask 静默套到 loss 上。
- `TrainingFlops` 负责解释 speech-to-speech 的模型、batch 与 objective；`PerformanceCallback` 只负责
  optimizer-step 聚合、计时、硬件峰值推断和 MFU 记录，不内置任务 batch schema。
- overfit performance composition 不包含 `TaskSampleLogger`、`GradLogger` 或 `GradNormLogger`；前者由
  配置显式关闭，后两者由入口自动省略。
- total loss 只由 LightningModule 以 `sync_dist=True` 记录一次，分项 logger 不重复记录。
- validation 是 teacher-forcing loss/accuracy 口径，不调用 autoregressive generation；真实生成质量
  仍由 generation callback 与独立结果文档验收。
- 正式 `scripts/train.py` 使用 `anytrain.lightning.ModelCheckpoint` 的默认异步保存；checkpoint
  目录、命名、保留数量和触发步数仍由本项目配置拥有。
- `SpeechToSpeechModule` 负责保存并校验固定 audio route 和 `peft-lora-v2` 配置 metadata。PEFT
  payload 来自完整 `LoraConfig.to_dict()` 与同版本官方默认值，不绑定 `peft_version`。共同字段
  严格比较；版本间新增或缺失字段只有保持官方默认值时才兼容。只有当前未启用 LoRA 时，才允许
  加载缺少 PEFT metadata 的旧 checkpoint。
