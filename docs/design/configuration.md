# configuration

Hydra 配置优先复用 `src` 的公开 Config，而不是在入口脚本中维护平行结构。目录只为真正可替换的
模块 preset 和运行编排建层级。root config 只选择入口基线和可选 experiment；入口自身的生产默认
写在 `entry`，完整链路测试的组合与预算写在 `experiment` 中。

## 源码模块

- `runtime`：完整映射 `runtime.Config`，统一拥有 codec、backbone、backbone initialization、
  audio tokenizer、device、dtype、attention backend 与 flow sampling。
  root-level `audio_sequence_layout=flattened|semantic` 是公开的音频序列轴：`flattened` 把完整 codec
  codes 作为 acoustic-first / semantic-last 的 token 序列训练；`semantic` 的逻辑输入输出仍是
  full codes，但 token sequence 只处理 semantic，acoustic 由 Flow/RVQ side module 或
  `semantic-acoustic-codec` artifact 补齐。`longcat`、`longcat_native`、`bicodec`、
  `unicodec` 表示相互兼容的资源 snapshot；不再拆分 `codec` 和 `sampler` 组。
- `model`：完整映射 `model.Config` 的 semantic adapter、可选 source-audio input adapter、
  `ToyConfig`、`AudioOutputAdapterConfig` 与 `peft.LoraConfig | None`。
  `model=toy` 只替换 backbone；`+model/lora@model.lora=qwen` 直接组合 Hugging Face PEFT 的官方字段并向现有
  backbone 注入 adapter，不维护项目内 LoRA config 或 layer；`model/acoustic` 选择 flow/RVQ composition，preset
  package 写入 `model.acoustic`，避免把 subtype 字段混入基础 `model.Config`。
- `datamodule`：speech 数据源 preset，通过
  `datamodule/dataset=<name>` 组合到统一的
  `SpeechConfig.dataset`；`datamodule/dataset=toy` 选择内存 codec samples，
  production/fixed-sample experiment 默认仍使用 WMT19 TTS prepared data。需要固定正式
  train/dev/test 子集时，通过 `DatasetConfig.split_manifest` 和 `split_label` 显式选择
  manifest 中的索引集合。task template 也在 `datamodule` 上：
  `datamodule.tasks.<task>.template`（`int` 固定下标，`null` 随机；默认 `0`）。
- `callback/parameter_policy`：写入 `callbacks.parameter_policy`，声明可训练参数组、
  冻结参数组和 `backbone_top_fraction`。这是通用训练 callback 能力，由 experiment 显式选择。
  正式 train 的 loader mix 不再是独立 Hydra group，而是由 train experiment 内联的
  `loader_plan` 持有。
- `scripts/create_split_manifest.py`：把 distribution candidate 与 formal-root audit 转成
  可追溯的 split manifest。它要求候选索引覆盖 audited `samples.parquet` 的全部行，绑定
  audit 文件 fingerprint，并把 split method 作为显式参数；它不是训练入口，也不会修改原始
  parquet 或 payload。
- `optim`：完整映射训练 optimizer 策略，包括 learning rate、weight decay、anytrain LLM
  optimizer 选择（`adamw` / `muon`）和 unit-based LR schedule。默认
  `optim.name=adamw`；Muon 对比时设 `optim.name=muon`。有 PEFT LoRA 时，anytrain 会把
  `muon` 自动路由到 LoRA-Muon（adapter）+ AdamW（其余可训参数）。
- `pl_module`：只保留 Lightning module 自身行为配置，例如 validation generation token budget；
  不再持有 optimizer 或 LR schedule knobs。

Acoustic-only codec screening 已迁入 `semantic-acoustic-codec`。本仓库不再维护对应的 Hydra root、
job wrapper 或 config schema；历史实验记录仍保存在 `docs/experiments/` 中。

`entry`、`trainer`、`logging`、`callback` 与 `experiment` 属于 Lightning/Hydra 运行编排，可以没有同名
`src` 包。overfit 的 sample index 和 train budget 位于 experiment；`sample_index` 是入口字段，
不再混入 `SpeechConfig`，数据源则通过 `datamodule.dataset: DatasetConfig` 选择。overfit 的
`callbacks.evaluation.enabled` 控制声学生成评估；真实
fixed-sample experiment 默认启用，随机输出不构成质量结论的 `toy_smoke` 显式关闭。共享
`callback/performance` preset 只暴露开关、
硬件峰值 override、记录 cadence、warmup、窗口、CUDA 同步和分布式起点对齐；训练 dtype 与 FLOPs
口径由实际入口和 provider 决定，不作为可脱离模型配置的 Hydra 字段。
speech-to-speech 自有日志 callback 按 `every_n_steps`（optimizer step）触发。正式 staged train
默认启用 `callbacks.text_retention`，用固定 T2TT
probe 在 fit 开始建立 reference-NLL baseline，并持续记录 greedy generation 与 NLL delta；启用时
至少配置一条非空 instruction/reference，cadence 和 generation budget 在入口解析时校验。
overfit 的可选 `text_retention`、`gradient_probe` 与 `flow_matching` 诊断同样由
`configs/overfit.yaml` 显式持有开关、cadence 和模型参数 probe；`OOMDiagnostics`、`OutputsLogger` 与
`LossSummary` 是两个训练入口固定拥有的结构 callback，不伪装成可关闭的实验选项。

`model.audio_input_adapter` 默认 `type=mlp`，让 source audio payload 先经过可学习的
pointwise 对齐层再 overlay 到 backbone embedding space。可显式选择 `none` 作为 ablation/smoke
baseline，或选择 `transformer` 并配置 `layers`、`heads`、`ffn_ratio` 与 `dropout`。启用的
input adapter 只作用于 `audio_input_positions` 标记的 source audio payload；它与
`semantic_audio_adapter`、`audio_output_adapter` 独立，不处理 target/generated audio，也不替换
Flow/RVQ acoustic decoder。启用 transformer 时要求 backbone hidden size 能被 `heads` 整除。

`model.audio_output_adapter` 默认 `type=none`。`none` / `linear` / `mlp` 是无序列混合的
pointwise 特例；`transformer` 启用因果 self-attention，并携带独立 KV cache（与 backbone cache
同步 compact）。可配置 `layers`、`heads`、`ffn_ratio`、`dropout`（仅 transformer 使用）。该模块
同时服务 teacher forcing、候选 logits 和 autoregressive generation。

## 生产默认与完整链路测试

Acoustic-only codec screening 的生产默认、smoke budget 和 MFU provider 由
`semantic-acoustic-codec` 维护。本仓库的生产默认只覆盖 joint token/Flow/RVQ 训练入口。

联合 token/Flow/RVQ 训练的 overfit root config 保持 `callbacks.performance.enabled: false`，避免短
fixed-sample 验收默认承担性能测试。显式启用时必须同时关闭 task sample logging，例如
`callbacks.performance.enabled=true callbacks.task_sample.enabled=false`；两者同时启用会在入口边界明确
失败。`TaskSampleLogger` 在 `on_train_batch_start` 只由 rank zero 执行 generation，DDP 其他 rank 会在
后续同步点等待，不能靠调整 callback 顺序可靠排除这段时间。

满足该前提后，入口使用 `speech_to_speech.performance.TrainingFlops` 组装
`anytrain.PerformanceCallback`，并沿用同一套硬件峰值 override、cadence、warmup、窗口和 CUDA 同步
配置。performance callback 位于 callback 列表首位，使其 step timer 在后续 batch-end 诊断前结束；
该模式不组装 `GradLogger`，因为 comparison probes 的额外 `autograd.grad` 会进入实测
step time，却不属于 provider 统计的训练 FLOPs。DDP 默认在下一 batch timer 启动前执行 barrier，
避免仅 rank zero 执行的 batch-end 诊断使各 rank 起点错位。

启用 `model.audio_input_adapter` 时，`TrainingFlops` 会把 source tower 的 dense 投影和 transformer
attention 纳入 forward 估算；由于标准 tower 在 padded `[batch, frames]` 张量上执行计算，统计按完整
frame 宽度计数，mask 不会把 padding 误报为零成本。

训练输出由 `repo_output_root`、相对的 `output_subdir` 和派生的 `output_dir` 组成。checkpoint、音频、
Hydra metadata 与 `metrics.json` 写入 `output_dir`；TensorBoard/CSV logger 的路径由 logging preset
统一计算。TensorBoard 运行目录为
`repo_output_root/tensorboard/output_subdir/version_*`，因此可以直接把整个项目的 TensorBoard 根
目录交给比较工具。`repo_output_root` 优先使用 `SPEECH_TO_SPEECH_TRAIN_ROOT`；未设置时由 workspace
解析出的 `DYNAMIC_HOME` 派生为 `$DYNAMIC_HOME/train/speech-to-speech`。job wrapper 统一 source
项目级 `jobs/env.sh`；该入口先组合 `workspace/jobs/env.sh`，再提供同一默认值，缺少
`DYNAMIC_HOME` 时显式失败，不回退到项目目录。workspace 环境不设置 `CUDA_VISIBLE_DEVICES`，GPU
默认值由具体 wrapper 持有，提交时的显式环境变量优先。
`output_subdir` 不允许绝对路径或 `..`，`output_dir` 也不允许独立 override。
正式 staged train 使用 `anytrain.lightning.ModelCheckpoint` 的默认异步落盘，把本机临时保存与目标
目录复制串行解耦；本项目的 checkpoint 配置只决定目录、文件名、归档 cadence 与保留策略。

五个 trainer preset 都使用 `devices: auto`，由 Lightning 使用 `CUDA_VISIBLE_DEVICES` 中的全部
可见设备；设备数量不再作为运行时配置契约重复校验。job wrapper 只提供机器相关的默认可见设备，
提交时可显式覆盖。`default`、`ddp`、`static_ddp` 保留通用 sampler 行为；正式 staged train
通过 `trainer=staged_static_ddp` 选择 static DDP 并明确关闭 distributed sampler；
`trainer=staged_ddp` 保留为 dynamic unused-parameter fallback。LongCat prepared map-style dataset 通过
`MapStyleABC.dataloader()` 暴露 deterministic shuffle 与 batch planning；UniCodec DDP smoke
同样要求每个 rank 重复读取同一个固定样本，因此其 experiment 也显式设置
`use_distributed_sampler: false`。
正式 staged train 的 DDP 策略由 `loader_plan.step_mode` 与 task coverage 共同决定。`fused_joint`
构造一次 backward 覆盖所有非零 loader/task branch；覆盖全部可训练参数时使用
`trainer=staged_static_ddp`，仍有未使用 adapter/head 时使用 `trainer=staged_ddp`。`serial_joint` 使用
`trainer=staged_ddp`，要求
`ddp_find_unused_parameters_true`，并约束 `loader_plan.accumulate_grad_batches == 非零 loader 数量`。
入口拒绝 serial mode 与 DDP unused-parameter 策略不一致的组合。`weighted_window` 中
`loader_plan.loaders.<name>.weight` 是 sampling weight；两个 joint mode 中每个 loader 固定执行一次，
同一字段改为归一化 task loss weight。

完整链路实验分别负责其 composition、数据范围、trainer、callback 和 step budget：

`configs/experiment/train/` 只放 `scripts/train.py` 消费的 staged experiments；仍位于
`configs/experiment/` 顶层的文件由 `scripts/overfit.py` 消费。这样两个入口不会共享一组含义不明的
flat experiment 名称。`configs/train.yaml` 组合 `entry=train`、默认 LoRA preset（须排在
experiment 之前，以便非 LoRA experiment 用 `model.lora: null` 覆盖）和可选 experiment；可复用的
百万步预算、数据加载和 callback cadence 位于 `configs/entry/train.yaml`。

- `overfit/unicodec`：UniCodec fixed-sample 100-step overfit。
- `unicodec_ddp_smoke`：UniCodec 显式 DDP 两步验收。
- `overfit`：TTS/S2ST fixed-sample 完整链路实验。
- `train/staged_joint/stage_0..3`：正式 staged joint experiments。每个文件显式内联
  `loader_plan` 并选择 `callback/parameter_policy`；`stage_0..3` 只是 experiment 目录名，
  不再生成运行身份字段或独立 stage config。`loader_plan.loaders` /
  `loader_plan.step_mode` / `loader_plan.accumulate_grad_batches` 等持有 loader/task 契约，再构造唯一 `DataModule`。每个 speech
  loader 使用 `LoaderSpec.speech(...)`，纯文本 MT loader 使用 `LoaderSpec.text(...)`（读
  `datamodule.tasks`），多 loader 调度由 `LoaderSchedule` 持有。四个正式 stage 都启用每 10,000
  optimizer steps 的 train fixed panels，并都包含 MT panel；正式 entry 还默认启用独立的
  text-retention baseline。MT panel 只允许 train split，validation panel 仍要求 speech loader 与
  独立 validation dataset。
  UniCodec smoke 只验证独立的 full-code 兼容路径，不属于 staged joint 课程；SAC 预训练和 artifact
  导出由仓外的 `semantic-acoustic-codec` 负责。
  新课程输出统一写入 `staged-joint/s2st/stage_0..3/`，与其他训练路线的 checkpoint、
  `last.ckpt` 和 TensorBoard 目录隔离。
- `train/smoke/parameter_policy`：参数冻结专用两步实验。它固定 toy model/data 与 CPU trainer，
  只通过 `callback/parameter_policy=<name>` 切换冻结策略，不借用正式长跑配置充当策略测试夹具。
- `toy_smoke`：正式 LongCat runtime 加 tiny model/in-memory dataset 的 CPU 两步训练契约测试；
  不读取真实 backbone 权重或 WMT19 prepared dataset，也不替代真实资源验收。

`jobs/002`、`jobs/005/02_unicodec.sh` 与 `jobs/005/05_unicodec_ddp.sh` 都显式传递对应的
`experiment=`；002 job 另行选择 TTS/S2ST task。training job 传递 `repo_output_root`、相对
`output_subdir` 和 `"$@"` 参数，测试预算因此由 experiment 单点维护，调用 smoke wrapper 时无需
再传 `train.max_steps=2`。`jobs/004/01_s2st.sh` 是独立的 generation smoke，不属于训练入口。

`jobs/011/03_staged_joint_train.sh` 是正式 staged joint training wrapper，调用
`scripts/train.py`，根据 `SPEECH_TO_SPEECH_EXPERIMENT=train/staged_joint/stage_0..stage_3` 选择对应 experiment，
并通过 `SPEECH_TO_SPEECH_STEP_MODE=fused_joint|serial_joint` 成对选择
`trainer=staged_static_ddp` 或 `trainer=staged_ddp`；未设置时默认
`train/staged_joint/stage_0` + `serial_joint`。Stage 0 的 TTS+MT 不覆盖可训练的 speech-input
分支，因此即使显式选择 `fused_joint` 也使用 find-unused DDP；Stage 1-3 默认使用
`fused_joint` + static DDP。该 wrapper 在启动 Python 前拒绝通过末尾 `"$@"`
覆写 `experiment`、`task`、`loader_plan` 或 `model.acoustic.init_artifact`，需要切换 identity 时必须使用
对应环境选择器并单独提交一次 wrapper。正式 train 入口通过
`train.ckpt_path=<checkpoint>` 显式恢复 Lightning checkpoint；默认值为空，普通训练不走 resume。
该字段只用于同一 stage、相同模型拓扑和 optimizer 参数组的断点续训；它不承担跨 stage 权重迁移，
且只属于 staged train，overfit 配置不接受它。

SAC 预训练、codec 筛选和 generator artifact 导出属于仓库外的
`semantic-acoustic-codec`。本仓库的 staged wrapper 不启动这些流程，也不切换到其他 codec
训练入口。正式 Flow/RVQ stage 必须通过
`SPEECH_TO_SPEECH_SAC_ARTIFACT` 指定仓外产物；wrapper 将其传给
`model.acoustic.init_artifact`，composition 再在构造 S2S model 前校验 artifact 的 route、frame
layout、decoder 配置和 acoustic backend metadata。该变量缺失或为空时 wrapper 在启动 Python 前失败。

正式 train 的 `validation` 默认关闭。启用时，`loader` 必须选择当前 `loader_plan` 的一个 speech loader，
且 `datamodule.dataset.split_manifest` 必须存在、`split_label` 必须与训练 split 不同。入口复制该 loader
的 task weights 与 speech data config，仅替换 dev `split_label`；配置的 `every_n_steps` 使用
optimizer-step 语义；`fused_joint` 下入口传给 Lightning 的 batch 级 `val_check_interval` 等于该
step 数，`serial_joint` 下乘以 `loader_plan.accumulate_grad_batches`。`sanity_steps=-1` 表示 fit 前遍历完整 dev split，非负值表示对应 sanity batch
数。为了让 step interval 不受 epoch 边界控制，入口同时设置
`check_val_every_n_epoch=None`。每次 sanity/interval 结果按 step 记录到 `metrics.json.validation`。

## 入口边界

`scripts/_config_common.py` 定义两个入口共享的 acoustic、Trainer、logging、callback 基础结构与
组合校验；`scripts/_overfit_config.py` 和 `scripts/_train_config.py` 分别拥有入口 schema 与其业务规则。
`scripts/_config_normalization.py` 隔离 OmegaConf 可写化、枚举规范化与 structured dataclass 合并，
不包含训练业务规则。`speech_to_speech.pl_module.composition` 负责 token/flow/RVQ 的
model/objective/module 组装、基于 `model.acoustic.type` 的统一分发，以及 runtime acoustic side-channel
约束；`scripts/_entry.py` 只放 overfit/train 共享的 runtime device、Trainer 与 performance callback
组装。
`runtime.Config`、`model.Config`、`pl_module.Config`、`model.DecoderConfig`、
`datamodule.config.SpeechConfig`、`DataLoaderConfig` 和 `datamodule.dataset.text.TextConfig` 直接进入 root
schema，不重复声明字段；`scripts/overfit.py` 与 `scripts/train.py` 都直接把解析后的 datamodule config 交给 `LoaderSpec`，不做
同构对象转换。`loader_plan.loaders` 使用 `LoaderConfig`：把字符串 task weights 暴露为 `Task` 映射，并根据非零任务
维护 text-only 与 speech loader 不可混合的不变量；配置解析校验 validation/panel 选择，训练组装不
重复这些条件。OmegaConf 对字符串枚举只接受成员
名，入口在合并前把公开的小写 value 转成 enum member name；除此之外不做兼容重写。
`audio_sequence_layout` 是 root/runtime schema 的公开结构，入口解析后由同一 `Runtime` 派生内部
route，再传给 DataModule、model 和 generation。Hydra 的字符串表示在入口归一化，再由公开
dataclass 校验。reference 不是独立 config 轴；需要 prompt-owned acoustic 的路径从输入 full
codes/context 中取得 reference acoustic。

两个入口分别解析为：

- `OverfitTokenConfig | OverfitFlowConfig | OverfitRVQConfig`
- `StagedTrainTokenConfig | StagedTrainFlowConfig | StagedTrainRVQConfig`

未知字段和错误 composition 在进入执行逻辑前失败，解析后的 dataclass 不再向 `src` 传递
`DictConfig`。

## 组合

- `model/acoustic=none|flow|rvq` 显式选择下游 acoustic path；`none` 只训练
  audio token，flow/RVQ 才启用 acoustic objective，RVQ schema 不接受 REPA。
- `audio_sequence_layout=flattened` 切换到完整 codebook 序列化格式，只允许
  `model/acoustic=none`，因为完整 codec codes 已作为 token objective 训练，不能再同时构造
  frame-aligned acoustic side channel。flattened layout 的 token 顺序固定为 acoustic-first /
  semantic-last。
- `audio_sequence_layout=semantic` 保持逻辑输入输出为 full codes，但模型 token sequence
  只预测 semantic。需要 waveform 时，acoustic 由 Flow/RVQ side module 或
  `runtime.semantic_codec_artifact` 提供。
- reference 不是公开 config 轴。BiCodec 的 semantic layout 从输入 full codes/context 取得
  reference acoustic；没有 reference 的路径只能依赖输出 full codes 或后置 acoustic provider。
- 非 semantic 单元的存储始终是 `SemanticAcousticCodes.acoustic`；frame-aligned 还是 fixed-length
  由 codec 的 `AcousticLayout` 决定，不由 grammar/prompt 再造第三种 stream。
- `runtime.backbone_initialization=random` 从 `runtime.backbone` 读取 tokenizer 与完整 HF config，
  但不读取预训练权重；它不能与 `model=toy` 组合，并要求
  `callback/parameter_policy=full`。
- `runtime.backbone_trust_remote_code`、`runtime.backbone_chat_template`、
  `runtime.backbone_readout`、`runtime.backbone_readouts` 与
  `runtime.backbone_supports_cache_position` 是替换 HF backbone 的显式兼容性开关；默认值保持
  Qwen/Qwen3 contract。`configs/runtime/kimi_audio.yaml` 使用 Kimi-Audio 的 remote-code
  双分支 readout，并显式配置本项目的单流 Jinja instruction template；这不是 Kimi 官方双流
  PromptManager 格式，后者需要独立的结构化 prompt adapter。
- `runtime.semantic_codec_artifact` 为 `semantic-acoustic-codec` 的 semantic-only waveform
  support artifact；LongCat 的 `semantic` token-only 路径可使用它。BiCodec 的 `semantic`
  layout 从 full-code input/context 复用 reference acoustic，并调用 backend `detokenize()`；它不接入
  Flow/RVQ composition。两份 BiCodec smoke 都选择
  `datamodule/dataset=qwen_tts_speaker` 和 `datamodule.shape=single`；可用
  `datamodule.dataset.speaker=<id>` 限制到一个 speaker。FrameCodec 的 token-only full-code
  baseline 使用 `audio_sequence_layout=flattened`。
- `model.acoustic.init_artifact` 是 Flow/RVQ 联合训练的 generator 初始化路径，与
  `runtime.semantic_codec_artifact` 不同。composition 加载 artifact 后校验 frame-aligned layout、
  decoder/REPA 配置和 acoustic backend metadata，再把已加载对象交给 model；semantic conditioner 不进入
  S2S。Flow 迁移 decoder 与 feature normalization，RVQ 当前只接受 `codebook_ar` artifact。
- UniCodec 也是 `FrameCodec`，`runtime=unicodec model/acoustic=none` 使用
  `audio_sequence_layout=flattened`，只是完整 frame 里只有一个 codebook。有独立 acoustic codebook 的 codec
  只有在提供 semantic-only artifact 或选择 full-code sequence 时才可以作为 token-only baseline。
- flow method、NFE 和 step 数直接覆盖 `runtime.flow_*`；RVQ/token 中保留这些字段是
  `runtime.Config` 的稳定 shape，不需要再为未使用字段创建 variant schema。
- flow/RVQ 必须有独立 acoustic codebook；`none` 不要求 codec 缺少 acoustic codebook，入口不自动改写 composition。

sequence layout 不是服务请求参数，而是实验与 checkpoint 的不变量。`SpeechToSpeechModule` 保存规范化后的
layout/route metadata；恢复 checkpoint 时要求 metadata 存在且与当前 `Runtime` 派生的 route 严格相等，
缺失或不一致都直接失败。reference/context 数据只属于样本或请求本身，不作为可在恢复时切换的配置轴。

## Loader plan 与参数策略

loader mix 属于 train experiment 内联的 `loader_plan`，由
`loader_plan.loaders`、`loader_plan.step_mode` 和
`loader_plan.accumulate_grad_batches` 描述；不再提供独立 `configs/stage/` Hydra group。
参数冻结和 backbone top-fraction 位于
`callbacks.parameter_policy`，由 `callback/parameter_policy` 组写入，并在 Trainer/optimizer
创建前一次性应用。一个正式 job 只选择一个 experiment，运行中不切换数据计划或参数冻结。
`staged_joint/stage_0..3` 只是人工训练里程碑目录名；它不生成运行身份字段，也不隐式选择
policy。约定组合：stage 0 使用 `lora`，stage 1 使用 `speech_interface`，stage 2 使用
`speech_interface_top_third`，stage 3 使用 `full`。这些非 LoRA 对照
experiment 会在配置体中显式写 `model.lora: null`，以覆盖正式 train 的默认 LoRA。需要只训练
semantic token interface 时在专用 experiment 中显式选择
`callback/parameter_policy=semantic_only`。

四个 stage 的 joint loss 初始权重为：Stage 0 使用 TTS `0.9`、MT `0.1`；Stage 1 使用
ASR/TTS 各 `0.45`、MT `0.1`；Stage 2 使用 ASR/S2TT/TTS/T2ST 各 `0.225`、MT `0.1`；
Stage 3 使用 S2ST `0.7`、ASR/S2TT/TTS/T2ST 各 `0.05`、MT `0.1`。Stage 1-3 默认
`fused_joint`，每个 loader 每个 optimizer step 各执行一次，`weight` 因而是归一化 loss
系数而非 sampling probability；Stage 0 的 `serial_joint` 也按相同 loss 权重契约缩放两个
microbatch。Stage 3 以直接 S2ST 为模型选择主目标，其余任务用于能力保持和分解诊断。

正式 train 入口默认选择 LoRA preset 与
`callback/parameter_policy=lora`（PiSSA 初始化）。
PEFT LoRA 必须同时选择 `+model/lora@model.lora=qwen` 与
`callback/parameter_policy=lora`，避免注入 adapter 后又选择
其它训练策略，或选择 policy 却没有 adapter。OmegaConf 2.3 不能把 PEFT 的复杂 union 字段直接作为
nested structured config 展开，因此 normalization 边界先用官方字段构造 `peft.LoraConfig`，再把
该对象写回公开 `model.Config`；它不复制字段或二次校验。PEFT 负责 backbone 内参数的冻结语义，
该 policy 再训练现有 speech/acoustic interface；当前 performance FLOPs provider 不支持 LoRA，入口
拒绝同时启用 performance callback。`optim.name=muon` 与 LoRA 组合时，要求
`init_lora_weights` 为 PiSSA 系满秩初始化，否则入口早失败。

overfit 入口继续默认 `callbacks.parameter_policy.name=full` 且不启用 LoRA，专门验收全参闭环。

`staged_joint/stage_0..3` 是 S2S 内部的数据/任务/参数策略里程碑命名，
不等同于“先在 SAC 预训练 generator、再在 S2S 用 hidden state 联合训练”的两个 phase。SAC
在仓库外预训练并导出 artifact；本仓库的初始化只在 model composition 时发生一次，loader plan
不拥有 artifact 训练、导出或切换逻辑。
