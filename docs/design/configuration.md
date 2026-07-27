# configuration

Hydra 配置优先复用 `src` 的公开 Config，而不是在入口脚本中维护平行结构。目录只为真正可替换的
模块 preset 和运行编排建层级；入口自身的生产默认写在 root config，完整链路测试的组合与预算
写在 `experiment` 中。

## 源码模块

- `runtime`：完整映射 `runtime.Config`，统一拥有 codec、audio representation、backbone、
  audio tokenizer、device、dtype、attention backend 与 flow sampling。`longcat`、`longcat_native`、
  `longcat_full_sequence`、`unicodec` 表示相互兼容的资源 snapshot；不再拆分 `codec` 和
  `sampler` 组。
- `model`：完整映射 `model.Config` 的三个 adapter 与可选 `ToyConfig`。`model=toy` 只替换
  backbone；`model/acoustic` 选择 flow/RVQ composition，preset package 仍是顶层 `acoustic`，
  避免把 subtype 字段混入基础 `model.Config`。
- `data`：overfit 数据源 preset；`data=toy` 使用 `DatasetConfig` 选择内存 codec samples，
  production/fixed-sample experiment 默认仍使用 WMT19 TTS prepared data。需要固定正式
  train/dev/test 子集时，通过 `DatasetConfig.split_manifest` 和 `split_label` 显式选择
  manifest 中的索引集合。
- `scripts/create_split_manifest.py`：把 distribution candidate 与 formal-root audit 转成
  可追溯的 split manifest。它要求候选索引覆盖 audited `samples.parquet` 的全部行，绑定
  audit 文件 fingerprint，并把 split method 作为显式参数；它不是训练入口，也不会修改原始
  parquet 或 payload。
- `pl_module`：完整映射 `pl_module.Config` 的 learning rate 与 weight decay；不再使用含义重复的
  `optimizer` 组。

Acoustic-only codec screening 已迁入 `semantic-acoustic-codec`。本仓库不再维护对应的 Hydra root、
job wrapper 或 config schema；历史实验记录仍保存在 `docs/experiments/` 中。

`trainer`、`logging`、`callback` 与 `experiment` 属于 Lightning/Hydra 运行编排，可以没有同名
`src` 包。overfit 的 sample index 和 train budget 位于 experiment；数据源通过公开
`DatasetConfig` 选择。overfit 的 `callbacks.evaluation.enabled` 控制声学生成评估；真实
fixed-sample experiment 默认启用，随机输出不构成质量结论的 `toy_smoke` 显式关闭。共享
`callback/performance` preset 只暴露开关、
硬件峰值 override、记录 cadence、warmup、窗口、CUDA 同步和分布式起点对齐；训练 dtype 与 FLOPs
口径由实际入口和 provider 决定，不作为可脱离模型配置的 Hydra 字段。
speech-to-speech 自有日志 callback 默认仍按 `every_n_steps` 触发；需要让 expensive callback 跟随
实际处理音频量时，配置对应的 `every_audio_seconds`，入口会用 `ModelBatch.audio_seconds` 统计 DDP
全局 processed audio seconds。该字段只适用于消费 `ModelBatch` 的项目 callback，不进入 anytrain
的 task-agnostic callback 契约。

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
该模式不组装 `GradLogger` 或重复计算全局 norm 的 `GradNormLogger`，因为这些额外计算会进入实测
step time，却不属于 provider 统计的训练 FLOPs。DDP 默认在下一 batch timer 启动前执行 barrier，
避免仅 rank zero 执行的 batch-end 诊断使各 rank 起点错位。

训练输出由 `repo_output_root`、相对的 `output_subdir` 和派生的 `output_dir` 组成。checkpoint、音频、
Hydra metadata 与 `metrics.json` 写入 `output_dir`；TensorBoard/CSV logger 的路径由 logging preset
统一计算。TensorBoard 运行目录为
`repo_output_root/tensorboard/output_subdir/version_*`，因此可以直接把整个项目的 TensorBoard 根
目录交给比较工具。`repo_output_root` 优先使用 `SPEECH_TO_SPEECH_TRAIN_ROOT`；未设置时由 workspace
解析出的 `DYNAMIC_HOME` 派生为 `$DYNAMIC_HOME/train/speech-to-speech`。job wrapper source
`workspace/jobs/env.sh` 后使用同一默认值，缺少 `DYNAMIC_HOME` 时显式失败，不回退到项目目录。
`output_subdir` 不允许绝对路径或 `..`，`output_dir` 也不允许独立 override。

两个 trainer preset 都使用 `devices: auto`，由 Lightning 使用 `CUDA_VISIBLE_DEVICES` 中的全部
可见设备；设备数量不再作为运行时配置契约重复校验。job wrapper 只提供机器相关的默认可见设备，
提交时可显式覆盖。共享 `trainer=ddp` 使用 Lightning 默认 distributed sampler。LongCat prepared
map-style dataset 通过 `MapStyleABC.dataloader()` 暴露 deterministic shuffle 与 batch planning；
UniCodec DDP smoke 则要求每个 rank
重复读取同一个固定样本，因此仅该 experiment 显式设置 `use_distributed_sampler: false`。

完整链路实验分别负责其 composition、数据范围、trainer、callback 和 step budget：

- `unicodec_overfit`：UniCodec fixed-sample 100-step overfit。
- `unicodec_ddp_smoke`：UniCodec 显式 DDP 两步验收。
- `overfit`：TTS/S2ST fixed-sample 完整链路实验。
- `011_qwen_rvq_native_p0_fixed_sample`：真实 Qwen、LongCat native token 与 RVQ decoder 的
  P0 TTS/S2ST 2-step fixed-sample 合同验收；该 experiment 只固化当前 P0 子项，不替代 011
  的正式 staged joint entry。
- `014_stage1_pilot_validation_smoke`：两卡 stage 1 的 1-step pilot 验收；fit 前遍历完整 dev split，
  step 1 再运行 interval validation，用于验证 token/RVQ CE 与各 codebook top-1 的真实 DDP 口径。
- `014_stage1_pilot_canary`：同一 1k split 的 100-step 两卡 canary；step 50/100 运行完整 dev，
  同步归档 checkpoint 并保留 `last.ckpt`，用于决定是否继续到计划中的 5k-step pilot。
- `014_stage1_pilot_resume_500`：从显式 `train.ckpt_path` 恢复 100-step canary，到 step 500
  为止每 100 steps 运行完整 dev 与归档 checkpoint；用于在扩大到 5k 前验证 RVQ CE 门槛。
- `014_stage1_pilot_resume_2000`：step 500 未达 5% RVQ CE 门槛后，从其 `last.ckpt` 恢复到
  step 2000，每 250 steps 运行完整 dev 与归档 checkpoint；作为是否继续到 5k 的中间 gate。
- `train`：正式 staged joint training root。它直接消费 `configs/stage/stage_*.yaml` 中的
  loader/task/freeze 契约，构造唯一 `DataModule`；每个 speech loader 使用
  `LoaderSpec.speech(...)`，纯文本 MT loader 使用 `LoaderSpec.text(...)`，多 loader 调度由
  `LoaderSchedule` 持有。
- `toy_smoke`：正式 LongCat runtime 加 tiny model/in-memory dataset 的 CPU 两步训练契约测试；
  不读取真实 backbone 权重或 WMT19 prepared dataset，也不替代真实资源验收。

`jobs/002` 与 `jobs/005/01-07` 都显式传递对应的 `experiment=`；002 job 另行选择 TTS/S2ST task，
training job 传递 `repo_output_root`、相对 `output_subdir` 和 `"$@"` 参数。测试预算因此由 experiment
单点维护，调用 smoke wrapper 时无需再传 `train.max_steps=2`。

`jobs/011/01_rvq_native_p0_fixed_sample.sh` 复用 `scripts/overfit.py` 作为唯一 Python 入口，
并行启动 TTS 与 S2ST 两个单卡 fixed-sample 子任务，分别写入 launcher log、pid 和 exit status。
真实 Qwen snapshot、prepared data root、输出根和 GPU 选择通过环境变量覆盖，避免把复旦机器的
临时 `/tmp` 路径写死进 Hydra preset。
`jobs/011/02_rvq_native_stage_smoke.sh` 仍使用 `scripts/overfit.py` 验证每个 stage 的 freeze
配置能完成 fixed-sample 两步训练。`jobs/011/03_staged_joint_train.sh` 是正式 staged joint
training wrapper，调用 `scripts/train.py`，默认 `trainer=static_ddp`，并可用
`SPEECH_TO_SPEECH_STAGE=stage_1..stage_4` 选择阶段。正式 train 入口通过
`train.ckpt_path=<checkpoint>` 显式恢复 Lightning checkpoint；默认值为空，普通训练不走 resume。
该字段只属于 staged train，overfit 配置不接受它；命名为 resume 的 experiment 必须把
`train.ckpt_path` 标为必填，避免绕过 wrapper 时静默从头训练。
`jobs/014/01_stage1_pilot_validation_smoke.sh` 复用同一 Python 入口；pilot data root 与 split manifest
分别由 `SPEECH_TO_SPEECH_STAGE_DATA_ROOT`、`SPEECH_TO_SPEECH_STAGE_SPLIT_MANIFEST` 显式提供。
`jobs/014/02_stage1_pilot_canary.sh` 使用相同资源入口运行 100-step canary，不从 smoke 输出恢复。
`jobs/014/03_stage1_pilot_resume_500.sh` 与 `04_stage1_pilot_resume_2000.sh` 额外要求
`SPEECH_TO_SPEECH_STAGE_CKPT_PATH`，避免隐式猜测 latest checkpoint。

正式 train 的 `validation` 默认关闭。启用时，`loader` 必须选择当前 stage 的一个 speech loader，
且 `data.dataset.split_manifest` 必须存在、`split_label` 必须与训练 split 不同。入口复制该 loader
的 task weights 与 speech data config，仅替换 dev `split_label`；`every_n_steps` 直接作为
optimizer-step `val_check_interval`，`sanity_steps=-1` 表示 fit 前遍历完整 dev split，非负值表示
对应 sanity batch 数。为了让 step interval 不受 epoch 边界控制，入口同时设置
`check_val_every_n_epoch=None`。每次 sanity/interval 结果按 step 记录到 `metrics.json.validation`。

## 入口边界

`scripts/_config.py` 只定义入口专属结构，例如 task、Trainer、logging、callback 与 flow/RVQ
acoustic config；`speech_to_speech.pl_module.composition` 负责 token/flow/RVQ 的
model/objective/module 组装；`scripts/_entry.py` 只放 overfit/train 共享的
runtime device、Trainer、performance callback 与 acoustic composition 边界校验。
`runtime.Config`、`model.Config`、`pl_module.Config` 和 `model.DecoderConfig` 直接进入 root
schema，不重复声明字段。OmegaConf 对字符串枚举只接受成员
名，入口在合并前把公开的小写 value 转成 enum member name；除此之外不做兼容重写。

两个入口分别解析为：

- `OverfitTokenConfig | OverfitFlowConfig | OverfitRVQConfig`
- `StagedTrainTokenConfig | StagedTrainFlowConfig | StagedTrainRVQConfig`

未知字段和错误 composition 在进入执行逻辑前失败，解析后的 dataclass 不再向 `src` 传递
`DictConfig`。

## 组合

- `model/acoustic=none|flow|rvq` 显式选择下游 acoustic path；`none` 只训练
  audio token，flow/RVQ 才启用 acoustic objective，RVQ schema 不接受 REPA。
- `runtime.audio_representation=full_codec_sequence` 只允许 `model/acoustic=none`，因为完整
  codec codes 已作为 token objective 训练，不能再同时构造 acoustic side channel。
- `runtime.semantic_codec_artifact` 为 `semantic-acoustic-codec` 的 semantic-only waveform
  support artifact；当前 S2S 只接入 LongCat structured backend，因此只允许
  `runtime=longcat|longcat_native model/acoustic=none`，并拒绝 full codec sequence、UniCodec、Flow
  和 RVQ composition。FrameCodec 的 token-only 路径使用 full-code sequence；artifact 路径只生成
  structured backend 的 semantic units。LongCat decoupled `model/acoustic=none` 必须配置该
  artifact；否则应改用 `runtime=longcat_full_sequence`。已有 `DECOUPLED + Flow/RVQ` 组合仍表示
  S2S 内部的 acoustic feature 训练路径，不是 semantic-only artifact 路径。
- UniCodec 也是 `FrameCodec`，`runtime=unicodec model/acoustic=none` 使用
  `full_codec_sequence`，只是完整 frame 里只有一个 codebook。有独立 acoustic codebook 的 codec
  只有在提供 semantic-only artifact 或选择 full-code sequence 时才可以作为 token-only baseline。
- flow method、NFE 和 step 数直接覆盖 `runtime.flow_*`；RVQ/token 中保留这些字段是
  `runtime.Config` 的稳定 shape，不需要再为未使用字段创建 variant schema。
- flow/RVQ 必须有独立 acoustic codebook；`none` 不要求 codec 缺少 acoustic codebook，入口不自动改写 composition。

## Stage 与参数策略

`configs/stage/stage_*.yaml` 只描述 loader/task schedule 与 `batches_per_step`。参数冻结和
backbone top-fraction 抽象为顶层 `parameter_policy` 组，入口解析为
`ParameterPolicyConfig`，并在 Trainer/optimizer 创建前一次性应用。一个正式 job 只选择一个
stage 和一个 parameter policy，运行中不切换数据计划或参数冻结。experiment 显式保留约定组合：
stage 0/4 使用 `full`，stage 1/2 使用 `speech_interface`，stage 3 使用
`speech_interface_top_third`；这不是 `StageName` 的隐式映射。需要只训练 semantic token
interface 时显式选择 `parameter_policy=semantic_only`。
