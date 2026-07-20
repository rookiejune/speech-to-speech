# codec oracle

为 codec screening 实验提供可测试的 oracle model 和 prepared-code 数据路径。Hydra
配置、codec 资源加载、logger 与 Trainer 组装仍由 `scripts/codec_oracle.py` 负责。

## 运行配置

裸 `scripts/codec_oracle.py` 是生产训练入口：默认遍历完整 prepared dataset、启用 LBA，使用
`bf16-mixed` 训练 1,000,000 steps，并每 10,000 steps 记录 sample 和归档 checkpoint。完整链路
短验收不修改这些默认值，而由
`experiment=acoustic_oracle_smoke` 和 `experiment=acoustic_oracle_ddp_lba_smoke` 分别提供单卡与
两卡 smoke 的数据上限、trainer、callback 和两步预算；`jobs/005/01_longcat.sh` 与
`jobs/005/04_longcat_ddp_lba.sh` 显式选择对应 experiment。

## 对外能力

- `Objective` / `Initialization`：使用字符串枚举固定 flow objective 与 embedding initialization
  模式；当前 oracle 入口只支持 `Objective.FLOW`，codec/random 权重由 `Initialization` 自身构造。
- `AcousticFlowScreening`：持有完整的 `SpeechToSpeechFlowModel`，prepared semantic codes 通过正式
  global ID、semantic audio embedding/adapter、target latent 与 acoustic flow 路径；wrapper
  只负责 acoustic-only objective、optimizer selection 与实验日志。构造时冻结完整正式模型，
  只解冻 optimizer 持有的 semantic audio embedding、adapter 与 acoustic flow，使静态路径可用
  `find_unused_parameters=False` 的 DDP。
- unified-token codec 不使用独立 screening model；其 codes 作为 semantic audio tokens，
  `acoustic_codes=None`，复用正式 DataModule、token model、objective 与 generation/decode。
- `DataModule`：仅供 LongCat acoustic-only screening 加载 prepared codec view，支持显式
  distributed sampler 和 LBA；它接收严格 `DataConfig` / `LBAConfig`，不依赖 OmegaConf。
  unified-token codec 不使用该入口。
- `codes()` / `collate()` / `single_batch_loader()`：单样本裁剪、变长 padding 与
  single-batch overfit 数据入口。
- `Config`：统一拥有 initialization、flow target normalization、decoder、optimizer 参数和
  `DataConfig`；Hydra `codec_oracle` preset 直接映射该公开契约。
- `Initialization` 自身负责 codec/random embedding initialization；flow target normalization
  statistics 仍由运行入口从 probe features 计算。
- `matched_random_weight(reference, seed, rows=None)`：在 reference device/dtype 上按其总体
  mean/std 生成可复现的 matched-normal 权重；默认保持 shape，`rows` 只覆盖首维。公开调用方
  提供二维 floating reference，`Initialization.RANDOM.weight()` 复用该函数。
- `Logger`：对固定 codes 聚合 flow loss 与 sample feature MSE，记录 oracle reconstruction 和
  sampled waveform，并在训练结束写 `metrics.json`。
- `WorldSizeContract` / `SamplerEpochSetter`：分别校验 world size 和推进显式 sampler epoch。
- `event()` / `timed()`：输出 codec oracle 专用的结构化阶段日志。

## 边界

- 本模块表达 codec oracle 的模型、数据和专用 callback，不选择完整链路测试 experiment 或输出
  目录。
- `scripts/codec_oracle.py` 是真实运行入口，不重复实现 model、collate 或 DataModule。
- 入口按顶层配置构造 `Runtime` 与完整 `SpeechToSpeechFlowModel`；`AcousticFlowScreening` 显式
  接收该 model、flow runtime、optimizer 参数与 target normalization statistics。codec 另行传给
  logger 执行 reconstruction/sample decode，不以零散 callable 复制进 wrapper 契约。
- prepared-code 裁剪和 LBA budget 使用 runtime codec 暴露的 frame rate；codec config
  只选择资源和 dataset view，不重复保存时间尺度。
- oracle runtime 不接受 audio tokenizer artifact；prepared semantic code 直接索引 codec 初始化的
  semantic embedding，CodecBPE token ID 与 raw codec code ID 不能混用。
- `runtime.codec` 同时选择资源和 dataset view；oracle objective 固定为 flow，不暴露无效的
  acoustic type。`codec_oracle` preset 提供 decoder、target normalization、optimizer 和 data；
  flow sampling 来自 `runtime.Config.flow_*`。
- oracle logger、grad norm、non-finite check 与 checkpoint 使用一个成套 callback preset；入口
  显式注入 codec、codes、metadata 和 output directory 等运行时依赖。
- initialization 的字符串只在配置边界转换一次；objective 由入口固定为 `Objective.FLOW`，内部
  不维护无效的字符串分派。
- 中间 metric、waveform 和 checkpoint 写入实验 output directory，不写入项目 repo。
