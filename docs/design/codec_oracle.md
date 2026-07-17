# codec oracle

为 codec screening 实验提供可测试的 oracle model 和 prepared-code 数据路径。Hydra
配置、codec 资源加载、logger 与 Trainer 组装仍由 `scripts/codec_oracle.py` 负责。

## 对外能力

- `Objective` / `Initialization`：使用字符串枚举固定 objective 与 embedding initialization
  模式，并由枚举自身处理对应的 code selection 与初始化规则。
- `AcousticFlowScreening`：持有完整的 `SpeechToSpeechFlowModel`，prepared semantic codes 通过正式
  global ID、semantic audio embedding/adapter、target latent 与 acoustic flow 路径；wrapper
  只负责 acoustic-only objective、optimizer selection 与实验日志。构造时冻结完整正式模型，
  只解冻 optimizer 持有的 semantic audio embedding、adapter 与 acoustic flow，使静态路径可用
  `find_unused_parameters=False` 的 DDP。
- unified-token codec 不使用独立 screening model；其 codes 作为 semantic audio tokens，
  `acoustic_codes=None`，复用正式 DataModule、token model、objective 与 generation/decode。
- `DataModule`：仅供 LongCat acoustic-only screening 加载 prepared codec view，支持显式
  distributed sampler 和 LBA；unified-token codec 不使用该入口。
- `codes()` / `collate()` / `single_batch_loader()`：单样本裁剪、变长 padding 与
  single-batch overfit 数据入口。
- `Initialization` 自身负责 codec/random embedding initialization；flow target normalization
  statistics 是运行入口的实验准备逻辑，不作为模块能力公开。
- `Logger`：对固定 codes 聚合 loss/probe metric；flow 记录 sampled waveform，token objective
  记录 teacher-forced prediction waveform。
- `WorldSizeContract` / `SamplerEpochSetter`：分别校验 world size 和推进显式 sampler epoch。
- `event()` / `timed()`：输出 codec oracle 专用的结构化阶段日志。

## 边界

- 本模块表达 codec oracle 的模型、数据和专用 callback，不选择实验配置或输出目录。
- `scripts/codec_oracle.py` 是真实运行入口，不重复实现 model、collate 或 DataModule。
- codec 对象仍由入口按顶层 codec 配置构造；oracle model 只接收需要的 codebook、
  dequantize callable 和 flow runtime。
- prepared-code 裁剪和 LBA budget 使用 runtime codec 暴露的 frame rate；codec config
  只选择资源和 dataset view，不重复保存时间尺度。
- `codec.name` 同时选择资源和 dataset view；`acoustic.type` 选择 objective 组合，flow preset 提供 decoder 结构与
  flow target normalization。optimizer、flow、trainer 和 logging 继续与正式训练共用。
- oracle logger、grad norm、non-finite check 与 checkpoint 分属独立 callback config；入口
  显式注入 codec、codes、metadata 和 output directory 等运行时依赖。
- objective 与 initialization 的字符串只在配置边界转换一次，内部使用枚举，不重复维护
  字符串分派。
- 中间 metric、waveform 和 checkpoint 写入实验 output directory，不写入项目 repo。
