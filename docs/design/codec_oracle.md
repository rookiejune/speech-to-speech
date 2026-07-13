# codec oracle

为 codec screening 实验提供可测试的 oracle model 和 prepared-code 数据路径。Hydra
配置、codec 资源加载、logger 与 Trainer 组装仍由 `scripts/codec_oracle.py` 负责。

## 对外能力

- `Objective` / `Initialization`：使用字符串枚举固定 objective 与 embedding initialization
  模式，并由枚举自身处理对应的 code selection 与初始化规则。
- `FlowOracle`：以 semantic codebook embedding 为条件，训练 acoustic feature flow。
- `TokenOracle`：对 unified codec token 做 causal token prediction；
  `teacher_forced_ids()` 明确返回使用真实前缀的逐位置 argmax，不表达自回归 sampling。
- `DataModule`：加载 prepared codec view，支持显式 distributed sampler 和 LBA。
- `codes()` / `collate()` / `single_batch_loader()`：单样本裁剪、变长 padding 与
  single-batch overfit 数据入口。
- `embedding_weight()` / `feature_stats()`：codec/random initialization 和 flow target
  normalization statistics。
- `Logger`：对固定 codes 聚合 loss/probe metric；flow 记录 sampled waveform，token objective
  记录 teacher-forced prediction waveform。
- `WorldSizeContract` / `SamplerEpochSetter`：分别校验 world size 和推进显式 sampler epoch。
- `event()` / `timed()`：输出 codec oracle 专用的结构化阶段日志。

## 边界

- 本模块表达 codec oracle 的模型、数据和专用 callback，不选择实验配置或输出目录。
- `scripts/codec_oracle.py` 是真实运行入口，不重复实现 model、collate 或 DataModule。
- codec 对象仍由入口按顶层 codec 配置构造；oracle model 只接收需要的 codebook、
  dequantize callable 和 flow runtime。
- objective 与 initialization 的字符串只在配置边界转换一次，内部使用枚举，不重复维护
  字符串分派。
- 中间 metric、waveform 和 checkpoint 写入实验 output directory，不写入项目 repo。
