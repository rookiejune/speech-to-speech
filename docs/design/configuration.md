# configuration

Hydra 配置优先复用 `src` 的公开 Config，而不是在入口脚本中维护平行结构。目录只为真正可替换的
模块 preset 和运行编排建层级；入口自身的生产默认写在 root config，完整链路测试的组合与预算
写在 `experiment` 中。

## 源码模块

- `runtime`：完整映射 `runtime.Config`，统一拥有 codec、backbone、audio tokenizer、device、dtype、
  attention backend 与 flow sampling。`longcat`、`longcat_native`、`unicodec` 表示相互兼容的资源
  snapshot；不再拆分 `codec` 和 `sampler` 组。
- `model`：完整映射 `model.Config` 的三个 adapter 与可选 `ToyConfig`。`model=toy` 只替换
  backbone；`model/acoustic` 选择 flow/RVQ composition，preset package 仍是顶层 `acoustic`，
  避免把 subtype 字段混入基础 `model.Config`。
- `data`：overfit 数据源 preset；`data=toy` 使用 `DatasetConfig` 选择内存 codec samples，
  production/fixed-sample experiment 默认仍使用 WMT19 TTS prepared data。
- `pl_module`：完整映射 `pl_module.Config` 的 learning rate 与 weight decay；不再使用含义重复的
  `optimizer` 组。
- `codec_oracle`：完整映射 `codec_oracle.Config`，统一拥有 objective、initialization、target
  normalization、decoder、optimizer 参数和 `codec_oracle.DataConfig`。LBA 是该模块的 data 能力，
  不再使用顶层 `oracle`、`init` 或 `data/oracle` 组。

`trainer`、`logging`、`callback` 与 `experiment` 属于 Lightning/Hydra 运行编排，可以没有同名
`src` 包。overfit 的 sample index 和 train budget 位于 experiment；数据源通过公开
`DatasetConfig` 选择。overfit 的 `callbacks.evaluation.enabled` 控制声学生成评估；真实
fixed-sample experiment 默认启用，随机输出不构成质量结论的 `toy_smoke` 显式关闭。oracle
callbacks 总是成套使用，因此合并为单个 preset。

## 生产默认与完整链路测试

裸 `scripts/codec_oracle.py` 组合 `configs/codec_oracle.yaml` 的生产训练默认：prepared dataset
不设 `sample_limit`，启用 LBA，训练 1,000,000 steps，并由默认 trainer 使用 `bf16-mixed`；
sample logging 与 checkpoint archive 都每 10,000 steps 触发。该入口不是 smoke test；需要短验收
时必须显式选择 experiment，避免生产默认被测试预算污染。

训练输出由 `repo_output_root`、相对的 `output_subdir` 和派生的 `output_dir` 组成。checkpoint、音频、
Hydra metadata 与 `metrics.json` 写入 `output_dir`；TensorBoard/CSV logger 的路径由 logging preset
统一计算。TensorBoard 运行目录为
`repo_output_root/tensorboard/output_subdir/version_*`，因此可以直接把整个项目的 TensorBoard 根
目录交给比较工具。`repo_output_root` 优先使用 `SPEECH_TO_SPEECH_TRAIN_ROOT`，job 默认将它设为项目
根，因此 `tensorboard/` 与 `src/` 同级。`output_subdir` 不允许绝对路径或 `..`，`output_dir` 也不允许
独立 override。

两个 trainer preset 都使用 `devices: auto`，由 Lightning 使用 `CUDA_VISIBLE_DEVICES` 中的全部
可见设备；设备数量不再作为运行时配置契约重复校验。job wrapper 只提供机器相关的默认可见设备，
提交时可显式覆盖。共享 `trainer=ddp` 使用 Lightning 默认 distributed sampler。LongCat LBA
直接暴露 dataset 与 DataLoader 构造参数供 Lightning 重建；UniCodec DDP smoke 则要求每个 rank
重复读取同一个固定样本，因此仅该 experiment 显式设置 `use_distributed_sampler: false`。

完整链路实验分别负责其 composition、数据范围、trainer、callback 和 step budget：

- `acoustic_oracle_smoke`：LongCat 默认策略两步验收。
- `acoustic_oracle_ddp_lba_smoke`：LongCat 显式 DDP LBA 两步验收。
- `acoustic_oracle_rvq_smoke`：LongCat RVQ oracle 默认策略两步验收。
- `acoustic_oracle_rvq_ddp_lba_smoke`：LongCat RVQ oracle 显式 DDP LBA 两步验收。
- `unicodec_overfit`：UniCodec fixed-sample 100-step overfit。
- `unicodec_ddp_smoke`：UniCodec 显式 DDP 两步验收。
- `overfit`：TTS/S2ST fixed-sample 完整链路实验。
- `011_qwen_rvq_native_p0_fixed_sample`：真实 Qwen、LongCat native token 与 RVQ decoder 的
  P0 TTS/S2ST 2-step fixed-sample 合同验收；该 experiment 只固化当前 P0 子项，不替代 011
  的正式 staged joint entry。
- `toy_smoke`：正式 LongCat runtime 加 tiny model/in-memory dataset 的 CPU 两步训练契约测试；
  不读取真实 backbone 权重或 WMT19 prepared dataset，也不替代真实资源验收。

`jobs/002` 与 `jobs/005/01-07` 都显式传递对应的 `experiment=`；002 job 另行选择 TTS/S2ST task，
training job 传递 `repo_output_root`、相对 `output_subdir` 和 `"$@"` 参数。测试预算因此由 experiment
单点维护，调用 smoke wrapper 时无需再传 `train.max_steps=2`。

`jobs/011/01_rvq_native_p0_fixed_sample.sh` 复用 `scripts/overfit.py` 作为唯一 Python 入口，
并行启动 TTS 与 S2ST 两个单卡 fixed-sample 子任务，分别写入 launcher log、pid 和 exit status。
真实 Qwen snapshot、prepared data root、输出根和 GPU 选择通过环境变量覆盖，避免把复旦机器的
临时 `/tmp` 路径写死进 Hydra preset。

`jobs/005/08-11` 是 LongCat Flow/RVQ 的正式默认策略与显式 DDP 入口。它们不选择 experiment，
直接继承 root config 的完整数据、LBA、1,000,000-step 预算和生产 callback 间隔；只选择
objective、trainer 和隔离的输出目录。wrapper 分别提供单卡、两卡可见设备默认值，DDP 入口显式
使用已验收的静态 `ddp` strategy。

## 入口边界

`scripts/_config.py` 只定义入口专属结构，例如 task、Trainer、logging、callback 与 flow/RVQ
composition。`runtime.Config`、`model.Config`、`pl_module.Config`、`model.DecoderConfig` 和
`codec_oracle.Config` 直接进入 root schema，不重复声明字段。OmegaConf 对字符串枚举只接受成员
名，入口在合并前把公开的小写 value 转成 enum member name；除此之外不做兼容重写。

两个入口分别解析为：

- `OverfitTokenConfig | OverfitFlowConfig | OverfitRVQConfig`
- `CodecOracleConfig`

未知字段和错误 composition 在进入执行逻辑前失败，解析后的 dataclass 不再向 `src` 传递
`DictConfig`。oracle 额外要求 `runtime.audio_tokenizer is None`，因为 prepared semantic IDs 是 raw
codec codes；入口显式拒绝 CodecBPE tokenizer，而不是静默忽略。

## 组合

- flow/RVQ 使用 `model/acoustic=flow|rvq`；RVQ schema 不接受 REPA。
- unified-token codec 使用 `runtime=unicodec ~model/acoustic`，明确选择 token-only composition。
- flow method、NFE 和 step 数直接覆盖 `runtime.flow_*`；RVQ/token 中保留这些字段是
  `runtime.Config` 的稳定 shape，不需要再为未使用字段创建 variant schema。
- codec capability 必须与 composition 一致，入口不自动把 flow/RVQ 改成 token model。
- codec oracle 通过 `codec_oracle.objective=flow|rvq` 选择 acoustic screening model；decoder、
  normalization 与 optimizer 位于 `codec_oracle.*`。flow objective 额外使用 runtime 的 flow
  sampling 和 normalization 字段，RVQ objective 不读取这些字段；两种 objective 的
  initialization 都只控制 semantic audio embedding。
