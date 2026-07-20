# configuration

Hydra 配置优先复用 `src` 的公开 Config，而不是在入口脚本中维护平行结构。目录只为真正可替换的
模块 preset 和运行编排建层级；入口自身的生产默认写在 root config，完整链路测试的组合与预算
写在 `experiment` 中。

## 源码模块

- `runtime`：完整映射 `runtime.Config`，统一拥有 codec、backbone、audio tokenizer、device、dtype、
  attention backend 与 flow sampling。`longcat`、`longcat_native`、`unicodec` 表示相互兼容的资源
  snapshot；不再拆分 `codec` 和 `sampler` 组。
- `model`：完整映射 `model.Config` 的三个 adapter。`model/acoustic` 选择 flow/RVQ composition，
  preset package 仍是顶层 `acoustic`，避免把 subtype 字段混入基础 `model.Config`。
- `pl_module`：完整映射 `pl_module.Config` 的 learning rate 与 weight decay；不再使用含义重复的
  `optimizer` 组。
- `codec_oracle`：完整映射 `codec_oracle.Config`，统一拥有 initialization、target normalization、
  decoder、optimizer 参数和 `codec_oracle.DataConfig`。LBA 是该模块的 data 能力，不再使用顶层
  `oracle`、`init` 或 `data/oracle` 组。

`trainer`、`logging`、`callback` 与 `experiment` 属于 Lightning/Hydra 运行编排，可以没有同名
`src` 包。overfit 的 fixed-sample data 和 train budget 没有独立替代项，直接位于 overfit
experiment；oracle callbacks 总是成套使用，因此合并为单个 preset。

## 生产默认与完整链路测试

裸 `scripts/codec_oracle.py` 组合 `configs/codec_oracle.yaml` 的生产训练默认：prepared dataset
不设 `sample_limit`，启用 LBA，训练 1,000,000 steps，并由默认 trainer 使用 `bf16-mixed`；
sample logging 与 checkpoint archive 都每 10,000 steps 触发。该入口不是 smoke test；需要短验收
时必须显式选择 experiment，避免生产默认被测试预算污染。

完整链路实验分别负责其 composition、数据范围、trainer、callback 和 step budget：

- `acoustic_oracle_smoke`：LongCat 单卡两步验收。
- `acoustic_oracle_ddp_lba_smoke`：LongCat 两卡 LBA 两步验收。
- `unicodec_overfit`：UniCodec fixed-sample 100-step overfit。
- `unicodec_ddp_smoke`：UniCodec 两卡两步验收。
- `overfit`：002 TTS/S2ST fixed-sample 实验。

`jobs/002` 与 `jobs/005` 都显式传递对应的 `experiment=`；002 job 另行选择 TTS/S2ST task，005
job 的 Python 调用只保留 experiment、项目输出目录和 `"$@"` 参数透传。测试预算因此由
experiment 单点维护，调用 005 smoke wrapper 时无需再传 `train.max_steps=2`。

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
- codec oracle 固定使用 flow model；decoder 与 normalization 位于 `codec_oracle.*`。
