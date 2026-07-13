# 005 Codec Oracle Screening

## 目标

在不在线编码 waveform 的前提下，分别验证 LongCat 的 acoustic flow 路径和 UniCodec
单码本 token 路径，并比较 codec codebook 与 matched-random 两种 audio embedding 初始化。
DAC 暂不纳入本轮。

## 数据与 Codec 边界

- 训练只从 prepared WMT19 TTS codec store 读取 `[frame, codebook]` 离散 codes。
- dataset、DataLoader 和训练入口不调用 waveform encoder，也不保存连续 features。
- LongCat 训练 step 允许调用冻结的 `acoustic_codes_to_features()`，将离散 acoustic codes
  转为 flow target；该 dequantize 不进入 optimizer。
- logging callback 允许调用 codec decoder 记录 oracle reconstruction 和训练中样本波形。
- native audio tokenizer 是本轮正式基线：一个 frame code 对应一个 audio token，不先引入
  codec-specific BPE 的压缩差异。

## 实验拆分

121 上 prepared store 的真实 shape 为：LongCat `[frame, 4]`，UniCodec `[frame, 1]`。
因此两者不能使用同一个 semantic/acoustic codebook 拆分。

### LongCat Acoustic Flow Oracle

- model：完整加载正式 `SpeechToSpeechFlowModel`，不单独构造 flow oracle model。
- condition ID：LongCat 第 0 个 semantic codebook，经正式 semantic audio embedding、adapter
  与 `target_frame_label_condition()` 生成 frame condition，不执行 Qwen forward。
- audio embedding：可训练，分别使用 LongCat semantic codebook 或同均值、同标准差的随机
  权重初始化。
- acoustic target：其余 3 个 codebooks 在训练 step 中冻结 dequantize 得到的 decoder
  features。
- optimizer：正式模型的 semantic audio embedding、adapter 与 `AcousticFlow`；Qwen backbone
  和无关 semantic head 不进入 optimizer。

### UniCodec Unified-Token Formal Path

- UniCodec 唯一 codebook 作为 `Speech.semantic_ids`，`acoustic_ids=None`。
- 复用正式 native audio tokenizer、DataModule、semantic backbone/head 与 `SemanticLoss`。
- audio target 不触发 flow/RVQ objective；generation 后将 semantic codes 直接交给 codec
  decode。
- 不再维护独立 causal Transformer、prepared-code collator 或 teacher-forced probe。

## 运行顺序

1. LongCat acoustic-only screening 验收 flow target 与 sample decode。
2. UniCodec 使用正常 TTS overfit 入口验收 semantic CE 与 direct decode。
3. 两条路径目标不同，数值不互相排名。

## DDP 与 LBA

- 多样本训练从 prepared codec store 读取完整 code sequence；LBA 的长度单位是 codec
  frames，显式 budget 为 `max_batch_seconds * codec.frame_rate`。
- map-style source loader 在每个 rank 使用显式 `DistributedSampler`；Trainer 关闭自动
  sampler 注入，避免无法穿透 LBA wrapper 的隐式替换。
- codes 用 `-1` padding。LongCat 在 dequantize 前替换为合法 ID 并用 frame mask 计算 flow
  loss；UniCodec 使用 Transformer key-padding mask 和 `ignore_index=-100` CE。
- DDP contract callback 在 fit start 校验实际 world size；121 wrapper 默认使用物理 GPU
  2、3。

```bash
jobs/005/04_longcat_ddp_lba.sh init=codec
jobs/005/05_unicodec_ddp_lba.sh init=codec
```

## 日志

- stdout JSON stage：dataset load、codec load、codebook extraction/dequantize probe、logger
  build、`Trainer.fit`、首次训练 dequantize、callback sample 和 waveform decode 均记录
  start/done/error 与耗时，用于定位远程卡点。
- TensorBoard：公共 `train/grad_norm`；LongCat 记录 `train/flow_loss`、`flow/time`、
  `oracle/sample_feature_mse`；UniCodec 记录 `train/token_loss`、`train/token_accuracy`、
  `oracle/teacher_forced_accuracy`；LongCat 记录 sampled waveform，UniCodec 记录
  teacher-forced prediction waveform。
- `metrics.json`：codec、objective、初始化、shape/scale 元数据、首末 loss 窗口和采样指标。
- non-finite callback：参数或梯度第一次出现非有限值时立即中止并暴露位置。

## 入口

```bash
jobs/005/01_longcat.sh train=smoke init=codec
jobs/005/01_longcat.sh train=smoke init=random
jobs/005/02_unicodec.sh train.max_steps=2
```

所有 wrapper 保留 Hydra overrides，例如：

```bash
jobs/005/02_unicodec.sh train.max_steps=2 data.sample_index=0
```

UniCodec 的 `fairseq==0.12.2` 与主 `py312` 不兼容；它使用 codec 专用 Python，通过
`SPEECH_TO_SPEECH_UNICODEC_PYTHON` 显式覆盖。121 当前验证入口为：

```bash
export SPEECH_TO_SPEECH_UNICODEC_PYTHON=/mnt/pami202/zhuyin/dynamic/debug/speech-to-speech/envs/unicodec/bin/python
```
