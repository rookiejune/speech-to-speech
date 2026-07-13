# 007 Flow 与 REPA 对比

## 目标

在相同真实样本、模型初始化、数据顺序、optimizer 和训练预算下，对比 DiT flow baseline
与 DiT flow + REPA，判断 REPA 是否改善 acoustic reconstruction 或 waveform 指标。训练 loss
只用于监督运行，不单独作为增益结论。

## 固定条件

- 数据：WMT19 TTS LongCat prepared store，sample 0，batch size 1。
- 模型：Qwen3-0.6B + 8-layer acoustic DiT；两组均保留 768 维 REPA projection，确保架构一致。
- seed：0；optimizer：AdamW SFT preset，learning rate `2e-5`，weight decay `0.01`。
- 两组先按相同顺序物化 lazy runtime 资源；teacher 加载后、model 构造前重置 PyTorch RNG，
  避免 WavLM 初始化改变 DiT 权重或 flow noise。
- baseline：semantic + flow；treatment：semantic + flow + `0.1 * REPA`。
- REPA teacher：frozen `microsoft/wavlm-base` layer 9；student：DiT layer 4。

## 顺序

1. 两组各运行 2-step real smoke，验证 teacher、feature alignment、backward 和 callback 闭环。
2. smoke 通过后，两组各运行固定 sample 的 100-step overfit。
3. 使用相同 step、target condition 和 noise seeds 的 sampled features 计算重建和 waveform
   指标；不横向比较随机单 step loss。
4. 100 steps 不能判断趋势时只增加训练步数，不同时修改权重、学习率或样本。

## Smoke 入口

```bash
jobs/002/01_tts.sh \
  train=smoke \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/007-flow-repa-comparison/smoke/baseline"

jobs/002/01_tts.sh \
  train=smoke \
  acoustic.repa.weight=0.1 \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/007-flow-repa-comparison/smoke/repa"
```

正式 100-step 对比关闭依赖自由 semantic generation 的 `SampleLogger`，只使用固定 target
condition 和 noise seeds 的 acoustic evaluator：

```bash
jobs/002/01_tts.sh \
  callbacks.sample.enabled=false \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/007-flow-repa-comparison/formal/baseline"

jobs/002/01_tts.sh \
  callbacks.sample.enabled=false \
  acoustic.repa.weight=0.1 \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/007-flow-repa-comparison/formal/repa"
```

## 验收

- smoke：两组均完成 2 steps；所有 objective、分项梯度和全局梯度有限；REPA 组 teacher
  frozen/eval；TensorBoard 包含 flow time、sample audio 与文本 retention 输出。
- 正式对比：两组均完成相同 optimizer steps，无持续显存增长；报告 semantic、flow、REPA、
  total loss 曲线以及同一指标实现下的 reconstruction/waveform 结果。
- 只有 treatment 的非训练指标相对 baseline 有稳定改善时，才得出 REPA 有增益的结论。

## 限制

- fixed-sample overfit 不验证泛化。
- smoke 不支持 REPA 效果结论。
- REPA loss 与 flow loss 数值尺度不同，不直接横向比较。
