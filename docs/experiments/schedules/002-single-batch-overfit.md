# 002 Single Batch Overfit

## 目标

在 121 的真实 WMT19/LongCat/Qwen3 环境中固定同一条 raw sample，验证 semantic CE
和 flow matching objective 不仅 finite，而且能够被当前优化器组合持续优化。

## 顺序

1. TTS 固定首条样本训练 100 steps，隔离 target semantic/acoustic 学习路径。
2. TTS 曲线正常后，以相同配置运行 S2ST，加入 source semantic/acoustic condition。
3. 100 steps 不能判断趋势时只增加 `--max-steps`，不同时修改学习率或样本。

## 记录

- TensorBoard 每 step 记录 total、semantic、flow matching loss 及 semantic text/audio details。
- `metrics.json` 记录各 objective 前 20 和后 20 steps 的均值与比值。
- 记录运行环境、显存、耗时以及实际命令。

## 验收

- 两个任务都完成预定 optimizer steps，loss 和梯度保持 finite，无持续显存增长。
- semantic 与 flow matching 的后 20 steps 均值低于前 20 steps；flow objective 有随机
  time/noise，不要求逐 step 单调。
- 结果写入 `docs/experiments/results/002-single-batch-overfit.md`，只把实际验证通过的
  结论移入阶段状态。

## 限制

- 本实验不验证泛化、音频质量、长时间稳定性或 generation 契约。
- 不使用当前 SampleLogger 作为验收依据，因为 token 与 waveform 尚未统一复用同一次生成。
