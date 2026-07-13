# 007 Flow 与 REPA 对比结果

> 状态：已完成。baseline/REPA 2-step smoke 与 100-step fixed-sample 对比均完成；当前预算
> 下未观察到稳定的 reconstruction/waveform 指标增益，REPA 不设为默认。

## Smoke 环境

- 日期：2026-07-13；机器：144，NVIDIA GeForce RTX 4090 24 GiB，物理 GPU 1。
- WMT19 TTS LongCat prepared store sample 0，batch size 1，seed 0，bf16 mixed。
- Qwen3-0.6B + 8-layer acoustic DiT；REPA weight 0.1，frozen WavLM-base layer 9，
  DiT layer 4 student representation。
- 输出位于 `.../007-flow-repa-comparison/smoke/{baseline,repa}`。

## Smoke 结果

| Group | Objective | Step 1 | Step 2 |
| --- | --- | ---: | ---: |
| baseline | semantic | 18.18750 | 14.50000 |
| baseline | flow matching | 2.525522 | 2.515621 |
| baseline | total | 20.713022 | 17.015621 |
| REPA | semantic | 18.18750 | 14.50000 |
| REPA | flow matching | 2.525522 | 2.515629 |
| REPA | REPA | 0.985914 | 0.983941 |
| REPA | total | 20.811613 | 17.114023 |

两组 step-1 semantic 与 flow 数值完全一致，证明首步使用相同 model 初始化、batch 和 flow
noise；step 2 起允许 REPA 梯度造成差异。两组均以 `max_steps=2` 正常结束，objective、
分项梯度和全局梯度保持 finite；REPA 组 94.4M WavLM 参数冻结，233 个 teacher module
保持 eval。TensorBoard 包含 flow time、分项梯度、sample audio 与 text retention 输出。

## 暴露与修复

第一次配对 smoke 中，REPA teacher 提前物化 lazy runtime 的 codec/backbone，导致两组虽然
使用相同 seed，model 初始化和 flow RNG 仍不一致。入口现按相同顺序预先物化 layout、codec、
backbone 和 flow runtime，再构造可选 teacher，最后重置 PyTorch RNG 后构造 model。

首次 baseline 100-step 运行在 step 8 因自由 semantic generation 没有产生可解码 audio token，
被 `SampleLogger` 明确中止。007 正式对比不依赖该随机 generation callback，改用固定 target
condition/noise 的 acoustic evaluator；入口增加显式 `callbacks.sample.enabled=false`，不在
`src` 中吞掉 generation 错误。

## Smoke 限制

smoke 不支持 REPA 增益结论；正式结果如下。

## 正式配置

- 两组复用 smoke 的 sample 0、seed 0、模型结构、optimizer、学习率和 bf16 mixed 配置。
- 各运行 100 optimizer steps；关闭自由 semantic generation 的 `SampleLogger`。
- step 0/20/40/60/80/100 使用相同 target condition 和 flow noise seeds 0-3，报告均值。
- acoustic evaluator 记录 masked feature MSE、MR-STFT spectral convergence/log-magnitude、
  waveform RMS、peak 和 duration。

## 正式结果

训练 objective 的前/后 20-step 均值：

| Group | Objective | First 20 | Last 20 | Last / first |
| --- | --- | ---: | ---: | ---: |
| baseline | semantic | 5.03010 | 0.02441 | 0.00485 |
| baseline | flow matching | 2.34726 | 1.61184 | 0.68669 |
| REPA | semantic | 5.13787 | 0.02462 | 0.00479 |
| REPA | flow matching | 2.34695 | 1.61523 | 0.68822 |
| REPA | REPA | 0.96140 | 0.69165 | 0.71942 |

固定 noise evaluator：

| Step | Feature MSE baseline / REPA | STFT log-mag baseline / REPA | Spectral convergence baseline / REPA |
| ---: | ---: | ---: | ---: |
| 0 | 2.52360 / 2.52360 | 2.07566 / 2.07566 | 1.15605 / 1.15605 |
| 20 | 2.17301 / 2.16983 | 1.82326 / 1.84367 | 1.04108 / 1.04213 |
| 40 | 2.10209 / 2.10369 | 1.68890 / 1.68085 | 1.07848 / 1.08082 |
| 60 | 2.02351 / 2.02384 | 1.61678 / 1.63031 | 1.06269 / 1.03103 |
| 80 | 1.97681 / 1.98411 | 1.57540 / 1.53266 | 0.94116 / 0.94422 |
| 100 | 1.94086 / 1.95096 | 1.42542 / 1.40563 | 1.13906 / 1.15652 |

两组 waveform duration 始终为 2.16 秒；RMS/peak 保持有限，没有静音、爆音或长度错误。
两组都正常达到 `max_steps=100`，GPU 退出后回落到 4 MiB。

## 结论

- REPA objective 能被优化，后 20-step 均值相对前 20 steps 下降约 28%。
- step 100 时，REPA 的 STFT log-magnitude 改善约 1.39%，但 feature MSE 恶化约 0.52%，
  spectral convergence 恶化约 1.53%；中间 checkpoints 的相对方向也不一致。
- 当前 fixed-sample、100-step 预算下没有稳定的非训练指标增益，不把 REPA 设为默认。
- 本实验不验证跨样本泛化；未来若扩大数据或预算，应新建实验，不覆盖本结论。

## 产物

- baseline：`.../007-flow-repa-comparison/formal/baseline/{metrics,evaluation}.json`
- REPA：`.../007-flow-repa-comparison/formal/repa/{metrics,evaluation}.json`
