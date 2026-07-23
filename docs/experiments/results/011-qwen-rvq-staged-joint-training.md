# 011 Qwen RVQ Staged Joint Training P0 Smoke

对应计划：[`schedules/011-qwen-rvq-staged-joint-training.md`](../schedules/011-qwen-rvq-staged-joint-training.md)。

状态：**P0 未完成，不允许晋级。** 本次只验证了真实 Qwen + native LongCat + RVQ 的
TTS/S2ST 单卡 fixed-sample 2-step 子项。两条训练和 teacher-forced acoustic sampling 均通过，
但训练后的端到端 generation 在公共能力检查处失败；P0 的 DDP、resume、100-step、1k pilot
和 checkpoint import 均尚未执行。

## 环境与范围

- 日期与机器：2026-07-21，复旦 `145`，TTS 使用 RTX 4090D GPU 1，S2ST 使用 GPU 2。
- 环境：Python 3.12、PyTorch `2.9.0+cu128`、Lightning `2.6.1`；远端源码为干净的
  `d5f69028ab57fc9862469283dccfb019ec603776`。
- 模型：真实 Qwen3-0.6B revision
  `c1899de289a04d12100db370d81485cdf75e47ca`，LongCat native semantic token，8 层 RVQ
  decoder；总参数 `949,827,052`，可训练参数 `949,826,028`，decoder 参数 `184,031,980`。
- 数据与预算：WMT19/LongCat prepared train sample 0，batch size 1，`bf16-mixed`，每个任务
  2 optimizer steps；关闭逐步 `TaskSampleLogger`，保留每步 4 个固定 seed 的 acoustic evaluation。
- 复旦共享 NAS 为 26 TB、使用率 100%，首轮任务在训练前发生 I/O stall。将 Qwen、
  LongCat cache 和 prepared data 复制到 `145:/tmp` 后，训练不再阻塞。该问题归类为运行环境
  容量问题，不归类为模型失败；所有 `/tmp` 产物均不是长期 artifact。

## 训练与声学采样

两条任务都完成 forward、backward 和 optimizer step；完整 TensorBoard 标量如下：

| task | total loss, step 1 -> 2 | audio token CE | RVQ CE | text probe NLL, step 0 -> 2 |
| --- | ---: | ---: | ---: | ---: |
| TTS | `18.24888 -> 15.93290` | `9.08193 -> 6.86867` | `9.16696 -> 9.06424` | `1.19907 -> 1.21947` |
| S2ST | `18.29995 -> 16.16632` | `9.07686 -> 7.04392` | `9.22309 -> 9.12240` | `1.19907 -> 1.14460` |

三个 RVQ codebook CE 均下降：TTS 为
`9.19965/9.24653/9.05469 -> 9.17622/9.04948/8.96701`，S2ST 为
`9.28906/9.27778/9.10243 -> 9.18056/9.13889/9.04774`。两步只验证优化信号，
不支持收敛结论。

Acoustic evaluation 使用 teacher-forced target token positions 构造 condition，再采样 RVQ
features 并通过 LongCat decode；它不是端到端 autoregressive generation：

| task | feature MSE, step 0 / 1 / 2 | STFT convergence, step 0 / 1 / 2 | step 2 RTF | waveform |
| --- | ---: | ---: | ---: | --- |
| TTS | `2.26636 / 2.30221 / 2.33197` | `1.29228 / 1.21122 / 1.13856` | `0.01470` | 2.16s，finite |
| S2ST | `2.28265 / 2.25843 / 2.31239` | `1.17058 / 1.16289 / 1.13349` | `0.01401` | 2.16s，finite |

每个任务的 3 个记录点乘 4 个 seed 均成功 decode，指标全部 finite。feature MSE 没有一致
改善，因此仍只作为诊断量。`GradLogger` 当前在 acoustic decoder 的 q-proj 上比较 token/RVQ
梯度；token objective 不经过该参数，token norm 为 0，故 cosine 记录为 `NaN`。这不是模型
梯度 non-finite。后续代码已把该 probe 改到 Qwen attention 的共享 q-proj，但本次历史结果
仍不可用于权重校准，必须在后续运行中重新采集。

## Generation 阻塞

两条任务均在 `Trainer.fit` 正常到达 `max_steps=2` 后，于训练后 greedy generation 报同一错误：

```text
TypeError: a codec with acoustic codebooks requires an AcousticFeatureGenerator.
```

因此 TTS、S2ST 和总退出码均为 `1`；`generation.json` 与 `metrics.json` 没有写出。关闭
acoustic evaluation 的 companion training-only run 能写出 `metrics.json`，且 loss 与本次
TensorBoard 完全一致，但不能据此把 generation gate 记为通过。

根因不是 RVQ model 缺少生成实现。`RVQModel` 已实现
`generate_audio_features()`，但 [`generate_responses()`](../../../src/speech_to_speech/generation/service.py#L51)
用 runtime-checkable Protocol 对整个 `TokenGenerator` 做 `isinstance`。Python 3.12 使用
`inspect.getattr_static()` 检查 Protocol 成员；真实 Qwen `backbone` 作为 `nn.Module` 子模块存于
`_modules`，普通访问存在但 static lookup 不可见，于是能力检查返回 false。当前 generation
测试替身把 backbone 设为普通 `SimpleNamespace` 属性，未覆盖真实 `nn.Module` 分支。

本地最小复现结果为：

```text
module_hasattr=True, module_static=None, module_protocol=False, plain_protocol=True
```

## P0 判定与后续

| P0 子项 | 判定 |
| --- | --- |
| 真实 Qwen/native/RVQ TTS 与 S2ST 单卡 forward/backward/optimizer | 通过 |
| S2ST source acoustic prompt 训练路径 | 通过 |
| teacher-forced RVQ sampling、LongCat decode、finite waveform | 通过 |
| 训练后端到端 semantic-to-RVQ-to-waveform generation | **失败** |
| 正常写出 `generation.json`、`metrics.json` 并以状态 0 退出 | **失败** |
| 2-GPU DDP、resume、32-sample 100-step、1k pilot、010 import | 未执行 |

下一步应把 optional acoustic generation 能力改为不会受 `nn.Module.__getattr__` 影响的明确
契约，并加入真实 registered backbone 的回归测试；随后原样重跑本节两条任务。只有两条都
写出 finite `generation.json`、`metrics.json` 且退出码为 0，才继续其余 P0 子项。

## 产物

- 失败前台 run：`145:/tmp/s2s-011-p0-rvq-native-20260721-041315-full-eval/`。
- 稳健重跑：`145:/tmp/s2s-011-p0-rvq-native-20260721-041315-full-eval-retry-1/`。
- 本地镜像：`debug/speech-to-speech/011-qwen-rvq-staged-joint-training/remote-full-eval-retry-1/`。
- companion training-only metrics：同一本地镜像下的 `tts-training-only-metrics.json` 与
  `s2st-training-only-metrics.json`。

任务退出后 GPU 1、2 均回到 15 MiB、0% utilization，没有遗留训练进程。
