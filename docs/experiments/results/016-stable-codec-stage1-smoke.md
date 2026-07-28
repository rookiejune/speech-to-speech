# 016 Stable Codec Stage 1 Smoke

2026-07-28 在复旦 `121` 的 A100 GPU 1 完成真实 Stable Codec、Qwen3-0.6B、
WMT19 prepared data 的 stage 1 单步验收。路线使用 native full-code sequence，
没有 audio BPE；ASR/TTS loader 权重各为 `0.5`。本结果只证明执行和 fixed-sample
日志契约，不支持质量或收敛结论。

## 真实资源

- Stable Codec：`stabilityai/stable-codec-speech-16k`，16 kHz，单码本 `46656` codes。
- 实测 `downsampling_ratio=640`，frame rate 为 `25 Hz`。
- 测试数据：`/tmp/stable-stage1-data-20260728`，32 pairs、64 audio items；Stable frame
  范围 `32..64`，code range `11..46504`。
- split manifest：`/tmp/stable-stage1-data-20260728/smoke-splits.json`。
- Python：`/tmp/stable-codec-py39-env`，Stable Codec 固定的 Torch 2.4 环境。
- 隔离代码：`/tmp/stable-stage1-repos`，没有修改共享 checkout。

真实 codec roundtrip 先独立通过：1 秒输入得到 `[1,25,1]` codes，token roundtrip 为 true，
decode 得到 `[1,1,16000]` finite waveform。

## 执行结果

入口为 `jobs/015/01_stable_codec_stage1.sh`，只把正式配置覆写成单步 smoke：

```bash
jobs/015/01_stable_codec_stage1.sh \
  train.max_steps=1 trainer.devices=1 trainer.strategy=auto \
  trainer.enable_checkpointing=false \
  data.dataloader.batch_size=1 data.dataloader.num_workers=0 \
  data.dataloader.pin_memory=false data.dataloader.persistent_workers=false \
  callbacks.task_sample.every_n_steps=1 \
  callbacks.task_sample.max_new_tokens=34
```

run7 exit code 为 `0`，`Trainer.fit` 因 `max_steps=1` 正常停止。每个 optimizer step
包含 ASR/TTS 两个 batch；总参数 `596099653`，stage 1 speech-interface 可训练参数 `49733`。
`metrics.json` 中 total loss 为 `9.965051651000977`，token loss 为
`8.360746383666992`，两者均 finite。

## Fixed Samples

TensorBoard event 位于：

```text
/tmp/stable-stage1-train-20260728/tensorboard/stable-codec-stage1/
stage_1-token/version_3/events.out.tfevents.1785200027.121.pami.group.2215007.0
```

ASR index 0 写出：

- `task_sample/asr/0/target`：`1929 or 1989?`
- `task_sample/asr/0/generated`：greedy 生成文本，metadata 为 `status=ok`。
- metadata 记录 source 为 Stable 单码本 54 frames，生成 25 tokens，未触及 34-token budget。

TTS index 0 写出：

| Tag | Sample rate | Samples | Duration | Finite |
| --- | ---: | ---: | ---: | --- |
| `task_sample/tts/0/target` | 16000 | 34560 | 2.16s | true |
| `task_sample/tts/0/generated` | 16000 | 20480 | 1.28s | true |

TTS metadata 为 `status=ok`，response 含两个 marker 和 32 个 code tokens，共 34 tokens。
event 的全部 audio tags 只有 target/generated；没有 `reference_generation`，符合 Stable
token-only 路线，不伪造 Flow/RVQ reference generation。

## 实测暴露并修复的边界

1. Stable backend 通过 `downsampling_ratio` 而非 `frame_rate/hop_length` 暴露 25 Hz。
2. Stable 固定的 Torch 2.4 没有 `nn.Buffer`，model buffer 注册保留非持久化回退。
3. `AudioView.STABLE` 按完整 frame codes 进入 parser，不产生 acoustic side channel。
4. ASR/TTS 两个 `TaskSampleLogger` 使用不同 state key 和 TensorBoard loader namespace。
5. 单码本 flattened generation 把 codec/codebook marker 加入 prompt，后续只允许 code range
   与 EOA；marker 保留在 response，并计入 `max_new_tokens`。
6. BF16 backbone 与 FP32 speech head 的组合 logits 使用 dtype promotion，避免 selected-head
   scatter 的 dtype mismatch，同时保留 FP32 speech logits。

本地最终验收为 297 tests OK / 1 CUDA skip、basedpyright 0 errors、Ruff、compileall 与
`git diff --check` 通过；远端 Torch 2.4 的 generation/mixed-dtype targeted tests 通过。
