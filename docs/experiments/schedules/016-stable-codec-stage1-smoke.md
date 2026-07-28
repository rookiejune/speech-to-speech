# 016 Stable Codec Stage 1 Smoke

## 目标

在复旦真实 GPU 上验收默认 Stable Codec stage 1 长跑入口的最小闭环，确认路线是
Stable Codec、native full-code sequence、无 audio BPE，并同时训练 ASR 与 TTS。

## 固定配置

- 入口：`jobs/015/01_stable_codec_stage1.sh`。
- codec：`stabilityai/stable-codec-speech-16k`。
- backbone：Qwen3-0.6B。
- audio representation：`full_codec_sequence`。
- audio tokenizer：单码本 `FlattenedAudioTokenizer`，不加载 BPE artifact。
- stage：stage 1，ASR/TTS loader 权重各 `0.5`。
- 正式默认：1,000,000 optimizer steps；每 10,000 steps fixed sample 与 checkpoint。

## Smoke 覆写

- 使用真实 prepared WMT19/Stable codes，batch size 1、workers 0。
- 单卡、1 optimizer step，关闭 smoke checkpoint。
- fixed sample 每 step 触发，ASR/TTS 各取 index 0。
- generation 设为 greedy、cache 开启、`max_new_tokens=34`。

## 验收条件

1. Stable Codec 与 Qwen 从真实 checkpoint 加载，训练完成 1 optimizer step，loss finite。
2. ASR fixed sample 写出带 loader namespace 的 target/generated text。
3. TTS fixed sample 写出带 loader namespace 的 target/generated 16 kHz audio，waveform finite。
4. Stable token-only 路线只记录 target/generated，不产生 Flow/RVQ `reference_generation`。
5. full-code response 使用合法的 codec/codebook marker prefix，且 marker 计入 generation budget。

本 smoke 只验证执行与日志契约，不验证语音质量、识别准确率或收敛。
