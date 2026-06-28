# Diagonal Acoustic Sampler Profile

## 结论

121 A100 上，真实 Qwen acoustic condition + 真实 1-layer DiT 的 sampler micro-profile
显示 diagonal wavefront 调度能明显减少 DiT 调用并加速 acoustic flow 部分。full
waveform 生成已经能通过同一个接口切换 serial/diagonal sampler；在当前随机未训练模型
和 codec decode 路径下，端到端收益被 semantic 自回归生成和 codec decode 稀释。

## 121 Profile

环境：

- dataset: `/mnt/pami202/zhuyin/datasets/wmt19-tts-longcat`
- config: `configs/wmt19_tts_longcat_acoustic_smoke.yaml`
- model: Qwen3-0.6B 4bit + 1-layer DiT, bf16 flow
- sample: index 0, source 27 frames, target 36 frames

Sampler micro-profile:

| frames | steps | chunk | serial DiT forwards | diagonal DiT forwards | serial seconds | diagonal seconds | speedup | max diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 8 | 16 | 32 | 11 | 0.0684 | 0.0284 | 2.41x | 0 |
| 128 | 16 | 32 | 64 | 19 | 0.1702 | 0.0567 | 3.00x | 0 |
| 256 | 32 | 32 | 256 | 39 | 0.5841 | 0.1188 | 4.92x | 0 |

Full waveform smoke, `max_new_tokens=64`, generated 127 acoustic frames:

| sampler | generation seconds | total seconds | audio shape |
| --- | --- | --- | --- |
| serial | 14.67 | 77.73 | `[1, 1, 121920]` |
| diagonal | 14.46 | 77.61 | `[1, 1, 121920]` |

Notes:

- `generation_seconds` includes semantic autoregressive generation, acoustic sampler, and codec decode.
- `total_seconds` also includes model/codec loading and setup.
- For generated frames below `chunk_size`, diagonal has no parallelism advantage; the 16-token smoke
  generated 31 frames and was effectively a single acoustic chunk.
