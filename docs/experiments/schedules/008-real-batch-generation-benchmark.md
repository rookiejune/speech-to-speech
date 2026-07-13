# 008 Real Batch Generation Benchmark

## 目标

在真实 Qwen3/LongCat S2ST 资源上验收变长 cached batch generation，并建立批量
推理的吞吐和显存基线。

## 配置

- 使用 WMT19/LongCat train split 连续样本，batch size 为 `1,2,4`。
- batch contract probe 为第 `i` 行前置 `i` 个 BOS token，并同步平移 source acoustic
  positions，以在真实 backbone 上覆盖变长 prompt 左 padding。
- 模型为 `Qwen/Qwen3-0.6B` + LongCat `16k_4codebooks`，bfloat16 + Flash Attention 2。
- semantic generation 使用 greedy + KV cache，最多生成 2 tokens；本实验只验证执行
  契约，不评价未训练模型的音频质量。
- 同一组请求分别执行 batch 和逐请求 cached generation。

## 验收

- 变长 prompt/source acoustic frames 可在同一 batch 中执行。
- batch 与逐请求的 greedy semantic token 逐行相同。
- 每行 waveform 成功 decode 且 finite。
- 记录 batch/逐请求耗时、tokens/s 和 peak allocated CUDA memory。
