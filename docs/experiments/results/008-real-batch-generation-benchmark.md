# 008 Real Batch Generation Benchmark Result

## 环境

- 日期：2026-07-14。
- 机器：144，NVIDIA GeForce RTX 4090，物理 GPU 0。
- 数据：WMT19/LongCat train split 连续 4 条 S2ST request。
- 模型：`Qwen/Qwen3-0.6B` 与 LongCat `16k_4codebooks`；模型未训练，
  `acoustic_prompt_gate=1`。
- 生成：greedy cached generation，每行最多 2 tokens。
- 原始 metrics 保存在 `debug/speech-to-speech/008-real-batch-generation-benchmark/`，
  不纳入 Git。

## Bfloat16 + Flash Attention 2

| Batch | Batch tokens/s | Serial tokens/s | Speedup | Peak allocated | Token parity |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 3.17 | 3.17 | 1.00x | 5.34 GB | pass |
| 2 | 4.34 | 3.14 | 1.38x | 5.34 GB | pass |
| 4 | 5.90 | 3.16 | 1.87x | 5.34 GB | fail |

三组的 waveform 均成功 decode 且 finite。Batch 4 与逐请求的 strict greedy
token parity 失败。同一未训练 audio head 的单样本首步 top-1 margin 仅
`0.0625`；bfloat16 batch GEMM 和单样本 GEMM 的数值路径足以改变近似并列
token 的排序。

## Float32 + Eager 复验

首先用原始同长 prompt 复验 batch 4，batch 与逐请求的 4 行 token 全部
相同。随后为第 `i` 行前置 `i` 个 BOS token，并平移 acoustic positions，
验收真实 backbone 上的变长契约：

- prompt tokens：`25 / 26 / 27 / 28`。
- source acoustic frames：`27 / 43 / 43 / 43`。
- response tokens：每行 2 tokens，batch 与逐请求逐行完全相同。
- 所有 waveform finite。
- batch 4：`1.481s`、`5.40 tokens/s`、peak allocated `7.47 GB`。
- serial：`2.635s`、`3.04 tokens/s`、peak allocated `7.44 GB`。
- batch 吞吐为 serial 的 `1.78x`，peak allocated 增加约 `22 MB`。

## 结论

- 左 padding、attention mask、source acoustic position 平移、KV cache、逐行响应
  裁剪和变长 frame decode 在 float32 真实资源上闭环通过。
- batch 4 在该短生成 probe 中提供 `1.78x`–`1.87x` 吞吐，没有显著增加
  peak allocated memory。
- bfloat16 对未训练且 top-1 margin 很小的 audio head 不保证 batch/single
  strict token parity。这不表明 batch 状态机错误；需在真实训练 checkpoint 上报告
  token agreement rate 与 logit margin，不应把随机 head 的逐 token 一致作为生产验收条件。
