# 009 Real RVQ Overfit And Generation Result

## 环境

- 日期：2026-07-14。
- 最终 formal run：125，NVIDIA RTX 3090 物理 GPU 2，PyTorch 2.11 + CUDA 13。
- Qwen checkpoint 复制到机器本地 `/tmp`，使用 eager attention；训练输出也写本地
  `/tmp`，避免当时 98% 容量下的 NAS mmap/page wait。
- 数据：WMT19/LongCat train split 第 0 条固定样本，TTS/S2ST 各 100 steps。
- 模型：`Qwen/Qwen3-0.6B` + 8-layer Qwen RVQ decoder，bfloat16 mixed
  precision。
- 原始 metrics 保存在 `debug/speech-to-speech/009-real-rvq-overfit-generation/`。

## LongCat Packed ID Contract

首次 smoke 在 RVQ embedding 报 acoustic ID 越界。LongCat 每个 acoustic RVQ codebook
使用两个 90-entry factor codebook，对外 token 是
`index_a * 90 + index_b`，因此 packed vocabulary 为 8100，不是 90。Prepared sample
的 acoustic IDs 均在 `[0, 8099]`。

anytrain `LongCat.codebook_sizes` 已修正为 `(8192, 8100, ...)`，并用 packed IDs
`8099/8006/7729` 验证 feature conversion。修复提交：`977de04`。

## Smoke

2-step TTS smoke 完成 forward/backward、optimizer、RVQ cached sampling 和 waveform
decode：

- semantic loss：`18.19 -> 14.53`。
- causal LM loss：`9.24 -> 9.16`。
- 2.16s waveform 的 acoustic sampling 为 `31–36 ms`，RTF 约 `0.015`。
- waveform 全部 finite。

S2ST 使用每步 SampleLogger 时，第 7 step 的未训练 semantic head 直接生成
EOA，generation 按契约报“无 codec-decodable token”。Formal run 禁用每步
SampleLogger，固定 teacher-forcing acoustic evaluation 每 20 steps 执行。

## Overfit

| Task | Metric | First | First 20 mean | Last | Last 20 mean | Last/first window |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TTS | semantic | 18.250 | 5.167 | 0.029 | 0.035 | 0.0068 |
| TTS | causal LM | 9.239 | 8.822 | 6.714 | 6.801 | 0.7710 |
| S2ST | semantic | 17.938 | 5.104 | 0.028 | 0.033 | 0.0064 |
| S2ST | causal LM | 9.243 | 8.780 | 6.691 | 6.783 | 0.7726 |

两个任务的 semantic objective 均已接近记忆固定样本。RVQ causal CE 的最后
20-step 均值相对最初窗口下降约 23%，收到有效优化信号，但在 100 steps
内远未记忆 acoustic packed IDs。

## Acoustic Evaluation

- acoustic decoder：`184,031,980` parameters。
- 完整模型：`856,846,828` parameters，其中 `856,845,804` trainable。
- 2.16s target 的 RVQ acoustic sampling 稳态约 `28–38 ms`，RTF 约
  `0.013–0.018`。
- TTS feature MSE：`2.357 -> 2.366`；S2ST：`2.384 -> 2.339`。
- TTS STFT spectral convergence：`1.298 -> 1.421`；S2ST：`1.262 -> 1.325`。
- 两个任务所有评估 waveform 都 finite，但 feature/STFT 轨迹非单调，不支持
  100-step waveform 质量改善结论。

## 结论

- RVQ 的真实训练、packed-code sampling、feature conversion 和 waveform decode 闭环通过。
- 100 steps 足以记忆 semantic target，但只使 acoustic causal CE 下降约 23%；需更长
  训练或单独 decoder optimization 才能评估 waveform 改善。

## 训练后生成

formal training 结束后在同一进程内对固定样本执行 greedy cached
semantic-to-waveform generation：

| Task | Audio BPE tokens | Frames | Feature shape | Duration | Elapsed | RTF |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| TTS | 1 | 36 | `[36, 1024]` | 2.16s | 1.086s | 0.503 |
| S2ST | 1 | 36 | `[36, 1024]` | 2.16s | 1.068s | 0.494 |

两条路径都生成 token `208296`，features 和 `[1, 34560]` waveform 全部
finite。该结果验证固定样本记忆后的完整执行契约，不表示泛化生成质量。
