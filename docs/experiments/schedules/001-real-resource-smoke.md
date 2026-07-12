# 001 Real Resource Smoke

## 目标

在 121 的真实 WMT19 TTS store、Qwen3、LongCat codec 和 Flash Attention 2 环境中，
验证 P1 数据与模型契约不只在 fake closure 中成立。

## 验收项

- 标准 store 的 LongCat view 是统一 `[T, K]` Tensor，batch 后保持 `[B, T, K]`。
- TTS 与 S2ST 各完成 forward、backward 和一次 optimizer step。
- semantic backbone 使用 bf16、CUDA 和 `flash_attention_2`。
- ground-truth semantic/acoustic 输出可通过真实 LongCat decoder 合成 finite waveform。
- S2ST prompt 可完成一次短 semantic generation、flow sampling 和 waveform decode。

## 限制

- 本次只做单 batch smoke，不验证收敛、质量或长时间训练稳定性。
- generation 使用当前过渡实现，只验证短单样本闭环；KV cache、source condition
  全程保持和标准变长 batch generation 仍按 Generation 待办推进。
