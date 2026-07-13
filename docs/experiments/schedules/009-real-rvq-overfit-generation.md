# 009 Real RVQ Overfit And Generation

## 目标

验收 8 层 Qwen RVQ acoustic decoder 在真实 WMT19/LongCat 固定样本上的训练、
acoustic sampling、waveform decode 和生成耗时，补齐与 Flow 相同的评估边界。

## 配置

- TTS 与 S2ST 分别使用 train split 第 0 条固定样本。
- `Qwen/Qwen3-0.6B` semantic backbone，LongCat `16k_4codebooks`。
- RVQ decoder 为 8 层，其他 optimizer、样本顺序和 100-step 预算沿用 002。
- 先运行 2-step smoke，再运行 100-step formal overfit。
- 固定间隔记录 semantic/causal LM loss、feature MSE、STFT distance、waveform
  RMS/peak、acoustic sampling seconds 和 RTF。

## 验收

- forward/backward、optimizer step、RVQ cached sampling 和 waveform decode 全部成功。
- semantic 与 causal LM objective 在固定样本上收到有效优化信号。
- TTS/S2ST 都产出 finite waveform 和完整 evaluation metrics。
- 报告 decoder 参数量、acoustic sampling RTF 和 waveform 指标；不将 RVQ token
  CE 数值与 Flow loss 直接比较。
