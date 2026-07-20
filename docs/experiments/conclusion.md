# Conclusion

## 适用范围

本页最新真实实验是 010（2026-07-21），其 LongCat codec oracle 结论对应代码快照
`9127e62`。010 只复验 Flow/RVQ oracle；008/009 之后 model/runtime/data/generation 和按模态
token CE 仍有调整，因此相应 generation/overfit 数值作为历史基线保留，未完成的当前复验项见
[todo, lines 15-28](todo.md#L15-L28)。

## 已验证结论

- 真实 Qwen3/LongCat 上，Flow 与 RVQ oracle 的单卡 fixed-sample、两卡静态 DDP + LBA 均完成
  2-step forward/backward/optimizer 和完整 callback；RVQ 静态 DDP 没有 unused-parameter 错误。
  该 smoke 只验证执行契约，不支持质量或收敛结论
  （[010 result, lines 20-40](results/010-codec-oracle-flow-rvq-smoke.md#L20-L40)）。
- 真实 Qwen3/LongCat 变长 batch 4 的 prompt、source acoustic frames、KV cache 和
  waveform decode 在 float32 下完成逐请求 token parity；该短生成 probe 的吞吐为
  serial 的 1.78x，peak allocated 只增加约 22 MB
  （[008 result, lines 27-46](results/008-real-batch-generation-benchmark.md#L27-L46)）。
- 8 层 Qwen RVQ decoder 在真实 TTS/S2ST 固定样本上完成 100-step 训练；
  semantic objective 接近记忆，acoustic causal CE 最后 20-step 均值相对首窗口
  下降约 23%，但 feature/STFT 轨迹不支持 waveform 质量改善结论
  （[009 result, lines 47-60](results/009-real-rvq-overfit-generation.md#L47-L60)）。
- 同一 RVQ formal run 训练后的 TTS/S2ST greedy cached generation 均生成 36
  acoustic frames 和 2.16s finite waveform，端到端 RTF 分别为 0.503/0.494；该结论
  只验证固定样本执行契约，不表示泛化质量
  （[009 result, lines 68-79](results/009-real-rvq-overfit-generation.md#L68-L79)）。
