# Conclusion

## 适用范围

本页最后一次真实实验更新止于 009（2026-07-14），当时的结论汇总代码快照为
`cec3a6c`。此后 model/runtime/data/generation/DDP 契约和按模态 token CE 均有调整；在当前
复验项完成前，下列数值和闭环结论作为历史基线保留，不作为当前 `HEAD` 的回归验收结果
（[待复验项，lines 15-29](todo.md#L15-L29)）。新复验应建立下一组一一对应的 schedule/result，
通过后再更新本页。

## 已验证结论

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
