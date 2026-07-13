# Conclusion

- 固定同一条真实样本时，TTS 与 S2ST 的 semantic objective 都能在 100 steps 内接近
  记忆；加入 source semantic/acoustic condition 后仍保持可优化性
  （[002 result, lines 49-58](results/002-single-batch-overfit.md#L49-L58)）。
- 同一实验中，flow matching 的后 20-step 均值相对前 20 steps 分别下降约 28%（TTS）
  和 30%（S2ST），说明两个路径都能收到有效优化信号
  （[002 result, lines 51-58](results/002-single-batch-overfit.md#L51-L58)）。
- 完整 `SpeechToSpeechFlowModel` 在 LongCat codec/random initialization 下均完成真实
  prepared-code 的 acoustic flow forward/backward、optimizer、sample 与 waveform decode
  2-step 闭环；该 smoke 不支持初始化优劣或收敛性结论
  （[005 result, lines 5-18](results/005-acoustic-oracle-codec-screening.md#L5-L18)）。
- 相同初始化、数据、optimizer 和 100-step 预算下，REPA 的 STFT log-magnitude 略有改善，
  但 feature MSE 与 spectral convergence 恶化，且中间 checkpoints 的相对方向不一致；
  当前没有稳定的非训练指标增益，REPA 不设为默认
  （[007 result, lines 58-85](results/007-flow-repa-comparison.md#L58-L85)）。
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
