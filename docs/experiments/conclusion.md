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
