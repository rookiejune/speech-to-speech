# Conclusion

## 适用范围

本页最新真实实验是 011 的 P0 fixed-sample 子项（2026-07-21），对应远端代码快照
`d5f6902`；它只通过真实 Qwen/native/RVQ 的单卡训练与 teacher-forced acoustic decode，
端到端 generation gate 仍失败，因此 011 P0 尚未完成。010 的 LongCat codec oracle 结论对应
代码快照 `9127e62`。008/009 之后 model/runtime/data/generation 和按模态 token CE 仍有调整，
因此相应 generation/overfit 数值作为历史基线保留，未完成项见
[todo, lines 20-34](todo.md#L20-L34)。

## 已验证结论

- 真实 Qwen3-0.6B、LongCat native token 与 8 层 RVQ decoder 上，TTS/S2ST fixed-sample
  均完成 2-step forward/backward/optimizer；两条 total、audio token CE 和各 RVQ codebook CE
  均下降。teacher-forced RVQ sampling 在 3 个记录点、每点 4 个 seed 上均可 decode 2.16s
  finite waveform，但 feature MSE 非单调，该 smoke 不支持收敛或质量结论
  （[011 result, lines 24-49](results/011-qwen-rvq-staged-joint-training.md#L24-L49)）。
- 同一 011 run 的训练后端到端 generation 尚未通过：Python 3.12 runtime Protocol 对真实
  registered `nn.Module` backbone 产生 false negative，两条任务退出码为 1，未写出
  `generation.json`/`metrics.json`。这是真实 P0 接口失败，不能由 training-only metrics
  或 teacher-forced waveform 代替
  （[011 result, lines 51-89](results/011-qwen-rvq-staged-joint-training.md#L51-L89)）。
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
