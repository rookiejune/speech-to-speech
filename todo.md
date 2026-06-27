# Speech-to-Speech TODO

## 开发顺序

1. 真实数据 smoke
   - 在 FDU 121 的 `fleurs-full-longcat` 上跑小规模 smoke。
   - 确认真实字段、BPE 缓存、batch 构造和单步训练都能工作。

2. translation 训练闭环
   - 跑混合任务训练。
   - 确认 autoregression 和 translation batch 都能进入同一个 model forward。
   - 检查两类任务的 loss 位置正确。

3. semantic generate 和简单评估
    - 实现 semantic token 生成的最小接口。
    - 做 sanity check：能从 prompt/source 生成 target semantic ids。
    - 确认 BPE decode / LongCat semantic ids 还原路径清晰。

4. DiT acoustic
    - 用真实 acoustic target features 跑通独立 acoustic loss。
    - 在 batch 契约里显式携带 source acoustic features 和 target acoustic features。
    - 实现 source acoustic features 池化到 DiT `acoustic_condition` 的 helper，并在 acoustic loss 中接入。
    - 增加 acoustic condition dropout 配置，训练时按概率将 source acoustic condition 替换为 null acoustic condition。
    - 实现 CFG 推理路径：分别跑 conditional / unconditional DiT forward，并用 guidance scale 融合速度预测。
    - 再考虑 semantic LM 和 acoustic loss 联合训练。
    - 接入真实 acoustic feature generator / DiT sampler，跑通 full-sequence waveform 生成闭环。
    - 对角线并行推理先预留接口或 scheduler 边界，等 full 路径能出声音并完成时延/显存 profile 后再实现。
