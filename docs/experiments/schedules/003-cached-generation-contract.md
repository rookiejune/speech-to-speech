# 003 Cached Generation Contract

## 目标

完成 P3 的单样本推理闭环，使真实推理不依赖训练 `ModelBatch`，并保证 semantic token、
acoustic output 与 waveform 来自同一次 cached generation。

## 验收

- inference service 接收独立的无 padding request，返回 token、acoustic features 和 waveform。
- cache 首步编码完整多模态 prompt，后续只输入一个新 token。
- source acoustic condition 通过 cache 持续影响后续 token。
- audio token 采样时在线收集 predictor hidden，不在结束后重新 full forward。
- tiny Qwen3 的 cache 与非 cache greedy sequence 一致。
- TaskSampleLogger 对每组样本只调用一次 generation。

## 限制

- 本轮 model primitive 只支持单样本，上层 service 逐 request 调用。
- 本轮不实现 padded batch、每行独立 stop 或 acoustic frame mask。
- 本地测试不替代真实 Qwen3/LongCat cached generation/decode smoke。
