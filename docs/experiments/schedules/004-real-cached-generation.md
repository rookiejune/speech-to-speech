# 004 Real Cached Generation

## 目标

在 121 的真实 Qwen3/LongCat S2ST 输入上验收 003 建立的单样本 cached generation
契约，确认 tiny/fake tests 覆盖的行为能迁移到真实 backbone、codec 和 waveform decode。

## 配置

- 使用 WMT19/LongCat train split 首条样本构造独立 S2ST request。
- semantic generation 使用 greedy，最多生成 2 个 token。
- cached 与 full-recompute 在各自运行前重置相同随机种子，使 flow source noise 一致。
- diagnostic 将 `acoustic_prompt_gate` 设为 1，确保未训练模型的 source acoustic prompt
  实际进入首步表示。

## 验收

- cached 首步输入完整 prompt，后续每步只输入一个 token；只有首步显式注入 source
  acoustic prompt，并且后续 step 携带 cache。
- full-recompute 每步输入完整增长序列并重新注入 source acoustic prompt。
- 两条路径 greedy semantic token 完全一致。
- 两条路径均生成 codec-decodable token、acoustic features 和 finite waveform。
- 记录 acoustic/waveform 最大绝对差、耗时与 peak allocated CUDA memory。

## 限制

- 随机初始化模型只验证执行契约，不评价 token、翻译或音频质量。
- bf16 cache 与 full-recompute 的 hidden 允许数值误差，acoustic/waveform 不要求逐元素完全相等。
- 本实验不验证 padded variable-length batch generation。
