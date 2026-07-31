# S2ST 内部结论文档

日期：2026-07-31  
状态：内部讨论稿入口

这组文档用于沉淀当前 LLM-based S2ST 路线的阶段性判断，并把“为什么做、接下来做什么、怎么判断成功”拆开维护。

## 文档结构

1. [简单介绍](plan/overview.md)：说明 V1 失败原因、当前路线判断、Codec 选型和四阶段总体思路。
2. [接下来要做的事](plan/next_steps.md)：列出近期执行顺序、每个阶段的任务、gate 和后备方案。
3. [具体指标](plan/metrics.md)：定义 ASR、MT/S2TT、TTS/T2ST/S2ST、效率和 LLM 能力保持的评估指标。

## 当前核心结论

短期不要直接把“端到端一次前向 S2ST”作为第一目标。更稳妥的路线是先做出统一模型内的 ASR + MT + TTS 强闭环，再逐步压缩为 S2TT、T2ST 和 S2ST。

V1 的 LongCat 显式「语义码 → 声学码」路线暂时不应继续追加主要预算；下一阶段以 BiCodec 为主方案，Stable Codec 和 UniCodec 作为对照。
