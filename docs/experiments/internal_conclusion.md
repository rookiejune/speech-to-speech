# S2ST 内部结论文档

日期：2026-07-31  
状态：内部讨论稿入口

这组文档用于沉淀当前 LLM-based S2ST 路线的阶段性判断，并把“为什么做、接下来做什么、怎么判断成功”拆开维护。

## 文档结构

1. [简单介绍](plan/overview.md)：说明 V1 失败原因、外部 SAC 边界和四阶段总体思路。
2. [接下来要做的事](plan/next_steps.md)：列出近期执行顺序、每个阶段的任务、gate 和后备方案。
3. [具体指标](plan/metrics.md)：定义 ASR、MT/S2TT、TTS/T2ST/S2ST、效率和 LLM 能力保持的评估指标。

## 当前核心结论

最终目标是直接 S2ST。Stage 0-2 先建立 TTS、ASR、MT 和跨模态分解能力，Stage 3 以直接
S2ST 为主损失和模型选择目标，辅助任务只用于能力保持、分解监督和故障定位。

SAC 的 conditioner、acoustic generator 预训练和 artifact 导出由仓库外的
`semantic-acoustic-codec` 完成。本仓库在 composition 时通过
`model.acoustic.init_artifact` 显式消费 frame-aligned SAC artifact，不把 SAC 训练或其他 codec
训练入口纳入 staged joint 路线。
