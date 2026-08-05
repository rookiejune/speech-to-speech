# S2ST 路线简单介绍

日期：2026-07-31  
更新：2026-08-04
状态：内部讨论稿

## 1. 背景

最终目标是优化 LLM 直接完成 S2ST（Speech-to-Speech Translation）的质量和效率。为避免直接训练时无法定位输入理解、翻译或语音生成侧的问题，路线按能力依赖逐步增加任务：

1. LLM 能否稳定生成语音 Codec Token；
2. LLM 能否同时具备 ASR、MT 和 TTS 能力；
3. LLM 原有文本翻译能力能否被语音输入和语音输出调用；
4. 如何用分解任务持续监督直接 S2ST，并避免端到端优化破坏已有能力。

## 2. V1 方案为什么失败

V1 方案失败的核心原因是：我们对 Codec 表征能力、LongCat 语义码可翻译性，以及「语义码 → 声学码」生成难度的估计过于乐观。

目前只能确认：使用 LongCat 语义码建模时，训练 Loss 低于常规机器翻译任务 Loss。但较低 Loss 不足以说明 LongCat 语义码：

- 完整保留翻译所需语言信息；
- 具备稳定跨语言映射能力；
- 能够支持后续高质量声学生成。

从语义码显式生成声学码的路线，在 RVQ 自回归预测和 Flow Matching 两类实验中均未达到预期：

- RVQ 各层声学码预测准确率较低，部分设置下甚至不如直接预测声学码；
- Flow Matching 训练 Loss 可以较低，但最终生成质量与 Codec 重建上限仍有明显差距；
- 时间偏移、连续表征偏差、音素时长和韵律不一致等误差，会在 Codec 解码阶段被放大。

因此，V1 的阶段性结论是：暂时不继续把主要预算投入 LongCat 显式「语义码 → 声学码」路线。

## 3. 新路线的核心判断

新方案应尽量绕过显式语义码到声学码生成，把主要建模难度转移到统一 LLM 上。

传统 S2ST 可写成三级级联：

~~~mermaid
flowchart LR
    A[源语言语音] -->|ASR| B[源语言文本]
    B -->|LLM / MT| C[目标语言文本]
    C -->|TTS| D[目标语言语音]
~~~

本方案希望逐步变成 Codec-based 统一建模：

~~~mermaid
flowchart LR
    A[语音 / Codec Token] --> L[统一 LLM]
    T[文本 Prompt] --> L
    L --> O1[文本输出]
    L --> O2[Codec Token 输出]
    O2 --> D[Codec Decoder]
    D --> S[语音输出]
~~~

需要注意：去掉显式 ASR/TTS 模块，不意味着系统不需要 ASR/TTS 能力，而是要把这些能力内化到 LLM 中。

## 4. 外部 generator plugin 前置依赖

Codec 预训练、筛选、重建上限评估和 generator artifact 导出由仓库外的
`semantic-acoustic-generator` 负责。本仓库不在 staged curriculum 中重新训练 codec，也不通过其他
codec 训练入口绕过这条边界。

Stage 0-3 的 Flow/RVQ S2S 训练在 composition 时显式读取外部导出的、frame-aligned
`AcousticGeneratorArtifact`。正式 wrapper 通过 `SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT` 接收路径并转写到
`model.acoustic.init_artifact`。generator plugin 的 conditioner、训练数据和导出流程不进入 S2S model；本仓库只
训练 hidden-state condition adapter、LLM 与 joint task objectives。artifact 的 route、decoder 配置、
frame layout 和 backend metadata 必须在进入正式训练前通过校验。

## 5. 四阶段路线

~~~mermaid
flowchart LR
    S0["Stage 0<br/>TTS + MT"] --> S1["Stage 1<br/>ASR + TTS + MT"]
    S1 --> S2["Stage 2<br/>S2TT + T2ST<br/>+ ASR + TTS + MT"]
    S2 --> S3["Stage 3<br/>S2ST + S2TT + T2ST<br/>+ ASR + TTS + MT"]

    S0 -.生成与文本能力保持.-> A[建立语音生成能力]
    S1 -.语音与文本双向对齐.-> B[形成统一基础模型]
    S2 -.跨模态分解监督.-> C[建立直接翻译能力]
    S3 -.以直接 S2ST 为主任务.-> D[优化端到端语音翻译]
~~~

各阶段的训练任务和损失权重为：

| 阶段 | 训练任务 | 损失权重 | 阶段目的 |
| --- | --- | --- | --- |
| Stage 0 | TTS + MT | TTS 0.9，MT 0.1 | 建立语音生成能力，同时保持文本翻译能力 |
| Stage 1 | ASR + TTS + MT | ASR 0.45，TTS 0.45，MT 0.1 | 建立语音与文本双向映射，同时保持文本翻译能力 |
| Stage 2 | S2TT + T2ST + ASR + TTS + MT | ASR、S2TT、TTS、T2ST 各 0.225，MT 0.1 | 用基础任务和跨模态分解任务建立完整翻译链路 |
| Stage 3 | S2ST + S2TT + T2ST + ASR + TTS + MT | S2ST 0.7；ASR、S2TT、TTS、T2ST 各 0.05；MT 0.1 | 以直接 S2ST 为主目标，其他任务提供能力保持和分解监督 |

这里的权重是多任务混合 Loss 的初始权重，不默认等同于任务采样比例；采样不均衡时应同时报告采样概率与 Loss 权重，避免重复放大或缩小任务的有效贡献。如不同任务的 Loss 尺度差异明显，应先做归一化或梯度量级校准，再按验证集结果调整。Stage 0-2 不是独立终点，而是 Stage 3 的能力初始化和可诊断基线。Stage 3 中 S2ST 占主导，ASR、S2TT、TTS、T2ST 用于约束输入理解、跨语言映射和语音生成能力，MT 用于保持 LLM 原有文本翻译能力。

## 6. 当前内部判断

直接 S2ST 是唯一最终优化目标，三级级联和 S2TT/T2ST 路径用于提供 teacher、质量上限和故障定位，不替代 Stage 3 的端到端目标。

如果前一阶段未通过 gate，应先修复对应基础能力再进入下一阶段；进入 Stage 3 后，则以直接 S2ST 指标作为主要模型选择依据，并把辅助任务指标作为退化约束。即使三级级联暂时质量更高，也应将其作为 baseline 和蒸馏来源，而不是改变最终目标。
