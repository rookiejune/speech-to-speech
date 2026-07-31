# S2ST 接下来要做的事

日期：2026-07-31  
状态：内部执行计划

## 1. 总体优先级

短期优先级是把 Stage 0 / Stage 1 做成强 baseline，而不是过早追求端到端一次前向 S2ST。

推荐执行顺序：

1. 以 BiCodec 为主复现或对齐 Spark-TTS，完成 Stage 0 TTS 验证；
2. 同步准备 Stable Codec / UniCodec 对照，验证单码本路线是否降低建模难度；
3. 在 Stage 0 成立后引入 ASR，验证 ASR、TTS、MT 能力是否能在同一模型中稳定共存；
4. 保留 Stage 1 的 ASR + MT + TTS 三级级联系统作为强 baseline；
5. 只有当 Stage 1 baseline 足够强时，再进入 Stage 2/3，讨论压缩为一次前向的价值。

## 2. Stage 0：TTS

目标：将 TTS 能力集成到 LLM，使模型能够根据文本 Prompt 直接生成 Codec Token。

需要做的事：

- 复现或对齐 Spark-TTS；
- 使用相同或相近的 BiCodec 配置；
- 建立 Stable Codec / UniCodec 的最小对照配置；
- 对特殊 Token 做消融；
- 比较强结构化输入、少量特殊 Token、纯文本 Prompt 三种接口。

需要回答的问题：

1. LLM 能否稳定完成文本到 Codec Token 的映射？
2. 不依赖强结构化特殊 Token 时，TTS 质量是否仍接近 Spark-TTS？
3. TTS 训练是否破坏 Qwen3 原有文本翻译能力？

通过条件：

- 语音可懂度、自然度和长文本稳定性接近 Spark-TTS；
- Codec Token 预测稳定，没有明显退化或崩溃模式；
- Qwen3 文本/MT 能力无明显破坏。

不通过时：

- 增加 TTS 专用层或轻量 Adapter；
- 保留少量必要特殊 Token；
- 在“保留文本能力”和“最大化 TTS 性能”之间寻找 Pareto 最优点。

## 3. Stage 1：TTS + ASR

目标：让同一个模型同时支持：

- 文本 → Codec Token：TTS；
- 语音 / Codec Token → 文本：ASR。

需要做的事：

- 在 Stage 0 基础上加入 ASR 任务；
- 设计 ASR/TTS 多任务采样比例；
- 比较直接输入 Codec Token 与引入 AuT 的效果；
- 对比 Spark-TTS、Qwen-TTS、Qwen-ASR；
- 验证文本 MT 能力是否保持。

需要回答的问题：

1. ASR 和 TTS 能否在同一套参数中稳定共存？
2. 语音输入是否会破坏 TTS 或 MT 能力？
3. 直接 Codec Token 输入是否足够，还是需要 AuT？

通过条件：

- ASR 接近同规模语音模型；
- TTS 接近 Stage 0；
- 原始文本能力和机器翻译能力无明显退化；
- ASR、MT、TTS 可以通过同一个模型三次前向形成闭环。

不通过时：

- 引入 AuT 作为语音输入侧 encoder；
- 冻结部分 Backbone；
- 调整 ASR/TTS/文本任务采样比例；
- 为语音输入或输出增加轻量 Adapter。

## 4. Stage 2：S2TT + T2ST

目标：验证跨语言、跨模态能力迁移。

- S2TT：源语言语音 → 目标语言文本，逻辑上等价于 ASR + MT；
- T2ST：源语言文本 → 目标语言语音，逻辑上等价于 MT + TTS。

需要做的事：

- 构建 S2TT 与 T2ST 数据和任务格式；
- 用 Stage 1 级联系统生成 teacher 输出；
- 比较直接生成结果和 Stage 1 级联结果；
- 监控 Qwen3 原有文本 MT 能力是否退化；
- 分别验证 S2TT 和 T2ST，再考虑联合训练。

需要回答的问题：

1. 语音输入能否调用 LLM 原有翻译能力？
2. 翻译后的语义能否直接驱动 Codec Token 输出？
3. 多模态翻译训练是否破坏原有文本能力？

通过条件：

- S2TT 接近 Stage 1 ASR + Qwen3 MT；
- T2ST 接近 Qwen3 MT + Stage 1 TTS；
- 实体、数字和否定关系稳定；
- Stage 1 ASR/TTS 能力没有明显退化。

不通过时：

- 用 Stage 1 级联系统蒸馏 S2TT/T2ST；
- 分别蒸馏 S2TT 与 T2ST，再联合训练；
- 调整多任务采样比例；
- 冻结部分 Backbone 或增加 Adapter。

## 5. Stage 3：S2ST

目标：把 Stage 1 的 ASR + MT + TTS 三次前向压缩为一次前向。

需要做的事：

- 以 Stage 1 三级级联系统作为 teacher；
- 训练源语言语音 / Codec Token → 目标语言 Codec Token 的 student；
- 保留目标语言文本或语义辅助监督；
- 比较 Stage 3 单次前向和 Stage 1 三级级联；
- 重点评估质量损失与延迟收益。

需要回答的问题：

1. 单次前向 S2ST 相对三级级联损失多少翻译质量？
2. 单次前向 S2ST 相对 Stage 1 TTS 损失多少语音质量？
3. 延迟、RTF、显存和计算量收益是否足够大？
4. 是否保留说话人、韵律和情绪信息？

通过条件：

- 翻译准确性接近 Stage 2 S2TT；
- 语音质量接近 Stage 1 TTS；
- 相对三级级联有显著效率收益；
- 质量损失和效率收益形成可接受 Pareto。

不通过时：

- 保留 Stage 1 三级级联系统作为主系统；
- Stage 3 作为长期探索；
- 优先优化 teacher 质量和蒸馏策略，而不是盲目扩大 student。

## 6. 近期最小可执行清单

第一批应优先落地：

1. 确认 BiCodec/Spark-TTS 复现实验入口；
2. 固化 Stage 0 数据格式和 Prompt 格式；
3. 做特殊 Token 消融设计；
4. 建立 TTS 自动评估与固定样本人工听测流程；
5. 建立 Qwen3 文本 MT 回归集，避免 TTS 训练把文本能力打坏；
6. 准备 Stable Codec / UniCodec 的最小对照配置。

完成这批之后，再进入 Stage 1 的 ASR + TTS 共存实验。
