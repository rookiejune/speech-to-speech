# S2ST 接下来要做的事

日期：2026-07-31  
更新：2026-08-04
状态：内部执行计划

## 1. 总体优先级

最终目标是优化直接 S2ST。Stage 0-2 用于逐步建立 TTS、ASR 和跨模态翻译能力，并为 Stage 3 提供初始化、teacher 和可诊断 baseline；进入 Stage 3 后，模型选择以直接 S2ST 为主，辅助任务只用于能力保持和分解监督。

推荐执行顺序：

1. 从外部 `semantic-acoustic-generator` 获取并校验 frame-aligned generator artifact，完成 Stage 0 TTS + MT 验证；
2. 完成 Stage 0 LoRA merge/export 和跨阶段 weight-only handoff 后引入 ASR，完成 Stage 1
   ASR + TTS + MT 联合训练；
3. 在 Stage 1 基础上加入 S2TT 和 T2ST，完成 Stage 2 的跨模态分解训练；
4. 在 Stage 3 加入并重点优化直接 S2ST，以 Stage 1/2 路径作为 teacher、baseline 和辅助监督。

## 2. Stage 0：TTS + MT

目标：将 TTS 能力集成到 LLM，使模型能够根据文本 Prompt 直接生成 Codec Token，同时通过 MT replay 保持原有文本翻译能力。

初始损失权重：TTS 0.9，MT 0.1。

需要做的事：

- 固定外部 generator artifact 的版本、route、frame layout 和 decoder contract；
- 通过 `SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT` 显式传入该 artifact，并完成 composition smoke；
- 对特殊 Token 做消融；
- 比较强结构化输入、少量特殊 Token、纯文本 Prompt 三种接口。
- 固定 MT 数据和评估集，验证 0.1 权重能否防止文本翻译能力退化。

需要回答的问题：

1. 在外部 acoustic generator contract 固定后，LLM 能否稳定完成文本到 Codec Token 的映射？
2. 不依赖强结构化特殊 Token 时，TTS 质量是否仍接近 Spark-TTS？
3. TTS + MT 联合训练能否在建立语音生成能力的同时保持 Qwen3 原有文本翻译能力？

通过条件：

- 语音可懂度、自然度和长文本稳定性接近 Spark-TTS；
- Codec Token 预测稳定，没有明显退化或崩溃模式；
- Qwen3 文本/MT 能力无明显破坏。

不通过时：

- 增加 TTS 专用层或轻量 Adapter；
- 保留少量必要特殊 Token；
- 在“保留文本能力”和“最大化 TTS 性能”之间寻找 Pareto 最优点。

## 3. Stage 1：ASR + TTS + MT

目标：让同一个模型同时支持：

- 文本 → Codec Token：TTS；
- 语音 / Codec Token → 文本：ASR。
- 文本 → 目标语言文本：MT 能力保持。

初始损失权重：ASR 0.45，TTS 0.45，MT 0.1。

需要做的事：

- 在 Stage 0 基础上加入 ASR 任务；
- 验证 Stage 0 LoRA merge/export 与 weight-only 初始化，切换参数策略时重新创建 optimizer；
- 按 ASR 0.45、TTS 0.45、MT 0.1 组合多任务 Loss，并独立记录任务采样比例；
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

## 4. Stage 2：S2TT + T2ST + ASR + TTS + MT

目标：在保留 ASR、TTS 和 MT 的同时，加入跨语言、跨模态分解任务，为直接 S2ST 建立输入理解、翻译和语音生成监督。

- S2TT：源语言语音 → 目标语言文本，逻辑上等价于 ASR + MT；
- T2ST：源语言文本 → 目标语言语音，逻辑上等价于 MT + TTS。

初始损失权重：ASR、S2TT、TTS、T2ST 各 0.225，MT 0.1。

需要做的事：

- 构建 S2TT 与 T2ST 数据和任务格式，同时保留 Stage 1 的 ASR、TTS 和 MT 数据；
- 用 Stage 1 级联系统生成 teacher 输出；
- 比较直接生成结果和 Stage 1 级联结果；
- 监控 Qwen3 原有文本 MT 能力是否退化；
- 先分别验证 S2TT 和 T2ST，再按统一权重联合训练全部五个任务。

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

## 5. Stage 3：S2ST + S2TT + T2ST + ASR + TTS + MT

目标：以源语言语音 / Codec Token → 目标语言 Codec Token 的直接 S2ST 为主要优化目标；保留 S2TT、T2ST、ASR、TTS 和 MT，提供能力保持和分解监督。

初始损失权重：S2ST 0.7；ASR、S2TT、TTS、T2ST 各 0.05；MT 0.1。

需要做的事：

- 以 Stage 1 三级级联和 Stage 2 S2TT/T2ST 路径作为 teacher；
- 训练源语言语音 / Codec Token → 目标语言 Codec Token 的直接 S2ST；
- 保留 ASR、S2TT、TTS、T2ST 和 MT 数据，并按辅助权重联合训练；
- 比较 Stage 3 直接 S2ST、Stage 1 三级级联和 Stage 2 分解路径；
- 重点优化直接 S2ST 的翻译准确性、语音质量和生成稳定性，同时评估延迟收益。

需要回答的问题：

1. 直接 S2ST 的翻译质量如何接近或超过三级级联和 Stage 2 分解路径？
2. 直接 S2ST 的语音质量如何接近 Stage 2 T2ST / TTS？
3. 辅助任务是否有效保持输入理解、翻译和语音生成能力？
4. 是否保留说话人、韵律和情绪信息？
5. 延迟、RTF、显存和计算量收益是否足够大？

通过条件：

- 翻译准确性接近 Stage 2 S2TT 和三级级联；
- 语音质量接近 Stage 2 T2ST / TTS；
- 相对三级级联有显著效率收益；
- ASR、S2TT、TTS、T2ST 和 MT 均无不可接受退化。

不通过时：

- 按 S2ST 错误来源回查 Stage 2 的 S2TT/T2ST 和 Stage 1 的 ASR/TTS 能力；
- 优化 teacher 质量、数据配对和蒸馏策略；
- 校准辅助任务的 Loss/梯度尺度，在保持 S2ST 0.7 主权重的前提下调整辅助任务。

## 6. 近期最小可执行清单

第一批应优先落地：

1. 固化外部 generator artifact 的版本和校验信息；
2. 固化 Stage 0 TTS/MT 数据格式、Prompt 格式和 0.9/0.1 权重；
3. 做特殊 Token 消融设计；
4. 建立 TTS 自动评估与固定样本人工听测流程；
5. 建立 Qwen3 文本 MT 回归集，避免 TTS 训练把文本能力打坏。

完成这批之后，再进入 Stage 1 的 ASR + TTS + MT 共存实验。
