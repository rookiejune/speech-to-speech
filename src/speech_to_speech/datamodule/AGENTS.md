# datamodule

## 对外职责

`datamodule` 负责把训练数据集中的原始 LongCat semantic codes 转成模型训练 batch。它向外部提供的是数据契约和 batch 构造能力，不暴露数据集内部字段遍历细节。默认训练数据集由 `speech_to_speech.dataset` 从 `zhuyin.datasets.wmt19_tts.wmt19_tts_longcat()` 取得。

对外能力：

- 定义 source autoregression、target autoregression、source-to-target translation 和 target-to-source translation task sample。
- 从 anydataset `Sample` 中提取双向 autoregression 和双向 translation task example。
- 将 `AudioView.LONGCAT` 的原始 `semantic_codes` 包装成单 codebook frame，并编码成 LongCat BPE ids。
- 根据配置启用 `autoregression` 和 `translation` 任务族，并按 task family 权重确定展开比例。
- 构造模型可消费的 `CausalLMBatch`，包含 `input_ids`、`attention_mask`、`labels`、`logits_to_keep` 和训练监控用 task family。
- 构造生成侧 `GenerationBatch` prompt，用于模型生成 acoustic condition hidden states。

## 输入输出契约

输入样本使用 anydataset canonical sample，至少包含：

- `(Role.SOURCE, Modality.AUDIO)` 持有 `AudioView.LONGCAT`
- `(Role.TARGET, Modality.AUDIO)` 持有 `AudioView.LONGCAT`
- 两侧 `AudioView.LONGCAT` 是至少包含 `semantic_codes` 和 `acoustic_codes` 的 dict

训练契约不读取文本和语言字段。文本只属于训练前过滤、数据准备或评估辅助链路。

task example 转换：

- source autoregression 读取 source 侧音频，并转成 `AutoregressionExample(audio_ids)`。
- target autoregression 读取 target 侧音频，并转成 `AutoregressionExample(audio_ids)`。
- source-to-target translation 转成 `TranslationExample(source_ids, target_ids)`。
- target-to-source translation 转成 `TranslationExample(source_ids=target_ids, target_ids=source_ids)`。
- BPE 训练语料使用 source/target pair 转成 `SpeechPair`。
- sample logging 使用 `semantic_codes` 的 BPE frame 展开结果和原始 `acoustic_codes` 解码音频；LongCat 当前要求单 codebook semantic frame，展开后 semantic length 必须和 acoustic time 一致。

speech-to-speech 不定义额外底层 dataset wrapper。datamodule 内部让 anydataset 负责读取和
rank/worker 分片，并用 `TaskSampleStream` 在每条 source/target raw sample 上按权重展开训练
task sample。DataLoader worker 只负责读取 raw sample 和展开 task sample；BPE encode、
prompt/label 构造留在主进程完成，`inputs_embeds` 由模型 forward 内部生成。

输出 batch 的约束：

- `input_ids` 使用全局 token id space。
- `attention_mask` 标记 padding 位置。
- audio segment 使用 idspace special token `BOA` 和 `EOA` 表达边界；LongCat BPE audio block 只包含真实 BPE token。
- `labels` 只覆盖需要计算 loss 的目标 token，包括目标 audio segment 的 `BOA` 和 `EOA`。
- `source_audio` / `target_audio` 可选携带 batch 对齐后的原始 LongCat semantic/acoustic codes 和 mask；semantic loss 不依赖它们，acoustic loss 和 feature 转换从这里读取。
- prompt 和 user/assistant 结构 token 默认不计算 loss。
- dataloader 对外只返回统一的 `CausalLMBatch`；`task_family` 只用于 loss/计数日志，不参与模型输入语义。
- Qwen3 文本模板必须通过 tokenizer 的 `apply_chat_template` 生成；不要手写 `<|im_start|>` / `assistant` / thinking token 序列。

生成 batch 的约束：

- `GenerationBatch` 只包含 `input_ids` 和 `attention_mask`，不包含 labels。
- autoregression 生成 prompt 以目标 `BOA` 结束，可选 prefix 只追加已有 audio BPE token，不追加 `EOA`。
- translation 生成 prompt 包含 source audio segment，并以 target `BOA` 结束。
- 生成 prompt 仍使用 Qwen3 chat template；不要为生成路径另写对话模板。

## 开发边界

- Do: 在数据层处理 BPE 缓存和样本结构归一；Don't: 让 batch builder 隐式猜数据集缺失字段。
- Do: batch builder 只消费 task example 并表达模板、任务结构和 label 位置；Don't: 在 batch builder 里做过滤、数据集扫描或 tokenizer 训练。
- Do: 让 anydataset 继续负责 rank/worker 分片；Don't: 在 task sample stream 里重复实现 DDP 或 worker shard。
- Do: 通过 `runtime.py` 获取 tokenizer/codec；Don't: 直接在本模块散落 `from_pretrained`。
- Do: 让模型模块只消费 `CausalLMBatch` 契约；Don't: 让模型依赖 Anydataset 样本结构。
