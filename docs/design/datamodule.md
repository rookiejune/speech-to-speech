# datamodule

把 anydataset 的 raw sample 组织成模型可直接消费的 `ModelBatch`。数据契约与 position 语义的权威定义见 [总览 §2](../model-design.md)。

## 对外能力

- `types.Speech` / `types.SpeechPair`：raw sample 的逻辑视图。`SpeechPair.from_raw()`
  按 runtime 配置读取统一 `[frame, codebook]` codec view；LongCat 的第 0 个码本和后续
  码本只在这里解释为 semantic/acoustic，数据集层不携带这组语义。
- `types.Sample` / `types.ModelBatch`：单条与 batch 级训练输入。`ModelBatch.from_samples()` 完成 padding；mask 由 padding 值派生并缓存。
- `types.Task` / `types.Language`：任务与语言枚举。`Task` 是 source/target modality、paired
  语义与 instruction template 的唯一事实来源。
- `task.build_sample()`：按任务把 `SpeechPair` 组装成 `Sample`，包括 chat template、
  boa/eoa 包装、labels 与 acoustic positions 的生成。
- `Collator`：按权重策略采样任务并 collate。
- `DataModule`：Lightning 数据入口；`Config` 只保存 codec/dataloader 数据，`setup()` 通过
  `wmt19_tts_codec(config.codec)` 选择当前工程使用的具体 codec view；持有可更新的
  `Collator`，`set_strategy()` 由外部 callback 控制任务权重。重复调用 `setup()` 不会
  重新加载已持有的数据集。

## 输入输出

- 输入：`anydataset.types.Sample`（source/target 两个 role，audio + text 两个 modality）。
- 输出：`ModelBatch`，字段语义与 padding 约定见总览 §2.3，position 语义见总览 §2.4。

## 边界

- task 层负责所有序列拼接与位置生成；model 不搜索 source audio token，也不判断 acoustic prompt 边界。
- chat template 先渲染为字符串并在字符串层切分 source placeholder，再分别 tokenize
  prefix/suffix；不能在 token IDs 中搜索单独编码的 placeholder，因为 BPE 分词受相邻文本影响。
- target 为 audio 时，boa 是结构性前缀不参与监督：`labels[len(input_ids) + 1:] = response_ids[1:]`，即只监督 semantic BPE tokens 和 eoa。
- audio-target task 必须提供 semantic audio target；acoustic target 由 codec/profile 决定。
  unified-token codec 使用 `acoustic_ids=None`，text-target task 仍不允许 acoustic target。
- `semantic_frame_labels` 保存 codec-native target semantic codebooks，与 `acoustic_labels`
  共享 frame 轴；它只表达完整 codec target，不包含或预计算任何 REPA teacher feature。
- `ModelBatch.from_samples()` 负责检查 acoustic IDs/positions 成对存在、所有 batch/frame 轴对齐以及 task 执行签名一致；不把错误推迟到 model/loss。
- `ModelBatch` 只表达训练或 teacher-forcing evaluation，不表达缺少 target 的真实推理请求。
- `Collator` 的策略校验要求同一策略内所有任务的 source/target modality 一致，保证 DDP 中不同 rank 走相同模型路径。
- DataModule 构造时必须提供初始 strategy；stage callback 只在 epoch 边界更新同一个 collator，阶段切换不依赖 Trainer 重建 DataLoader。
- `DataModule.setup()` 在加载数据前校验 datamodule 与 runtime 的 codec 身份一致；不允许
  prepared codec view 和模型 codec 使用不同配置。
- `DataModule.train_samples()` 是 callback 按索引读取已 setup 训练样本的公开边界；callback
  通过 `trainer.datamodule` 使用它，不读取私有 dataset 字段。
- `Speech.bpe_spans` 通过 audio tokenizer 的 `frame_spans()` 取得，并校验 spans 与 semantic
  frame 严格双射；不满足直接报错，不做静默修复。
- raw language 在构造 `Speech` 时转成 `Language`，未知语言直接报错；task template 不消费
  数据集各自的语言别名。
