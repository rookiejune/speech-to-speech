# datamodule

把 anydataset 的 raw sample 组织成模型可直接消费的 `ModelBatch`。数据契约与 position
语义的权威定义见 [总览 §2](../model-design.md)。

## 对外能力

- `protocol.DataRuntime`：datamodule 所需资源的最小只读协议，公开 codec identity/view、
  text/audio tokenizer、layout 和 special token ID。正式 `Runtime` 与测试 fake 都通过该协议
  显式注入。
- `parser.parse_sample()`：把 `anydataset.types.Sample` 解析为 `SpeechPair`。它解释当前
  `AudioView`，将 LongCat codebooks 分成 semantic/acoustic codes，并生成 text/audio token
  IDs 与 audio token spans。
- `sample.build_sample()`：根据 `Task` 把 `SpeechPair` 组装成 `ModelSample`，负责 chat
  template、BOA/EOA/EOS、global ID 映射、token labels、acoustic prompt 和 target frame
  positions。
- `types.Speech` / `types.SpeechPair`：raw sample 的 codec、token 和语言逻辑视图。
- `types.ModelSample` / `types.ModelBatch`：单条和 batch 级模型输入；
  `ModelBatch.from_samples(..., pad_token_id=...)` 完成校验与 padding，mask 由 padding 字段
  派生并缓存。
- `task.Task` / `types.Language`：任务与语言枚举。`Task` 是 source/target modality、
  `uses_source_role` 和 instruction template 的唯一事实来源。
- `Collator(runtime, task_weights)`：按任务权重为 raw samples 选择任务，依次调用 parser、
  sample builder 和 batch padding；`set_task_weights()` 原地更新后续 batch 的任务分布。
- `DataModule(config, runtime, task_weights)`：Lightning 数据入口。`Config` 只保存
  codec/dataloader 配置；`setup()` 加载 prepared dataset，并在加载前校验 config 与 runtime
  的 codec identity。重复调用不会重新加载已持有的数据集。

## 输入输出

输入是 `anydataset.types.Sample`，包含 source/target 两个 role 及 audio/text modality。
内部转换顺序为：

```text
raw Sample
    -> parser.parse_sample(runtime) -> SpeechPair
    -> sample.build_sample(task, runtime) -> ModelSample
    -> ModelBatch.from_samples(pad_token_id=runtime.pad_token_id) -> ModelBatch
```

`ModelSample` 和 `ModelBatch` 使用同一组核心字段：

```python
input_ids: Tensor
token_labels: Tensor
acoustic_prompt: AcousticPrompt | None
acoustic_target: AcousticTarget | None
```

`AcousticPrompt` 包含 `codes`、`token_positions`；`AcousticTarget` 包含
`semantic_codes`、`codes`、`token_positions`。分组使必须共同存在的 tensor 不能形成半完整状态。

`ModelBatch` 额外保存 `tasks: list[Task]` 和 `pad_token_id`，并公开
`attention_mask`、`acoustic_prompt_mask` 与 `acoustic_target_mask`。

## 边界

- `DataRuntime` 必须由组合入口显式传给 `DataModule`/`Collator`，再沿 parser 和 sample
  builder 传递；datamodule 不自行选择 tokenizer、layout 或 special tokens。
- `parser.py` 只解释 raw dataset representation；`sample.py` 只实现任务序列规则；
  `types.py` 保存结构并处理局部校验、padding 和 mask。三层不反向读取彼此的私有逻辑。
- LongCat 的第 0 个 codebook 和后续 codebooks 只在 parser 边界解释为 semantic/acoustic。
  unified-token codec 的完整 codes 是 `semantic_codes`，`acoustic_codes=None`。
- audio tokenizer 的输出统一称为 `audio_token_ids`；codec codebook index 统一称为
  `semantic_codes` / `acoustic_codes`。只有 layout global IDs 使用 `input_ids` 和
  `token_labels`。
- chat template 先渲染为字符串并在字符串层切分 source placeholder，再分别 tokenize
  prefix/suffix；不能在 token IDs 中搜索单独编码的 placeholder，因为 BPE 分词受相邻文本
  影响。
- target 为 audio 时，BOA 是结构性 response prefix，不参与监督：
  `token_labels[len(input_ids) + 1:] = response_ids[1:]`，只监督 audio tokens 和 EOA。
- `acoustic_prompt` 整体表示 source acoustic frames 及其 token sequence 注入位置。
- `acoustic_target` 内各 tensor 共享 frame 轴；`token_positions` 将每个 acoustic frame
  对齐到 target audio token。它只表达
  codec target，不保存或预计算 REPA teacher features。unified-token codec 没有独立
  acoustic side channel，因此这些 target code 字段为 `None`。
- `ModelBatch.from_samples()` 显式接收 `pad_token_id`，在 padding 前要求 acoustic/semantic
  codes 是非空、二维、非负有符号整数 tensor，并检查 prompt/target 内部 frame 轴；
  `ModelBatch` 自身要求 input/label 是非空、对齐的有符号整数二维 batch、每行恰有一个
  `Task`，并维护单一 task execution signature。codebook 上界由持有具体 codec size 的下游
  负责。
- `ACOUSTIC_PAD_ID=-1` 只由 batch padding 引入，不能出现在未 padding 的 `ModelSample`
  中；因此派生的 frame mask 只包含右侧 padding，不会形成内部空洞。
- `ModelBatch` 只表达训练或 teacher-forcing evaluation，不表达缺少 target 的真实推理请求。
- 同一 `task_weights` 中的任务必须具有相同 source/target modality，保证 DDP 各 rank 走相同
  模型路径。每项权重必须有限且非负，总和必须有限且为正；非法 stage 更新在替换现有权重前
  报错。DataModule 构造时必须提供初始权重；stage callback 只在 epoch 边界调用
  `set_task_weights()`，不依赖 Trainer 重建 DataLoader。
- `DataModule.train_samples()` 是 callback 按索引读取已 setup 训练样本的公开边界；callback
  不读取私有 dataset 字段。
- parser 生成 `Speech.audio_token_spans`，`Speech` 校验 spans 与 semantic frame 完整对齐；
  不满足时直接报错，不做静默修复。
- raw language 在 parser 边界转换为 `Language`，未知语言直接报错；task template 不消费
  dataset 各自的语言别名。
