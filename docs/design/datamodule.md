# datamodule

把 anydataset 的 raw sample 组织成模型可直接消费的 `ModelBatch`。数据契约与 position
语义的权威定义见 [总览 §2](../model-design.md)。

## 对外能力

- `protocol.DataRuntime`：datamodule 所需资源的最小只读协议，公开 codec identity/view、
  text/audio tokenizer、layout 和 special token ID。正式 `Runtime` 与测试 fake 都通过该协议
  显式注入。
- `DataRuntimeSnapshot`：DataLoader worker 使用的可 pickle 数据视图，只保存 tokenizer、layout
  blocks 和 special token ID；不携带 runtime 已缓存的 backbone、codec 或 CUDA module。
- `DatasetRuntime`：在 `DataRuntime` 上增加正式 codec object，仅供 dataset factory 根据
  codebook metadata 构造 toy prepared-code samples。
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
- `TextDataModule` / `TextCollator`：纯文本 MT 数据路径，只读取 source/target text，当前可配置为
  anydataset `WMT19` preset 或 deterministic toy text samples，不消费 codec/audio tokenizer。
- `JointDataModule` / `ScheduledDataLoader`：组织多个 homogeneous dataloader。默认按
  optimizer step 确定性轮转；配置 `batches_per_step > 1` 时，一个 optimizer step 返回多个子
  batch，供静态 DDP 覆盖多条可训练执行路径。每个子 dataloader 自己保持单一 execution
  signature。
- `DatasetConfig` / `load_dataset()`：显式选择 `wmt19_tts` prepared data 或确定性的内存
  `toy` data。toy codes 根据正式 codec 的 semantic/acoustic codebook 数量和值域构造。
- `ToyDataset`：提供完整 source/target audio+text raw sample，不读取文件、不修改全局 RNG。
- `DataLoaderConfig(batch_size, num_workers, pin_memory, persistent_workers)` /
  `Config(codec, dataloader, dataset)`：公开的 DataLoader、dataset 与 DataModule 配置结构。
- `DataModule(config, runtime, task_weights)`：Lightning 数据入口；`setup()` 加载所选 dataset，
  并在加载前校验 config 与 runtime 的 codec identity。重复调用不会重新加载已持有的数据集。
- `FixedDataModule(codec, runtime, task_weights, sample_index, dataset=...)`：fixed-sample
  overfit/验收数据入口，只暴露一个 selected sample 的训练 loader，并复用同一公开
  `train_samples()` 边界供 callback 读取 raw sample。

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

- runtime 必须由组合入口显式传入：`DataModule` 接收 `DatasetRuntime`，`Collator` 及下游 parser
  和 sample builder 只消费较小的 `DataRuntime`；datamodule 不自行选择 tokenizer、layout 或
  special tokens。
- toy dataset 只读取正式 runtime 的 codec identity 与 codebook metadata；它不提供 tokenizer、
  codec、layout 或 special token，因此不存在 toy runtime 分支。
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
  acoustic target 的 `token_positions` 必须至少为 1，保证每个 frame 都有 causal predictor；
  `ModelBatch` 自身要求 input/label 是非空、对齐的有符号整数二维 batch、每行恰有一个
  `Task`，并维护单一 task execution signature。codebook 上界由持有具体 codec size 的下游
  负责。
- `ACOUSTIC_PAD_ID=-1` 只由 batch padding 引入，不能出现在未 padding 的 `ModelSample`
  中；因此派生的 frame mask 只包含右侧 padding，不会形成内部空洞。
- `ModelBatch` 只表达训练或 teacher-forcing evaluation，不表达缺少 target 的真实推理请求。
- 同一 `task_weights` 中的任务必须具有相同 source/target modality，保证 DDP 各 rank 走相同
  模型路径。0 权重任务不会参与 batch 分配；每项权重必须有限且非负，总和必须有限且为正；
  按 batch size 固定分配时，任一非 0 权重任务拿不到至少 1 条 sample 会直接报错。非法 stage
  更新在替换现有权重前报错。DataModule 构造时必须提供初始权重；stage callback 只在 epoch
  边界调用 `set_task_weights()`。task weights 使用进程共享数组，因此持久 worker 会在下一次
  collate 时看到更新，不要求 Trainer 重建 DataLoader。
- `LoaderSchedule.batches_per_step=1` 保留单子 batch 轮转；`batches_per_step > 1` 使用固定
  loader 分配，任一非 0 权重 loader 拿不到至少 1 个子 batch 会报错。loss 聚合不使用 loader
  权重，权重只改变数据进入训练 step 的频率。
- `DataModule` 在构造 loader 前把 collator 的完整 runtime 替换为 `DataRuntimeSnapshot`；主进程
  仍持有正式 runtime 供 dataset setup 使用。`persistent_workers` 只在 `num_workers > 0` 时启用，
  `pin_memory` 由入口显式配置。
- `DataModule.train_samples()` 是 callback 按索引读取已 setup 训练样本的公开边界；callback
  不读取私有 dataset 字段。
- parser 生成 `Speech.audio_token_spans`，`Speech` 校验 spans 与 semantic frame 完整对齐；
  不满足时直接报错，不做静默修复。
- raw language 在 parser 边界转换为 `Language`，未知语言直接报错；task template 不消费
  dataset 各自的语言别名。
