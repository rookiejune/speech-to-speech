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
  `AudioView` 与 runtime `audio_representation`，将解耦路线的 LongCat codebooks 分成
  semantic/acoustic codes，或在 `full_codec_sequence` 下把完整 codec codes 作为 token
  supervision，并生成 text/audio token IDs 与 audio token spans。
- `single.SingleCollator` / `single.parse_single_sample()`：处理 single utterance 数据形态，
  即同一条 utterance 同时提供 text 与 audio/codes。TTS、ASR、TEXT_AR、AUDIO_AR 共用该
  path，由 task 决定同一 utterance 的哪个 modality 作为 source/target；pair translation path
  不再承载 single-only 数据契约。
- `sample.build_sample()`：根据 `Task` 把 `SpeechPair` 组装成 `ModelSample`，负责 chat
  template、BOA/EOA/EOS、global ID 映射、source token prompt、token labels 和 target frame
  positions。
- `single.build_single_sample()`：复用同一 `ModelSample` / `ModelBatch` 输出契约，把 single
  utterance 组装成 text->audio 或 audio->text 序列；`pl_module` 不区分 batch 来源。
- `types.Speech` / `types.SpeechPair`：raw sample 的 codec、token 和语言逻辑视图。
- `types.ModelSample` / `types.ModelBatch`：单条和 batch 级模型输入；
  `ModelBatch.from_samples(..., pad_token_id=...)` 完成校验与 padding，mask 由 padding 字段
  派生并缓存。
- `task.Task` / `types.Language`：任务与语言枚举。`Task` 是 source/target modality、
  `uses_source_role` 和 instruction template 的唯一事实来源。
- `Collator(runtime, task_weights)`：按任务权重为 raw samples 选择任务，依次调用 parser、
  sample builder 和 batch padding；正式训练在构造时固定 task weights，`set_task_weights()` 只保留
  为显式的低层控制入口。
- `LoaderSpec.text(...)` / `TextCollator`：纯文本 MT loader，只读取 source/target text，当前可
  配置为 anydataset `WMT19` preset 或 deterministic toy text samples，不消费 codec/audio
  tokenizer。
- `LoaderSchedule` / `ScheduledDataLoader`：为唯一 `DataModule` 组织多个 homogeneous loader。
  默认按 optimizer step 确定性轮转；配置 `batches_per_step > 1` 时，一个 optimizer step 返回
  多个子 batch，供静态 DDP 覆盖多条可训练执行路径。每个子 loader 自己保持单一 execution
  signature。
- `DatasetConfig` / `load_dataset()`：显式选择 `wmt19_tts` prepared data 或确定性的内存
  `toy` data。toy codes 根据正式 codec 的 semantic/acoustic/full-sequence codebook 数量和值域构造。
- `ToyDataset`：提供完整 source/target audio+text raw sample，不读取文件、不修改全局 RNG。
- `DataLoaderConfig(batch_size, num_workers, pin_memory, persistent_workers)` /
  `Config(codec, dataloader, shape, encode_missing_codes, dataset)`：公开的 DataLoader、
  dataset 与 DataModule 配置结构。prepared speech 的 map-style dataset 通过
  `MapStyleABC.dataloader()` 使用 anydataset 的 cost planner；普通或 iterable dataset 使用
  PyTorch `DataLoader`。`shape=pair` 是默认路径；`shape=single` 显式选择 single utterance path。
- `DataModule(runtime, loaders, schedule=None)`：唯一 Lightning 数据入口。`loaders` 是
  `name -> LoaderSpec` 映射；speech loader 使用 `LoaderSpec.speech(config, task_weights,
  sample_index=...)`，纯文本 loader 使用 `LoaderSpec.text(config, task_weights)`。`setup()` 加载
  所选 dataset，并在加载前校验 speech config 与 runtime 的 codec identity；重复调用不会重新
  加载已持有的数据集。fixed-sample overfit 只是 speech spec 的 `sample_index` 变体，仍复用
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

single path 输入是 `Role.DEFAULT` 下的一条 text+audio utterance。正式训练要求 audio item 已经
materialize 出当前 runtime `AudioView` 的 codec codes：

```text
raw Sample(Role.DEFAULT)
    -> single.parse_single_sample(runtime) -> SpeechUtterance
    -> single.build_single_sample(task, runtime) -> ModelSample
    -> ModelBatch.from_samples(pad_token_id=runtime.pad_token_id) -> ModelBatch
```

debug fallback 可显式设置 `encode_missing_codes=true`，在缺少 codec codes 但存在
`AudioView.WAVEFORM` 时让 collator 返回 `RawSingleBatch`。该 batch 不能直接进入 objective；
训练入口必须给 `SpeechToSpeechModule` 挂 `OnDeviceCodecMaterializer`，在 GPU/device 上调用
runtime codec encode 后再构造标准 `ModelBatch`。

`ModelSample` 和 `ModelBatch` 使用同一组核心字段：

```python
input_ids: Tensor
token_labels: Tensor
acoustic_target: AcousticTarget | None
```

`AcousticTarget` 包含 `semantic_codes`、`codes`、`token_positions`。分组使必须共同存在的 tensor
不能形成半完整状态。

`ModelBatch` 额外保存 `tasks: list[Task]` 和 `pad_token_id`，并公开
`attention_mask` 与 `acoustic_target_mask`。speech batch 还保存
`audio_seconds: Tensor[B]`，表示每条训练样本按当前 task 实际消费的 source/target 音频秒数之和；
纯文本样本为 0。

## 边界

- 包级 `speech_to_speech.datamodule` 只导出唯一运行入口 `DataModule`。`LoaderSpec`、配置结构、
  schedule、parser、sample、types、protocol、collator、dataset factory 等契约从对应子模块导入，
  不提升为包级稳定 API。
- runtime 必须由组合入口显式传入：含 speech loader 的 `DataModule` 接收 `DatasetRuntime`，纯文本
  loader 只需要 `TextRuntime`；`Collator` 及下游 parser 和 sample builder 只消费更小的 runtime
  协议。datamodule 不自行选择 tokenizer、layout 或 special tokens。
- datamodule 按数据形态拆分 pair/single，而不是按 TTS/ASR 拆分。pair path 拥有 source/target
  role 选择；single path 拥有同一 utterance 内 text/audio 的方向选择。`Task.uses_source_role`
  只服务 pair path，single path 不用它推断 dataset role。
- 正式训练路径优先使用预先 materialize 的 codec codes。训练时 wav->codes 只作为显式 debug
  fallback：普通 DataLoader worker 不持有 codec/CUDA module，fallback batch 必须在
  `pl_module` loss 前经 on-device materializer 转为 `ModelBatch`。
- toy dataset 只读取正式 runtime 的 codec identity 与 codebook metadata；它不提供 tokenizer、
  codec、layout 或 special token，因此不存在 toy runtime 分支。
- `parser.py` 只解释 raw dataset representation；`sample.py` 只实现任务序列规则；
  `types.py` 保存结构并处理局部校验、padding 和 mask。三层不反向读取彼此的私有逻辑。
- LongCat 的第 0 个 codebook 和后续 codebooks 只在 parser 边界解释为 semantic/acoustic。
  `full_codec_sequence` 不拆 semantic/acoustic side channel，而是把完整 codec codes 放入
  `semantic_codes` 并设置 `acoustic_codes=None`；unified-token codec 的完整 codes 也是
  `semantic_codes`，`acoustic_codes=None`。fixed-length structured codec 不属于这条 frame-code
  parser 路径。
- audio tokenizer 的输出统一称为 `audio_token_ids`；codec codebook index 统一称为
  `semantic_codes` / `acoustic_codes`。只有 layout global IDs 使用 `input_ids` 和
  `token_labels`。
- chat template 先渲染为字符串并在字符串层切分 source placeholder，再分别 tokenize
  prefix/suffix；不能在 token IDs 中搜索单独编码的 placeholder，因为 BPE 分词受相邻文本
  影响。
- target 为 audio 时，BOA 是结构性 response prefix，不参与监督：
  `token_labels[len(input_ids) + 1:] = response_ids[1:]`，只监督 audio tokens 和 EOA。
- `acoustic_target` 内各 tensor 共享 frame 轴；`token_positions` 将每个 acoustic frame
  对齐到 target audio token。它只表达
  codec target，不保存或预计算 REPA teacher features。unified-token codec 没有独立
  acoustic side channel，因此这些 target code 字段为 `None`。
- `ModelBatch.from_samples()` 显式接收 `pad_token_id`，在 padding 前要求 acoustic/semantic
  codes 是非空、二维、非负有符号整数 tensor，并检查 target 内部 frame 轴；
  acoustic target 的 `token_positions` 必须至少为 1，保证每个 frame 都有 causal predictor；
  `ModelBatch` 自身要求 input/label 是非空、对齐的有符号整数二维 batch、每行恰有一个
  `Task`，并维护单一 task execution signature。codebook 上界由持有具体 codec size 的下游
  负责。
- `ACOUSTIC_PAD_ID=-1` 只由 batch padding 引入，不能出现在未 padding 的 `ModelSample`
  中；因此派生的 frame mask 只包含右侧 padding，不会形成内部空洞。
- `ModelBatch` 只表达训练或 teacher-forcing evaluation，不表达缺少 target 的真实推理请求。
- `AudioMeta.DURATION` 的单位是秒，不是 codec frame 或 waveform sample。parser 优先读取并校验
  该元数据；缺失时用当前 codec view 的 frame count 除以 `runtime.codec_frame_rate` 推导
  `Speech.duration_seconds`。task sample builder 按 source/target modality 决定哪些角色计入
  `ModelBatch.audio_seconds`；不能把真实音频静默计为 0。
- 同一 `task_weights` 中的任务必须具有相同 source/target modality，保证 DDP 各 rank 走相同
  模型路径。0 权重任务不会参与 batch 分配；每项权重必须有限且非负，总和必须有限且为正；
  按 batch size 固定分配时，任一非 0 权重任务拿不到至少 1 条 sample 会直接报错。非法权重
  更新在替换现有权重前报错。DataModule 构造时必须提供初始权重；正式入口不会在运行中调用
  `set_task_weights()`。权重使用进程共享数组，因此显式更新时持久 worker 会在下一次 collate
  看到新值，不要求重建 DataLoader。
- `LoaderSchedule.batches_per_step=1` 保留单子 batch 轮转；`batches_per_step > 1` 使用固定
  loader 分配，任一非 0 权重 loader 拿不到至少 1 个子 batch 会报错。loss 聚合不使用 loader
  权重，权重只改变数据进入训练 step 的频率。每个子 loader 独立维护从 0 开始的 cycle；耗尽后
  先推进到下一 cycle，再通过 loader 的 `set_epoch()` 或其 `batch_sampler.set_epoch()` 更新
  deterministic shuffle，然后重建 iterator。同一 schedule 和 per-rank batch count 下，各 rank
  会在相同 optimizer step 推进相同子 loader 的 epoch。
- `DataModule` 在构造 loader 前把 collator 的完整 runtime 替换为 `DataRuntimeSnapshot`；主进程
  仍持有正式 runtime 供 dataset setup 使用。`persistent_workers` 只在 `num_workers > 0` 时启用，
  `pin_memory` 由入口显式配置。
- 对 anydataset `MapStyleABC`，`DataModule` 使用其 `dataloader()` 公开入口负责 deterministic
  shuffle、runtime shard 和固定的 sample-cost batch 规划；store-backed dataset 会额外保留
  payload locality。DataLoader 仍索引原始外层 dataset，因此 `AnyDataset` transform 不会被绕过。
  普通或 iterable dataset 使用 PyTorch `DataLoader`。多 loader train 的外层
  `ScheduledDataLoader` 不接受 Lightning 注入 sampler；正式 distributed sample partition
  由各 loader 的公开 dataloader 契约负责。
- `DataModule.train_samples()` 是 callback 按索引读取已 setup 训练样本的公开边界；callback
  不读取私有 dataset 字段。
- parser 生成 `Speech.audio_token_spans`，`Speech` 校验 spans 与 semantic frame 完整对齐；
  不满足时直接报错，不做静默修复。
- raw language 在 parser 边界转换为 `Language`，未知语言直接报错；task template 不消费
  dataset 各自的语言别名。
