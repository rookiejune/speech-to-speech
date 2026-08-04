# datamodule

把 anydataset 的 raw sample 组织成模型可直接消费的 `ModelBatch`。数据契约与 position
语义的权威定义见 [总览 §2](../model-design.md)。

## 对外能力

- `protocol.DataRuntime`：datamodule 所需资源的最小只读协议，公开 codec identity/view、
  `audio_sequence_layout`（`flattened | semantic`）、semantic artifact、acoustic
  layout/unit length、text/audio tokenizer、layout 和 special token ID。正式 `Runtime` 与测试
  fake 都必须显式满足完整协议，不为缺失字段推断默认语义。
- `DataRuntimeSnapshot`：DataLoader worker 使用的可 pickle 数据视图，只保存 tokenizer、layout
  blocks、codec 数据解释字段、`audio_sequence_layout`、semantic/acoustic metadata 和 special token ID；不携带
  runtime 已缓存的 backbone、codec 或 CUDA module。
- `DatasetRuntime`：在 `DataRuntime` 上增加正式 codec object，仅供 dataset factory 根据
  codebook metadata 构造 toy prepared-code samples。
- `parse.parser.parse_sample()`：把 `anydataset.types.Sample` 解析为 `SpeechPair`。它解释当前
  `AudioView` 与 runtime `audio_sequence_layout`：逻辑 audio 输入/输出始终保留完整
  semantic/acoustic codes；`flattened` 把完整 codes 投影为 audio token sequence（acoustic-first、
  semantic-last），`semantic` 只把 semantic 放进 sequence，并保留 acoustic codes 给 side module、
  SAC。BiCodec 固定使用 self-describing `flattened` sequence，prompt/output ownership 由 sample
  builder 根据 source/reference 是否提供 global stream 处理，不由 parser 另造 semantic route。
- `parse.parser.parse_task_sample()`：按 `Task` 只解析实际消费的 source/target modality。pair/single
  只决定 role 映射；codec view 缺失时是否使用 waveform fallback 与数据 shape 无关。
- `build.single.SingleCollator` / `build.single.parse_single_sample()`：处理 single utterance 数据形态，
  即同一条 utterance 同时提供 text 与 audio/codes。TTS、ASR、TEXT_AR、AUDIO_AR、
  PARALLEL_AR、INTERLEAVED_AR 共用该 path；AR 序列由 `datamodule.build.ar` 组装。pair
  translation path 不再承载 single-only 数据契约。
- `build.sample.build_sample()`：根据 `Task` 把 `SpeechPair` 组装成 `ModelSample`，负责 chat
  template、BOA/EOA/EOS、global ID 映射、source token prompt、token labels 和 target frame
  positions。
- `build.single.build_single_sample()`：复用同一 `ModelSample` / `ModelBatch` 输出契约，把 single
  utterance 组装成 text->audio 或 audio->text 序列；`pl_module` 不区分 batch 来源。
- `types.Speech` / `types.SpeechPair`：prepared sample 的 codec、token 和语言逻辑视图；
  `AudioContextSample` 为 raw sample 绑定独立 audio reference；`RawSpeech` / `SpeechTaskSample` /
  `RawSpeechBatch` 表达 task 已选择、但部分 audio item 或 reference 尚待 codec encode 的中间状态。
- `types.ModelSample` / `types.ModelBatch`：单条和 batch 级模型输入；
  `ModelBatch.from_samples(..., pad_token_id=...)` 完成校验与 padding，mask 由 padding 字段
  派生并缓存。
- `task.Task` / `prediction.PredictionModality` / `source.SourceLayout`：`Task` 拥有
  `source_layout`、默认 `prediction_modality`、`allowed_predictions`、`uses_source_role` 和
  instruction template。loader 可对白名单任务覆写 `prediction`（如 T2ST/S2ST 的
  `audio|parallel`）。同构 batch 比较 `(source_layout, prediction)`。`MASKED_AR` 使用
  `TEXT_AUDIO` source（mask 后）与 mixed prediction。instruction 文案在
  `speech_to_speech.templates` 的每任务 paraphrase 池中；`SpeechConfig.tasks.<task>.template`
  为每 task 的 `int|null`（`null`=该 task 池内随机，整数=固定下标；默认 `0`）。loader
  schedule（`loaders` / `step_mode` / `accumulate_grad_batches`）同属
  `SpeechConfig`。训练构建调用 `sample_template(index)`；generation 要求固定下标，可用
  `evaluation_template_index()`（把 `null` 钉成 `0`）。`datamodule.tasks=null` 表示所有任务
  使用模板 `0`；显式提供 task 映射时，所有正权重 loader task 都必须在该映射中声明。
- `Collator(runtime, task_weights, prediction=...)`：按任务权重为 raw samples 选择任务，依次调用
  parser、sample builder 和 batch padding；可选 loader 级 prediction override 写入
  `SpeechTaskSample.prediction`。
- `LoaderSpec.text(...)` / `TextCollator`：纯文本 MT loader，只读取 source/target text，当前可
  配置为 anydataset `WMT19` preset 或 deterministic toy text samples，不消费 codec/audio
  tokenizer。
- `LoaderSchedule` / `ScheduledDataLoader`：为唯一 `DataModule` 组织多个 homogeneous loader。
  未指定 `step_mode` 时保留历史 weighted accumulation window 行为。`fused_joint` 返回覆盖所有非零
  loader 的 `FusedBatch`，task coverage 完整时可用于 static DDP，存在未使用参数时也可使用
  find-unused DDP；`serial_joint` 每个 optimizer step 串行消费每个非零 loader 一次，必须使用
  find-unused DDP。weighted mode 的 loader weight 表示采样频率；joint mode 的 loader
  weight 表示归一化 task loss 权重。每个子 loader 自己保持单一 execution signature。
- `DatasetConfig` / `load_dataset()`：显式选择 `wmt19_tts`、`qwen_tts_speaker` prepared data
  或确定性的内存 `toy` data。`qwen_tts_speaker` 通过 workspace 加载
  `SpeakerAudioGrid`，再由 `SpeakerGridCellsDataset` 暴露 `Role.DEFAULT` flat cells；默认覆盖
  所有 speaker，也可用 `speaker` 显式选择一列。BiCodec prepared cell 使用
  `AudioView.BICODEC` 的 anydataset structured mapping；parser 进入 S2S 后立即把 semantic 与
  fixed-length speaker slots 规范化为 `AudioCodes.semantic_codes/global_codes`，两者保留独立轴；
  可选的 `split_manifest` + `split_label` 把已加载的 map-style dataset 限制到
  manifest 声明的非重复、非负索引；manifest 不替换底层 anydataset split，也不绕过其公开
  dataloader/batch-planning 契约。底层是 `MapStyleABC` 时，split view 委托其 `_shuffle()` 并
  映射回子集位置，以保留 store-backed payload locality。toy codes 根据正式 codec 的
  semantic/acoustic/full-sequence codebook 数量和值域构造。

Qwen speaker grid 只允许 `bicodec` / `longcat` runtime，并强制使用 `shape=single`。训练不读取
grouped rows，因此不会把 speaker 轴或 semantic padding 带入模型 batch。指定 speaker 时 adapter
把底层 flat store 的局部分组映射回 text-row 索引，再在过滤后执行 rank 分片，避免 speaker-minor
排列把某一列集中到单个 distributed rank。该接入只确认 prepared-data 与模型输入契约；真实
checkpoint 的收敛和生成音质仍需单独验收。
显式 reference 的数据表示与 task source 分开处理：

- speaker-grid（`qwen_tts_speaker` / `shape=single`）：adapter 为每个 target cell 绑定同
  speaker 的下一 text row 作为 `AudioContextSample.audio_context`，最后一行循环到第一行；
  reference 与 target 必须是不同 row。它仍通过 flat-cell 索引读取，不把 grouped `rows` 或
  另一 speaker 混入训练样本。
- pair（例如 WMT19）：parser 只解析显式 `AudioContextSample`，不会把 pair 的
  `Role.SOURCE` audio 自动改写成 reference。builder 只把 task source layout 实际可见的 audio
  当作输入 global；因此 S2ST 可复用 source global，而 TTS/T2ST 即使底层 pair 携带 source audio，
  仍从 `<begin_of_global>` 生成 global。显式 reference 则序列化进 prompt。
- split manifest 的生成属于审计/部署入口，不属于 dataset loader：
  `scripts/create_split_manifest.py` 只消费 candidate、root audit 和 data-root 路径，输出带
  source artifact 与 root fingerprint 的 JSON；训练前必须先在 stable root 上完成该产物的独立
  校验。
- `ToyDataset`：提供完整 source/target audio+text raw sample，不读取文件、不修改全局 RNG；它实现
  `MapStyleABC`，因此 DDP smoke 与正式 prepared data 使用同一 rank-local batch planner。
- `config.DataLoaderConfig(batch_size, num_workers, pin_memory, persistent_workers, costs)` /
  `DataLoaderCostsConfig(enabled, max_batch_frames, planning_window)` /
  `SpeechConfig(codec, dataloader, shape, encode_missing_codes, tasks, loaders, dataset, …)`：公开的 DataLoader、
  dataset 与 DataModule dataclass 配置结构，字段和值域校验只在这里维护；Hydra staged train
  直接复用 `Config` 与 `TextConfig`，入口不声明同构 schema 或转换 dict。`costs` 默认关闭；
  开启后只适用于 speech `MapStyleABC` 路径，样本 cost 为全部 audio item 的
  `ceil(duration_seconds * codec_frame_rate)` 之和，并以 `max_batch_frames` 作为
  `max_batch_memory`。开启 costs 要求 cost_row / manifest 提供 `AudioMeta.DURATION`，不会像
  parser 那样从 codec frames 回退；text loader、fixed-sample subset 与非 MapStyle dataset
  开启 costs 直接报错。prepared speech 的 map-style dataset 通过
  `MapStyleABC.dataloader()` 使用 anydataset 的 cost planner；普通或 iterable dataset 使用
  PyTorch `DataLoader`。`shape=pair` 是默认路径；`shape=single` 显式选择 single utterance path。
- `_helper.task` 私有承载 loader task weights 校验与 batch task 分配；`_helper.text` 私有承载 text
  dataset 的 DataLoader 构造。相邻模块复用其中公开命名的 `TaskWeights`、`allocate_tasks` 与
  `TextLoader`，不跨模块导入函数级私有名，也不把这些实现细节提升为包级 API。
- `_helper.duration` 统一校验显式音频时长，并在 metadata 缺失时按 codec frames 或 waveform samples
  推导秒数；pair、single 与 raw waveform parser 不各自维护同一数值约束。
- `DataModule(runtime, loaders, schedule=None, validation=None)`：唯一 Lightning 数据入口。`loaders` 是
  `name -> LoaderSpec` 映射；speech loader 使用 `LoaderSpec.speech(config, task_weights,
  sample_index=...)`，纯文本 loader 使用 `LoaderSpec.text(config, task_weights)`。`setup()` 加载
  所选 dataset，并在加载前校验 speech config 与 runtime 的 codec identity；重复调用不会重新
  加载已持有的数据集。fixed-sample overfit 只是 speech spec 的 `sample_index` 变体，仍复用
  `diagnostic_samples()` 边界供 callback 读取 raw sample。可选 `validation` 是独立的 `LoaderSpec`；
  `val_dataloader()` 不进入 train schedule，也不复用 train loader instance。
  `diagnostic_samples()` 显式选择 train/validation 数据源；`diagnostic_collator()` 为 panel 指定的
  单一 task 构造独立 collator，不修改训练 loader 的共享 task weights。speech 与 text train loader
  都提供这两个 diagnostic 边界；text loader 对 WMT19 iterable dataset 使用 global shard 的固定
  索引读取，不受 DDP rank 分片影响。validation 仍只接受独立的 speech loader。
  schedule 在 DataModule 构造时固定；切换 stage 必须启动绑定新 stage 的 run，不提供运行时
  loader-weight setter。

## 输入输出

输入是 `anydataset.types.Sample`，包含 source/target 两个 role 及 audio/text modality。
内部转换顺序为：

```text
raw Sample
    -> parse.parser.parse_sample(runtime) -> SpeechPair
    -> build.sample.build_sample(task, runtime) -> ModelSample
    -> ModelBatch.from_samples(pad_token_id=runtime.pad_token_id) -> ModelBatch
```

single path 输入是 `Role.DEFAULT` 下的一条 text+audio utterance。正式训练要求 audio item 已经
materialize 出当前 runtime `AudioView` 的 codec codes：

```text
raw Sample(Role.DEFAULT)
    -> build.single.parse_single_sample(runtime) -> Speech
    -> build.single.build_single_sample(task, runtime) -> ModelSample
    -> ModelBatch.from_samples(pad_token_id=runtime.pad_token_id) -> ModelBatch
```

debug fallback 可显式设置 `encode_missing_codes=true`。pair 与 single collator 都只检查当前 task
实际消费的 audio item：已有 runtime codec view 时直接解析；缺少 codes 但存在
`AudioView.WAVEFORM` 时返回 `RawSpeechBatch`；既没有 codes 也没有 waveform 时明确报错。该 batch
不能直接进入 objective；训练入口必须给 `SpeechToSpeechModule` 挂
`OnDeviceCodecMaterializer`，在 GPU/device 上调用 runtime codec encode 后再构造标准
`ModelBatch`。materializer 把 waveform 显式转为 FP32，并在关闭当前 device autocast 的上下文内
执行完整 codec encode/tokenize，因此不继承 Trainer 的 `bf16-mixed` 计算精度。BiCodec view 走
structured `tokenize()`；LongCat、Stable Codec 与 UniCodec frame view 走完整 `encode()`，不能仅因
LongCat 同时暴露 structured capability 就改变数据表示。同一 batch 可以混合 prepared-code item
和 raw waveform item。

`ModelSample` 拆成与 generation 共用的输入侧 `Request`，以及仅训练使用的 `Labels`；collate
再导出 teacher-forcing 视图供 loss / backbone 消费：

```python
@dataclass
class ModelSample:
    request: Request   # generation.types.Request
    labels: Labels

# Request（与推理共用）
prompt_ids: Tensor
task: Task
prediction: PredictionModality | None   # 训练必填；推理可空=用 task 默认
audio_input_positions: Tensor | None

# Labels（仅训练）
response_ids: Tensor
token_labels: Tensor          # 与 cat(prompt_ids, response_ids) 等长；prompt 段为 -100
token_groups: Tensor | None
acoustic_target: AcousticTarget | None
source_ctc: CTCTarget | None
target_ctc: CTCTarget | None
audio_seconds: float
```

`ModelBatch.from_samples` 分别 pad request / labels，再令
`input_ids = cat(prompt_ids, response_ids)`，并令 `generation_prompt_lengths = len(prompt_ids)`。
batch 仍暴露对齐的 `input_ids` / `token_labels` / `token_groups` / `acoustic_target` /
`source_ctc` / `target_ctc` / `audio_input_positions` / `predictions`，供现有 loss
与 bridge 使用。

`AcousticTarget` 包含 `semantic_codes`、`codes`、`token_positions`。分组使必须共同存在的 tensor
不能形成半完整状态。

`Speech` 使用三分字段：`semantic_codes`、`global_codes`、`acoustic_codes`。fixed-layout anycodec
输入在 parser/构造边界归入 `global_codes`，frame-aligned 输入才进入 `acoustic_codes`。

BiCodec sample builder 先按“task 实际可见的 audio source 或显式 reference”是否提供 global stream
决定序列开局。prompt 已有 global 时，response 以 `<begin_of_semantic>` 开局；prompt 没有 global
时，response 以 `<begin_of_global>` 开局并按 global-first / semantic-last 生成两者。装配完成后
不再另存一份 context codes 给 model batch 或 decode，target semantic 也不会被放进 prompt。
`token_groups` 只标记实际预测的 semantic、semantic-or-end 或各 global codebook payload；forced
codec/stream marker 与外层 EOA 不进入监督。

`ModelBatch` 额外保存 `tasks: list[Task]`、`predictions: list[PredictionModality]` 和
`pad_token_id`，并公开 `attention_mask` 与 `acoustic_target_mask`。speech batch 还保存
`audio_seconds: Tensor[B]`，表示每条训练样本按当前 task 实际消费的 source/target 音频秒数之和；
纯文本样本为 0。batch padding 把单条 prompt 边界聚合为 `generation_prompt_lengths`；raw sample
的显式 `audio_context` 已在装配阶段消费，不进入 batch。
teacher-forcing generation bridge（`requests_from_batch`）切出 `prompt_ids` 并带上同批
`prediction`，不从第一个非 `-100` label 反推。BiCodec reference global stream 已在该 prompt
边界内；audio-target 路径把结构 BOA 写入 `prompt_ids`，因此即使后续 codec marker
不受监督，真实生成仍从相同状态开始。

`audio_input_positions` 是每条序列中 source audio payload token 的位置，按 `[frames]` 保存，batch
padding 后为 `[batch, frames]`，右侧填充 `-1`。sample builder 只为
`task.source_modality == Modality.AUDIO` 的 source payload 记录位置；source BOA/EOA、target audio
response、generated token 以及额外序列化的 BiCodec reference global span 不进入该字段。它只服务
backbone input tower；BiCodec decode ownership 完全由 prompt/response token marker 表达。

`source_ctc` / `target_ctc` 各自包含 audio span 的完整序列 `token_positions` 与 tokenizer-local
`text_token_ids`。sample builder 不按“任务输出是否为 audio”这一条粗规则决定 CTC，而是按 transcript
visibility 编译：ASR/S2TT/S2ST 有 source CTC；T2ST/S2ST 的纯 audio prediction 与 AUDIO_AR 有
target CTC；TTS、MT、parallel/interleaved output 没有 target CTC。source positions 交给 non-causal
route 读取 `h[p]`，target positions 保留 token 自身位置并由 loss 读取 `h[p-1]`。audio span 指
tokenizer 序列化 payload：BiCodec 内部 stream/end marker 也保留为可预测 blank 的 CTC step，外层
BOA/EOA 则排除。

## 边界

- 包级 `speech_to_speech.datamodule` 只导出唯一运行入口 `DataModule`。`LoaderSpec`、配置结构、
  schedule、parser、sample、types、protocol、collator、dataset factory 等契约从对应子模块导入，
  不提升为包级稳定 API。
- runtime 必须由组合入口显式传入：含 speech loader 的 `DataModule` 接收 `DatasetRuntime`，纯文本
  loader 只需要 `TextRuntime`；`Collator` 及下游 parser 和 sample builder 只消费更小的 runtime
  协议。datamodule 不自行选择 tokenizer、layout 或 special tokens。
- datamodule 按数据形态拆分 pair/single，而不是按 TTS/ASR 拆分。pair path 拥有 source/target
  role 选择；single path 拥有同一 utterance 内 text/audio 的方向选择。`Task.uses_source_role`
  只服务 pair path，single path 不用它推断 dataset role。在线 codec encode 不属于 shape 规则；
  它只由 task 实际消费的 audio item 是否缺少 runtime codec view 决定。
- `SpeakerAudioGrid.rows` 是检查/对比 speaker 轴的 grouped view，不进入训练。Qwen TTS 训练只消费
  `cells`，并要求每个 cell 在 `Role.DEFAULT` 下同时提供 text/audio；adapter 不静默重写 role。
- 正式训练路径优先使用预先 materialize 的 codec codes。训练时 wav->codes 只作为显式 debug
  fallback：普通 DataLoader worker 不持有 codec/CUDA module，fallback batch 必须在
  `pl_module` loss 前经 on-device materializer 转为 `ModelBatch`。materializer 对 S2ST 编码 source
  和 target，对 S2TT/ASR 只编码音频 source，对 T2ST/TTS 只编码音频 target；纯文本 task 不调用
  codec。在线编码是 FP32 预处理边界，不属于 backbone/acoustic decoder 的 autocast graph。
- toy dataset 只读取正式 runtime 的 codec identity 与 codebook metadata；它不提供 tokenizer、
  codec、layout 或 special token，因此不存在 toy runtime 分支。
- `parse.parser` 只解释 raw dataset representation；`build.sample` 只实现任务序列规则；
  `types.py` 保存结构并处理局部校验、padding 和 mask。三层不反向读取彼此的私有逻辑。
- LongCat 的第 0 个 codebook 和后续 codebooks 只在 parser 边界解释为 semantic/acoustic。
  `flattened` sequence layout 不拆 semantic/acoustic side channel，而是把完整 codec codes 放入
  `semantic_codes` 并设置 `acoustic_codes=None`；Stable Codec 与 UniCodec 的完整 frame codes
  也使用相同表示。fixed-length structured codec 不属于这条 frame-code parser 路径，其 prompt、
  output 和 decode ownership 由序列 marker 自描述，不作为 datamodule 公共配置轴。
- audio tokenizer 的输出统一称为 `audio_token_ids`；codec codebook index 统一称为
  `semantic_codes` / `global_codes` / `acoustic_codes`。只有 layout global IDs 使用 `input_ids` 和
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
- `CTC_PAD_ID=-1` 分别 pad position 与 transcript；有效 position 必须严格递增，target position
  必须至少为 1。batch 校验 CTC 的必要时间步，包括连续重复 transcript token 所需的额外 blank。
- `ModelBatch` 只表达训练或 teacher-forcing evaluation，不表达缺少 target 的真实推理请求。
- `audio_input_positions` 只表达可见 source audio payload 的 overlay 位置；其值必须唯一、落在
  当前序列内并指向 runtime codec audio range。没有 source audio 时必须为 `None`。
- `ModelBatch.generation_prompt_lengths` 是 teacher-forcing 到真实推理的显式 prompt 桥接字段；
  BiCodec reference global stream 必须已位于该边界内，decode 不从 batch side channel 或 target codes
  回填缺失 ownership。
- `AudioMeta.DURATION` 的单位是秒，不是 codec frame 或 waveform sample。parser 优先读取并校验
  该元数据；缺失时用当前 codec view 的 frame count 除以 `runtime.codec_frame_rate` 推导
  `Speech.duration_seconds`。task sample builder 按 source/target modality 决定哪些角色计入
  `ModelBatch.audio_seconds`；不能把真实音频静默计为 0。
- 同一 `task_weights` 中的任务必须具有相同 `(source_layout, prediction)` 执行签名，保证 DDP 各 rank 走相同
  模型路径。0 权重任务不会参与 batch 分配；每项权重必须有限且非负，总和必须有限且为正；
  task allocator 把 weighted round-robin credit 跨 collate 调用保存在进程共享状态中。小 batch
  可以暂时不含某个低权重 task，但不会丢弃尾批样本，并会在后续 batch 归还配额。DataModule
  构造时必须提供 task weights，collator 构造后不可修改；切换任务组合必须构造新的 loader。
- 未指定 `step_mode` 的 `LoaderSchedule.accumulate_grad_batches=1` 保留逐 batch 轮转；大于 1 时，
  每个 accumulation window 按 loader 权重分配并交错排列 microbatch，任一非 0 权重 loader 拿不到至少
  1 个 microbatch 会报错。`fused_joint` 不按权重重复采样，而是按声明顺序从每个正权重 loader 取一次，
  返回带归一化 loss weights 的 `FusedBatch`；`serial_joint` 同样每个 loader 取一次，并要求
  `accumulate_grad_batches` 等于正权重 loader 数量。serial 返回的 `LoaderBatch.loss_scale` 等于
  `loader_count * normalized_weight`，抵消 Lightning automatic accumulation 对 loss 除以 loader
  count 的缩放，使其 optimizer-step gradient 与 fused weighted loss 一致。
  每个子 loader 独立维护从 0 开始的 cycle；耗尽后
  先推进到下一 cycle，再通过 loader 的 `set_epoch()` 或其 `batch_sampler.set_epoch()` 更新
  deterministic shuffle，然后重建 iterator。同一 schedule 和 per-rank batch count 下，各 rank
  会在相同 accumulation-window 位置推进相同子 loader 的 epoch。
- `token_weighted` 把权重解释为长期监督 token 比例：调度器在每次取 batch 后，以
  `ModelBatch.token_labels != -100` 或 MIMO 的 shifted text/audio target masks 计数，更新累计 deficit，
  再确定性地选择最大 deficit 的 loader。它允许任意 `accumulate_grad_batches`，但不能与
  `fuse_loaders_per_step` 同时开启；raw waveform batch 没有 token 计数契约，会在首次取样时报错。
  已初始化 DDP process group 时，计数会先 all-reduce 为全局平均值，避免 rank-local 序列长度差异导致
  各 rank 选择不同 loader；新 batch 类型可实现 `supervised_token_count` 属性，或通过
  `ScheduledDataLoader(token_counter=...)` 注入计数器。
- `DataModule` 在构造 loader 前把 collator 的完整 runtime 替换为 `DataRuntimeSnapshot`；主进程
  仍持有正式 runtime 供 dataset setup 使用。`persistent_workers` 只在 `num_workers > 0` 时启用，
  多个 train spec 复用同一个 `SpeechConfig` 时只加载一份 speech dataset；复用同一个
  `DataLoaderConfig` 时，`num_workers` 是该配置组的总预算，并按 loader schedule weight
  分配给各子 loader，不随逻辑 loader 数量成倍启动进程。
  `pin_memory` 由入口显式配置。
- 对 anydataset `MapStyleABC`，`DataModule` 使用其 `dataloader()` 公开入口负责 deterministic
  shuffle、runtime shard 和 sample-cost batch 规划；默认 costs 关闭时以 unit cost 与
  `batch_size` 对齐，开启后按 audio-frame cost 与 `max_batch_frames` 规划。store-backed
  dataset 会额外保留 payload locality。DataLoader 仍索引原始外层 dataset，因此
  `AnyDataset` transform 不会被绕过；split-manifest view 的 cost planning 也委托底层
  `cost_row()`，不会为了估算 batch cost 加载完整 sample。普通或 iterable dataset 使用 PyTorch `DataLoader`，
  不能开启 costs。多 loader train 的外层 `ScheduledDataLoader` 不接受 Lightning 注入
  sampler；正式 distributed sample partition 由各 loader 的公开 dataloader 契约负责。
- validation speech loader 使用同一公开 batch planner 和 distributed partition，但显式关闭
  shuffle。正式 train 入口从一个现有 stage speech loader 复制 task weights 与 speech config，
  只把复制后的 `DatasetConfig.split_label` 改为 dev；训练 spec 与 dataset config 保持不变。
  validation diagnostic panel 读取该独立数据源，但使用 panel 的 task-specific collator，因此同一
  paired speech split 可同时监督 ASR 与 TTS generation。
- train loader 与 validation loader 使用独立的窄 Protocol。没有 validation spec 时
  `DataModule.val_dataloader()` 返回空 iterable，Lightning 不运行 validation；text loader 不提供
  `validation_dataloader()`，把 text spec 作为 validation 传入时在 DataModule 构造边界直接报错，
  不用 training loader 伪装 validation。
- `DataModule.diagnostic_samples()` 是 callback 按 split、loader 和索引读取已 setup 样本的公开边界；
  callback 不读取私有 dataset 字段。text loader 的 iterable dataset 通过 `iter_shard(1, 0)` 读取固定
  global indices，避免 callback 样本随 world size 变化。诊断代码通过 `diagnostic.source_item()` / `target_item()` 按 task
  解析 raw sample：pair sample 只接受 source/target role，single sample 只接受
  `Role.DEFAULT`，缺项或混用 role 直接报错。`ModelBatch.row()` 提供与该 raw sample 同行的
  teacher-forcing batch，不由 callback 手工切 tensor。
- parser 生成 `Speech.audio_token_spans`，`Speech` 校验 spans 与 semantic frame 完整对齐；
  不满足时直接报错，不做静默修复。
- raw language 在 parser 边界转换为 `Language`，未知语言直接报错；task prompt 不消费
  dataset 各自的语言别名。
