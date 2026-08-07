# datamodule

把 anydataset 的 raw sample 组织成模型可直接消费的 `ModelBatch`。跨模块所有权与依赖方向见
[设计总览](../model-design.md)。

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
  generator plugin。BiCodec 固定使用 self-describing `flattened` sequence；source 与 target 分别由
  input/output runtime 编码，parser 和 builder 都不检查 source stream 来决定 target serialization。
- `parse.parser.parse_task_sample()`：按 `Task` 只解析实际消费的 source/target modality。pair/single
  只决定 role 映射；codec view 缺失时是否使用 waveform fallback 与数据 shape 无关。
- `build.single.SingleCollator` / `build.single.parse_single_sample()`：处理 single utterance 数据形态，
  即同一条 utterance 同时提供 text 与 audio/codes。TTS、ASR、TEXT_AR、AUDIO_AR、
  PARALLEL_AR、INTERLEAVED_AR 共用该 path；AR 序列由 `datamodule.build.ar` 组装。pair
  translation path 不再承载 single-only 数据契约。
- `build.sample.build_sample()`：根据 `Task` 把 `SpeechPair` 组装成 `ModelSample`，负责 chat
  template、typed ASR/MT controls、BOA/EOA/EOS、global ID 映射、source token prompt、token labels 和 target frame
  positions。
- `build.single.build_single_sample()`：复用同一 `ModelSample` / `ModelBatch` 输出契约，把 single
  utterance 组装成 text->audio 或 audio->text 序列；`pl_module` 不区分 batch 来源。
- `sample.Speech` / `sample.SpeechPair`：prepared sample 的 codec、token 和语言逻辑视图；
  `RawSpeech` / `SpeechTaskSample` / `RawSpeechBatch` 表达 task 已选择、但部分 audio item 尚待 codec
  encode 的中间状态。旧的 `AudioContextSample` 只保留为数据边界类型，训练 parser/collator 明确拒绝；
  voice clone 必须表示为普通 source/target pair。
- `batch.ModelSample` / `batch.ModelBatch`：单条和 batch 级模型输入；
  `ModelBatch.from_samples(..., pad_token_id=...)` 完成校验与 padding，mask 由 padding 字段
  派生并缓存。
- `task.TaskProgram` / `ResponseSpec` / `ResponseStep`：声明 task 的可见 context、有序 response
  fields 及其 control framing、序列 layout、内部 prediction route 和 objective；`Task` 继续作为稳定的配置/指标 ID，
  loader 只通过 `trace` 选择一个 `ResponseSpec`，prediction 由该 response 派生，不再接受独立
  override。同构 batch 比较解析后的 `(source_layout, response.prediction)`；`MASKED_AR` 使用
  `TEXT_AUDIO` context 与 reconstruction objective。instruction 文案在
  `speech_to_speech.task.templates` 的每任务 paraphrase 池中；`SpeechConfig.tasks.<task>.template`
  为每 task 的 `int|null`（`null`=该 task 池内随机，整数=固定下标；字段默认 `0`，正式
  train entry 显式使用 `null`）。loader
  schedule（`loaders` / `step_mode` / `accumulate_grad_batches`）同属
  `SpeechConfig`。训练构建调用 `sample_template(index)`；generation 要求固定下标，可用
  `evaluation_template_index()`（把 `null` 钉成 `0`）。`datamodule.tasks=null` 表示所有任务
  使用模板 `0`；显式提供 task 映射时，所有正权重 loader task 都必须在该映射中声明。
- `Collator(runtime, task_weights, trace=...)`：按任务权重为 raw samples 选择任务，依次调用 parser、
  sample builder 和 batch padding；loader 级 trace 会解析为明确的 `ResponseSpec`，其 prediction
  只作为 loss/generation 的内部执行路由。
- `LoaderSpec.text(...)` / `TextCollator`：纯文本 MT loader，只读取 source/target text，当前可
  配置为 anydataset `WMT19` preset 或 deterministic toy text samples，不消费 codec/audio
  tokenizer。
- `LoaderSchedule` / `ScheduledDataLoader`：为唯一 `DataModule` 组织多个 homogeneous loader。
  未指定 `step_mode` 时保留历史 weighted accumulation window 行为。`fused_joint` 返回覆盖所有非零
  loader 的 `FusedBatch`，task coverage 完整时可用于 static DDP，存在未使用参数时也可使用
  find-unused DDP；`serial_joint` 每个 optimizer step 串行消费每个非零 loader 一次，必须使用
  find-unused DDP。weighted mode 的 loader weight 表示采样频率；joint mode 的 loader
  weight 表示归一化 task loss 权重。每个子 loader 自己保持单一 execution signature。
- `DatasetConfig` / `load_dataset()`：显式选择 `wmt19_tts`、`qwen_tts_speaker` prepared data、
  canonical online `covost2` / `libritts` validation source，或确定性的内存 `toy` data。
  CoVoST 2 固定返回 source audio/transcript + target translation 的 pair sample；LibriTTS 固定返回
  `Role.DEFAULT` waveform + normalized text 的 single sample。两者调用 workspace pinned loader，
  不接受本地 root/filter 覆写。`qwen_tts_speaker` 通过 workspace 加载
  `SpeakerAudioGrid`，再由 `SpeakerGridCellsDataset` 暴露 `Role.DEFAULT` flat cells；默认覆盖
  所有 speaker，也可用 `speaker` 显式选择一列。它的 `filter` 是绑定当前 waveform/codec
  grid snapshot 的 source-level selection revision，默认 `null`；不复用 WMT19-TTS 生成前的
  `speech_translation_v1` source filter。BiCodec prepared cell 使用
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
TTS 与 voice-clone TTS 使用同一 target-audio 输出路径，只在 context 上不同：

```text
TTS:             target text                -> full target audio
TTS_VOICE_CLONE: target text + source audio -> full target audio
```

voice-clone 数据必须使用普通 pair：`Role.SOURCE/AUDIO` 是音色条件，`Role.TARGET/TEXT` 是要朗读的
文本，`Role.TARGET/AUDIO` 是完整监督。两段 audio 可以使用不同 runtime/tokenizer；builder 只按 task
program 排列 target text 与 source audio，不假设 source audio 含有 global codes，也不从 source 拷贝
任何 target token。Qwen speaker grid 的 flat-cell adapter 仍服务 `shape=single` 普通 TTS；若要从
speaker grid 构造 voice-clone 数据，应在 dataset 边界产出上述 pair，而不是 `audio_context` wrapper。

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
  `diagnostic_samples()` 边界供 callback 读取 raw sample。可选 `validation` 接受单个 `LoaderSpec` 或
  `name -> LoaderSpec` 映射；named loaders 按映射顺序返回给 Lightning，均不进入 train schedule，
  也不复用 train loader instance。speech validation spec 可用 `max_samples` 取确定性前缀。
  `diagnostic_samples()` 显式选择 train/validation 数据源；`diagnostic_collator()` 为 panel 指定的
  单一 task 构造独立 collator，不修改训练 loader 的共享 task weights。speech 与 text train loader
  都提供这两个 diagnostic 边界；text loader 对 WMT19 iterable dataset 使用 global shard 的固定
  索引读取，不受 DDP rank 分片影响。validation 仍只接受独立的 speech loader。
  schedule 在 DataModule 构造时固定；切换 stage 必须启动绑定新 stage 的 run，不提供运行时
  loader-weight setter。

## Workspace codec 读穿补产

`SpeechConfig.materialization` 为 prepared workspace 增加 request-scoped read-through 路径。开启时
必须同时设置 `encode_missing_codes=true`，并让可选的 `codec_view` 与 runtime `AudioView` 完全一致。
runtime 的 input/output `AudioView` 都从对应 `audio_input.tokenizer` /
`audio_output.tokenizer` 推导；`materialization.codec_view` 只是针对 output 推导值的可选
防错校验，不是另一个表示选择轴。
`DatasetName.WMT19_TTS` 继续读取原 `moss_tts` 资源及其 selection。这个 read-through 路径面向有限的
prepared dataset；新的 live S2ST 入口不使用它。`zhuyin.datasets.s2st:source` 直接发布 waveform
catalog，训练 runtime 按 batch 在线编码。
解析顺序固定为：

1. 先按 workspace root、split、codec view 和 `DatasetConfig.filter` 读取现有 codec store；命中后
   直接复用，不创建后台任务。
2. codec store 或该 filtered selection 尚未发布时，读取同一个 filter 下的 waveform dataset 作为
   epoch 0 fallback，并创建完整补产任务。store 已存在但 manifest、payload 或 provenance 损坏时
   直接失败，不把损坏资产当成 miss。
3. workspace 连 filtered waveform selection 都不存在时，不允许退化为 `filter(None)`。调用方必须用
   `materialization.source_factory="module:attribute"` 注册包含目标过滤规则的流式 source，并显式提供
   `input_id`；否则无法安全确定输入身份并直接报错。该 source 在训练 setup 阶段先解析出稳定的
   filtered dataset，filter 本身不在训练 batch 内做稀疏增量发布。factory 必须在所有 rank 上返回
   相同长度、顺序和过滤结果，并自行保证并发调用安全。这里是两段式接口：所有 rank 都必须能 import
   的 module-level builder 接收 `AssetRequest` 并返回 dataset factory；builder 和返回值都必须是可由
   `spawn` pickle 的确定性对象，不能使用 lambda、局部函数或捕获不可 pickle 状态的 closure。factory
   会在每个 rank 的 setup 调用一次，并在 global owner 的 non-daemon spawn worker 中再次调用，因此
   必须幂等、可重入且只读。non-daemon worker 允许 materializer 在内部继续启动多设备/DataLoader
   子进程。返回值必须是有限、稳定且有 `__len__` 的 dataset，
   并提供 map-style `__getitem__` 或 anydataset 支持的显式 shard iteration；Sample 必须包含内置
   codec provider 所需的 waveform/file audio，以及 workspace pair schema 所需的 source/target text、
   `TextView.TEXT` 和 `TextMeta.LANG`。完整 frame codec view 使用 anydataset
   `CodecProvider.encode()`；GLM-4 等 tokenizer-only frame view 使用
   `AudioTokenizerProvider.tokenize()`；BiCodec 使用 structured `tokenize()`，并支持同一 WMT19
   sample 内 source/target 两个 audio reference。

两侧 backend/derived view 不同时，resolver 会分别查 input/output store，再按同一
sample index 合并。同 backend 但 BPE 不同时虽然 model token space 独立，prepared
codec codes 仍只读同一份 store。GLM-4 input / BiCodec output 的边界是：

- GLM-4 prepared store 通过 `AnyDataset.from_store(.../glm4)` 直接加载，并复用与 output
  相同的 WMT19 filter selection 索引；两侧 store 都 ready 时直接返回合并数据，不启动
  waveform fallback 或后台 job。
- GLM-4 input ready 但 BiCodec output 缺失时，fallback 把 ready GLM-4 view 合入同一
  waveform sample。训练 batch 只在 output 侧调 BiCodec online materializer，后台也只补产
  BiCodec store；到 epoch 边界仍要重载并验证两侧等长后才切 ready dataset。
- GLM-4 input store 缺失时，内置 workspace materializer 使用独立 tokenizer provider 补产，绝不
  复用 BiCodec output provider 伪造 input codes。GLM-4 loader 接受任意 checkout/fork，不绑定源码
  Git commit，但严格校验源码/API/模型契约、固定 weights revision 和 `transformers==4.44.1`；若
  output provider 的依赖不能与之共存，应在兼容环境中运行独立
  AnyDataset producer，再由本 resolver 消费持续发布的 store。

补产结果保持 workspace codec dataset 的读取格式，但它本身已经是过滤后的 composite store，因此
ready store 加载时不再二次应用 filter。逻辑请求由 dataset/source root、split、backend
tokenizer identity、由它推导的 codec view、
filter、`input_id`、`provider_id` 和 source factory 共同确定 request ID，写入目录为
`<output_root>/<request-id>/<codec-store-dir>`。这样不同过滤规则、输入版本或 codec provider 不会共用
同一补产目录；manifest provenance 必须精确匹配 `input_id` 和 `provider_id` 才能复用。这两个 ID 是
调用方维护的语义版本：source 内容、filter 实现或样本顺序变化时必须更新 `input_id`，codec checkpoint、
provider 配置或 code 语义变化时必须更新 `provider_id`，否则系统会按旧 provenance 合法复用旧资产。
BiCodec store 使用 anydataset 的 `{"semantic": Tensor, "global": Tensor}` schema。parser 直接把
两条流映射到 `AudioCodes.semantic_codes/global_codes`，不会经过 acoustic 字段或 layout 转换。
旧 `semantic/acoustic` BiCodec store 必须显式离线迁移，运行时不会静默兼容。

生命周期跨一个明确的 epoch 边界：epoch 0 的 DataLoader 使用 filtered waveform fallback，同时只有
global owner 在后台写完整 store；epoch 结束时 owner 的 `finish` 等待所有缺失样本写完，并把异常广播
给全部 rank。成功后先执行 DDP barrier，再由每个 rank `refresh_materialized_assets()`。Trainer 每个
epoch 重建 DataLoader，因此 epoch 1 开始读取 ready codec store，不再走 waveform fallback。这个边界
保证完整跑完一次 epoch 后，请求所需的全部 codec 数据已生成或训练以明确错误终止。通用后台进程、
状态机、跨 rank 同步和异常回收由 `anytrain.lightning` 提供；本项目只保留业务 request、DataModule
方法和 `AssetMaterialization` 薄适配。

`finish` 会在 epoch 边界等待全量补产完成，不设置短超时；因此卡死的 source/provider 也会让该边界持续
等待。训练若在到达 epoch end 前异常退出，callback/teardown 会 best-effort 终止后台 worker，本次不承诺
ready；
ViewMaterializer 的 resume state 会供后续同一请求继续补产。直接构造 DataModule/Trainer 的调用方必须
自行安装 `AssetMaterialization` callback，并设置 `reload_dataloaders_every_n_epochs=1`；正式
`scripts/train.py` 已自动配置这两个条件。

当前实现只允许唯一的 training loader，且它必须是启用补产的 speech loader，以提供有限且唯一的 reload
边界；validation 若也启用补产，必须与 training 指向同一个 codec source request。内置 provider 当前
支持 `wmt19_tts` 的 GLM-4 tokenizer-only frame view，DAC、LongCat、Stable Codec、UniCodec 完整
frame-code view，以及 semantic/global BiCodec view。WMT19 workspace 为每个 codec store 单独解析 filter selection；BiCodec selection
缺失时仍走同一 filtered waveform/stream fallback，不会拿 `base` selection 冒充 `bicodec` selection。
Qwen speaker-grid read-through 仍需要独立 backend：它拥有不同的 waveform/codec root、DEFAULT-role flat
cell 和 `speaker_grid_manifest.jsonl` 契约，不能沿用 WMT19 pair materializer 猜测格式。
`source_root` 和 `output_root` 必须在所有 rank 解析成完全相同的绝对路径；global owner 对 output 可写、
全部 rank 可读，且共享文件系统在 barrier 后必须一致可见。即使两个挂载前缀指向同一存储，路径字符串
不同也会形成不同 request ID。`device` 禁止使用 `auto`，并应显式选择不会与训练争抢显存且 codec backend
确实支持的单一设备。`write_workers` 只控制该后台进程内的 writer 线程，不创建额外顶层 materializer
进程；布局/吞吐字段 `max_shard_samples`、`max_shard_bytes`、`batch_size`、`commit_samples`、
`write_workers` 和 `write_prefetch` 不属于逻辑 request identity。

## 流式 S2ST 合成消费

正式入口在 CUDA runtime、distributed launch 和模型构造前解析
`datamodule.source.factory`。workspace source 统一提供 `access()`、`generate()` 和 `toy()`；
`access().load()` 在已有 snapshot 和尚无首版 snapshot 时都返回同一个 `LiveS2STDataset` facade。
训练入口只做路由和设备切分，不读取生成目录，也不把 dataset split、root 或 codec 配置注入 source。
invalid/corrupt access 在所有 mode 下直接失败，不允许用 toy 或 generation 掩盖损坏。

auto route 的行为固定为：

- 已有 sealed snapshot：只走 access，全部可见设备用于训练；
- 已有未 sealed snapshot：训练立即读取当前前缀；多卡恢复 generation，单卡只训练当前前缀；
- 尚无首版 snapshot：单卡使用 toy 数据测真实模型与在线 codec 性能；多卡启动 generation，训练侧
  live dataset 等待首个最终 snapshot；
- 显式 `access|generate|toy` 只用于诊断，不能改变 invalid/corrupt 的失败语义。

顶层 `devices` 只列 workspace 暴露的 factory 名和 `CUDA_VISIBLE_DEVICES` 内相对 id；当前 factory
只有 `translation` 与 `tts`。列表长度就是该 factory 的 replica 数，用来调节上下游吞吐；未列出的
设备全部留给 Lightning，且必须至少保留一张。最终 S2ST snapshot 发布 source/target waveform，
GLM4 semantic、BiCodec global/target 等 runtime view 都在训练进程中在线编码。

### Source family、增长和发布

workspace 配置声明语言、每种语言的有序 source slots、一个通用 translator、一个 TTS、speaker list
或 reference audio，以及 `initial_sources` / `interval_sources`。同一个物理来源可以重复声明；每个
slot 用稳定 name 持有独立 cursor。source row 被接纳后形成一个 source family，并为其他语言各产生
一条 source/target pair。source dataset 中已有的平行译文不作为 label，target text 始终由配置的
translator 生成。

family 接纳时确定 voice condition。speaker-list 模式从当前 speaker pool 确定性选择一次 speaker，
并用同一个 TTS/speaker 合成 source 与所有 target audio；reference-audio 模式保留原始 source
waveform，并把它作为所有 target audio 的同一个 TTS condition。新增 speaker 只影响以后接纳的
family；新增语言时先为旧 family 补新目标语言，再接纳新语言来源，最后恢复各语言 source slot 的
稳定轮转。

增长单位是 source family，而不是扁平 sample：

- `initial_sources` 控制首版最多接纳多少 family，可以设成刚好完成一个 optimizer step 或小规模
  overfit 的数量；
- `interval_sources` 控制后续每个 revision 最多新增或回填多少 family。

一个 revision 的逻辑依赖固定为：

```text
source@r -> translation@r -> tts@r
```

三个阶段共享 revision，但发布时间可以不同。下游只消费 manifest 中明确记录的 upstream snapshot，
不能重新查询某个模糊的 latest。`translation@r` 逻辑包含 `source@r`，`tts@r` 逻辑包含
`translation@r`；只有最终 `tts` snapshot 会推进训练可见 catalog。依赖关系、staging 恢复、精确
parent、store 校验和原子发布都由 workspace/anydataset 实现，不出现在训练配置中。

### DataModule、live refresh 和 cursor

source preflight 通过现有 `training_datasets` seam 把 `access.load()` 返回的同一个 live dataset 注入
唯一 speech loader。DataModule 只持有该对象并持续取 batch，不访问 workspace source、factory、stage
manifest 或 snapshot 路径，也不需要收到“请更新 dataset”的控制消息。`LiveS2STDataset` 在首版等待和
iterator/catalog 安全边界自行 refresh；每次观察到更长的 append-only catalog 前缀时只输出一条明确日志：

```text
data.snapshot.updated previous=... current=... added_samples=...
total_samples=... cursor=... wait_seconds=...
```

旧前缀、sample identity 或 lineage 被改写时 refresh 直接失败。live facade 自己按稳定 global index 做
rank 分片；入口强制一个 speech loader、`num_workers=0`、`persistent_workers=false`、关闭 cost batching
和 distributed sampler，避免 worker 或 sampler 冻结旧长度。validation 使用独立的 sealed dataset，
不从正在增长的训练 catalog 抽取固定索引。

`LiveS2STCursor` 把 dataset 的 `lineage_id`、`snapshot_id` 和 `pair_cursor` 交给 Lightning checkpoint，
并只在 `trainer.global_step` 真正前进后 acknowledge，因此 gradient accumulation 中间 batch 不会提前
提交。等待数据期间，dataset 的 stop predicate 同时检查 SIGTERM 和 generation service 健康状态；
factory 提前失败会立即暴露对应日志路径。`ManagedServiceCallback` 单独拥有 generation service 的
start/check/close，DataModule teardown 是 live dataset 的唯一 close owner。

### 日志边界

- toy：`data.plan` 标记 `formal_training=false`，输出独立 `toy-perf/`、warmup、measurement window、
  step time、throughput 和最终 perf report；不保存正式 checkpoint，也没有 generation stage 日志；
- generate：普通训练 loss、吞吐和 checkpoint 日志不变；`generation/translation.log` 与
  `generation/tts.log` 明确记录 plan、依赖/反压等待、每阶段计算时间和 snapshot publish，训练日志
  只增加首版等待与 `data.snapshot.updated`；
- access：保持普通正式训练日志，不安装 toy perf 或 generation telemetry。

## 输入输出

### 任务程序与响应 trace

response trace 是 loader 级数据课程配置，同一次训练可以用多个 homogeneous loader 分别采样 direct、
target CoT 和 full CoT。任务表示就是 CoT：lexical prompt 只用自然语言说明步骤，response 再按该
步骤顺序生成并监督对应字段；`trace` 只是选择 CoT 变体的配置键，不会映射成额外的 task token。协议
token 只保留字段分隔、路由、语言选择和音频/codec framing 所必需的部分，不承担 `<s2st>`、`<cot>`
一类 task identity。`trace` 是唯一 response 选择轴；未指定时使用 program 的 `default_response`，旧
`prediction` 配置会作为未知字段失败。S2ST 的三个逻辑序列为（`|` 左侧属于 prompt，右侧属于完整、
自回归监督的 response）：

```text
direct:     ... | BOA output_schema target-codec-sequence EOA
target_cot: ... | <mt><lang_en> target text </mt> BOA output_schema target-codec-sequence EOA
full_cot:   ... | <asr> source text </asr> <mt><lang_en> target text </mt> BOA output_schema target-codec-sequence EOA
```

中文目标把 `<lang_en>` 换成 `<lang_zh>`。当前 UniSS-style BiCodec 路径中，`target_cot` 的必需
response protocol 是 `<mt>`、目标语言 selector、target text、`</mt>`，随后是 BOA、output schema、
BiCodec private marker/order/payload grammar 和 EOA；`full_cot` 只在其前额外增加
`<asr> source text </asr>`。两者都不增加 `<s2st>` 或 `<cot>` task token。S2TT 的 `full_cot` 同理为
`... | <asr> source text </asr> <mt><lang_en> target text </mt>`。ASR/MT 不再借用通用 EOS：Runtime
在 lexical tokenizer 词表之后固定追加 `<asr>`、`</asr>`、`<mt>`、`</mt>`、`<lang_en>`、
`<lang_zh>` 六个可组合控制 ID；它们不写入或扩充 HF/Kimi tokenizer。普通 `TEXT_AR` 仍使用
tokenizer 自己的 EOS。audio 使用通用 BOA/EOA 包络，并在 BOA 后显式生成当前 output schema
selector；selector 之后才是 codec tokenizer 自己的 marker/order/payload grammar。

lexical prompt 只用自然语言描述任务和 response 顺序，不出现协议 token；完整 prompt 还包含
source/context 的音频协议序列，但不包含任何 response begin。builder 按 resolved response steps 编译
完整 CoT trace；从第一个 `<asr>`、`<mt>` 或 BOA 到最后一个 end token 都是普通 teacher-forcing
target。grammar 可以把某一步候选收窄为一个合法 token，但不能替模型插入该 token。

一条具体的 S2ST full CoT 样本如下。正式训练直接从已 materialize 的 `anydataset` codec store
读取 tensor。以 LongCat 的第 0 条样本为例，source/target codec payload 分别位于对应 audio view
的 tar shard 中；同名 `.pt` 是两个不同 view 下的成员：

```text
<snapshot>/longcat/
  samples.parquet
  source/audio/longcat/
    manifest.parquet
    shards/000000.tar :: 000000000000.pt  # source codec Tensor
  source/text/text/
    manifest.parquet
    shards/000000.tar :: 000000000000.txt # "今天天气很好。"
  target/audio/longcat/
    manifest.parquet
    shards/000000.tar :: 000000000000.pt  # target codec Tensor
  target/text/text/
    manifest.parquet
    shards/000000.tar :: 000000000000.txt # "The weather is nice today."
```

loader 解包后交给 collator 的逻辑样本等价于：

```python
{
    (Role.SOURCE, Modality.AUDIO): AudioItem(
        views={AudioView.LONGCAT: source_codec_codes},
    ),
    (Role.SOURCE, Modality.TEXT): TextItem(
        views={TextView.TEXT: "今天天气很好。"},
        meta={TextMeta.LANG: Lang.ZH},
    ),
    (Role.TARGET, Modality.AUDIO): AudioItem(
        views={AudioView.LONGCAT: target_codec_codes},
    ),
    (Role.TARGET, Modality.TEXT): TextItem(
        views={TextView.TEXT: "The weather is nice today."},
        meta={TextMeta.LANG: Lang.EN},
    ),
}
```

该样本由 loader 的 `task + trace` 选择 full CoT，不在 sample 中额外保存 task/prediction：

```yaml
loader_plan:
  loaders:
    s2st_full:
      weight: 1.0
      task_weights: {s2st: 1.0}
      trace: full_cot
```

这里 `source_codec_codes` 和 `target_codec_codes` 就是对应 `.pt` payload 经 CPU 上的安全
`torch.load` 得到的 tensor。任务 context 只暴露 source audio；source text、target text 和 target
audio 按 program 声明的顺序成为 response。正式配置保持 `encode_missing_codes=false`；只有显式开启
debug/materialization fallback 时才允许从 `AudioView.WAVEFORM` 现场生成缺失 codes。

该实例把自然语言 lexical prompt、prompt-owned source audio protocol 和监督 response protocol
明确分开：

```text
lexical prompt:
  Translate the following speech into English speech.
  Respond in this exact order:
  1. transcribe the source speech as text
  2. produce the English translation as text
  3. generate the corresponding English speech

source audio context (prompt-owned protocol):
  BOA input_schema <source .pt codec tokens> EOA

response protocol:
  <asr> 今天天气很好。 </asr>
  <mt> <lang_en> The weather is nice today. </mt>
  BOA output_schema <LongCat marker/order + target .pt payload> EOA

labels:
  lexical prompt 和 source audio context 全部为 -100
  response 全部监督：所有 begin/end、language selector、BOA、output_schema、
  codec-private marker/payload 和 EOA 的 label 都等于自身 token ID
```

其中 full CoT 的任务表示是自然语言三步 CoT 及其顺序监督，不是响应协议字面量；`trace=full_cot` 只
选择这套 CoT 编译规则。若选择 `target_cot`，lexical prompt 只描述“先输出目标译文、再生成目标语音”，response 从
`<mt> <lang_en> ... </mt>` 开始，不含 ASR step。两条 CoT 路径都不使用 `<s2st>` 或 `<cot>`。

上面的 source/target audio payload 直接来自各自 tar shard 内的 `.pt` tensor；正常路径不会先还原
WAV，也不会重新调用 codec encoder。WAV 只属于显式开启的缺失-code materialization/debug fallback。

AR 任务也使用同一个 program contract：`TEXT_AR` / `AUDIO_AR` 是空 context 的单字段 causal
continuation，`PARALLEL_AR` / `INTERLEAVED_AR` 声明多模态 response layout，`MASKED_AR` 使用独立
reconstruction objective。新增仍属于 causal serialized-sequence 的 AR 变体时，优先增加 program /
response preset；如果需要新的训练目标或执行算法（例如不同的 masked、diffusion executor），则必须
增加明确 objective/executor，不能只靠 prompt 假装成同一任务。
当前 serialized response grammar 支持 audio-only，或若干 text step 后跟至多一个末尾 audio step；
多个独立 audio step 必须先增加对应 executor，`ResponseSpec` 会在定义处拒绝这类未实现布局。

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

解耦模式不允许缺失 input view 时调用 output codec waveform fallback。普通显式 debug
fallback 若处理 raw input waveform，必须调用可加载的 `runtime.input_codec`；workspace read-through
则遵循上述方向性 provider 边界。若 input store 已 ready、output store
尚在后台补产，composite fallback 会把 ready input view
合入 waveform sample，只让缺失的 output target/audio context 进入现有 output codec materializer，
并在 epoch 边界等两侧 store 都严格可加载且等长后再切换到 ready dataset。

`ModelSample` 拆成与 generation 共用的输入侧 `Request`，以及仅训练使用的 `Labels`；collate
再导出 teacher-forcing 视图供 loss / backbone 消费：

```python
@dataclass
class ModelSample:
    request: Request   # task.contract.Request
    labels: Labels

# Request（与推理共用）
prompt_ids: Tensor
task: Task
trace: str | None                       # 训练保存 resolved name；推理可空=用 program default
target_language: str | None             # 含 MT step 时必填；其余 response 为空/省略
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
`source_ctc` / `target_ctc` / `audio_input_positions`，并由每行 `task + trace` 派生只读的
`predictions` 属性，供 loss 使用。

`AcousticTarget` 包含 `semantic_codes`、`codes`、`token_positions`。分组使必须共同存在的 tensor
不能形成半完整状态。

`Speech` 使用三分字段：`semantic_codes`、`global_codes`、`acoustic_codes`。fixed-layout anycodec
输入在 parser/构造边界归入 `global_codes`，frame-aligned 输入才进入 `acoustic_codes`。

audio target 始终直接使用 output runtime 已产生的 `target.audio_token_ids`。BiCodec target 因而是
`global + semantic` 的完整 self-describing sequence，外面包
`BOA + schema selector + ... + EOA`；source audio 的 codec 结构不会改变 response。外层控制 token、
codec-private marker 与 payload 全部属于 response 监督；`token_groups` 只负责为不同位置选择合法候选
范围，不负责代写 marker。

`ModelBatch` 额外保存 `tasks: list[Task]`、`traces: list[str]`、
`target_languages: list[str | None]` 和 `pad_token_id`，并公开
`predictions`（从 task/trace 派生）、`attention_mask` 与 `acoustic_target_mask`。speech batch 还保存
`audio_seconds: Tensor[B]`，表示每条训练样本按当前 task 实际消费的 source/target 音频秒数之和；
纯文本样本为 0。batch padding 把单条 prompt 边界聚合为 `generation_prompt_lengths`；raw sample
的显式 `audio_context` 会在 parser/collator 边界被拒绝，不进入 batch。
teacher-forcing generation bridge（`requests_from_batch`）切出 `prompt_ids`，并带上同批
resolved `trace` 与对应的 `target_language`，不从第一个非 `-100` label 反推。voice-clone 的 target
text 与 source audio 已在该 prompt 边界内；audio-target BOA/schema 不在 prompt，真实生成必须从
BOA 开始预测完整 response trace。

`audio_input_positions` 是每条序列中 source audio payload token 的位置，按 `[frames]` 保存，batch
padding 后为 `[batch, frames]`，右侧填充 `-1`。sample builder 为 task program 中可见的 source audio
payload 记录位置，包括 `AUDIO` 与 `TEXT_AUDIO` context；source BOA/EOA、target audio response 和
generated token 不进入该字段。它只服务 backbone input tower，不规定 input/output codec 的内部排布。

`source_ctc` / `target_ctc` 各自包含 audio span 的完整序列 `token_positions` 与 tokenizer-local
`text_token_ids`。sample builder 不按“任务输出是否为 audio”这一条粗规则决定 CTC，而是按 transcript
visibility 编译：ASR/S2TT/S2ST 有 source CTC；T2ST/S2ST 的纯 audio prediction 与 AUDIO_AR 有
target CTC；这里的 prediction 来自 resolved response，而不是独立配置。TTS、TTS_VOICE_CLONE、MT、
parallel/interleaved output 没有 target CTC。source positions 交给 non-causal
route 读取 `h[p]`，target positions 保留 token 自身位置并由 loss 读取 `h[p-1]`。audio span 指
tokenizer 序列化 payload：BiCodec stream marker 也可保留为 blank CTC step；外层
BOA/schema selector/EOA 则排除。

## 边界

- 包级 `speech_to_speech.datamodule` 只导出诊断坐标 `SampleSplit`。`DataModule`、`LoaderSpec`、
  配置结构、schedule、parser、sample、target、batch、protocol、collator 和 dataset factory
  从对应子模块导入，不提升为宽 facade。
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
  和 target，TTS_VOICE_CLONE 同样分别编码 source 与 target；对 S2TT/ASR 只编码音频 source，对
  T2ST/TTS 只编码音频 target；纯文本 task 不调用 codec。在线编码是 FP32 预处理边界，不属于
  backbone/acoustic decoder 的 autocast graph。
- toy dataset 只读取正式 runtime 的 codec identity 与 codebook metadata；它不提供 tokenizer、
  codec、layout 或 special token，因此不存在 toy runtime 分支。
- `parse.parser` 只解释 raw dataset representation；`build.sample` 只实现任务序列规则；
  `sample.py`、`target.py`、`batch.py` 分别保存领域样本、监督结构与训练 batch，私有
  `_batch_ops.py` 集中 padding/transfer/assembly。各层不反向读取彼此的私有逻辑。
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
- target 为 audio 时，完整 `BOA + schema selector + codec-private sequence + EOA` 都属于 response，
  `token_labels[len(input_ids):] == response_ids`；prompt 段才全部为 `-100`。
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
  voice-clone source audio 必须已位于该边界内，decode 不从 batch side channel 或 target codes 回填
  条件。
- `AudioMeta.DURATION` 的单位是秒，不是 codec frame 或 waveform sample。parser 优先读取并校验
  该元数据；缺失时用当前 codec view 的 frame count 除以 `runtime.codec_frame_rate` 推导
  `Speech.duration_seconds`。task sample builder 按 source/target modality 决定哪些角色计入
  `ModelBatch.audio_seconds`；不能把真实音频静默计为 0。
- 同一 `task_weights` 中的任务必须在当前 trace 下具有相同
  `(source_layout, response.prediction)` 执行签名，保证 DDP 各 rank 走相同模型路径。0 权重任务不会
  参与 batch 分配；每项权重必须有限且非负，总和必须有限且为正；
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
  shuffle。canonical `s2tt` / `tts` loaders 分别持有 CoVoST 2 pair config 和 LibriTTS single config；
  raw waveform collate 为 `RawSpeechBatch`，统一由 `OnDeviceCodecMaterializer` 编码。删除 named mapping
  时，旧单-loader入口仍可从现有 stage loader 复制 task/config 并切换 manifest `split_label`。
  validation diagnostic panel 按 loader name 读取对应独立数据源，并使用 task-specific collator。
- train loader 与 validation loader 使用独立的窄 Protocol。没有 validation spec 时
  `DataModule.val_dataloader()` 返回空 iterable，Lightning 不运行 validation；一个 validation 返回
  单 loader，多个 validation 返回稳定顺序的 loader list。
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
