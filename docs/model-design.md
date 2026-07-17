# Speech-to-Speech 设计总览

本文只维护跨模块契约：总体结构、数据契约、position 语义、任务定义和生成契约。各模块对外提供的能力、输入输出和边界见 `docs/design/`：

- [datamodule](design/datamodule.md)：raw sample 到 `ModelBatch` 的构造。
- [model](design/model.md)：semantic backbone、embedding 注入和 acoustic decoder。
- [loss](design/loss.md)：objective 组合和监督契约。
- [runtime](design/runtime.md)：已加载资源的单一入口。
- [pl_module 与 callback](design/pl_module.md)：训练循环、生成路径和日志。
- [codec oracle](design/codec_oracle.md)：codec screening 的正式模型复用与 prepared-code 数据边界。

阶段状态与工程待办不在设计文档中维护，见 `docs/experiments/todo.md`。

## 1. 总体结构

```text
Raw Sample
    │
    ▼
Task / Prompt Builder
    │
    ▼
ModelBatch
    ├── semantic input / labels
    ├── acoustic prompt
    └── acoustic target
         │
         ▼
SpeechToSpeechFlowModel / SpeechToSpeechRVQModel
    ├── Input Embedding
    ├── Semantic Backbone
    ├── Text / Audio Logits
    └── Acoustic Decoder
         ├── Flow Matching Decoder
         └── Causal RVQ Decoder
```

设计原则：

1. semantic stream 是主序列。
2. acoustic stream 只为已经可见的 speech semantic span 提供 side-channel。
3. response 的 acoustic target 不作为 semantic backbone 的输入。
4. semantic backbone 和 acoustic decoder 通过稳定的 hidden-state contract 连接。
5. `SpeechToSpeechFlowModel` 和 `SpeechToSpeechRVQModel` 是显式的模型组装入口。
6. codec 是实验级共享身份，同时决定 runtime codec、dataset codec view 和 raw audio view。

## 2. 数据契约

### 2.1 Speech

```python
@dataclass
class Speech:
    semantic_ids: Tensor                    # [L, M_semantic]
    acoustic_ids: Tensor | None             # [L, M_acoustic]
    text_ids: Tensor
    language: Language

    @cached_property
    def bpe_ids(self) -> Tensor: ...

    @cached_property
    def bpe_spans(self) -> Tensor: ...
```

`semantic_ids` 和 `acoustic_ids` 共享同一个 frame 轴，但可以有不同的 codebook 数量。
`None` 表示当前 codec/profile 没有独立 acoustic side-channel；unified-token codec 的唯一
codebook 放在 `semantic_ids [L, 1]`，`acoustic_ids=None`。

`bpe_ids` 是 audio tokenizer 产生的本地 BPE ids。`bpe_spans` 通过
`audio_tokenizer.frame_spans(bpe_ids)` 取得，与 `bpe_ids` 等长，记录每个 BPE token
覆盖的 semantic frame 数量，且必须完整覆盖全部 semantic frame；span 查询不解码和重建
frame 内容。

`audio_tokenizer.decode(bpe_ids)` 返回 frame-level semantic units，形状为 `[T, K_semantic]`；每个 frame unit 包含全部 semantic codebooks。`frame_spans()` 只返回各 token 覆盖的 frame
数量。tokenizer 只负责单条序列的 token 到 frame 解码；batch 各行的 `T` 对齐和 padding
由调用方负责。

`global_ids` 不属于 `Speech` 的缓存属性。datamodule 在拼接 backbone 输入时使用 layout 将 `bpe_ids` 转成 global ids。

`language` 在 raw sample 边界转换为 `Language`；`zh` / `zh-cn` / `chinese` 与
`en` / `en-us` / `english` 分别归一化为 `ZH` / `EN`，instruction 使用枚举值
`Chinese` / `English`。

### 2.2 ID 空间与 special token

所有跨模块接口必须区分 local ID 与 global ID：

- `Speech.text_ids` 是 text tokenizer local ID。
- `Speech.bpe_ids` 是 audio tokenizer local ID。
- `Speech.acoustic_ids` 是 codec local ID。
- `ModelBatch.input_ids`、`labels` 和 generation sequence 是 layout global ID。

audio layout block 与 audio head 覆盖完整的 `semantic audio tokens + BOA + EOA`。但以下三个集合语义不同，不能混用：

- audio head block：semantic audio tokens、BOA、EOA；用于模型输出维度。
- audio generation allowed IDs：semantic audio tokens、EOA；BOA 已由 prompt builder 添加，不能再次生成。
- codec-decodable audio IDs：仅 semantic audio tokens；BOA、EOA 不能传给 audio tokenizer 或 codec。

text BOS/EOS 属于 text tokenizer 原生 vocabulary，因此不需要额外追加 head 行；正式 text
generation 直接选择 text head，并屏蔽 PAD/BOS、保留 EOS。Runtime 负责公开上述范围或判断
能力，消费方不自行从 layout block 推导。

### 2.3 ModelBatch

```python
ACOUSTIC_PAD_ID = -1


@dataclass
class ModelBatch:
    input_ids: Tensor
    labels: Tensor
    acoustic_input_ids: Tensor | None
    acoustic_input_positions: Tensor | None
    semantic_frame_labels: Tensor | None
    acoustic_labels: Tensor | None
    acoustic_label_positions: Tensor | None
    tasks: list[Task]
```

字段职责：

- `acoustic_input_*`：输入条件，只覆盖 prompt 中已经可见的 audio semantic span，供 semantic backbone 注入 source acoustic prompt。
- `semantic_frame_labels`：target codec semantic codebooks，与 acoustic target 组成可在线
  解码的完整 codec codes；不携带特定 teacher feature。
- `acoustic_labels`：输出监督，只用于训练 acoustic decoder，不能注入 semantic backbone。
- `acoustic_label_positions`：输出 frame 对应的 semantic label position（见 2.4）。

padding 与 mask 约定：

- `input_ids` 使用 `runtime().pad_token_id` padding；semantic `labels` 按 Transformers causal LM 契约使用 `-100` padding，且与 `input_ids` 未移位对齐（shift 在 loss 内部完成）。
- acoustic IDs 是未加 global offset 的 codec 局部 ID，使用 `ACOUSTIC_PAD_ID = -1` padding；position tensor 同样以 `-1` padding，有效值必须小于对应序列长度。
- mask 不作为独立字段存储，由 padding 值派生并以 `cached_property` 缓存；`ModelBatch` 在完成 padding 和 device transfer 后视为不可变。
- acoustic IDs 传给 codec 或 embedding 前，必须先通过派生 mask 把 `-1` 替换成合法的占位 ID，并在得到 feature 后重新应用 mask。
- 一个 batch 的 acoustic 字段必须整体存在或整体不存在，不支持部分 row 为 `None`。由 task sampler/bucketing 保证同一 batch 的 source/target modality 一致，避免 DDP 中不同 rank 走不同参数路径。

`ModelBatch.from_samples()` 是进入模型前建立跨字段不变量的唯一边界：

- `input_ids` 与 `labels` shape 相同。
- `acoustic_input_ids` 与 `acoustic_input_positions` 必须同时存在或同时不存在。
- `acoustic_labels` 与 `acoustic_label_positions` 必须同时存在或同时不存在。
- acoustic 字段存在时，batch/frame 轴严格对齐，有效 position 指向非 padding token。
- 同一 batch 的 task 必须具有相同的 source/target modality，即相同模型执行路径。

`ModelBatch` 是训练或 teacher-forcing evaluation 的完整监督结构，不表达缺少 target 的真实推理请求。真实推理使用独立的 generation 输入接口。

### 2.4 Position 语义

设 target semantic BPE token `bpe_k` 在完整序列中的位置为 `p`，则 `labels[p] = bpe_k`（labels 未移位）。

- `acoustic_input_positions`：source acoustic frame 对应的输入序列位置，指向该 frame 所属 source BPE token 的位置。
- `acoustic_label_positions`：target acoustic frame 记录 `p`，即该 frame 所属 target BPE token 自己的位置。

所有 data、loss、teacher-forcing 和 generation 调用方统一传 **token 自身位置 `p`**。causal LM 的 position `p - 1` 预测 `labels[p]`，这一 shift 只由 model 内部处理，不暴露给调用方：

- backbone hidden 路径：`target_frame_condition(hidden_states, acoustic_label_positions)`，内部取 `hidden[p - 1]`。
- oracle 路径：`target_frame_label_condition(labels, acoustic_label_positions)`，直接读取并嵌入 `labels[p]`。

两条路径的调用方约定完全对称。生成时每一步的最后一个 hidden 预测本步采样 token；若采样出 audio BPE token，则立即按该 token 的 span 展开并收集 hidden。EOA/EOS 的 hidden 不进入 acoustic frame condition。

## 3. 任务定义

任务与输出类型必须保持一致：

| Task | 输入 | semantic target | acoustic target |
| --- | --- | --- | --- |
| ASR | audio | text | no |
| S2TT | audio | text | no |
| S2ST | audio | audio semantic | yes |
| TTS | text | audio semantic | yes |
| T2ST | text | audio semantic | yes |
| T2TT | text | text | no |
| TEXT_AR | none | text | no |
| AUDIO_AR | none/audio context | audio semantic | yes |

任务能力以 `Task` 枚举为唯一事实来源。`Task` 公开 source modality、target modality、
paired 语义和 instruction template；task builder、collator、generation、loss 与 callback
不各自维护 audio-task 集合。modality 使用 `anydataset.types.Modality`，不使用裸字符串。

所有 audio-target task（AUDIO_AR、S2ST、T2ST、TTS）必须提供 semantic audio target；是否
提供 acoustic target 由 codec/profile 决定。具有独立 acoustic representation 的 codec 必须
提供完整 target，unified-token codec 使用 `acoustic_ids=None`。所有 text-target task 都不允许
acoustic target。T2TT 使用 paired source/target text，作为纯文本训练与 backbone retention
evaluation 的正式任务语义。

## 4. Runtime 与模块所有权

Runtime 聚合一套相互兼容的已加载资源：backbone、text/audio tokenizer、codec、layout、special IDs 和 flow runtime。layout 依赖 backbone/tokenizer vocabulary 与 codec/audio-tokenizer vocabulary，因此这些资源属于同一个不可替换的 runtime snapshot。

Runtime 不是 `nn.Module`，持有模块不等于注册模块。训练所有权由 model 的显式 `nn.Module` 属性决定：

- 同一可训练模块只在 model 中注册一条路径；backbone text embedding 不能同时作为 backbone 子模块和 multimodal embedding 子模块重复注册。
- 一个 Runtime 对应一个训练模型组合；不承诺从同一 Runtime 构造多个相互独立训练的模型。
- 冻结 codec 可以留在 Runtime；若 codec 需要训练，model 必须显式注册并纳入 optimizer/checkpoint。
- model、loss、datamodule 必须使用同一个 runtime snapshot 或同一个顶层 codec 身份，不能分别隐式选择不兼容资源。

## 5. Model 与 Loss 契约

Model 提供 semantic forward、condition 对齐和具体 acoustic decoder 能力；Loss 只依赖结构化 Protocol，不依赖具体组合类，也不向下读取 `model.runtime` 获取 objective 资源。

公开 `forward()` 保持完整 global logits 契约。正常训练使用 `semantic_hidden()` 取得完整序列
表示，Loss 只选取有效 target 的 predictor position，再调用 `semantic_logits()` 计算 global
text+audio 分布；因此 objective 数值语义不变，但不为 prompt/padding position 构造大词表 logits。

正常训练只表达一种合法组合：所有 batch 计算 semantic CE；存在 acoustic target fields 时
额外计算模型对应的 acoustic objective。audio target 但 `acoustic_ids=None` 的 unified-token
batch 只计算 semantic CE。`semantic`、`flow_matching` 等布尔开关不用于表达正常训练路径；
REPA 通过显式 teacher 与数值权重加入。

REPA teacher 不属于 dataset：loss 接收可替换的在线 teacher。当前 WavLM teacher 把完整
target codec codes 解码为 waveform，重采样至 16 kHz，使用冻结 WavLM-base 第 9 层作为
目标；8 层 DiT 使用第 4 层作为 student representation。

codec acoustic representation 是 Runtime 的固定契约：codebook 输入、feature dimension 和 waveform decode 必须匹配。model 不通过 `acoustic_codebooks` 任意切取 batch codebook；比较不同 representation 时选择不同 codec/profile。

## 6. 生成契约

生成分成两条清晰路径：

- `ModelBatch -> semantic_hidden -> sparse semantic_logits -> loss/metrics`：训练或 teacher-forcing evaluation。
- `prompt + task + AcousticPrompt | None -> semantic generation -> acoustic generation -> decode`：真实推理。

semantic generation 使用 KV cache 作为正式实现：首步编码完整多模态 prompt 并注入 source acoustic condition，后续只输入新 token 并复用 `past_key_values`。cache 只属于一次 generation 调用，不保存在 model 实例上；公开 generation API 不暴露具体 backbone cache 类型。

`AcousticPrompt` 只允许用于 audio-source task；text-source 或无 source 的 task 携带该结构
属于非法请求，由 generation service 在调用 model 前显式拒绝。

状态机约定：

- text target：`prompt -> text tokens -> EOS`。
- audio target：`prompt + BOA -> semantic audio tokens -> EOA`。
- model 层返回包含 prompt 与 stop token 的完整 sequence；上层 service 裁剪为不含 BOA/EOA 的 response。
- semantic token 被采样时在线收集用于预测它的 hidden；设备侧 span lookup 在循环结束后一次
  展开 acoustic frame condition，不逐 token 同步 CPU，也不追加全序列 forward。

Model 提供 semantic generation 与可选 acoustic sampling 原语；上层 generation service
负责任务状态机、modality head、变长裁剪、frame 对齐和 decode。具有独立 acoustic
representation 时采样 features；unified-token codec 直接 decode semantic codes，并返回
`AudioOutput(features=None, waveform, sample_rate)`。采样率由生成所用 codec 提供。

batch generation 使用标准批量自回归：变长 prompt 左 padding 并使用 attention
mask，每行独立跟踪 EOS/EOA；frame 轴以 padding + frame mask 贯穿 acoustic
sampling，decode 前按每行实际 frame 数裁剪。generation service 按 target modality
和 acoustic prompt 执行签名分组，保留请求的原始返回顺序。

## 7. 数据与阶段配置

`workspace` 提供多种已处理逻辑对象的加载入口；具体工程负责选择实际训练数据。speech-to-speech 的 DataModule 通过 `wmt19_tts_codec(config.codec)` 取得 codec view，而不是让 workspace 替工程选择数据集。

codec 是实验级共享身份：同一个顶层配置值同时传给 Runtime 与 DataModule，并决定 runtime codec、dataset codec view 和 raw audio view。DataModule 不通过全局 runtime 偷读 codec。

DataModule 持有一个可更新的 Collator。初始 strategy 在构造时确定；stage callback 只在
epoch 边界调用 `Collator.set_strategy()` 替换任务权重。`persistent_workers=False` 使下一
epoch 的 worker 获得更新后的 collator 状态，阶段切换不依赖 Trainer 隐式重建 DataLoader。

正式多任务 DDP 允许 stage 间使用不同子模块，采用 `find_unused_parameters=True`。每个 batch
的所有 rank 必须使用相同执行签名；静态 codec oracle 显式冻结目标路径外的参数，继续使用
`find_unused_parameters=False`。后续在正式训练路径稳定后再评估冻结策略或 DDP 优化。
