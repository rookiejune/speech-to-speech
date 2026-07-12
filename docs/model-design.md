# Speech-to-Speech 设计总览

本文只维护跨模块契约：总体结构、数据契约、position 语义、任务定义和生成契约。各模块对外提供的能力、输入输出和边界见 `docs/design/`：

- [datamodule](design/datamodule.md)：raw sample 到 `ModelBatch` 的构造。
- [model](design/model.md)：semantic backbone、embedding 注入和 acoustic decoder。
- [loss](design/loss.md)：objective 组合和监督契约。
- [runtime](design/runtime.md)：已加载资源的单一入口。
- [pl_module 与 callback](design/pl_module.md)：训练循环、生成路径和日志。

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
    language: str

    @cached_property
    def bpe_ids(self) -> Tensor: ...

    @cached_property
    def bpe_spans(self) -> Tensor: ...
```

`semantic_ids` 和 `acoustic_ids` 共享同一个 frame 轴，但可以有不同的 codebook 数量。single-codebook codec 仍使用 `[L, 1]` 的 `acoustic_ids`；`None` 只表示当前任务不需要 acoustic side-channel，不表示 codec 没有 acoustic code。

`bpe_ids` 是 audio tokenizer 产生的本地 BPE ids。`bpe_spans` 与 `bpe_ids` 等长，记录每个 BPE token 覆盖的 semantic frame 数量，且必须完整覆盖全部 semantic frame。

`audio_tokenizer.decode(bpe_ids)` 返回 frame-level semantic units，形状为 `[T, K_semantic]`；每个 frame unit 包含全部 semantic codebooks。tokenizer 只负责单条序列的 token 到 frame 解码；batch 各行的 `T` 对齐和 padding 由调用方负责。

`global_ids` 不属于 `Speech` 的缓存属性。datamodule 在拼接 backbone 输入时使用 layout 将 `bpe_ids` 转成 global ids。

`language` 当前为 `str`；`Language` 枚举已提供归一化逻辑但尚未接入（见 todo）。

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

text BOS/EOS 属于 text tokenizer 原生 vocabulary，因此不需要额外追加 head 行；generation 仍通过统一的 allowed-token 接口约束目标 modality。Runtime 负责公开上述范围或判断能力，消费方不自行从 layout block 推导。

### 2.3 ModelBatch

```python
ACOUSTIC_PAD_ID = -1


@dataclass
class ModelBatch:
    input_ids: Tensor
    labels: Tensor
    acoustic_input_ids: Tensor | None
    acoustic_input_positions: Tensor | None
    acoustic_labels: Tensor | None
    acoustic_label_positions: Tensor | None
    tasks: list[Task]
```

字段职责：

- `acoustic_input_*`：输入条件，只覆盖 prompt 中已经可见的 audio semantic span，供 semantic backbone 注入 source acoustic prompt。
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
| TEXT_AR | none | text | no |
| AUDIO_AR | none/audio context | audio semantic | yes |

任务能力以 `Task` 枚举为唯一事实来源。`Task` 公开 source modality、target modality 和 paired 语义；task builder、collator、generation、loss 与 callback 不各自维护 audio-task 集合。modality 使用 `anydataset.types.Modality`，不使用裸字符串。

所有 audio-target task（AUDIO_AR、S2ST、T2ST、TTS）都必须提供 semantic audio target 与 acoustic target；缺失时在 task 构造 `Sample` 时立即报错。所有 text-target task 都不允许 acoustic target。

## 4. Runtime 与模块所有权

Runtime 聚合一套相互兼容的已加载资源：backbone、text/audio tokenizer、codec、layout、special IDs 和 flow runtime。layout 依赖 backbone/tokenizer vocabulary 与 codec/audio-tokenizer vocabulary，因此这些资源属于同一个不可替换的 runtime snapshot。

Runtime 不是 `nn.Module`，持有模块不等于注册模块。训练所有权由 model 的显式 `nn.Module` 属性决定：

- 同一可训练模块只在 model 中注册一条路径；backbone text embedding 不能同时作为 backbone 子模块和 multimodal embedding 子模块重复注册。
- 一个 Runtime 对应一个训练模型组合；不承诺从同一 Runtime 构造多个相互独立训练的模型。
- 冻结 codec 可以留在 Runtime；若 codec 需要训练，model 必须显式注册并纳入 optimizer/checkpoint。
- model、loss、datamodule 必须使用同一个 runtime snapshot 或同一个顶层 codec 身份，不能分别隐式选择不兼容资源。

## 5. Model 与 Loss 契约

Model 提供 semantic forward、condition 对齐和具体 acoustic decoder 能力；Loss 只依赖结构化 Protocol，不依赖具体组合类，也不向下读取 `model.runtime` 获取 objective 资源。

正常训练只表达一种合法组合：所有 batch 计算 semantic CE；audio-target batch 额外计算模型对应的 acoustic objective。`semantic`、`flow_matching` 等布尔开关不用于表达正常训练路径；oracle、REPA 等 diagnostic 或消融使用独立 objective/入口。

codec acoustic representation 是 Runtime 的固定契约：codebook 输入、feature dimension 和 waveform decode 必须匹配。model 不通过 `acoustic_codebooks` 任意切取 batch codebook；比较不同 representation 时选择不同 codec/profile。

## 6. 生成契约

生成分成两条清晰路径：

- `ModelBatch -> forward -> loss/metrics`：训练或 teacher-forcing evaluation。
- `prompt + task + source acoustic condition -> semantic generation -> acoustic generation -> decode`：真实推理。

semantic generation 使用 KV cache 作为正式实现：首步编码完整多模态 prompt 并注入 source acoustic condition，后续只输入新 token 并复用 `past_key_values`。cache 只属于一次 generation 调用，不保存在 model 实例上；公开 generation API 不暴露具体 backbone cache 类型。

状态机约定：

- text target：`prompt -> text tokens -> EOS`。
- audio target：`prompt + BOA -> semantic audio tokens -> EOA`。
- model 层返回包含 prompt 与 stop token 的完整 sequence；上层 service 裁剪为不含 BOA/EOA 的 response。
- semantic token 被采样时在线收集用于预测它的 hidden，并按 BPE span 展开为 acoustic frame condition；不再生成完成后额外做一次全序列 forward。

Model 提供 semantic generation 与 acoustic sampling 原语；上层 generation service 负责任务状态机、allowed tokens、变长裁剪、frame 对齐和 decode。同一次请求的 token、acoustic output 与 waveform 必须来自同一次生成结果。

batch generation 的目标契约是标准批量自回归：变长响应通过 padding 和 attention mask 处理，每行独立跟踪 eoa；frame 轴同样以 padding + frame mask 贯穿 flow sampling 和 `decode_features()`，不要求 batch 内各行 frame 数相等。

当前实现是过渡状态：`pl_module` 的 generation 按行循环调用单样本路径，`generate_audio` 对 batch 内不等长的 frame 展开直接报错；这些限制属于实现欠账（见 todo），不属于契约。

## 7. 数据与阶段配置

`workspace` 提供多种已处理逻辑对象的加载入口；具体工程负责选择实际训练数据。speech-to-speech 的 DataModule 通过 `wmt19_tts_codec(config.codec)` 取得 codec view，而不是让 workspace 替工程选择数据集。

codec 是实验级共享身份：同一个顶层配置值同时传给 Runtime 与 DataModule，并决定 runtime codec、dataset codec view 和 raw audio view。DataModule 不通过全局 runtime 偷读 codec。

DataModule 持有一个可更新的 Collator。初始 strategy 在构造时确定；stage callback 只在 epoch 边界调用 `Collator.set_strategy()`，清空 task/weight 缓存。`persistent_workers=False` 使下一 epoch 的 worker 获得更新后的 collator 状态，阶段切换不依赖 Trainer 隐式重建 DataLoader。

第一版 DDP 允许 stage 间使用不同子模块，采用 `find_unused_parameters=True`。每个 batch 的所有 rank 必须使用相同执行签名；后续在路径稳定后再评估冻结策略或 DDP 优化。
