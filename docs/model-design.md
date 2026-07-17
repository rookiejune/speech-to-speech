# Speech-to-Speech 设计总览

本文维护跨模块契约：总体结构、数据结构、position 语义、模型组合和生成边界。模块级能力见 `docs/design/`：

- [datamodule](design/datamodule.md)：raw sample 到 `ModelBatch`。
- [model](design/model.md)：token backbone、embedding 注入和 acoustic decoder。
- [loss](design/loss.md)：objective 组合与监督。
- [runtime](design/runtime.md)：已加载资源及窄协议。
- [pl_module 与 callback](design/pl_module.md)：训练集成、generation service 和日志。
- [codec oracle](design/codec_oracle.md)：codec screening 的实验边界。

阶段状态与待办见 `docs/experiments/todo.md`。

## 1. 总体结构

```text
Raw Sample
    -> parser.parse_sample(runtime)
    -> SpeechPair
    -> sample.build_sample(task, runtime)
    -> ModelSample
    -> ModelBatch
    -> SpeechToSpeechFlowModel | SpeechToSpeechRVQModel
         -> token backbone
         -> text / semantic-audio token heads
         -> flow | RVQ acoustic decoder
```

设计原则：

1. 全局 token 序列同时容纳 text token 与 semantic-audio token。
2. acoustic stream 只为已经可见的 speech token span 提供 side channel；response acoustic target 不注入 backbone。
3. backbone 和 acoustic decoder 通过 frame-aligned hidden-state contract 连接。
4. runtime 在入口创建并显式传给 model 与 datamodule；底层 model/data 代码不读取 singleton。
5. flow 与 RVQ 是显式组合，非法配置不能通过未消费字段静默进入模型。

## 2. 数据契约

### 2.1 Speech

```python
@dataclass
class Speech:
    semantic_codes: Tensor       # [frames, semantic_codebooks]
    acoustic_codes: Tensor | None
    text_token_ids: Tensor       # text tokenizer local IDs
    audio_token_ids: Tensor      # audio tokenizer local IDs
    audio_token_spans: Tensor    # semantic frames per audio token
    language: Language
```

`semantic_codes` 与 `acoustic_codes` 共用 frame 轴。unified-token codec 没有独立 acoustic side channel，因此使用 `acoustic_codes=None`。

parser 在 raw sample 边界完成以下工作：

- 根据 `DataRuntime.audio_view` 解释 codec view。
- 用 text/audio tokenizer 生成 local token IDs。
- 用 `frame_spans(audio_token_ids)` 生成 span，并校验 span 完整覆盖 semantic frames。
- 把 raw language 转成 `Language`；未知值显式报错。

`Speech` 只保存解析后的数据，不持有 runtime，也不通过 cached property 隐式编码。

### 2.2 ID 空间

跨模块名字遵循固定词汇：

- `*_token_ids`：tokenizer 或 layout 序列。
- `*_codes`：codec codebook index。
- `*_labels`：直接参与 token CE 的 target。

具体字段：

- `Speech.text_token_ids`、`Speech.audio_token_ids` 是 tokenizer local ID。
- `Speech.semantic_codes`、`Speech.acoustic_codes` 是 codec local code。
- `ModelBatch.input_ids`、`ModelBatch.token_labels` 和 generation sequence 是 layout global token ID。

audio layout block 包含 semantic-audio tokens、BOA、EOA；以下集合不能混用：

- audio head block：semantic-audio tokens、BOA、EOA。
- audio generation allowed IDs：semantic-audio tokens、EOA。
- codec-decodable audio IDs：仅 semantic-audio tokens。

text generation 使用 text head，屏蔽 PAD/BOS 并保留 EOS。集合与 range 由 Runtime 暴露，消费方不重复推导。

### 2.3 ModelBatch

```python
@dataclass
class ModelBatch:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_prompt: AcousticPrompt | None
    acoustic_target: AcousticTarget | None
    tasks: list[Task]
    pad_token_id: int
```

字段职责：

- `acoustic_prompt`：`codes` 与 `token_positions` 共同表示 source acoustic condition。
- `acoustic_target`：`semantic_codes`、`codes` 与 `token_positions` 共同表示 decoder target、
  codec/REPA 输入和逐帧全局 audio token 位置。

padding 与 mask：

- `input_ids` 使用 batch 自带的 `pad_token_id`；`token_labels` 使用 `-100`，shift 由 token loss 完成。
- codec codes 与 frame positions 使用 `ACOUSTIC_PAD_ID=-1`。
- `attention_mask`、`acoustic_prompt_mask` 和 `acoustic_target_mask` 由 padding 值派生并缓存。
- codec 接口只接收合法 code；调用前把 padding 替换为安全值，得到 feature 后重新应用 mask。

`ModelBatch.from_samples(samples, pad_token_id=...)` 是跨字段校验边界：

- input 与 token label 必须是对齐的一维序列。
- acoustic prompt/target 以完整结构出现，内部 tensor 共用 frame 轴。
- position 必须指向序列内非 padding token。
- 同一 batch 的 task 必须具有相同 source/target modality 执行签名。

真实推理不使用缺 target 的半成品 `ModelBatch`，而使用独立的 `generation.Request`。

### 2.4 Position 语义

设 target audio token 在完整序列中的位置为 `p`，则 `token_labels[p]` 是该 token，label 未移位。

- `acoustic_prompt["token_positions"]` 指向 source frame 所属的可见 prompt token。
- `acoustic_target["token_positions"]` 记录 target frame 所属 token 自身的位置 `p`。

所有调用方统一传 token 自身位置：

- `target_frame_condition(hidden_states, positions)` 在 model 内取 causal predictor `hidden[p - 1]`。
- `target_frame_label_condition(token_labels, positions)` 直接读取并嵌入 `token_labels[p]`。

generation 每采样出一个 codec-decodable audio token，就收集预测该 token 的最后一个 hidden，并按 `audio_token_spans` 展开为 frame condition。EOA/EOS 不进入 acoustic condition。

## 3. 任务定义

| Task | source | token target | acoustic target |
| --- | --- | --- | --- |
| ASR | audio | text | no |
| S2TT | audio | text | no |
| S2ST | audio | semantic audio | codec-dependent |
| TTS | text | semantic audio | codec-dependent |
| T2ST | text | semantic audio | codec-dependent |
| T2TT | text | text | no |
| TEXT_AR | none | text | no |
| AUDIO_AR | none | semantic audio | codec-dependent |

`Task` 是 source modality、target modality、`uses_source_role` 和 instruction template 的唯一事实来源。task builder、collator、generation 与 objective 不维护重复的任务集合。

## 4. Runtime 与所有权

Runtime 聚合互相兼容的 backbone、text/audio tokenizer、codec、layout、special token IDs 与 flow runtime。

- model 接收满足 `TokenModelRuntime`（flow 额外满足 `FlowModelRuntime`）的显式 runtime。
- datamodule/collator 只依赖窄 `DataRuntime` Protocol。
- 入口可通过 singleton 完成一次组装；parser、sample builder、batch padding 不读取 singleton。
- DataModule 在加载 prepared dataset 前比较 `config.codec` 与 `runtime.codec_name`。
- 同一可训练 `nn.Module` 只注册在 model 的一条 ownership path 下。

## 5. Model 与 Objective

`model.Config` 只配置 token backbone 周边的 semantic-audio adapter、output adapter 和 acoustic prompt adapter。acoustic composition 使用独立结构：

```python
class DecoderConfig(TypedDict):
    hidden_dim: int | None
    layers: int
    heads: int
    ffn_ratio: int

class FlowRepaConfig(TypedDict):
    feature_dim: int
    student_layer: int | None
```

`SpeechToSpeechFlowModel` 接收 `decoder` 与可选 `repa`；`SpeechToSpeechRVQModel` 只接收 `decoder`，因此 RVQ 无法接收后被忽略的 REPA 字段。Hydra 使用单一 `acoustic.type=flow|rvq`，flow preset 独占 teacher 与 student REPA 配置。

model 的训练能力是：

- `token_hidden_states()`：返回完整 backbone 表示，不构造 vocabulary logits。
- `token_logits()`：在有效 predictor rows 上构造 global text+audio logits。
- `target_frame_condition()`：把 target token position 对齐到 acoustic frame。
- flow/RVQ 各自提供 acoustic target 与 decoder 能力。

`TokenObjective`、`FlowObjective`、`RVQObjective` 只依赖结构化 Protocol。所有 batch 计算 token CE；存在 acoustic target 时，组合对应的 flow 或 RVQ objective。REPA 只属于 flow，通过显式 teacher 与正数 weight 加入。

## 6. Generation

训练与推理是两条独立路径：

- `ModelBatch -> token_hidden_states -> sparse token_logits -> objective`
- `Request -> generation service -> token/audio generation -> decode -> Result`

`speech_to_speech.generation` 拥有 `Request`、`Result`、service、decode 与 text evaluation；`pl_module` 只负责 Lightning 集成。

model 对外提供：

- `generation_step()`：供私有自回归循环使用的单步、目标 head 前向契约。
- `generate_tokens()`：text 或 semantic-audio token generation。
- `generate_audio_condition()`：生成 audio tokens 及 frame-aligned condition。
- `generate_audio_features()`：flow/RVQ 组合返回 sequence 与 codec acoustic features。

通用 `generate_sequence()` 自回归循环位于私有 `model/_generation.py`，具体模型不跨文件调用
基类私有方法。循环首步编码完整多模态 prompt，后续复用 KV cache；cache 只属于单次调用。

`AcousticPrompt` 使用 `codes` 与 `token_positions`，只允许出现在 audio-source task。service 按 target modality 和是否存在 acoustic prompt 分组，左 padding 变长 prompt，逐行追踪 EOS/EOA，并恢复原请求顺序。

状态机：

- text：`prompt -> text tokens -> EOS`。
- audio：`prompt + BOA -> semantic-audio tokens -> EOA`。

service 把 model sequence 裁剪为不含 stop token 的 `Result.response_ids`。有独立 acoustic representation 时用 features 解码；unified-token codec 直接解码 semantic codes，返回 `AudioOutput(features=None, waveform, sample_rate)`。

## 7. Data 与阶段配置

DataModule 显式持有 runtime 与一个可更新的 Collator。初始 `task_weights` 在构造时确定；`StageSwitcher` 在 epoch 边界调用 `set_task_weights()`。`persistent_workers=False` 使下一 epoch worker 获得更新状态。

同一组 task weights 只能包含相同 source/target modality 的任务，以保证 batch 执行签名稳定。正式多任务 DDP 使用 `find_unused_parameters=True`；静态 codec oracle 冻结目标路径外参数并使用静态 DDP 契约。
