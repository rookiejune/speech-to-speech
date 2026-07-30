# Speech-to-Speech 设计总览

本文维护跨模块契约：总体结构、数据结构、position 语义、模型组合和生成边界。模块级能力见 `docs/design/`：

- [datamodule](design/datamodule.md)：raw sample 到 `ModelBatch`。
- [model](design/model.md)：token backbone、embedding 注入和 acoustic decoder。
- [loss](design/loss.md)：objective 组合与监督。
- [runtime](design/runtime.md)：已加载资源及窄协议。
- [generation](design/generation.md)：独立 `Request -> Result` 推理、batching、评估与 decode。
- [pl_module 与 callback](design/pl_module.md)：Lightning 训练集成与日志。
- [reporting](design/reporting.md)：实验入口复用 `anytrain.lightning.window_summary` 的窗口摘要边界。

已验证结论见 [experiments/conclusion](experiments/conclusion.md)，尚未完成的复验与工程欠账见
[experiments/todo](experiments/todo.md)。

## 1. 总体结构

```text
Raw Sample
    -> parser.parse_sample(runtime)
    -> SpeechPair
    -> sample.build_sample(task, runtime)
    -> ModelSample
    -> ModelBatch
    -> FlowModel | RVQModel
         -> source audio payload -> AudioInputTower (optional) -> selected input embeddings
         -> token backbone
         -> text / semantic-audio token heads
         -> aligned hidden state -> HiddenConditionAdapter -> flow | RVQ acoustic decoder
```

设计原则：

1. 全局 token 序列同时容纳 text token 与 semantic-audio token。
2. source audio 的可配置 input tower 只 overlay 已显式标记的 payload 位置；BOA/EOA、target/generated
   audio 和 route reference context 不进入该 tower。
3. acoustic stream 只为已经可见的 speech token span 提供 side channel；response acoustic target 不注入 backbone。
4. backbone 和 acoustic decoder 通过 frame-aligned hidden-state contract 连接；adapter 显式隔离
   backbone hidden dimension 与 acoustic generator condition dimension。
5. runtime 在入口创建并显式传给 model 与 datamodule；底层 model/data 代码不读取 singleton。
6. flow 与 RVQ 是显式组合，非法配置不能通过未消费字段静默进入模型。

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
    acoustic_target: AcousticTarget | None
    audio_input_positions: Tensor | None
    tasks: list[Task]
    pad_token_id: int
```

字段职责：

- `acoustic_target`：`semantic_codes`、`codes` 与 `token_positions` 共同表示 decoder target、
  codec/REPA 输入和逐帧全局 audio token 位置。
- `audio_input_positions`：`[batch, frames]` 的 source audio payload 位置；右侧 `-1` 是 batch
  padding，不包含 BOA/EOA、target/generated audio 或 route reference context。

padding 与 mask：

- `input_ids` 使用 batch 自带的 `pad_token_id`；`token_labels` 使用 `-100`，shift 由 token loss 完成。
- codec codes 与 frame positions 使用 `ACOUSTIC_PAD_ID=-1`。
- `attention_mask` 和 `acoustic_target_mask` 由 padding 值派生并缓存。
- codec 接口只接收合法 code；调用前把 padding 替换为安全值，得到 feature 后重新应用 mask。

`ModelBatch.from_samples(samples, pad_token_id=...)` 是跨字段校验边界：

- input 与 token label 必须是对齐的一维序列。
- acoustic target 以完整结构出现；未 padding 的 codes 必须是非空二维非负整数 tensor，
  内部 tensor 共用 frame 轴。
- position 必须指向序列内非 padding token。
- `audio_input_positions` 中的有效位置必须唯一，并指向 runtime codec audio range；source 不是
  audio 时该字段为 `None`。
- 同一 batch 的 task 必须具有相同 source/target modality 执行签名。

真实推理不使用缺 target 的半成品 `ModelBatch`，而使用独立的 `generation.Request`。

### 2.4 Position 语义

设 target audio token 在完整序列中的位置为 `p`，则 `token_labels[p]` 是该 token，label 未移位。

- `acoustic_target["token_positions"]` 记录 target frame 所属 token 自身的位置 `p`。

所有调用方统一传 token 自身位置：

- `target_frame_condition(hidden_states, positions)` 在 model 内取 causal predictor `hidden[p - 1]`。
- `target_frame_label_condition(token_labels, positions)` 直接读取并嵌入 `token_labels[p]`。

generation 每采样出一个 codec-decodable audio token，就收集预测该 token 的最后一个 hidden，并按 `audio_token_spans` 展开为 frame condition。EOA/EOS 不进入 acoustic condition。

`audio_input_positions` 是另一套 position contract：它记录 source audio payload token 自身在完整
prompt 中的位置，只供 `AudioInputTower` 做输入 embedding overlay；不能与 target 的
`acoustic_target.token_positions` 混用。

## 3. 任务定义

| Task | source | token target | acoustic target |
| --- | --- | --- | --- |
| ASR | audio | text | no |
| MT | text | text | no |
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
- 组合入口显式构造并传递 `Runtime`；parser、sample builder、batch padding 不读取全局状态。
- DataModule 在加载 prepared dataset 前比较 `config.codec` 与 `runtime.codec_name`。
- 同一可训练 `nn.Module` 只注册在 model 的一条 ownership path 下。

## 5. Model 与 Objective

`model.Config` 配置 token backbone 周边的 semantic-audio input/output adapter，以及可选的 source-audio
input tower。semantic-audio output adapter 是逐 token 的 hidden-to-audio projection，必须保持
pointwise 以兼容 cached generation；acoustic composition 使用独立结构：

```python
@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: int | None = None
    layers: int = 8
    heads: int = 8
    ffn_ratio: int = 4

class FlowRepaConfig(TypedDict):
    feature_dim: int
    student_layer: int | None
```

`FlowModel` 接收 `decoder` 与可选 `repa`；`RVQModel` 接收
`decoder` 与可选 `codebook_embeddings`，但无法接收后被忽略的 REPA 字段。Hydra 使用
`model/acoustic=none|flow|rvq`，`none` 只训练 semantic audio token，flow preset 独占
teacher 与 student REPA 配置；训练组装由 `speech_to_speech.pl_module.composition` 持有，
入口脚本只传入解析后的配置；root schema 直接复用基础 `model.Config`。UniCodec 也按
`FrameCodec` 处理，`runtime=unicodec model/acoustic=none` 走 full-code token 序列，只是完整
frame 里只有一个 codebook。有独立 acoustic codebook 的 codec 只有在配置 semantic-only artifact
或选择 full-code sequence 时才可以作为 token-only baseline。ODE sampler 由 `runtime.Config.flow_*` 统一拥有；
入口只校验 flow/RVQ 所需的 codec capability，不自动改写 composition。

semantic-only 路线的跨仓库训练分为两个 phase：

```text
Phase A: semantic codes -> SAC conditioner -> acoustic generator pretraining
Phase B: aligned backbone hidden state -> HiddenConditionAdapter -> initialized generator joint training
```

Phase A 由 `semantic-acoustic-codec` 拥有。Phase B 通过 `acoustic.init_artifact` 加载 SAC 的
`AcousticGeneratorArtifact`，只迁移 generator 权重和其 condition/acoustic contract；SAC semantic/reference
conditioner 不进入 S2S model。artifact I/O、route/backend/config 校验由 `pl_module.composition` 持有，
`FlowModel`/`RVQModel` 构造器只接收已加载对象，不读取路径。

`HiddenConditionAdapter` 固定为 `LayerNorm + Linear(backbone_hidden_dim, condition_dim)`，teacher-forcing 与
autoregressive generation 共用同一路径，并归入现有 `ACOUSTIC_DECODER` 参数组。当前初始化仅支持
frame-aligned generator：Flow 同时迁移 decoder 和 feature normalization；RVQ 只支持 SAC
`codebook_ar`，`MTP` 与 fixed-length artifact 显式失败。联合初始化只校验 acoustic metadata；artifact
里的 semantic vocab/embedding 是来源记录，不约束 hidden-state consumer。

`runtime.semantic_codec_artifact` 是另一条独立用途：它配合 `model/acoustic=none` 加载完整 semantic
support 做 waveform reconstruction，不初始化联合训练 decoder，并继续校验 semantic + acoustic 全部
backend metadata。新增 adapter 改变了 Flow/RVQ checkpoint schema；旧 S2S checkpoint 缺少
`acoustic_condition.*`，strict resume 会显式失败，不做隐式补参。

source-audio input tower 的数据流是：

```text
source audio semantic token IDs
    -> semantic_audio_embedding
    -> AudioInputTower (none | mlp | transformer)
    -> overlay inputs_embeds at audio_input_positions
    -> Qwen backbone
```

`mlp` 是逐帧 gated projection；`transformer` 是保持帧数的非 causal encoder。tower 只服务输入
表示，不能读取或修改生成中的新 token，也不参与 `audio_output_adapter`、Flow/RVQ decoder
或 audio response grammar。显式位置由 datamodule/sample builder 和 generation request 传递；
没有 source audio 时为 `None`。

model 的训练能力是：

- `token_hidden_states()`：返回完整 backbone 表示，不构造 vocabulary logits。
- `token_logits(hidden, modality)`：在有效 predictor rows 上只构造 target modality 的局部
  vocabulary logits；省略 modality 时为通用 forward 构造 global text+audio logits。
- `target_frame_condition()`：把 target token position 对齐到 acoustic frame。
- flow/RVQ 各自提供 acoustic target 与 decoder 能力。

`TokenObjective`、`FlowObjective`、`RVQObjective` 只依赖结构化 Protocol。所有 batch 计算 token CE；存在 acoustic target 时，组合对应的 flow 或 RVQ objective。REPA 只属于 flow，通过显式 teacher 与正数 weight 加入。
token CE 的 softmax 只覆盖 task 的 target modality，不让 text/audio head 跨模态竞争；flow、RVQ
与 REPA 只对 boolean mask 选中的有效 frame 计算非线性 loss，padding NaN/Inf 不参与梯度。

## 6. Generation

模块级 API、输入校验和 service/model/runtime 边界见 [generation](design/generation.md)。

训练与推理是两条独立路径：

- `ModelBatch -> token_hidden_states -> sparse modality token_logits -> objective`
- `Request -> generation service -> text generation | audio strategy -> decode -> Result`

语义 seq2seq 是基础且完整的模型能力：`model/acoustic=none` 只预测 text token 或 audio token。
音频重建按 backend capability 分成两条路径：FrameCodec 使用
`FULL_CODEC_SEQUENCE` 展开全部 codebooks，生成后调用 `FrameCodec.decode(full_codes)`；
SemanticAcousticCodec 只生成 semantic units，再由 `semantic-acoustic-codec` 的
`SemanticCodecRuntime` 预测缺失 acoustic units 并重建波形。普通 FrameCodec 的 `decode()`
不接收 semantic-only codes。两条路径不改变 token model、objective 或 `Request -> Result` 契约。
训练数据只要求基础 `Codec`，codec table 初始化要求 `CodebookCodec`，Flow/RVQ 的 acoustic feature
训练显式要求 `AcousticCodec`；semantic-only waveform decoder 不属于 anytrain。UniCodec 虽然只有
一个 codebook，也按 `FrameCodec` 的 full-code path 解码。

`speech_to_speech.generation` 拥有 `Request`、`Result`、通用 service、audio capability strategy、
decode 与 text evaluation；`pl_module` 只负责 Lightning 集成。

model 对外提供：

- `generation_step()`：供私有自回归循环使用的单步、目标 head 前向契约。
- `generate_tokens()`：text 或 semantic-audio token generation。
- `generate_audio_condition()`：生成 audio tokens 及 frame-aligned condition。
- `generate_audio_features()`：flow/RVQ 组合返回 sequence、padded codec acoustic features 与
  每行有效 frame count。

通用 `generate_sequence()` 自回归循环位于私有 `model/_generation.py`，具体模型不跨文件调用
基类私有方法。循环首步编码完整多模态 prompt，后续复用 KV cache；已结束行会同步从输入和 cache
移除，只让 active rows 继续计算。cache 只属于单次调用。

当前 generation 只接收已经映射到 layout global ID 空间的 semantic-token prompt；audio-source
内容和 text-source 内容都编码在 `Request.prompt_ids` 中。audio-source request 可额外携带
`audio_input_positions`，让可配置 `AudioInputTower` 只覆盖 source payload 的 embedding，不改变
序列长度或 generation grammar。service 只按 target modality 分组，左 padding 变长 prompt，逐行
追踪 EOS/EOA，并恢复原请求顺序；KV cache 首步之后不再重复运行 source tower。

状态机：

- text：`prompt -> text tokens -> EOS`。
- audio：`prepared prompt ending in BOA -> semantic-audio tokens -> EOA`；service 不追加 BOA。

service 把 model sequence 裁剪为不含 stop token 的 `Result.response_ids`。有独立 acoustic
representation 时直接复用 model 返回的 frame count 裁剪 features；`(token count, frame count)`
相同的行合并执行 codec decode。unified-token codec 直接解码 semantic codes，返回
`AudioOutput(features=None, waveform, sample_rate)`。

## 7. Data 与阶段配置

DataModule 显式持有 runtime 与 Collator。一个正式 job 只运行一个 stage；loader/task 权重在
DataModule 构造时确定，训练过程中保持不变。task weights 位于进程共享数组，持久 worker 在
collate 时读取；worker 侧 runtime 是不含 backbone/codec 的数据快照。

同一组 task weights 只能包含相同 source/target modality 的任务，权重必须有限、非负且总和为
正，以保证每个子 batch 的执行签名稳定。task 与 loader 权重只控制进入训练 step 的数据频率，
不额外乘到 loss 上；每个 microbatch 独立按有效 token/frame 归约 token、flow、RVQ 与 REPA loss，
再由 Lightning 在 accumulation window 内累积梯度。

`scripts/overfit.py` 只用于 fixed-sample overfit、smoke 和参数冻结合同验收；正式训练入口是
`scripts/train.py`。`configs/stage/stage_*.yaml` 是 Stage 0-4 的数据计划契约：每个 stage 显式声明 loader
权重、loader 内 task 权重和 `accumulate_grad_batches`。多 loader schedule 在每个 accumulation
window 内按权重交错单个 homogeneous microbatch，不构造联合 batch tuple。独立的
`parameter_policy` 显式声明可训练参数组、
冻结参数组和 `backbone_top_fraction`，入口在 Trainer 创建前应用一次。正式
`experiment=train/staged_joint_stage_1..4` 当前约定
Stage 1-2 使用 speech-interface policy，Stage 3 解冻 Qwen 顶部 1/3 block 与 final norm，
Stage 4 使用 full policy；stage 本身不隐式选择 policy。RVQ decoder 的结构性冻结参数始终保持
frozen。正式 joint entry 使用 `ddp_find_unused_parameters_true`，因为一个 microbatch 只执行自身
task 分支；optimizer step 在配置数量的 microbatch 后发生。

需要参数高效适配时，`model.lora` 使用 Hugging Face PEFT 向现有 Qwen backbone 注入 LoRA，并与
`parameter_policy=lora` 成对选择；项目不维护轻量 LoRA 层。原始 backbone 保持冻结，adapter 与
speech/acoustic interface 按 policy 训练。LoRA 的正式文本保真度先由固定
`TextRetentionLogger` baseline 验证。

这里的 Stage 0-4 只表示 S2S 数据、任务和参数策略日程，不是上文 Phase A/B。SAC generator pretraining
在进入任一使用 `acoustic.init_artifact` 的 S2S experiment 之前独立完成；S2S stage 不在运行中创建或
替换 SAC artifact。

正式 train 只在配置了独立 speech validation spec 时让 `val_dataloader()` 返回真实 loader；
没有 spec 时返回空 iterable，text train loader 不被复用为 validation。teacher-forcing 指标由 loss 层的
`validation_metrics()` 解释 objective 输出，通用 count-weighted epoch/DDP aggregation 和可恢复历史
由 `anytrain.lightning.validation` 提供。
