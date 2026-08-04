# loss

组合训练 objective，消费 `ModelBatch` 和模型公开接口，产出含标量 `loss` 的 mapping。
跨模块 position 所有权见 [设计总览](../model-design.md)。

## 对外能力

- `Objective[ModelT]`：统一描述 model/objective 配对的泛型 `nn.Module` 契约，确保 objective
  的参数和子模块参与设备迁移、checkpoint 与 DDP。`forward(batch, model)` 只产出训练所需
  loss；`validation(batch, model)` 默认复用 forward，允许具体 objective 在同一次前向中补充
  teacher-forcing validation 指标。
- `TokenObjective`：只组合 text/audio token CE；`FlowObjective`：组合 token CE、
  acoustic flow matching 和可选 REPA；`RVQObjective`：组合 token CE 与 acoustic RVQ CE。
  三者的 `forward(batch, model)` 都返回含标量总损失的 `Outputs`，直接满足 Lightning
  训练契约。
- `TokenLoss`：按 `ModelBatch.prediction_modality`（有效 loader prediction，不是
  `Task.prediction_modality` 默认值）展开监督 head（TEXT / AUDIO / 两者），在对应
  局部词表上计算 CE，每行必须至少包含一个非 `-100` target；causal shift 在此完成，只把有效
  predictor hidden states 交给 `model.token_logits(hidden, modality)`，text/audio head 不做跨模态
  softmax 竞争。`target_modality` 只是单模态 prediction 的便捷属性，mixed 时为 `None`，不作为
  loss 入口。BiCodec grammar 额外消费逐位置 `token_groups` 与 `model.selected_logits()`，只在当前
  semantic、semantic-or-end 或 acoustic codebook candidate group 上计算 restricted CE。
- `CTCAlignmentLoss`：对 transcript-latent audio span 做冻结文本头监督。source route 在 audio
  自身位置读取 `h[p]` 并使用非因果 decoder，target route 在 causal predecessor 读取 `h[p-1]` 并使用
  因果 decoder；两侧可分别选择 backbone readout、pooling 和 identity/linear/transformer topology，
  最终都通过冻结 tied text readout 得到 tokenizer-local text logits。blank 是 runtime PAD 在 text
  block 内的 local ID。每条 route 的 CTC 先按 transcript token 数归一；source/target 权重项在同一
  row 内相加，组合项再按有效 `sequences`（有任一 active route 的样本行数，而不是 audio span 数）聚合。
- `FlowLoss`：直接从 `semantic-acoustic-generator.loss` 包级导出；S2S 只保留 joint
  token/acoustic objective 的组合，不再维护独立 loss 子模块或重命名 alias。
- `MaskedCodebookCrossEntropyLoss`：直接从 `anytrain.loss` 包级导出；训练 forward
  的 `details` 只保留逐行 `codebook_N` 和有效 frame 数。`RVQObjective.validation()` 显式请求
  `codebook_N_top1`，训练 step 不额外执行大码本 argmax；acoustic padding ID 不进入 decoder
  embedding、loss 或 accuracy。
- `WavLMTeacher`：由 `semantic-acoustic-generator.loss` 提供；按 boolean frame mask 在线解码 target
  semantic/acoustic codes，以 16 kHz waveform 运行冻结 WavLM，取得配置层的 hidden states 并插值、
  写回原有效 frame 位置。
- `MaskedCosineAlignmentLoss`：把选定 DiT block 的逐帧表示投影到 WavLM hidden dimension，再转给
  `anytrain.loss.MaskedCosineAlignmentLoss` 与 stop-gradient teacher features 做 masked cosine distance；
  输出必须携带逐行有效 `frames`，总 REPA loss 按该计数加权。
- `types.Outputs`：上层日志与训练消费的 S2S objective mapping；`LossItem` 和通用 output 聚合来自
  `anytrain.loss`。
- `types.loss_items()`：按 token、flow matching、REPA、RVQ 的稳定顺序遍历实际存在的
  分项，供 callback 和实验 summary 复用。
- `validation_metrics()`：把 objective outputs 转成 `anytrain.evaluator.weighted.Metric`，并统一
  拥有 token/RVQ/Flow/REPA validation 指标名和有效 token/frame count 语义；通用加权聚合不在本模块
  重复实现。

## Objective 组合

三个组合入口共享 token forward；ownership 不在 objective 内动态切换，sample builder 已经把
self-describing prompt/response streams 编译成 labels 和 prediction groups：

```python
hidden = model.objective_hidden_output(
    batch.input_ids,
    ctc_routes=active_ctc_routes,
    attention_mask=batch.attention_mask,
)
token = self.token(
    hidden.token,
    batch.token_labels,
    batch.prediction_modality,
    model.token_logits,
)
result = {
    "loss": loss_item_mean(token, unit="tokens", fallback_to_mean=False),
    "token": token,
}

if self.ctc is not None:
    ctc = self.ctc(
        hidden.token,
        source_hidden_states=hidden.source_ctc,
        target_hidden_states=hidden.target_ctc,
        source=batch.source_ctc,
        target=batch.target_ctc,
        decode=model.ctc_logits,
    )
    result["ctc"] = ctc
    result["loss"] += loss_item_mean(
        ctc,
        unit="sequences",
        fallback_to_mean=False,
    )
```

存在独立 acoustic target codes 时，flow 与 RVQ 入口再执行各自分支：

```python
target_data = batch.acoustic_target

# FlowObjective
condition = model.target_frame_condition(
    hidden.token,
    target_data["token_positions"],
)
target = model.acoustic_target_latent(target_data["codes"])
acoustic = self.flow_matching(
    model.acoustic_decoder,
    condition,
    target,
    batch.acoustic_target_mask,
    self.flow_runtime,
)

# RVQObjective
teacher_forced_codes = target_data["codes"].masked_fill(
    ~batch.acoustic_target_mask[..., None],
    0,
)
logits = model.acoustic_logits(
    hidden.token,
    target_data["token_positions"],
    teacher_forced_codes,
)
rvq = self.rvq(logits, target_data["codes"], batch.acoustic_target_mask)
```

所有 batch 都计算 token CE。是否增加 acoustic objective 只由
`batch.acoustic_target is not None` 决定，不通过 task modality 猜测 codec
representation，也不通过模式布尔开关表达组合。BiCodec grammar 的 structured global payload 是
token objective 的 grouped CE，不是 frame-aligned `acoustic_target`；结构化 target fields 或
prediction groups 不完整时直接报错。

BiCodec grouped CE 的 candidate group 由 tokenizer 唯一拥有：codec/stream markers 使用 forced
group，不计算 loss；semantic 首 token 使用 semantic group，后续 semantic tokens 与 sequence
end 使用 semantic-or-end group；每个 global slot 使用对应 global codebook
range。`selected_logits()` 返回与该候选集合同序的 logits，target 不在集合内时显式失败。

`TokenObjective` 不要求 model 提供 acoustic 能力。`FlowObjective` 固定组合 token CE 与
flow matching；传入包含正数 `weight` 和 `teacher` 的 `RepaConfig` 时显式加入 REPA。
`RVQObjective` 固定组合 token CE 与 codebook causal CE。训练入口显式选择 model/objective
配对，不在 objective 内按具体模型类型猜组合。

REPA 只属于 flow 组合。teacher 显式接收 `acoustic_target["semantic_codes"]`、
`acoustic_target["codes"]` 和 `acoustic_target_mask`；dataset 不绑定 WavLM 型号、层号或
teacher features。acoustic-only codec screening 与 oracle artifact 导出由
`semantic-acoustic-generator` 维护，本仓库只组合 joint token/Flow/RVQ objective。

## 边界

- 训练侧有效 prediction 以 `ModelSample.prediction` / `ModelBatch.predictions`（`batch.prediction_modality`）
  为准；`Task.prediction_modality` 只是 loader 未覆写时的默认值。loss、FLOPs、sample/batch 校验都消费
  有效 prediction，不回退到 task 默认。
- `TokenObjective`、`FlowObjective` 和 `RVQObjective` 只依赖结构化 Protocol 的
  `layout`、`token_hidden_states()`、`objective_hidden_output()`、`ctc_logits()`、
  `token_logits(hidden, modality)`、`target_frame_condition()`、`acoustic_decoder` 等公开能力，
  不依赖具体模型类。
- target position 表示 token 自身位置 `p`；causal predictor shift `p - 1` 由 model 的
  `target_frame_condition()` 统一处理，objective 不重复偏移。
- CTC position 同样表示 audio token 自身位置 `p`，但 source/target shift 由 CTC route 自身拥有：
  source 不偏移，target 偏移一次。CTC 不从 task 名动态推断 route；datamodule 已按 transcript
  visibility 编译 `source_ctc` / `target_ctc`。
- `SpeechToSpeechModule` 通过泛型 `Objective` 保留 model/objective 的配对类型，不在训练循环中
  cast。
- validation 指标名、RVQ codebook detail 解释和有效单位由 loss 模块唯一负责；名称与训练
  TensorBoard 路径对齐（`token/loss`、`acoustic/rvq/...`、`acoustic/flow_matching/loss`、
  `acoustic/repa/loss`），pl_module 只消费 `validation.Metric`，通过
  `anytrain.lightning.validation.log()` 加 `val/` 前缀接入 Lightning epoch/DDP aggregation。
- flow runtime 等 objective 资源在 `FlowObjective` 构造时显式传入，不通过
  `model.runtime` 向下读取。
- 子 objective 在 `__init__` 中构造完毕，forward 不挂载新 submodule。
- flow matching、RVQ CE 和 REPA 在非线性 loss 计算前把无效 frame 替换为安全值，并只归约
  boolean mask 选中的 frame；padding 位置的 NaN/Inf 不参与 forward，也不产生梯度。
- token、flow matching、RVQ 与 REPA `LossItem` 必须分别携带 `tokens` 或 `frames` 有效单位；
  objective 不在单位缺失时静默退回逐行平均。
- CTC `LossItem` 携带 `sequences`，inactive/padded row 的 count 为 0；全 padding CTC row 的 loss
  精确为 0，总损失使用 zero-safe sequence mean，不产生 `0 * NaN`。日志和 validation 不把这些 row
  纳入 task mean。稳定路径为 `alignment/ctc/loss`，route detail 分为 source/target loss、transcript
  tokens 与 audio steps。
- token 行损失是有效 token 的加权平均；`details` 中的 `text_loss` / `audio_loss` 仅供观测，不改变
  训练标量。validation 暴露聚合 `token/loss`（经 `val/` 前缀写入 logger），暂不拆
  `token/text_loss` / `token/audio_loss`。
- generation 按有效 `Request.prediction`（缺省则 `task.prediction_modality`）分组；训练 bridge
  会把 `ModelBatch.predictions` 写入 Request。
- BiCodec prompt/output ownership 不改变 Flow/RVQ acoustic objective 的 frame-aligned contract；
  target acoustic stream 不会静默泄漏到 reference prompt，prompt codes 也不作为 token labels。
- `causal_lm.py` 只实现离散 acoustic RVQ objective，不读取 model/runtime 或重复 condition
  对齐；其稳定输出键是 `rvq`。
- REPA teacher 始终保持 eval/frozen；teacher features detach，梯度只进入 DiT 与 student
  projector。声学 route 的 batch-free 训练入口由 `semantic-acoustic-generator` 维护，S2S 不再复制
  route-specific loss/teacher 实现。
