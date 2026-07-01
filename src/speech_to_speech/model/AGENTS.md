# model

## 对外职责

`model` 负责提供训练和生成所需的模型对象。调用方优先通过 `Orchestrator` 使用模型，不直接拼接 Qwen3、embedding、LM head 和 DiT 的内部模块。

对外能力：

- 加载或构建 Qwen3 semantic decoder。
- 将 Qwen3 text token 和 LongCat BPE audio token 放入同一个 token space。
- 将 `BOA`/`EOA` 注册为 idspace special token，并提供覆盖 audio token 和 BOA/EOA 的 LM head。
- 接收 data module 构造的 batch，执行 semantic token 训练。
- 从 semantic labels 推导目标侧 BPE token，使用 Qwen3 shifted hidden states 和 LongCat BPE 展开得到 frame-level acoustic condition。
- 提供基于 DiT 的连续 acoustic flow loss 入口；LongCat discrete acoustic code 到连续 target feature 的转换由调用方或后续数据层显式提供，source acoustic condition 只在调用方显式传入 feature extractor 时从 batch source side 池化得到。
- 提供 semantic-only 生成入口，将生成的 LongCat BPE token 显式展开回原始 semantic ids，用于最小生成 sanity check 和后续评估。
- 提供 supervised semantic token accuracy 计算入口，统计范围与 semantic loss 使用的 supervised positions 对齐。
- 提供基于 Qwen3 hidden states 的 DiT acoustic condition 生成接口；离散 BPE token 只用于自回归反馈、停止条件和可选 debug。
- 提供 full-sequence waveform 生成编排入口：semantic BPE 生成、frame-level acoustic condition 展开、外部 acoustic feature generator 调用和 LongCat codec `decode_features()`。
- 提供 full-sequence serial、BPE-boundary diagonal 和旧 fixed-size diagonal acoustic flow 调度边界。`serial` 是语义解码完成后的常规整段 acoustic flow；`diagonal_bpe` 按每个 BPE token 展开的帧数作为非均匀 chunk 做 wavefront 调度。旧 fixed-size `diagonal` 和 `causal_window` 暂保留用于对照，不作为默认路径。

## 模块边界

- `orchestrator.py` 是模型对外入口，负责 forward、semantic/acoustic condition、flow loss 和生成编排。
- `builder.py` 负责把配置和运行期资源组装成 Qwen3、token embedding、LM head、adapter、DiT 和训练开关。
- `runtime.py` 负责构造 Qwen3/LongCat 共享 `IdSpace`；`token_space.py` 负责 embedding 替换、special token embedding 和 LM head；`audio_embedding/` 负责 audio BPE embedding 的 lookup 和 semantic-composition 两种内部实现；`trainable.py` 负责模型级参数训练策略。
- `semantic/` 负责 semantic token 增量生成、EOA 停止、acoustic condition hidden 收集，以及 semantic LM supervised positions、loss weights 和 batch row loss 归约。
- `acoustic/` 是 acoustic 公开入口；调用方从 `model.acoustic` import，不依赖内部文件。
- `acoustic/condition.py` 负责训练侧 acoustic condition 展开、连续 acoustic flow loss 和相关校验。
- `acoustic/generation.py` 负责 DiT acoustic feature generator、`AcousticSampler` 和 waveform 生成侧 acoustic sampler 选择。
- `acoustic/flow.py` 负责对接 `anytrain.framework.flow_matching`，提供 acoustic flow source 和 full-sequence sampler 适配。
- `acoustic/diagonal.py` 负责 acoustic flow 的 full-sequence baseline、BPE token 非均匀 chunk baseline、diagonal wavefront 调度和 synthetic Euler 采样验证。
- `acoustic/condition_encoder.py` 负责 repeated frame-level condition hidden 送入 DiT 前的轻量 temporal encoder。
- `qwen3.py` 是 Hugging Face Qwen3 相关类的本地导入层，避免其他文件到处依赖 transformers 的深层路径。
- `DiT/` 是 acoustic decoder 子模块，外部优先通过 `Orchestrator` 调用。

## 输入输出契约

训练侧输入来自 `types.CausalLMBatch`：

- `input_ids` 用于 token 对齐、mask 和生成。
- `attention_mask` 用于屏蔽 padding。
- `inputs_embeds` 由模型内部的 `IdSpaceEmbedding` 从 `input_ids` 生成。
- `labels` 只包含需要计算 loss 的目标 token。
- `logits_to_keep` 指明每行需要保留的尾部 supervised token 数量，或显式位置索引。

acoustic 侧输入输出契约：

- acoustic condition 从 `labels != IGNORE_INDEX` 的目标 audio segment 推导，排除 `BOA/EOA`。
- BPE token 对应的 condition hidden 使用其下一位 input token 的 Qwen3 hidden state。
- BPE hidden 通过 `CodecBPE.repeat_interleave(..., mask=...)` 展开到原始 semantic frame 粒度；LongCat 当前只接受单 codebook semantic ids，模型层会显式把 `[B, T, 1]` frame 压成 `[B, T]`。
- `acoustic_flow_loss` 接收连续 `target_features`，形状为 `[batch, time, acoustic_dim]`，并要求 time 维与展开后的 condition mask 对齐。
- `acoustic_flow_inputs` 暴露与 `acoustic_flow_loss` 相同的准备结果，供训练日志复用同一份 Qwen hidden、source acoustic condition、noise 和 mask。
- `acoustic_flow_loss` 可选接收 `source_feature_extractor`，将 `batch.source_audio` 的 LongCat acoustic codes 转为连续 features 后按 mask mean 池化成 DiT 的 batch-level `acoustic_condition`；缺失 source 的行使用 DiT null acoustic condition。
- `ModelConfig.acoustic.condition_dropout` 只作用于训练态、由 source features 池化得到的 acoustic condition；显式传入的 `acoustic_condition` 不被隐式替换。
- Lightning 联合训练由 `TrainConfig.acoustic_loss_weight` 开启；权重为 0 时保持 semantic-only，权重大于 0 时必须显式传入 BPE 和 LongCat acoustic feature extractor。
- LongCat discrete acoustic codes 到连续 features 的转换由 `anytrain.codec.longcat` 显式提供，模型层只消费连续 `target_features`。

生成侧输入来自 `types.GenerationBatch`：

- `input_ids` 和 `attention_mask` 表达已经构造好的 Qwen3 prompt。
- `generate_acoustic_condition` 内部采样目标 audio BPE token 维持自回归，但对外主输出是 `AcousticConditionGeneration.hidden_states` 和 `mask`。
- `generate_semantic` 返回生成的全局 token ids、展开后的 LongCat semantic ids 和 semantic mask，作为 semantic-only 评估入口。
- 生成侧的 hidden condition 使用每个 sampled BPE token 的下一步 Qwen hidden state，与训练侧 acoustic condition 的 shifted hidden 契约对齐。
- 离散 token ids 只在 `return_token_ids=True` 时返回，用于 debug、EOA 停止检查或简单 sanity check；不要把它当作 DiT 的主要条件输入。
- `generate_waveform` 先走 full-sequence 路线，必须显式接收 acoustic feature generator；当前模型层不隐式把 condition 变成 LongCat acoustic features。
- `teacher_forced_waveform` 用训练侧 labels 的 shifted hidden states 生成 waveform，主要用于诊断 acoustic/DiT 是否能在接近正确 semantic hidden 的条件下产生可听音频，不替代最终 free-running waveform 评估。
- `acoustic_velocity(..., guidance_scale=...)` 是 CFG 速度预测边界；`guidance_scale=1` 只跑 conditional DiT，其他值会再跑 null acoustic condition 并做 `uncond + scale * (cond - uncond)`。
- `Orchestrator.acoustic_feature_generator(...)` 返回可传给 `generate_waveform` 的 DiT acoustic feature generator，按 `AcousticSampler` 选择 `serial` 整段 acoustic flow 或 `diagonal_bpe` BPE-boundary wavefront sampler 生成连续 LongCat acoustic features，再交给 codec decode。

模型层不负责：

- 读取 Anydataset。
- 训练或查找 BPE 缓存。
- 决定任务采样权重。
- 做音频质量过滤。
- 隐式解释 LongCat acoustic codebook 到连续 acoustic target feature 的映射；需要转换时调用 `anytrain.codec.longcat` 或 `runtime.longcat_acoustic_features()`。

## 配置边界

模型配置按职责分层：

- `ModelConfig.backbone` 表达 Qwen3 权重来源、4bit 加载、backbone full/LoRA 训练策略。
- `ModelConfig.token_space` 表达 text/audio embedding 和 audio boundary special token 是否训练。
- `ModelConfig.acoustic` 表达是否构建 DiT acoustic decoder、是否训练 acoustic decoder/condition adapter、source acoustic condition dropout 和 DiT 尺寸。
- `ModelConfig.token_space.audio_embedding_type` 表达 audio BPE embedding 的内部实现，支持直接 lookup 和 LongCat semantic code 组装；外部调用方只依赖 audio embedding 的 forward/weight 契约，不依赖内部实现。
- `ModelConfig.token_space.input_adapter` 表达 audio embedding `weight` 内部的输入侧投影；LM head 通过 tied `head_view` 读取同一张 audio weight。`ModelConfig.token_space.output_adapter` 属于 semantic logits 路径，负责把 Qwen hidden 投影到这张 tied weight 对应的空间。`ModelConfig.acoustic.condition_adapter` 负责 Qwen/DiT hidden 对齐；semantic-composed audio embedding 的低秩 shift 留在 `audio_embedding/` 和 token space 配置里。
- `ModelConfig.acoustic.condition_encoder` 表达 repeated frame-level condition hidden 送入 DiT 前的轻量 Qwen3 decoder layer temporal encoder。
- `ModelConfig.acoustic.attention_mode` 表达 acoustic branch 的时序注意力约束；DiT 和 condition encoder 共享该策略。`causal` 保持旧实验语义并适合 streaming/causal-window 对照，`bidirectional` 用于 offline full-sequence acoustic flow 对照。
- `ModelConfig.acoustic.dit.norm_time`、`norm_hidden`、`norm_acoustic` 表达 DiT 内部三路条件相加前的可选无 affine LayerNorm。

训练入口只用 `TrainConfig.acoustic_loss_weight` 决定 acoustic loss 是否参与训练；当权重大于 0 时，`ModelConfig.acoustic.enabled` 必须为 true。评估或 smoke 脚本需要加载 acoustic decoder 时，应显式构造带 acoustic decoder 的 `ModelConfig`，不要把 DiT 是否存在隐含在 Qwen/LoRA preset 名字里。

## 开发边界

- Do: 修改 token space、embedding 或 LM head 时同步更新本文件的对外契约；Don't: 让 data module 依赖模型内部 layout 细节。
- Do: 让 data module 和 model 共享 runtime 构造的 `IdSpace`；Don't: 让 batch builder 依赖 `IdSpaceEmbedding` 权重对象。
- Do: 把 Qwen3/DiT/token-space 构建逻辑收敛在 `builder.py`，让 `Orchestrator` 保持运行期编排入口；Don't: 在训练入口重复拼装子模块。
- Do: 保持 `qwen3.py` 作为兼容导入层；Don't: 在跨模块代码里直接依赖 transformers 的私有路径。
