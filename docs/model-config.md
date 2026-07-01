# Model Config

模型配置按职责分成三层：

- `model.backbone`：Qwen3 权重来源、4bit 加载、backbone full/LoRA 训练策略。
- `model.token_space`：text embedding、audio embedding 和 audio boundary special token 是否训练。
- `model.acoustic`：是否构建 DiT acoustic decoder、是否训练 acoustic decoder/condition adapter、source acoustic condition dropout 和 DiT 尺寸。
- `model.train_mode`：控制参数训练范围；默认 `default` 按 backbone/token/acoustic 配置分别控制，`acoustic_only` 用于声学消融，只训练 audio embedding、DiT acoustic decoder 和 acoustic condition adapter。
- `model.token_space.audio_embedding_type`：控制 audio BPE embedding 的内部实现。`lookup` 使用直接 BPE lookup table；`semantic_composition` 使用 LongCat BPE 展开的 semantic code embedding 组装 BPE embedding，并叠加低秩 shift。
- `model.token_space.input_adapter` 表达 audio embedding `weight` 内部的输入侧投影；LM head 通过 tied `head_view` 读取同一张 audio weight。`model.token_space.output_adapter` 属于 semantic logits 路径，负责把 Qwen hidden 投影到这张 tied weight 对应的空间。`model.acoustic.condition_adapter` 负责 Qwen/DiT hidden 对齐。三者公开字段只包含 `type`、`in_features` 和 `out_features`。
- `model.acoustic.condition_encoder`：可选的 frame-level condition temporal encoder，使用轻量 Qwen3 decoder layer，在 repeated hidden 经过 condition adapter 后、送入 DiT 前运行一次。
- semantic-composed audio embedding 的低秩 token shift 由 `model.token_space.semantic_shift_rank` 表达，不放进 hidden adapter。
- `model.acoustic.condition_source`：控制 DiT 的 frame-level hidden condition 来源。默认 `qwen_hidden` 使用 target BPE label 对齐后的 Qwen shifted hidden；`target_audio_embedding` 直接用 target BPE id 查 audio embedding 后按 BPE 展开，且不需要跑 Qwen forward。
- `model.acoustic.attention_mode`：控制 acoustic branch 的时序注意力，DiT 和 condition encoder 共享该策略，支持 `causal` 和 `bidirectional`。preset 显式保持 `causal` 旧实验语义；offline full-sequence acoustic flow 对照建议 override 为 `bidirectional`。
- `model.acoustic.dit.norm_time`、`norm_hidden`、`norm_acoustic`：控制 DiT 内部三路条件在相加前是否做无 affine LayerNorm；默认关闭以保持旧实验语义。

`train.semantic_loss_weight` 决定是否计算语义 LM loss；等于 0 时训练 step 不跑 Qwen forward，也不记录 semantic accuracy/task semantic loss。`train.acoustic_loss_weight` 决定 acoustic loss 是否进入训练目标。权重大于 0 时，`model.acoustic.enabled` 必须为 true；权重等于 0 时可以关闭 acoustic decoder，semantic-only smoke 配置应显式写：

```yaml
model:
  acoustic:
    enabled: false
```

临时评估或 smoke 脚本需要加载 acoustic decoder 时，使用 `with_acoustic_decoder(...)` 派生运行期 `ModelConfig`，不要在 Qwen/LoRA preset 名字里隐含 DiT 是否存在。

## Preset 含义

- `qwen3_0_6b_lora`：Qwen3-0.6B fp/bf16 加载，backbone 冻结，只训练 LoRA 和 token space。
- `qwen3_0_6b_lora_4bit`：Qwen3-0.6B 4bit 加载，backbone 冻结，只训练 LoRA 和 token space。
- `qwen3_0_6b_full`：Qwen3-0.6B 非 4bit 加载，backbone 全量训练，不启用 LoRA。

这些 preset 当前都默认 `model.acoustic.enabled=true`，用于 quality/acoustic 训练。semantic-only smoke 实验在 experiment 配置里覆盖关闭 acoustic decoder。

## 常用 override

```bash
# 改 acoustic/FM loss 权重
python scripts/train.py experiment=wmt19_quality_100k_muon train.acoustic_loss_weight=0.03

# 临时关闭 acoustic decoder，只跑 semantic 训练闭环
python scripts/train.py experiment=wmt19_mixed_smoke model.acoustic.enabled=false

# 切 full backbone 对照
python scripts/train.py experiment=wmt19_quality_100k_full_adamw

# 打开 Qwen hidden 和 source acoustic 条件 norm
python scripts/train.py experiment=wmt19_quality_100k_muon model.acoustic.dit.norm_hidden=true model.acoustic.dit.norm_acoustic=true

# offline full-sequence DiT attention 对照
python scripts/train.py experiment=wmt19_quality_100k_muon model.acoustic.attention_mode=bidirectional

# 打开 1-layer Qwen3 condition temporal encoder
python scripts/train.py experiment=wmt19_quality_100k_muon model.acoustic.condition_encoder.enabled=true

# target BPE audio embedding 条件的 acoustic-only 消融
python scripts/train.py experiment=wmt19_acoustic_target_embed_100k_muon

# 使用 semantic code 组装 audio BPE embedding
python scripts/train.py experiment=wmt19_quality_100k_muon model.token_space.audio_embedding_type=semantic_composition
```

acoustic 训练会记录 `condition/{time,hidden,acoustic}_{mean,std}`，统计的是实际送入 DiT 条件融合前的三路张量；`hidden` 按有效 acoustic frame 统计，`time` 和 `acoustic` 按包含有效帧的 batch 行统计。

训练过程中的 checkpoint、sample logging 和 generation logging 属于 trainer callback
配置，见 `docs/trainer-config.md`。
