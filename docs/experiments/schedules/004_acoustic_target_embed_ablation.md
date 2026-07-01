# 004 Acoustic Target Embedding Ablation

## 目标

验证 acoustic decoder 是否能在不使用 Qwen hidden state 的情况下，仅依赖 target
speech BPE token 的 audio embedding 条件学习 acoustic flow。

## 方案

- 数据和任务分布沿用 mixed 配置。
- 完整初始化 Qwen、token space 和 DiT。
- `model.train_mode=acoustic_only`：
  - freeze Qwen backbone、text embedding、audio special tokens 和语义 LM 路径。
  - train audio embedding、DiT acoustic decoder 和 acoustic condition adapter。
- `model.acoustic.condition_source=target_audio_embedding`：
  - target BPE labels 直接查 audio embedding。
  - BPE embedding 按 tokenizer 展开到 frame 粒度后作为 DiT condition。
- `train.semantic_loss_weight=0.0`，训练 step 不跑 Qwen forward。
- `train.acoustic_loss_weight=1.0`，只优化 acoustic/FM loss。

## 入口

```bash
jobs/004/01_acoustic_target_embed_100k_lora_muon.sh
```

对应配置：

```bash
python scripts/train.py experiment=wmt19_acoustic_target_embed_100k_muon
```

## 期望对照

与 003 中使用 Qwen hidden state 条件的 acoustic 训练对照，观察 acoustic loss、
teacher-forced waveform sample 和后续 free-running 评估差异。
