# Parallel Free-Running Ablation

## 背景

001 已经把主路线定为 bidirectional semantic AR + acoustic-guided FM，并给出
S0/S1/S2 阶段和 LoRA/full 对照。003 不再设计必须串行衔接的阶段训练，而是设计一组
可以从同一起点并行启动的长周期对照。free-running 句子级能力需要足够训练周期才能判断，
早期 checkpoint 只用于确认 loss 是否稳定下降和训练链路是否正常。

本轮优先服务两个问题：

1. free-running 不能成句时，瓶颈更可能来自任务比例、acoustic/FM 权重还是 backbone 更新能力。
2. source-to-target 主方向是否被双向训练或 acoustic/FM 常驻拖偏。

003 固定以下前提：

- acoustic/FM loss 必须常驻，不再做 `acoustic_loss_weight=0` 对照。
- 100k LongCat BPE 已经验证为正确方向，本轮作为默认 BPE 粒度，不再和 10k BPE 做并行对照。
- backbone 更新能力只用 full backbone 对照判断，不在同批次加入 LoRA rank sweep。

配置解释：

- `wmt19_quality_100k_muon` 使用 `model.backbone.lora.enabled=true`，`model.backbone.train=false`。
- `wmt19_quality_100k_full_adamw` 使用 `model.backbone.lora.enabled=false`，`model.backbone.train=true`。
- 两个 quality 100k 入口都启用 `model.acoustic.enabled=true`；P1-P6 只通过
  `train.acoustic_loss_weight` 调整 acoustic/FM loss 权重。
- 模型配置分层契约见 [../../model-config.md](../../model-config.md)。

## 前置评估入口

训练对照可以先并行启动，但在解释结果前必须固定 checkpoint 生成评估入口。

评估入口至少保存：

- fixed sample 的 source/reference/generated waveform
- 生成 semantic token 数、展开后的 semantic frame 数
- EOA 是否命中、是否提前停止
- source-to-target 和 target-to-source 分方向结果
- 人工听感备注：是否成句、是否只翻关键词、是否重复、是否断裂

当前训练内置 `TaskGenerationLogger` 记录的是 teacher-forced waveform，适合诊断 acoustic/DiT
路径，但不能替代 free-running 结论。因此 003 的训练结果不以 teacher-forcing loss 或
teacher-forced waveform 单独定胜负。

003 使用 `scripts/evaluate_free_running.py` 作为 checkpoint 后评估入口。每个 run 在
`jobs/003/eval_*.sh` 有对应 wrapper；调用时显式传入 `--ckpt-path` 和 `--output-dir`，
避免脚本猜测 checkpoint 位置。输出结构为每个 sample/direction 的
`source.wav`、`reference.wav`、`generated.wav` 和 `summary.jsonl`。

## 并行实验矩阵

所有 run 尽量使用同一批 canary samples、同一 `max_steps`、同一 checkpoint 间隔和同一生成评估协议。
如果 GPU 资源不足，优先跑 P1 到 P4。

| ID | 变量 | 配置草案 | 目的 |
| --- | --- | --- | --- |
| P1 | S1 balanced baseline | `experiment=wmt19_quality_100k_muon tasks=s1_bidirectional_mixed train.acoustic_loss_weight=0.01` | 作为 003 的 100k LoRA 基线 |
| P2 | S2 weak reverse | `experiment=wmt19_quality_100k_muon tasks=s2_translation_weighted train.acoustic_loss_weight=0.01` | 判断主方向加权且保留少量反向约束是否优于 balanced |
| P3 | remove reverse translation | `experiment=wmt19_quality_100k_muon tasks=s2_translation_weighted tasks.weights.target_to_source=0.0 train.acoustic_loss_weight=0.01` | 判断反向 translation 是否伤害主方向 |
| P4 | stronger acoustic/FM | `experiment=wmt19_quality_100k_muon tasks=s1_bidirectional_mixed train.acoustic_loss_weight=0.03` | 验证更强 acoustic/FM 是否给 free-running 生成更多约束 |
| P5 | stronger acoustic with main direction | `experiment=wmt19_quality_100k_muon tasks=s2_translation_weighted train.acoustic_loss_weight=0.03` | 检查 P2 任务比例和 P4 acoustic 权重是否能叠加 |
| P6 | full backbone | `experiment=wmt19_quality_100k_full_adamw tasks=s1_bidirectional_mixed train.acoustic_loss_weight=0.01` | 判断 LoRA 是否限制 codec token free-running 动力学 |

## 推荐优先级

### 第一优先级

P1、P2、P3、P4。

这四个 run 直接回答当前最关键的训练目标问题：

- bidirectional mixed 是否稳定；
- 主方向是否应该加权，以及是否需要减少或移除 target-to-source；
- acoustic/FM 权重应该继续维持 `0.01`，还是提高到 `0.03`；

### 第二优先级

P5、P6。

P5 检查任务比例和 acoustic 权重两个已经有希望的干预是否能叠加。P6 是对 001 中 LoRA
瓶颈假设的直接检查，但成本较高。

## 训练周期

003 不按 10k 或 20k 这类短程结果决定变量优劣。当前默认 `warmup_ratio=0.01`，
在 `max_steps=5000000` 时等价于 50000 step warmup。早期 free-running 结果会同时受到
warmup、随机初始化的 audio token/DiT 和采样噪声影响，不能作为任务比例或 acoustic/FM
权重的确定性结论。

训练应尽量拉长到 warmup 结束后再比较 checkpoint。中间 checkpoint 的用途是：

- 观察 total/semantic/acoustic loss 是否稳定下降。
- 检查是否出现 NaN、loss 爆炸、长期不下降或生成长度明显异常。
- 固定同一批样本做过程记录，但不把早期 waveform 当作最终判定。

只有当 loss 崩掉、出现持续 NaN/Inf、或者同等资源下长期没有任何下降信号时，才提前停止某个 run。
否则保持同一训练周期后再进入 free-running 评估和人工听感比较。

## 不放进 003 的实验

- S0 -> S1/S2 串行 warmup：它有 checkpoint 依赖，不满足本轮“可以并行”的要求。
- `acoustic_loss_weight=0`：已经验证 acoustic/FM 必须常驻，不再占用 003 并行名额。
- 10k vs 100k BPE：100k BPE 已验证为正确方向，本轮固定 100k。
- LoRA rank sweep：它和 full backbone 对照都在解释 backbone 更新能力，同批次并行会让结论混淆。
- Qwen3-8B LoRA：当前没有现成 8B 配置，且容量变量和资源变量同时变化。除非 P1/P6 都失败，
  否则不作为 003 首批并行项。
- scheduled sampling、prefix dropout、短段 free-running 训练：这些需要新增训练逻辑，应等
  P1-P4 在足够训练周期后判断当前损失和任务比例是否已经足够再做。
- LBA 质量对照：LBA 主要影响吞吐和长度 batching，不应和模型质量变量混在同一个实验结论里。

## 运行约束

- 每个并行 run 必须使用不同 `trainer.name`。
- 多实例同机运行时显式设置 `CUDA_VISIBLE_DEVICES`，避免多个 run 抢同一张卡。
- 如果启用 LBA，必须给每个 run 单独设置 `datamodule.lba.log_dir`。
- checkpoint 间隔保持长跑默认值，避免 NAS I/O 干扰训练吞吐。
- 生成评估使用同一组 sample index；如果 sample 太短，应补充至少一个中等长度样本和一个长样本。

评估 wrapper 示例：

```bash
jobs/003/eval_p1_s1_100k_lora_muon.sh --ckpt-path <ckpt> --output-dir <dir> --sample-indices 0 8 32
```

## 判定标准

优先级从高到低：

1. source-to-target free-running waveform 是否形成完整短句。
2. 是否只翻关键词、重复、断裂或提前 EOA。
3. generated semantic frame 数是否接近 reference 的合理范围。
4. teacher-forcing loss、semantic AR loss、translation loss 和 acoustic/FM loss 的走势。
5. teacher-forced waveform 是否可听。

判定必须基于相近训练周期的 checkpoint。早期 checkpoint 只能用于排查训练是否崩掉；
如果 loss 没有崩，不用短程 waveform 直接淘汰配置。

如果 P2 明显优于 P1，说明主方向加权和弱反向约束比 balanced 更适合 source-to-target 主线。  
如果 P3 明显优于 P2，说明反向 translation 对主方向是负约束，后续 source-to-target 主线应移除
target-to-source。  
如果 P4 优于 P1 且没有退化到词级对齐，后续把 S1 acoustic/FM 权重提高到 `0.03`。  
如果 P5 优于 P2/P4，说明主方向加权和更强 acoustic/FM 可以叠加。  
如果 P6 明显优于 P1，说明 backbone 更新能力是第一轮瓶颈。
