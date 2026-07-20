# 011 Qwen RVQ Staged Joint Training

## 目标

在真实 Qwen3-0.6B、LongCat 和 WMT19 zh-en speech pair 上建立可恢复、可晋级、可止损的
分阶段联合训练路径。最终目标是 S2ST，不把 codec oracle 的 acoustic-only smoke 当作 Qwen
联合训练或质量结论。

首轮固定使用 LongCat native semantic token 和 RVQ acoustic decoder。Flow 只保留 2-step
合同复验，不与 RVQ 同时展开正式长跑。

## 当前基线与决策

- 010 已验证 Flow/RVQ oracle 的单卡和两卡 2-step 执行闭环，但 Qwen backbone 被冻结，
  condition 直接来自 teacher-forced semantic label embedding；它不验证 Qwen hidden 到
  acoustic decoder 的联合路径，也不支持收敛或质量结论。
- 010 checkpoint 只有 2 optimizer steps 且位于远端临时目录。它只用于验证严格权重导入
  契约，不能作为有训练价值的初始化。
- 2026-07-21 对 010 临时 v2 数据根的只读审计显示 train 只有 1000 条；canonical `base`
  仍是 anydataset schema v1。当前数据只能做 pilot。
- 同次审计中，现有 100k CodecBPE 把 1000/1000 条 source 和 1000/1000 条 target 都压成
  恰好一个 audio token。该 artifact 近似把整段语音变成样本 ID，不进入本计划。
- native token 保留逐 semantic frame 的组合结构，也与 RVQ oracle 的 semantic embedding
  shape/语义一致。若未来使用 BPE，需在完整 train split 上重训并作为独立实验从头训练，
  不能在本计划中途切 tokenizer。
- RVQ 是首轮主路径：它已有真实 fixed-sample joint overfit/generation 历史基线，oracle 与
  joint target 都是相同 packed acoustic code。010 的 Flow 包含 normalized 和 raw-feature
  smoke，但都只有 2 steps；本轮为降低变量不迁移 Flow checkpoint，也不展开 Flow 长跑。

## 长跑前置门槛

### 数据

1. 把 canonical WMT19 `base` 迁移为 schema v2，并验证已为 v2 的 LongCat view 与该 base
   逐行对齐；只有对齐失败才重物化 LongCat。正式任务只读取稳定 NAS root，不依赖 `/tmp`。
2. 在 v2 store materialization 后以 `(store manifest fingerprint, immutable global row index)`
   作为 split manifest key，pair 的 source/target 必须留在同一 row/split。1000 条 pilot 使用
   800/100/100；正式数据为 dev/test 各保留 `max(1000, 1%)`，test 只在最终选定 checkpoint
   后读取一次。
3. 记录每个 split 的样本数、语言方向、文本长度、source/target frame 与 native token 分布、
   parse/span 错误和质量过滤版本。parse/span 错误必须为 0。
4. split 后 train 至少 5 万条通过质量过滤的 prepared pair 才允许 scaled canary；1M-step joint
   budget 的目标和默认门槛是约 50 万 train pair。未达到 5 万时只执行最多 5k-step pilot，
   不对泛化作结论。
5. 所有 A-D stages 记录 `pair_exposures = cumulative drawn raw pairs / train pair count`。
   各阶段表中的 step 数是上限；实际 `max_steps` 取 step 上限与剩余 64 pair exposures 的较小值。
   只有 batch ladder 证明整个 A-D 预算不超过该上限时，才允许启动 1M joint run。

### 训练接口

现有 `scripts/overfit.py` 只允许 fixed-sample、batch size 1，不作为正式入口。长跑前需要一个
直接调用真实训练逻辑的 joint entry，并补齐：

- train/dev/test DataLoader、确定性 shuffle、distributed sampler，以及按构造后
  `ModelSample` token/frame 成本规划的 LBA。task family 必须在 LBA 之前确定，不能在 collator
  内随机决定后再让 planner 猜长度。
- 每个 task family 使用固定 `len_fn`：
  `ceil(sequence_tokens / token_unit) + ceil((source_frames + target_frames) / frame_unit)`；
  `token_unit`、`frame_unit` 和 LBA scalar budget 由 P0 ladder 固化到 resolved config。另设
  sequence/source-frame/target-frame hard cap；超 cap 的记录写入版本化 length-filter manifest
  并显式统计，不能静默裁剪或丢弃。
- 不重叠的 model-relative state-dict prefix 组：`backbone.*`、
  `semantic_audio_embedding.*`、`semantic_audio_adapter.*`、
  `acoustic_prompt_adapter.*` 加裸 `acoustic_prompt_gate`、
  `semantic_audio_output_adapter.*`、`acoustic_decoder.*`。入口断言每个
  `requires_grad` parameter 恰好归属一组；其中 semantic embedding 同时是 audio 输入与 head 权重。
- 每组 freeze/LR scale、Qwen `top_fraction` 解冻，以及 RVQ 结构性 unused 参数冻结。
- RVQ 的 `acoustic_decoder.decoder.embed_tokens`、最后一个 codebook embedding 与其 projection
  固定冻结，和 oracle 的静态路径保持同一参数边界。
- warmup + WSD/cosine scheduler、gradient accumulation、token/acoustic objective weight。
- 按 optimizer step 在不同 execution signature 的 DataLoader 间轮转，并保证 DDP rank 在
  同一步选择相同 task family。同一 batch 仍只含相同 source/target modality signature。
- 多任务 DDP 初始固定使用 `find_unused_parameters=True`；只有参数路径和 task schedule 稳定后
  才评估切换到静态 DDP，不把 unused-parameter 错误当作可忽略 warning。
- `initialize_from` 只加载模型白名单并重建 optimizer；`resume_from` 只用于同一 stage，恢复
  model/optimizer/scheduler/global step。不能复用一个含糊的 `ckpt_path` 语义。
- TensorBoard、non-finite check、分组 grad/update norm、validation generation、周期 sample、
  step archive checkpoint 与 `last.ckpt`。

### Batch 与显存

在目标机器运行 batch `1/2/4/8` ladder，选 peak allocated 小于单卡显存 90% 的最大配置。
记录实际 token、source/target frame、padding ratio、audio seconds、samples、loader wait 和
step time；通过 gradient accumulation 固定 effective batch。随后完成两卡 2-step DDP smoke。
tokenizer、task family 或 decoder 变化后必须重测，不能沿用名义 batch size。

## 阶段总览

| 阶段 | 数据门槛 | Qwen | 主要训练对象 | 预算 |
| --- | --- | --- | --- | ---: |
| P0 合同 pilot | 1k | full model smoke | native-token joint contract | oracle 最多 5k，smoke/overfit 另计 |
| A RVQ oracle pretrain | >=50k train，按 exposure 缩放 | frozen/bypassed | semantic embedding/adapter + RVQ decoder | 100k-200k ceiling |
| B frozen-Qwen bridge | >=50k train，按 exposure 缩放 | frozen | 全部 speech interface + RVQ decoder | 50k ceiling |
| C partial-Qwen joint | >=50k train，按 exposure 缩放 | top 1/3 + final norm | S2ST + component replay | 150k ceiling |
| D full multitask joint | ~500k train + C 晋级 | full，需晋级 | S2ST 为主的多任务联合 | 800k ceiling |

B+C+D 的 nominal 上限为 1M joint optimizer steps，A 不计入 joint 预算；所有数字仍受
pair-exposure cap 约束。若 C 不满足文本保持或 S2ST 晋级条件，D 不全解冻 Qwen；可以沿用
C 的 top-1/3 策略继续剩余预算，但必须明确记录。

## P0：真实合同与 1k Pilot

1. 使用真实 Qwen、LongCat native token 和 prepared pair，按当前 TODO 重跑 TTS/S2ST 的
   Flow/RVQ 各 2 steps，确认按模态 token CE、backward、generation 与 waveform finite。
2. RVQ 在 32 个固定 train sample 上跑 100-step overfit；token CE 和逐 codebook CE 都应
   下降，waveform 必须可 decode。该结果只验证优化信号。
3. 用 800/100/100 split 运行最多 5k-step RVQ oracle pilot，每 250 steps 验证、每 1k steps
   归档 checkpoint。它验证 held-out 曲线和 stage checkpoint，不替代正式数据训练。
4. 验证从 010 checkpoint 导入 `semantic_audio_embedding.*`、`semantic_audio_adapter.*` 和
   `acoustic_decoder.*` 时，仅对选中 prefix 的 expected/actual key 集合做双向严格校验；完整
   冻结模型的其他 source key 必须显式排除并记录，不能误报为 unknown。companion manifest
   必须绑定 checkpoint SHA、010 resolved-config SHA、commit、`longcat`、
   native tokenizer、semantic vocab `8192+2`、RVQ `[8100, 8100, 8100]`、8-layer/hidden-1024
   decoder 和 key mapping。任何选中 key、shape 或 metadata 不匹配都直接失败；随后丢弃该
   2-step 权重。

P0 晋级条件：单卡/两卡 optimizer step、resume、loss/grad 全部通过；fixed-sample 可 decode 率
100%；held-out RVQ CE 相对初始至少下降 5%，多数 codebook 的 teacher-forced top-1 accuracy 高于
随机基线。feature MSE 仅作为固定 seed 的诊断，先用 P0 重复运行确定方差，不作为晋级阈值。5k
steps 仍无 held-out CE 改善时停止并检查数据/condition，不扩大预算。

## A：RVQ Oracle Pretrain

- 从 codec-derived native semantic embedding 初始化，不加载 010 的 2-step 训练值。
- 只训练 `semantic_audio_embedding`、`semantic_audio_adapter` 和 `acoustic_decoder`；condition 绕过
  Qwen transformer，不把 oracle 的梯度当作 Qwen 联合梯度。
- 初始 AdamW LR `3e-4`、weight decay `0.01`，2% warmup 后 WSD/cosine；最终值以 P0
  held-out 曲线为准，配置冻结后再启动正式 run。
- 每 1k steps 验证 RVQ CE/accuracy/feature MSE，每 10k steps 保存 sample、完整归档
  checkpoint 和 `last.ckpt`。100k 先审计，只有 held-out 指标仍改善才继续到 200k。

A 到 B 的 `initialize_from` 白名单只包含 native `semantic_audio_embedding`、
`semantic_audio_adapter` 与 RVQ `acoustic_decoder`。Qwen、text embedding/head、audio output、
acoustic prompt 和 optimizer state 均不导入。

## B：Frozen-Qwen Bridge

- Qwen backbone、text embedding/head 全部 frozen；训练全部 audio 参数组与 RVQ decoder。
- 60% steps 使用 text-to-audio stream，`TTS:T2ST=2:1`；40% 使用 audio-to-text stream，
  `ASR:S2TT=1:1`。先分别做 input/output 1k-step probe，再启用轮转。
- audio 参数组初始 LR `1e-4`、weight decay `0.01`；2% warmup，随后 cosine 到 0.1 倍。
- 前 500 个 audio-target step 仅在 TTS/T2ST 上，对真正共享的 `semantic_audio_embedding` 与
  `semantic_audio_adapter` 测 token/RVQ grad norm；其它 task family 记录 `N/A`。固定
  `rvq_weight` 使 median ratio 落在 `[0.5, 2.0]`，正式 stage 中不动态改变 loss weight。

B 晋级条件：dev semantic token CE 相对起点下降至少 10%；ASR/S2TT held-out 指标改善；
audio-target 请求非空且可 decode 比例至少 95%，waveform finite 率 100%。因 Qwen/text path
全冻结，双向 text probe NLL delta 绝对值应不超过 `1e-4`，超出视为契约错误。

## C：Partial-Qwen Joint

- 解冻 Qwen 顶部 `ceil(num_hidden_layers / 3)` 个 block 与 final norm；text embedding/head 和
  其余 block 继续冻结。
- step 比例：40% `TTS/T2ST`、25% S2ST、25% `ASR/S2TT`、10% T2TT。DDP rank 共享同一个
  确定性 family schedule；所有后续 stage 固定 `TTS:T2ST=2:1`、`ASR:S2TT=1:1`。
- Qwen LR `1e-5`，audio 参数组 LR `5e-5`；2% warmup、78% stable、20% cosine，gradient
  clip `1.0`。
- 在 audio-target step 的 Qwen 顶部共享参数上记录 token/RVQ grad norm 与 cosine；其它 step
  记录 `N/A`。objective weight 只允许在独立 calibration run 后变更。
- 每 1k steps 做轻量 dev loss/NLL，每 10k steps 在固定 dev manifest、固定 generation seed 和
  固定 evaluator checkpoint 上做完整生成评估。

C 晋级条件：主指标 `dev/s2st_asr_chrf` 相对 B 的最佳 checkpoint 至少改善
`max(0.5 chrF points, 2%)`，且连续三次完整验证均不低于当前 best 1.0 chrF point；非空/可
decode 率至少 99%，waveform finite 率 100%；双向 text probe mean NLL 不超过 pretrained
baseline 的 1.05 倍，任一 probe 不超过 1.10 倍。超过 1.10 倍立即停止。

## D：Full Multitask Joint

只有 C 全部通过才全解冻 Qwen；否则沿用 partial-unfreeze 参数边界继续。

- step 比例：50% S2ST、30% `TTS/T2ST`、10% `ASR/S2TT`、10% T2TT。
- full Qwen LR 从 `2e-6` 起，audio 参数组从 `2e-5` 起；1% warmup、89% stable、10%
  cosine。以 C 最佳 checkpoint 初始化并重建 optimizer/scheduler。
- 每 1k steps 做轻量 dev loss/NLL；每 10k steps 做生成、ASR/translation/quality 评估、
  sample 与归档 checkpoint。所有 10k 档位都保留，并维护 `last.ckpt`。
- text mean NLL 超过 1.05 倍、任一 probe 超过 1.10 倍，或连续三次完整验证低于当前 best
  1.0 chrF point 时，回到最近满足约束的 checkpoint。完整验证以 `min_delta=0.5 chrF points`
  和 `patience=5` 选择 best；五次完整验证均无该幅度改善时停止当前 run。

## 评估与选择

训练侧记录 total/token/RVQ total 与逐 codebook CE/accuracy、有效 token/frame、各参数组
grad/update norm 和 LR，以及 samples/tokens/audio-seconds per second、loader wait、padding ratio、
step time 和 peak memory。

held-out 生成记录：

- 非空率、EOA/EOS 完成率、decode/finite 率、生成/target 长度比、feature MSE、multi-scale STFT。
- `max_new_tokens` 按 dev target-token p99 加固定 margin 配置；被截断的 response 单独计为失败，
  不能从成功率中删除。
- P0 固定评估器为 `WhisperASREvaluator(model_name="large-v3-turbo", language="en",
  temperature=0.0)` 和 `UTMOSEvaluator(repo="tarepan/SpeechMOS:v1.2.0",
  model_name="utmos22_strong")`；评估器代码 commit、模型 revision、sample rate 与 language
  写入 resolved config。当前 WMT19 zh->en target 使用 `language="en"`；加入反向 speech
  pair 时必须单独固定其 language/evaluator config。后续不能在同一曲线中更换 evaluator。
- TTS 使用该固定 ASR 的 CER/WER 与 UTMOS。S2ST 将生成语音经该 ASR 后，对 target text
  计算 corpus `dev/s2st_asr_chrf` 与 BLEU，并报告 UTMOS。decode failure 以空转写计入
  该固定 dev manifest，不能从分母删除。
- ASR 报 CER/WER；S2TT/T2TT 报 chrF/BLEU 与 reference NLL。
- 至少覆盖 zh->en 与 en->zh 的固定纯文本 probe，记录 pretrained baseline、NLL delta 与
  deterministic generation。没有反向 speech pair 时，不把 text probe 误报成反向 S2ST 结果。

最终 checkpoint 以 dev S2ST ASR-chrF 为主排序，但必须先满足 text retention、decode/finite、
UTMOS 和长度约束：decode/finite 率至少 99%，至少 95% 样本的生成/target 长度比在 `[0.5, 2.0]`，
median UTMOS 不低于本 stage 起点 best checkpoint 减去 `0.10`。P0 固化 generation seed、dev
manifest 与上述聚合方式；不按 train loss 或单个主观 sample 选模。test 只对最终候选运行一次。
结果文档记录代码 commit、resolved config、dataset/tokenizer manifest hash、环境与 checkpoint
路径。

## 全局止损

以下任一条件立即停止，不自动吞错或扩大预算：

- loss、grad、weight、feature 或 waveform 出现 NaN/Inf。
- checkpoint 不能同 stage 完整 resume，或 stage import 未通过严格 key/shape/metadata 校验。
- 生成可 decode 率低于 95%，或空响应被当作成功样本。
- 完整验证按本计划定义的 `min_delta`/`patience` 已穷尽，或连续三次低于当前 best 1.0 chrF
  point。
- partial/full joint 的 text NLL 超过上述 5%/10% 边界。

长跑启动时同步启动 TensorBoard 后端并完成端口转发；训练期间以监督曲线和阶段门槛为准，
不能只等待最终 `metrics.json`。
