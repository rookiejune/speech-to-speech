# Conclusion

## 适用范围

本页最新真实实验是 016 的 Stable Codec stage 1 单步验收（2026-07-28）：真实
`stabilityai/stable-codec-speech-16k`、Qwen3-0.6B 和 WMT19 prepared data 完成无 audio BPE 的
ASR/TTS optimizer step 与 fixed-sample TensorBoard 闭环；loss 与 target/generated waveform finite，
Stable 路线没有 Flow/RVQ `reference_generation`。该 smoke 不支持质量或收敛结论。015 的
SAC generator 到 S2S hidden-state joint 验收（同日）使用真实 Qwen3-0.6B、LongCat、WMT19 sample 0
与 8 层 Flow/RVQ `codebook_ar` artifact，完成两条 route 的初始化、forward/backward/optimizer 和
finite generation；hidden adapter 与 decoder 均有 finite 非零梯度。Flow 另完成正式 Stage 1
step 1 -> 2 resume 和 checkpoint delta；RVQ `codebook_ar` 随后也在已推送代码的隔离 checkout
完成相同的正式 resume 验收。该结果不支持质量或收敛结论。014 的 pami201 1k stage 1
FP32-storage step-500 -> 1000 resume 与 reducer
修复复验（2026-07-27）中，step 1000 的两卡 NCCL dev RVQ CE 为 `8.773024`，相对修正后的
step-0 估计下降 `4.116%`，仍未达到既定 5% gate `8.694203`；step 900 的局部最低值为
`8.747676`，末步略有回升。step-500 condition ablation 中 shuffled/zero condition 相对 correct
condition 的 CE 分别恶化 `0.207262/0.301940`，证明 decoder 已使用 semantic condition。四组
目标可训练参数在 step 100 到 500 间有 `99.181562%` 发生逐位变化，参数与 AdamW moment 均为
FP32；这明显优于旧 BF16 checkpoint 的 `20.19%` 更新覆盖。当前停止继续追加预算；decode、
长跑监督和质量/收敛仍未验证。历史 BF16 step-2000 run 仍作为执行和趋势证据保留，但其
validation artifact 使用了
会把 `4265` 截断为等效 `4264` 分母的旧 Lightning reducer，不再视为精确全局加权值。
同日的 32-sample smoke/canary 另外证明：`/mnt/pami202` 超时时，可以用
145 本地 runtime/HF cache 和 pami201 小数据根完成 stage 1 TTS/S2ST 两步入口、
正式 `scripts/train.py` 100-step canary，以及重分片 root 的两卡 DDP/resume。
014 的 LongCat native stage 1 P0 验收（2026-07-25）只证明 debug copy 上历史 duration
workaround、targeted tests、复旦 P0 wrapper 和 generation gate 已通过；当前代码已在缺失
`AudioMeta.DURATION` 时从 codec frame count 和 runtime frame rate 推导音频秒数；原 pami202 root
的 fingerprint 与 native token/RVQ 分布审计已经完成，pami201 1k pilot 已固化
split 并通过 DDP/resume 和 dev 指标链路验收，但学习门槛和长跑仍未完成。013 的
FDU stage_2 joint LBA DDP smoke（2026-07-23）只验证正式 staged joint train entry 的两卡
DDP + LBA 两步执行闭环，不替代长跑 distributed sample partition、resume、质量或收敛验收。
011 的 P0 fixed-sample 子项（2026-07-21，对应远端代码快照 `d5f6902`）只通过真实
Qwen/native/RVQ 的单卡训练与 teacher-forced acoustic decode，端到端 generation gate 仍失败，
因此 011 P0 尚未完成。010 的 LongCat codec oracle 结论对应代码快照 `9127e62`。008/009 之后
model/runtime/data/generation 和按模态 token CE 仍有调整，因此相应 generation/overfit 数值作为
历史基线保留。旧 schedule 已删除；新的未完成路线见 [todo](todo.md)。

## 已验证结论

- 真实 Stable Codec stage 1 无 BPE 路线已完成单步 smoke：Stable backend 为 16 kHz、25 Hz、
  单码本 46656 codes；ASR/TTS 每个 optimizer step 各提供一个 batch，total/token loss 为
  `9.965052/8.360746`，均 finite。TensorBoard 的 ASR index 0 写出 target/generated text；TTS
  index 0 写出 2.16s target 与 1.28s generated 16 kHz finite audio，tag 按 loader 隔离，且没有
  `reference_generation`。该结论覆盖真实 codec/Qwen/data 的入口、生成语法、mixed dtype、训练和
  fixed-sample 日志契约，不覆盖质量或收敛；正式 1M-step 长跑仍未执行
  （[016 result, lines 3-19](results/016-stable-codec-stage1-smoke.md#L3-L19)，
  [lines 35-64](results/016-stable-codec-stage1-smoke.md#L35-L64)，
  [lines 68-78](results/016-stable-codec-stage1-smoke.md#L68-L78)）。
- 真实 SAC frame-aligned Flow artifact 已完成 S2S `acoustic.init_artifact` joint smoke：
  generator-only loader 在旧 conditioner state 不兼容时仍严格加载 162662400 个 generator
  参数；WMT19 TTS sample 0 的 total/token/Flow loss 为 `13.177253/10.386825/2.790428`。
  训练后生成 `[64,1024]` finite features 与 `[1,61440]` finite waveform；speech-interface
  策略下 hidden adapter 4 个参数和 acoustic decoder 124 个参数的梯度 norm 分别为
  `0.589404/1.639094`，均 finite。正式 Stage 1 checkpoint 从 step 1 恢复到 step 2，optimizer
  counter 全部从 1 推进到 2；hidden adapter `4/4` 个 key、decoder `124/124` 个 key 发生 finite
  变化，固定 `last.ckpt` 正确指向 `global_step=2`。该结论覆盖 Flow Phase B 初始化、单步执行和
  checkpoint resume，不覆盖质量或收敛
  （[015 result, lines 3-23](results/015-sac-flow-joint-init-smoke.md#L3-L23)，
  [lines 43-102](results/015-sac-flow-joint-init-smoke.md#L43-L102)）。
- 真实 SAC frame-aligned RVQ `codebook_ar` artifact 已完成 S2S joint smoke：generator-only loader
  严格加载 `184031980` 个 `AcousticRVQDecoder` 参数；WMT19 TTS sample 0 的 total/token/RVQ loss
  为 `18.904053/9.745777/9.158276`。训练后生成 `[64,1024]` finite features 与 `[1,61440]`
  finite waveform，RTF 为 `1.788941`；speech-interface 策略下 hidden adapter 4 个参数和 acoustic
  decoder 98 个参数的梯度 norm 分别为 `2.216846/5.768480`，均 finite。正式 Stage 1 checkpoint
  从 step 1 恢复到 step 2，107 个 optimizer counter 全部从 1 推进到 2，AdamW 两类 moment 的
  107 个 entry 全部变化且 finite；hidden adapter `4/4`、decoder `98/100` 个 state key 变化，
  backbone `0/311` 变化，checkpoint 不含 codec/runtime key，固定 `last.ckpt` 指向
  `global_step=2`。该结论验证 RVQ Phase B 初始化、单步执行和 checkpoint resume，不覆盖质量或收敛
  （[015 result, lines 104-149](results/015-sac-flow-joint-init-smoke.md#L104-L149)，
  [lines 151-182](results/015-sac-flow-joint-init-smoke.md#L151-L182)）。
- 014 的 FP32-storage 500-step A/B 与 500 -> 1000 resume 已完成：修复后的两卡 NCCL validation
  使用真实 `4265` frame 分母，step 500 RVQ CE 为 `8.780052`，step 1000 为 `8.773024`；step
  1000 相对修正初始估计只下降 `4.116%`，仍未达到 5% gate。step 900 的局部最低值为 `8.747676`
  且末步回升，因此按止损规则不再追加预算。step-500 condition ablation 证明 correct condition
  比 shuffled/zero 分别低 `0.207262/0.301940` CE；四组目标参数更新覆盖为 `99.181562%`，参数
  与 optimizer moment 均为 FP32，支持 storage 改善但不支持质量或收敛结论
  （[014 result, lines 440-519](results/014-longcat-stable-stage1.md#L440-L519)）。
- 014 的历史 pami201 1k pilot validation、100-step canary 与 BF16 resume 到 step 2000 均完成，
  step 750/1000/1250/1500/1750/2000 与 `last.ckpt` 均已保留；但历史 Lightning 2.6.x reducer
  会把全局 `4265` frame 的整数 count mean 截断为等效 `4264`，所以原始表只保留为执行和趋势
  证据，不再作为精确全局加权值。旧 BF16 checkpoint 的预计可训练参数更新覆盖只有 `20.19%`；
  top-1 又接近无条件众数，不能单凭该历史 run 证明条件使用或收敛
  （[014 result, lines 281-438](results/014-longcat-stable-stage1.md#L281-L438)）。
- 014 的 pami201 1k pilot 已固化：数据根包含 1000 条样本、20 个各 50 条的
  payload group，LongCat tensor 为 rank-2 integer、4 个 codebook，frame 范围 `14..43`，
  无 symlink，总大小 `504357694` bytes。split manifest SHA256 为
  `ef3f1009bfb1f1c885ec0cfbab6d06875a7678f164865bba89e9011e8a0dc728`；每组
  按 `40/5/5` 分到 train/dev/test，连续 5 个 epoch 的两卡 rank counts 均为
  train `[400,400]`、dev/test `[50,50]`。正式 train entry 完成两卡 DDP 2-step
  与 step 2 -> 3 resume，两者 exit 0，日志明确记录 state restore；恢复后
  total/token/RVQ 为 `16.085545`/`6.741939`/`9.137355`。这只验证
  split、distributed partition、checkpoint/resume 和 finite loss 契约；独立 dev validation
  结论见上一项，长跑、质量和收敛均未验证
  （[014 result, lines 256-279](results/014-longcat-stable-stage1.md#L256-L279)）。
- 014 的 pami201 32-sample root smoke/canary 通过：在 202 NFS 超时时，`145` 本地 runtime、
  本地 Qwen/LongCat cache 与 `/mnt/pami201` 数据根可完成 stage 1 TTS/S2ST 2-step
  forward/backward/optimizer、metrics 写出和训练后 teacher-forced acoustic generation；正式
  `scripts/train.py` stage 1 也完成 100-step 小数据 canary，窗口口径 total/token/RVQ loss 均下降。
  单 payload group root 暴露了 rank 空数据问题；按 8 条重写成 4 个 payload group 后，两卡 DDP
  完成 2 steps，并通过新增 `train.ckpt_path` 从 step 2 恢复到 step 3。同时修复了
  `load_codec("longcat")` 在运行时求值 `LongCat` 的 `NameError` 并补测试。该结论只验证小数据入口、
  checkpoint/resume 执行契约和 finite 指标，不支持质量或收敛结论
  （[014 result, lines 172-224](results/014-longcat-stable-stage1.md#L172-L224)，
  [lines 226-254](results/014-longcat-stable-stage1.md#L226-L254)）。
- 014 的 P0 在 debug-migrated copy 上通过：代码迁移的本地/远端 targeted tests 通过，远端
  targeted tests 为 `Ran 96 tests ... OK`、exit `0`；历史 debug duration workaround 更新
  2000 个 audio item 且只写入 debug copy；当前代码缺失 `AudioMeta.DURATION` 时可从 codec
  frame count 和 runtime frame rate 推导音频秒数；复旦 `145` 上 TTS/S2ST wrapper、metrics、
  generation 和 waveform decode 均为 finite。该结论只接受 debug copy P0，不允许直接晋级正式
  stable root 或 native stable P1 长跑
  （[014 result, lines 3-16](results/014-longcat-stable-stage1.md#L3-L16)，
  [lines 18-39](results/014-longcat-stable-stage1.md#L18-L39)，
  [lines 41-69](results/014-longcat-stable-stage1.md#L41-L69)，
  [lines 133-161](results/014-longcat-stable-stage1.md#L133-L161)）。
- 真实 Qwen3-0.6B、LongCat native token 与 8 层 RVQ decoder 上，TTS/S2ST fixed-sample
  均完成 2-step forward/backward/optimizer；两条 total、audio token CE 和各 RVQ codebook CE
  均下降。teacher-forced RVQ sampling 在 3 个记录点、每点 4 个 seed 上均可 decode 2.16s
  finite waveform，但 feature MSE 非单调，该 smoke 不支持收敛或质量结论
  （[011 result, lines 24-49](results/011-qwen-rvq-staged-joint-training.md#L24-L49)）。
- 同一 011 run 的训练后端到端 generation 尚未通过：Python 3.12 runtime Protocol 对真实
  registered `nn.Module` backbone 产生 false negative，两条任务退出码为 1，未写出
  `generation.json`/`metrics.json`。这是真实 P0 接口失败，不能由 training-only metrics
  或 teacher-forced waveform 代替
  （[011 result, lines 51-89](results/011-qwen-rvq-staged-joint-training.md#L51-L89)）。
- 真实 Qwen3/LongCat 上，Flow 与 RVQ oracle 的单卡 fixed-sample、两卡静态 DDP + LBA 均完成
  2-step forward/backward/optimizer 和完整 callback；RVQ 静态 DDP 没有 unused-parameter 错误。
  该 smoke 只验证执行契约，不支持质量或收敛结论
  （[010 result, lines 20-40](results/010-codec-oracle-flow-rvq-smoke.md#L20-L40)）。
- FDU `145` 上，真实 Qwen3/LongCat stage_2 RVQ 的正式 staged joint train entry 已完成
  两卡 DDP + joint LBA 2-step smoke；ASR、TTS 和 toy text MT 子 loader 均写出两个 rank 的
  LBA summary，`metrics.json` 中 total/token/RVQ 均为 finite。该 smoke 只验证 DDP/LBA 执行
  契约，不验证长跑 partition、resume、质量或收敛
  （[013 result, lines 7-45](results/013-fdu-codec-oracle-and-stage-smoke.md#L7-L45)，
  [lines 55-61](results/013-fdu-codec-oracle-and-stage-smoke.md#L55-L61)）。
- 真实 Qwen3/LongCat 变长 batch 4 的 prompt、source acoustic frames、KV cache 和
  waveform decode 在 float32 下完成逐请求 token parity；该短生成 probe 的吞吐为
  serial 的 1.78x，peak allocated 只增加约 22 MB
  （[008 result, lines 27-46](results/008-real-batch-generation-benchmark.md#L27-L46)）。
- 8 层 Qwen RVQ decoder 在真实 TTS/S2ST 固定样本上完成 100-step 训练；
  semantic objective 接近记忆，acoustic causal CE 最后 20-step 均值相对首窗口
  下降约 23%，但 feature/STFT 轨迹不支持 waveform 质量改善结论
  （[009 result, lines 47-60](results/009-real-rvq-overfit-generation.md#L47-L60)）。
- 同一 RVQ formal run 训练后的 TTS/S2ST greedy cached generation 均生成 36
  acoustic frames 和 2.16s finite waveform，端到端 RTF 分别为 0.503/0.494；该结论
  只验证固定样本执行契约，不表示泛化质量
  （[009 result, lines 68-79](results/009-real-rvq-overfit-generation.md#L68-L79)）。
