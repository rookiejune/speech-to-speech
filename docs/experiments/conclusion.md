# Conclusion

## 适用范围

本页最新真实实验是 014 的 pami201 1k stage 1 step-500 resume（2026-07-27）：已在
1000-sample、20 payload-group 的正式 split manifest 上完成两卡 step 0/50/100/200/300/400/500
完整 dev validation。step 500 相对初始 token/RVQ CE 下降 `19.602%`/`2.701%`，3 个 acoustic
codebook top-1 均明显高于随机，并保留 step 200/300/400/500 与 latest checkpoint。RVQ CE
尚未达到 5% 门槛；step-2000 gate、最多 5k-step pilot、decode、长跑监督和质量/收敛仍未验证。
同日的 32-sample smoke/canary 另外证明：`/mnt/pami202` 超时时，可以用
145 本地 runtime/HF cache 和 pami201 小数据根完成 stage 1 TTS/S2ST 两步入口、
正式 `scripts/train.py` 100-step canary，以及重分片 root 的两卡 DDP/resume。
014 的 LongCat stable stage 1 P0 验收（2026-07-25）只证明 debug copy 上历史 duration
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

- 014 的 pami201 1k pilot validation、100-step canary 与 step-500 resume 通过：两卡分别读取 50 条 dev sample，
  指标按有效 token/frame 跨 batch/rank 加权，并区分 sanity/interval 写入 `metrics.json`。
  step 0 -> 500 的 token CE 为 `9.883604 -> 7.946233`，RVQ CE 为
  `9.151793 -> 8.904623`；3 个 acoustic codebook top-1 从 `1/4264,0,0` 提升到
  `163/4264,132/4264,155/4264`。step 200/300/400/500 与 `last.ckpt`、TensorBoard event 均已保留。
  该 pilot 证明明确学习方向，但 RVQ CE 只下降 `2.701%`，尚未达到 5% 晋级门槛，
  不支持质量或收敛结论
  （[014 result, lines 276-311](results/014-longcat-stable-stage1.md#L276-L311)，
  [lines 313-372](results/014-longcat-stable-stage1.md#L313-L372)）。
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
  （[014 result, lines 251-275](results/014-longcat-stable-stage1.md#L251-L275)）。
- 014 的 pami201 32-sample root smoke/canary 通过：在 202 NFS 超时时，`145` 本地 runtime、
  本地 Qwen/LongCat cache 与 `/mnt/pami201` 数据根可完成 stage 1 TTS/S2ST 2-step
  forward/backward/optimizer、metrics 写出和训练后 teacher-forced acoustic generation；正式
  `scripts/train.py` stage 1 也完成 100-step 小数据 canary，窗口口径 total/token/RVQ loss 均下降。
  单 payload group root 暴露了 rank 空数据问题；按 8 条重写成 4 个 payload group 后，两卡 DDP
  完成 2 steps，并通过新增 `train.ckpt_path` 从 step 2 恢复到 step 3。同时修复了
  `load_codec("longcat")` 在运行时求值 `LongCat` 的 `NameError` 并补测试。该结论只验证小数据入口、
  checkpoint/resume 执行契约和 finite 指标，不支持质量或收敛结论
  （[014 result, lines 167-203](results/014-longcat-stable-stage1.md#L167-L203)，
  [lines 205-219](results/014-longcat-stable-stage1.md#L205-L219)，
  [lines 221-249](results/014-longcat-stable-stage1.md#L221-L249)）。
- 014 的 P0 在 debug-migrated copy 上通过：代码迁移的本地/远端 targeted tests 通过，远端
  targeted tests 为 `Ran 96 tests ... OK`、exit `0`；历史 debug duration workaround 更新
  2000 个 audio item 且只写入 debug copy；当前代码缺失 `AudioMeta.DURATION` 时可从 codec
  frame count 和 runtime frame rate 推导音频秒数；复旦 `145` 上 TTS/S2ST wrapper、metrics、
  generation 和 waveform decode 均为 finite。该结论只接受 debug copy P0，不允许直接晋级正式
  stable root 或 native stable P1 长跑
  （[014 result, lines 3-5](results/014-longcat-stable-stage1.md#L3-L5)，
  [lines 20-34](results/014-longcat-stable-stage1.md#L20-L34)，
  [lines 38-64](results/014-longcat-stable-stage1.md#L38-L64)，
  [lines 128-156](results/014-longcat-stable-stage1.md#L128-L156)）。
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
