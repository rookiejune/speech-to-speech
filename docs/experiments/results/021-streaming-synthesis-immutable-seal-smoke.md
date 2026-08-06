# 021 Streaming synthesis and immutable seal smoke

## Scope

2026-08-04 在复旦 `125` 的 6 张 RTX 3090 上验证一个 8-sample 流式 S2ST 数据闭环：

```text
GPU 0: T_s -> Qwen3-TTS -> A_s
GPU 1: T_s -> Qwen3-0.6B -> T_t
GPU 2: T_t + ref A_s -> MOSS-TTS v1.5 -> A_t
GPU 3: A_s/A_t -> LongCat 16 kHz 4-codebook codes
GPU 4-5: sealed LongCat store -> 2-rank DDP S2ST overfit
```

本次实现是 `/tmp` 下的一次性 probe，不是仓库生产实现。代码 checkout 保持 clean，未同步本地
未提交修改，也未 commit/push 实验记录。

## Pipeline contract exercised

- SQLite 是运行状态事实源；task 使用 `UNIQUE(kind, cache_key)`、lease/claim 和显式 retry。
- AS 与 TT 可并行；AT 只在 AS/TT join 后创建，CODEC 只在 AT ready 后创建。
- artifact 使用内容寻址路径、临时文件加原子发布；已有 artifact 可恢复而不重算。
- 改变失败阶段的生成配置会生成新的 cache key，不会把旧失败错误地视为新配置结果。
- final seal 只读取全部 ready 的样本，写出标准 `base/` 与 `longcat/` immutable store，并发布
  `sealed.json`。

## Results

最终状态为 8 个 sample 全部完成 AS、TT、AT、CODEC；四个阶段的 attempts 均为 8，
`ready_samples=8`。seal 写出 8 条标准 store，`sealed.json` SHA256 为
`e77361fa445034f07296a64380c6c7b262492262887837fb708229edc72b45df`。

对四个 worker 原样重跑后，AS、TT、AT、CODEC 均报告 `computed: 0`；随后 status 中 attempts
仍为 8，验证成功结果不会被重复计算或重复记账。

现有 workspace/anydataset loader 用 `moss_tts.codec("longcat", root=..., split="train")`
并显式 `filter(None)` 成功读取 8 条。sample 0 的 source/target LongCat tensor shape 分别为
`[163, 4]` 和 `[128, 4]`，文本与音频 code 四元组均可读。

GPU 4-5 随后使用现有 `scripts/overfit.py`、Qwen3-0.6B、`longcat_native`、flattened layout、
`task=s2st` 和 full parameter policy 完成 2-rank DDP 2 steps。total/token loss 均 finite，
由 `11.628868103027344` 降到 `10.658365249633789`；训练参数为 `495610112`，结果文件 SHA256
为 `71baefbe0c96930d11bc76a7604e4141c833ed4705fa83bb3fa8dd0958d02d5a`。

## Resource findings

| Stage | Probe peak | Observed latency | Placement conclusion |
| --- | ---: | ---: | --- |
| Qwen source TTS | 4074.75 MiB | 8.49 s/sample | 24 GB card has ample memory; throughput is the concern. |
| Qwen3-0.6B translation | 1152.81 MiB | 1.12 s/sample | Can share a producer GPU if strict stage isolation is not required. |
| MOSS target TTS | 23219.89 MiB | 19.77 s source/reference setup, 3.85 s target generation in the isolated probe | 24 GB has no production safety margin. |
| LongCat encode | 2968.02 MiB | 1.09 s source, 0.06 s target in the isolated probe | Can share a light producer GPU or run as a separate scalable worker. |

MOSS 在 3090 上使用较长输入和 `max_new_tokens=256` 时仍因申请额外 72 MiB OOM；启用
expandable allocator 并改为 `max_new_tokens=128` 后 8 条全部完成。这个 workaround 可能截断
长音频，因此生产 target TTS 应优先使用 40 GB-class GPU，并在发布前增加长度和质量门禁。

## Ownership boundary

- `anytrain`：无状态模型 adapter/protocol（TTS、可复用 text generation、codec）、evaluator、
  Lightning schedule/performance/checkpoint primitive。它不拥有本 pipeline 的 task 状态和 cache key。
- `anydataset`：canonical `Sample`、通用 artifact/store metadata、immutable manifest/shard、
  materializer/merge/seal 和稳定 reader contract。它不拥有服务队列、lease 或训练消费 cursor。
- `speech-to-speech`：AS/TT/AT/CODEC DAG 语义、完整 cache key、join/retry/backpressure、质量策略、
  snapshot 发现、streaming DataModule 和精确 checkpoint cursor。
- `workspace`：WMT19/Qwen/MOSS 具体资产、物理路径、环境和生产 job wrapper。

不要把 SQLite/Redis/Kafka queue、lease/backpressure 或训练 cursor 放入 anytrain/anydataset core；
这些属于项目 orchestration 和训练消费语义。

## Baseline limits (2026-08-04)

上一轮结果只证明流式合成/codec、去重恢复、immutable seal、现有 loader 读取和新数据 DDP
optimizer step 的执行闭环，不证明翻译或语音质量，也不证明 MOSS 128-token 输出完整。当时
Lightning 仍是对 sealed root 的一次性启动，尚未实现长驻消费、精确 cursor checkpoint 或 DDP
restart 验证；以下续跑 probe 覆盖了这些缺口。

## Resumable streaming DDP probe (2026-08-05)

在复旦 145 的 GPU `5,6` 上以 revision `37e7a36` 运行正式
`scripts/train.py experiment=train/streaming_s2st` 入口，2-rank DDP、`batch_size=1`、
`trainer.max_epochs=1`、8 个双向 probe sample。首轮 producer 发布 `[0..3]` 后在 600 秒延迟中
被终止；第二次相同入口设置 `producer_options.retry=true`，从同一 `last.ckpt` 恢复并继续合成。

- 训练日志明确恢复 `checkpoints/last.ckpt`、`Restored all states`，随后从 epoch 内剩余两步跑到
  `4/4`；最终 `global_step=4`、`epoch=0`。
- checkpoint cursor 为 `committed_position=8`、`committed_batches=4`、`world_size=2`、
  `batch_size=1`、`next_snapshot_sequence=2`，并保存完整 catalog digest；`SynthesisSampleLogger`
  恢复并保留 `logged=[0,1,2]`。
- stream seal 为 `sample_count=8`、`snapshot_count=2`；snapshot membership 严格为
  `[0,1,2,3]` 与 `[4,5,6,7]`，无重复或遗漏，seal digest 与 checkpoint 一致。
- TensorBoard `version_0` 含 9 个文本 summary 和 6 个音频标签（均 `16 kHz`、`3840 frames`），
  文本包含生成目标 translation；续跑产生的 `version_1` 没有重复 synthesis 标签，证明 callback
  状态随 checkpoint 恢复。训练结束后 GPU 5/6 均为 `15 MiB / 0%`，无残留 rank 或 producer。

本 probe 只验证数据边界、resume 和日志契约，不代表 probe 文本/音频的翻译质量；生产长跑仍需
保留 `max_epochs=1`，每次入口从同一 stream root 和 `last.ckpt` 继续合成与消费。

## Streaming telemetry resume probe (2026-08-05)

在复旦 145 的 GPU `5,6` 上以隔离 revision `9d42a47` 再跑正式 streaming 入口。第一次启动在
`committed_position=4` 且 `last.ckpt` 已完整写入后终止训练，第二次相同入口自动恢复并跑到
`4/4`、`committed_position=8`。本次 60 秒 producer delay 恰好在终止窗口结束，第二 snapshot 和
seal 已由首个 producer 发布；因此训练/checkpoint resume 由本次覆盖，producer 从半成品继续合成仍
由上一节 600 秒 delay probe 覆盖。

- TensorBoard scalar 跨两个 event 文件保持同一 `version_0`。step 1-4 的 committed position 为
  `2,4,6,8`，published samples 为 `4,4,8,8`；`batch_wait_seconds` 为
  `1.504,0,0,0`，`wait_seconds_total` 在 checkpoint/resume 后保持 `1.504`。
- step 1-4 的 `batch_fetch_seconds` 为 `1.580,0.024,0.042,0.015`，`batch_load_seconds` 为
  `0.076,0.024,0.042,0.015`，`step_seconds` 为 `2.230,0.259,1.470,0.265`；step 1 的
  `wait_ratio=0.397`。
- `streaming_gpu.csv` 在第一次进程终止前为 251 行，第二次启动后沿用同一个 header/路径追加到
  309 行；包含且只包含物理 GPU 5/6。第二进程 summary 有 29 个 poll，平均 GPU utilization
  `43.83%`、平均显存 `10497 MiB`、平均功耗 `66.49 W`。summary 的 `scope=current_process`；
  跨启动完整时间轴以 CSV 和两个 event 文件为准。
- step 1-4 的最近一次双卡平均 GPU utilization scalar 为 `50%,50%,9%,50%`，显存为
  `6649,11512,9873,10143 MiB`。原始 CSV 继续覆盖 checkpoint/sample callback 和数据等待期间，
  而 `step_seconds` 只覆盖 Lightning train batch hook 内的计算，不应解释成完整 wall time。
- producer JSON 事件完整记录两个 `snapshot_publish`（`0.313s`、`2.021s`）、60 秒
  `resume_probe_delay` 和 `stream_seal_validation`（`0.00084s`），均带 timestamp、sample count、
  device 和可见 GPU ids。真实 AS/TT/AT/codec producer 不在本仓库；只有它们显式使用同一 stage
  helper 后，才能得到真实生产阶段的分段事件，当前不能把 probe publish 阶段冒充模型阶段。
- sample logger 在同一 TensorBoard run 中保留 9 个文本 summary 和 6 个音频 tag。sample 0 为
  `流式探针源句 0 -> streaming probe generated translation 0`，sample 1 为反向
  `streaming probe source sentence 0 -> 流式探针生成译文 0`。
- 训练正常完成后 GPU 5/6 均回到 `15 MiB / 0%`，无残留 rank 或 producer。

## Interruptible SIGTERM and producer resume probe (2026-08-05)

revision `cf1452c` 先让 streaming dataset poll 最多每 0.5 秒检查一次 Lightning stop request。
只向 Lightning launcher PID 发送一次 SIGTERM 的对照组在 7.314 秒内关闭两 rank 和 producer，保留
首个 snapshot 与完整 `last.ckpt`、不写 seal；第二次相同入口自动恢复并完成 `4/4`。向整个 DDP
process group 发送 SIGTERM 时则暴露了 Lightning 2.6.1 的信号重入：rank 0 的 launcher fan-out 会
再次向已收到 group signal 的 rank 1 发 TERM，rank 1 因此重复进入 distributed broadcast 并挂住。

revision `c4f3e32` 在每个 rank 的 train start 后用 one-shot guard 包装 Lightning 已注册的 SIGTERM
handler。复旦 145 GPU `5,6` 上重新从空 stream 运行，训练停在
`read=4 / committed=4 / published=4 / expected=8` 且 `last.ckpt` 完整时，只发送一次 process-group
SIGTERM：rank 0/1 日志各出现一次 `Received SIGTERM`，两 rank 和 producer 在 7.115 秒内全部退出，
仅保留首个 snapshot、没有 `sealed.json`，GPU summary 已落盘，GPU 回到 `15 MiB / 0%`；本轮没有
使用 SIGKILL。

随后用完全相同的 stream/root/run id、`expected_samples=8`、DDP2、`batch_size=1` 和
`producer_options.retry=true` 再次进入正式 streaming 入口：日志明确从 `last.ckpt` 恢复并显示
`Restored all states`，producer 从 4-sample durable prefix 继续发布 `[4..7]`，写出第二个 snapshot
与 seal，训练在 epoch 0 到达 `4/4` 后因 `max_epochs=1` 正常结束。最终 checkpoint 为
`global_step=4`、`epoch=0`、`committed_position=8`、`committed_batches=4`、`world_size=2`、
`batch_size=1`、`next_snapshot_sequence=2`；等待状态也随 checkpoint 恢复到
`wait_seconds=60.126`、`wait_events=2`、`poll_count=2`。

- step 1-4 的 `batch_wait_seconds` 为 `30.077,0,30.049,0`，累计 `wait_seconds_total` 为
  `30.077,30.077,60.126,60.126`，`wait_ratio` 为 `0.936,0,0.965,0`；对应 `step_seconds` 为
  `2.006,0.271,1.036,0.261`。cursor 为 `2,4,6,8`，published samples 为 `4,4,8,8`。
- 两次启动仍写入同一个 TensorBoard `version_0`，其中有两个 event 文件；`streaming_gpu.csv`
  追加为 404 行且只含物理 GPU 5/6。第二进程 summary 的平均 GPU utilization 为 `27.13%`、
  平均显存 `9515 MiB`、平均功耗 `55.91 W`；完整跨启动时间轴仍以追加 CSV 为准。
- sample logger 保留 9 个文本 summary 与 6 个音频 tag；sample 0 为
  `流式探针源句 0 -> streaming probe generated translation 0`，sample 1 为
  `streaming probe source sentence 0 -> 流式探针生成译文 0`。
- producer stage 事件记录 snapshot publish `0.050s,0.015s,0.045s` 和 seal validation
  `0.00029s`。真实 AS/TT/AT/codec producer 仍在本仓库外；只有那些 producer 显式接入同一个
  stage helper 后，才能得到真实模型阶段的分段耗时并与 GPU CSV 对齐。

## Real A100 producer, training, and sealed resume probe (2026-08-06)

revision `2f36d69` 已把真实 WMT19 producer 接入正式 streaming 入口；checkpoint link 修正使用
revision `94bbac5`。复旦 121 的运行根为：

```text
/mnt/pami202/zhuyin/dynamic/debug/streaming_s2st_train_concurrent_a100_20260806_a
```

资源映射保持 GPU 0 上的既有任务不动：

```text
GPU 1: Qwen source TTS + Qwen3-0.6B translation + LongCat codec
GPU 2: MOSS-TTS v1.5 target TTS worker
GPU 3: Qwen3-0.6B backbone training
```

输入选择两个 WMT19 pair，workspace 双向展开后得到 4 个训练 sample。producer 以 batch size 2
发布两个 snapshot，最终 seal 为 `sample_count=4`、`snapshot_count=2`，catalog SHA256 为
`d6942f97e2bcc39c1c07488729db0bfb9507cff26bbe331fd5d6b4b935899fc4`。4090 上相同 MOSS
batch size 2 会在 24 GB 显存 OOM；A100 40 GB 上两批 target TTS 峰值约 24.46/24.52 GB，均完成，
因此当前 MOSS batch size 2 的最低生产档位应为 40 GB-class GPU。

真实 producer 分段 telemetry 如下。GPU utilization、memory 和 power 来自 1 秒采样；codec 等
短于 1 秒的 stage 样本不足，不能用 utilization 数值判断 kernel 效率。

| Stage | Batch 0 | Batch 1 | GPU utilization | Mean used memory | Mean power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen source TTS, 2 samples | 32.223 s | 22.653 s | 26.15% / 27.65% | 8.73 / 9.71 GB | 42.74 / 43.09 W |
| Qwen translation, 2 samples | 3.386 s | 1.213 s | 7.33% / 41.00% | 8.69 / 9.71 GB | 41.91 / 46.86 W |
| MOSS target TTS, 2 samples | 7.164 s | 5.711 s | 49.83% / 54.75% | 24.46 / 24.52 GB | 120.22 / 124.34 W |
| LongCat codec, 4 audios | 0.893 s | 0.281 s | under-sampled | 9.71 / 13.32 GB | 37.40 / 37.19 W |
| Snapshot publication, 2 samples | 0.731 s | 0.877 s | CPU/NAS stage | n/a | n/a |

producer 从 telemetry 启动到结束共 `120.273s`。训练在首批发布前等待，第二批发布并 seal 后完成
唯一一个 optimizer step；Lightning 明确打印 `Trainer.fit stopped: max_epochs=1 reached`。首轮
`metrics.json` 保留 finite loss：total `172.18854`、token `12.18853`、CTC `158.0`。TensorBoard
关键 streaming 指标为：

```text
batch_wait_seconds       150.0114
batch_fetch_seconds      150.2223
batch_load_seconds         0.2109
step_seconds               3.0175
wait_ratio                  0.9789
read_position               4
committed_position          4
committed_batches           1
published_samples           4
expected_samples            4
poll_count_total             5
wait_events_total            2
```

同一 TensorBoard event 含 20 条文本和 8 条音频：每个 sample 都有 source text、model
translation、dataset translation、三行 comparison、metadata、source audio 和 target audio。
4/4 的 model translation 都与 WMT19 dataset translation 不同，证明训练 target 是 backbone
前的 Qwen 模型译文，原 WMT target 只作为 reference sidecar。sample 0 的对比为：

```text
source:
巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。

model translation:
Paris - As the economic crisis continues to deepen and spread, the world is searching for similar historical events to help us understand the current situation.

dataset translation:
PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.
```

首轮 checkpoint 使用旧的 `save_last=true`，因此 async saver 先发布 3,616,093,609-byte
`step-00000001.ckpt`，再完整复制一份 `last.ckpt`。训练计算结束后 NAS 尾延迟超过 30 分钟，期间
checkpoint thread 位于 `folio_wait_bit_common`。revision `94bbac5` 将 streaming 配置改为
`save_last=link`，正常有新 step 的 checkpoint publication 只需写一份 persistent checkpoint，
`last.ckpt` 使用 link；本地 28 项定向测试、Ruff、basedpyright 以及复旦 py312 的 12 项
streaming entry unittest 均通过。

随后用完全相同的 stream/root/run、batch size、world size、codec 和原 HF cache 绝对路径再次进入
`jobs/016/01_streaming_s2st.sh`：日志明确出现 `Restoring states from .../last.ckpt` 和
`Restored all states`，随后直接因 `max_epochs=1` 结束，没有启动 producer/MOSS worker，没有新增
snapshot，也没有 optimizer step。最终 checkpoint 仍为 `global_step=1`、`epoch=1`，DataModule
cursor 为 `committed_position=4`、`committed_batches=1`、`next_snapshot_sequence=2`，catalog
digest 与 seal 一致；sample logger 保留 `logged=[0,1,2,3]`。producer log、telemetry、GPU CSV、
seal 和两个 snapshot 的大小/mtime 均未变化，GPU 3 最终释放到 10 MiB。

完成态 no-op resume 仍暴露一个独立的 anytrain/Lightning 欠账：新 callback 实例没有恢复
`_last_checkpoint_saved`/persistent publication，`on_train_end` 会再次完整刷新一次 `last.ckpt`；本次
因此又产生约 20 分钟 NAS 尾延迟，并把 `metrics.json` 暂时覆盖为空 metrics。首轮 metrics 已备份
并恢复为 canonical `metrics.json`，no-op 结果保存在 `metrics.sealed-resume.json`。这个问题不影响
未 seal 的真实故障续跑：只要恢复后产生新 step，`save_last=link` 会复用该 persistent checkpoint；
但完成态重复进入的无写入快速退出仍应在 anytrain checkpoint state 恢复层解决。
