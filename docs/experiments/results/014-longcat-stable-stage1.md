# 014 LongCat Stable Stage 1 P0 Acceptance

对应 [014 schedule](../schedules/014-longcat-stable-stage1.md)。本文记录 2026-07-25
在 FDU `145` 上完成的 P0 验收和当前边界。状态：**P0 在 debug-migrated copy 上通过**；
正式 stable data root 的 parquet 指纹审计、无 duration parse/map-style dataloader probe、
native token/RVQ 分布审计和 pilot split candidate 已补齐。2026-07-27 另用 pami201
32-sample root 完成 stage 1 TTS/S2ST 2-step smoke、正式 `scripts/train.py` 100-step canary，
并在重分片 root 上完成两卡 DDP 2-step 与 resume 到 step 3，用来绕开 `/mnt/pami202` 超时
验证入口闭环。随后又固化 pami201 1k pilot split manifest，验证两卡连续 5 个 epoch
的 rank 均衡，并完成 2-step DDP 及 step 2 -> 3 resume。这些结果仍只验证数据与
训练执行契约；dev CE、每 codebook top-1、长跑监督和质量/收敛尚未验证。

## 范围与代码状态

本轮先把通用 loss 与 loss logging 下沉到 `anytrain`，`speech-to-speech` 只保留 batch、
task 和 objective order 的薄适配。共享远端 repo 有未提交改动，因此没有覆盖
`145:/mnt/pami202/zhuyin/repos`；复旦验证使用独立临时目录：
`145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725`。

本地针对性验证已通过：

| Package | Scope | Result |
| --- | --- | --- |
| `third_party/anytrain` | loss + lightning callback targeted unittest | 55 tests OK |
| `speech-to-speech` | logging + model loss contracts + acoustic DiT targeted unittest | 38 tests OK |
| `third_party/anytrain` | basedpyright + ruff + `git diff --check` | 通过 |
| `speech-to-speech` | basedpyright + `git diff --check` | 通过 |

远端 targeted tests 在同一临时目录重跑，日志写入
`145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/logs/remote_targeted_20260725_doc_verify.log`，
退出码写入同目录 `remote_targeted_20260725_doc_verify.exit`。本次远端验证运行
`anytrain/tests/test_lightning.py`、`anytrain/tests/test_loss.py`、
`speech-to-speech/tests/test_logging.py`、`test_model_loss_contracts.py` 和
`test_acoustic_dit.py`，结果为 `Ran 96 tests ... OK`，exit 为 `0`。

## Debug Duration Migration

本节记录的是 2026-07-25 的历史 debug 验收路径。后续 datamodule contract 已改为
duration metadata 可选：缺失时从 codec frame count 和 runtime frame rate 推导，因此正式 stable
root 不再需要先补写 `AudioMeta.DURATION` 才能进入 parse/LBA/training 验证。

P0 数据验收使用 debug copy，不修改正式 stable root。debug 数据目录为
`145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/datasets/wmt19_tts_duration_p0_20260725_040054`。
duration migration 只作用于该目录下的 `longcat` view：

| Artifact | Path |
| --- | --- |
| migration summary | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/logs/migrate_20260725_040054.json` |
| migration exit | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/logs/migrate_20260725_040054.exit` |
| backup before duration write | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/datasets/wmt19_tts_duration_p0_20260725_040054/longcat/samples.before-duration.parquet` |
| migrated samples | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/datasets/wmt19_tts_duration_p0_20260725_040054/longcat/samples.parquet` |

Migration summary: `dry_run=false`、`frame_rate=50.0`、`pending_audio_items=2000`、
`updated_audio_items=2000`、exit `0`。只读核对显示 backup 与 migrated parquet 均在
debug 目录内，inode 分别为 `754480618`、`754480619`，hardlink count 均为 `1`。
因此本轮的 duration migration 是历史 debug workaround，只证明该 debug copy 曾可补写
`AudioMeta.DURATION`；当前代码已在缺失 `AudioMeta.DURATION` 时从 codec frame count 和
runtime frame rate 推导音频秒数，正式 root 不再需要 duration migration。正式 root 剩余工作
见下方正式根审计与 probe 记录。

debug 数据 summary 写在
`145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/datasets/wmt19_tts_duration_p0_20260725_040054/summary.json`。
该 copy 的 `limit=1000`，sample 0 source semantic/acoustic shape 为 `[27]` / `[3,27]`，
target semantic/acoustic shape 为 `[36]` / `[3,36]`；CUDA 可用，设备数 `4`。

## Formal Root Parquet/Fingerprint Audit

2026-07-26 对正式 LongCat stable root 做只读 parquet/fingerprint 审计。正式根为
`145:/mnt/pami202/zhuyin/datasets/wmt19_tts/longcat`，审计 JSON 写在
`145:/mnt/pami202/zhuyin/dynamic/debug/s2s-014-formal-root-parquet-audit-20260726-074012/audit.json`。
审计环境使用 `pyarrow 24.0.0`，`failure_count=0`。这一步只证明正式根的关键文件可读、
行数和 sha256 已固化；不包含 split manifest、native token/RVQ 分布或训练验收。

| Artifact | Size | Rows | SHA256 | Columns |
| --- | ---: | ---: | --- | --- |
| `dataset.json` | `97` | — | `ff9e2ce9e9e28495481566d8c2aaa65c9713ef03e8f3e010d51f89b62040eeb6` | — |
| `samples.parquet` | `13225` | `1000` | `caaae1793a81de3c763e906c533e15b9fe310a1bd39d50faab2ab77d58e14b52` | `sample_id,sample_index,items` |
| `source/audio/longcat/manifest.parquet` | `14697` | `1000` | `3c984281251de87e854775094dbc06ed2c83a365d582a096c2e42f8548081ec9` | — |
| `target/audio/longcat/manifest.parquet` | `14697` | `1000` | `4ff9e550b3b90e8e0eec02c9e5ae63d24bc9e6d7a5ed55edc8cf09c3e0a6dcb3` | — |
| `source/text/text/manifest.parquet` | `14683` | `1000` | `8c6d6a5ca7931e354584803b2b7ac82835757e42d55662a178c3911481a4b976` | — |
| `target/text/text/manifest.parquet` | `14683` | `1000` | `c3dbba304b765d2f5f75462bf17d9b65cb29bdd85b171094774ae94c6a5bb770` | — |

## Formal Root Parse and DataLoader Probe

2026-07-26 对同一正式 root 做 parse 与 map-style DataLoader probe。输出目录为
`145:/mnt/pami202/zhuyin/dynamic/debug/s2s-014-formal-root-parse-dataloader-20260726-124930`。
probe 读取到 `dataset_len=1000`，抽样 index 为 `0,1,2,3,4,500,999`；这些样本的
source/target 均缺失 duration metadata，计数为 `source=7`、`target=7`。

所有抽样 source/target LongCat tensor 都成功解析为 integer、rank-2、4 codebooks。DataModule
的 S2ST dataloader batch 通过，`input_shape=[1,434]`、`label_shape=[1,434]`、
`audio_seconds=[1.7200000286102295]`。这证明本次正式根 probe 没有把真实音频静默计为
0 秒；但它仍只是抽样 parse/dataloader 验证，不等同于 split、分布、训练 step、overfit、
DDP 或 resume 验收。

## Formal Root Distribution and Pilot Candidate Audit

2026-07-26 对正式 LongCat train split 做 1000-sample distribution/pilot candidate 审计。
输出目录为
`145:/mnt/pami202/zhuyin/dynamic/debug/s2s-014-distribution-pilot-20260726T070417Z`，
脚本为
`145:/mnt/pami202/zhuyin/dynamic/debug/s2s-audit-scripts/s2s_014_distribution_pilot.py`。
本轮没有修改正式 root；这里的 pilot split 只是 debug candidate artifact，不是最终 formal
split manifest，也不能直接作为 stage 1 长跑入口。

| Artifact | Size | Role |
| --- | ---: | --- |
| `stats.json` | `10088` | token/frame/duration distribution audit |
| `split_candidate.json` | `9054` | 800/100/100 顺序 pilot candidate |

审计读取 `wmt19_tts_codec(codec='longcat', split='train')`，`processed=1000/1000`；
`parse_errors=0`、`validation_errors=0`、`valid_sample_count=1000`。source/target LongCat
tensor 均为 rank-2 integer，且均为 4 个 codebooks。source frame count 为
min `14`、p50 `43`、p95 `43`、max `43`、mean `42.634`；target frame count 为
min `20`、p50 `43`、p95 `43`、max `43`、mean `42.507`。

正式 root 中 `AudioMeta.DURATION` 仍然缺失，计数为 source `1000`、target `1000`；但
从 frame count 推导出的正数 duration 计数为 source `1000`、target `1000`，与前一节
parse/dataloader probe 的 optional-duration contract 一致。`stats.json` 中已记录各 codebook
的 min/max 与 top counts；source codebook 范围包含 cb0 `0..8191`、cb1-cb3 `0..8099`，
target cb0 为 `2..8186`、cb1-cb3 为 `0..8099`。

`split_candidate.json` 使用 `sequential_no_sample_id` 方法生成候选切分：train `800`、
dev `100`、test `100`。由于这是从当前 1000 条 train view 顺序生成的 debug candidate，
它不作为后续训练入口。split 生成规则、指纹和路径后续已在 pami201 1k
pilot manifest 中固化；companion metadata 与 P1 长跑仍是后续工作。

## P0 Wrapper Run

复旦 P0 wrapper 复用现有 `jobs/011/01_rvq_native_p0_fixed_sample.sh`，但输出写入 014 路线的
debug 验收根：
`145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/train-p0-duration/run_20260725_040807`。
wrapper log 为同一 debug root 下的 `logs/p0_duration_20260725_040807.wrapper.log`，
其中记录 output root 与 launcher dir。TTS、S2ST 和 overall exit 均为 `0`：

| Exit file | Value |
| --- | ---: |
| `train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/launcher/overall.exit` | `0` |
| `train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/launcher/tts.exit` | `0` |
| `train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/launcher/s2st.exit` | `0` |

两条任务均使用 `max_steps=1`、`stage_0`、`parameter_policy=full`，日志显示
`Trainer.fit stopped: max_steps=1 reached.`。本次只验证真实资源、forward/backward、
optimizer step、metrics 写出和训练后 generation gate；一步 run 不支持收敛结论。

| Task | total loss | token | RVQ | Metrics |
| --- | ---: | ---: | ---: | --- |
| TTS | `18.403934` | `9.250000` | `9.153935` | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/tts/rvq-8l/metrics.json` |
| S2ST | `19.127556` | `9.939189` | `9.188368` | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/s2st/rvq-8l/metrics.json` |

训练后的 generation 均写出 finite waveform：

| Task | finite | Duration | Feature shape | Waveform shape | RTF | Artifact |
| --- | --- | ---: | --- | --- | ---: | --- |
| TTS | `true` | `3.84s` | `[64,1024]` | `[1,61440]` | `2.113459` | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/tts/rvq-8l/generation.json` |
| S2ST | `true` | `3.84s` | `[64,1024]` | `[1,61440]` | `1.938283` | `145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/train-p0-duration/run_20260725_040807/011-qwen-rvq-native-p0-fixed-sample/s2st/rvq-8l/generation.json` |

## Toy Smoke

为验证下沉后的 `LossSummaryCallback` / `LossItemLoggerCallback` 在训练闭环中可用，远端还运行
`scripts/overfit.py experiment=toy_smoke train.max_steps=1 hydra.job.chdir=false`。输出 metrics
写入
`145:/mnt/pami202/zhuyin/dynamic/debug/codex-s2s-anytrain-migration-20260725/train/toy-smoke/tts/toy-flow-1l/metrics.json`。
该 run 为 `max_steps=1`、`task=tts`、`stage_0`，metrics 全部 finite：
`loss=11.212008`、`token=8.792907`、`flow_matching=2.419101`。

## PAMI201 Small Root Stage 1 Smoke

2026-07-27 因 `/mnt/pami202` NFS 操作超时，另走 `145` 本地运行目录与 `/mnt/pami201`
数据根做一批 32-sample 小数据，用来先验证 stage 1 入口不再依赖 202。新数据根为
`145:/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_small_32_20260727`；其中 `base/` 从
`145:/tmp/s2s-oracle-010-data/base` 的前 32 条重写为 schema v3 store，`longcat/` 用
`145:/tmp/s2s-011-hf/longcat-audio-codec` 权重在 GPU 1 重新 materialize，避免复用旧的
无文本 LongCat store。`base/.ready` 与 `longcat/.ready` 均存在，总大小约 `16M`，且未发现
指向 `/mnt/pami202` 的 symlink。

数据验收使用 `145:/home/zhuyin/local-runs/s2s-small-data-20260727/runtime` 下同步的本地源码，
显式设置 `HF_HOME=/tmp/s2s-011-hf`、`ANYDATASET_HOME`、`ANYTRAIN_HOME` 和本地
`DYNAMIC_HOME`，不 source 会回落到 202 的 wrapper。`wmt19_tts_codec()` 与
`speech_to_speech.datamodule.dataset.load_dataset()` 均读取到 `32` 条样本；64 个 source/target
LongCat tensor 全部为 rank-2 integer、4 codebooks，frame 数范围为 `22..43`，并通过码本范围
检查。Qwen 快照使用本地 HF cache：
`145:/tmp/s2s-011-hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`。

首次 stage 1 smoke 在 runtime codec 初始化处失败：`speech_to_speech.runtime.codec` 将
`LongCat` 放在 `TYPE_CHECKING` 下，但 `cast(LongCat, ...)` 运行时仍会求值，触发
`NameError: name 'LongCat' is not defined`。修复方式是将 `LongCat` 作为运行时导入，并补充
`tests/test_runtime_codec.py` 覆盖 `load_codec("longcat")` 的 adapter 构造。修复后本地验收为：
`basedpyright` 0 errors、`unittest discover` 229 tests OK / 1 skipped、`compileall` OK、
`git diff --check` OK。

修复后在同一 32-sample root 上运行 `011_rvq_native_stage_1_smoke`，单卡 GPU 1、
`max_steps=2`、`parameter_policy=speech_interface`、`runtime.audio_tokenizer=null`，输出写入
145 本地盘，避免训练产物回落 202。TTS 与 S2ST 均完成真实 Qwen/native LongCat/RVQ 的
forward、backward、optimizer step、metrics 写出与训练后 teacher-forced acoustic generation。

| Task | First loss | Last loss | First token | Last token | First RVQ | Last RVQ | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TTS | `18.403934` | `18.029600` | `9.250000` | `8.955236` | `9.153935` | `9.074364` | `145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-stage1-retry/011-rvq-native-stage-smoke/tts/stage_1-rvq-8l/metrics.json` |
| S2ST | `19.127556` | `18.530514` | `9.939189` | `9.391047` | `9.188368` | `9.139467` | `145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-stage1-s2st/011-rvq-native-stage-smoke/s2st/stage_1-rvq-8l/metrics.json` |

两条任务均写出 `generation.json`，路径分别位于上表同目录。该 smoke 只证明绕开 202 后，
32-sample pami201 root 可以完成 stage 1 的 TTS/S2ST 2-step overfit 入口和 finite 指标。

随后直接调用正式 `scripts/train.py`，不用 `jobs/011/03_staged_joint_train.sh`，因为 wrapper 会默认
进入 static DDP 且容易把 train root 带回 202。本轮显式设置 `trainer.strategy=auto`、
`trainer.devices=1`、`trainer.enable_checkpointing=false`、`data.dataset.root=<32-sample root>`、
`data.dataloader.batch_size=1`、`num_workers=0`、`runtime.audio_tokenizer=null`。`stage_1` 的正式
ASR/TTS 两个 speech loader 均参与，`batches_per_step=2`。

| Run | Max steps | First loss | Last loss | Last mean loss | Token last mean | RVQ last mean | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| formal train smoke | `2` | `15.921749` | `16.752659` | `16.337204` | `6.792932` | `9.106468` | `145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-formal-stage1-2step/011-staged-joint-train/stage_1/stage_1-rvq-8l/metrics.json` |
| formal train canary | `100` | `15.921749` | `15.254683` | `15.080203` | `5.879787` | `8.870380` | `145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-formal-stage1-100step/011-staged-joint-train/stage_1/stage_1-rvq-8l/metrics.json` |

100-step canary 约 55 秒完成，窗口口径下 total loss `last_to_first=0.898619`、token
`last_to_first=0.840348`、RVQ `last_to_first=0.967496`。该 run 证明 32-sample 小数据上正式
stage 1 train entry 可以持续 100 optimizer steps；该 run 本身没有 dev loader、split manifest、
1k pilot、DDP 或 resume，因此不满足 P1 晋级条件。小数据 DDP/resume 另见下节。

## PAMI201 Small Root DDP and Resume

首次用该 root 启动两卡 DDP 时，训练在首个 batch 前失败。原因不是样本数本身，而是
`longcat/` 只有一个覆盖全部 32 条样本的 payload group；store 的 DDP shuffle 按 payload group
分 rank，结果 rank 0 得到 32 条、rank 1 得到 0 条。anydataset 为保持等长 DDP step 在 rank 0
丢弃 32 个尾 batch，随后 staged loader 报 `scheduled loader 'asr' produced no batches`。
`drop_distributed_tail=false` 只会把这一不等长条件改为显式错误，不能形成可运行的 DDP 输入。

为继续验证而不改 third-party batching，另把同一 32 条 `base/` 与 `longcat/` store 重写到
`145:/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_small_32_ddp4_20260727`，写入时设置
`max_shard_samples=8`。两个 store 均为 32 条且 `.ready` 存在；LongCat payload groups 为
`(0,8)`、`(8,16)`、`(16,24)`、`(24,32)`，可均匀分到两个 rank。

正式 train 入口新增可选 `train.ckpt_path`，默认空，并传给
`trainer.fit(..., ckpt_path=config.train.ckpt_path)`。在 GPU 1/2、
`NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1`、batch size 1、checkpoint 每 step 保存的条件下，
两卡 DDP 完成 2 optimizer steps；随后从 `last.ckpt` 恢复，日志明确记录
`Restoring states from the checkpoint path` 与 `Restored all states`，并继续到 step 3。

| Run | Result | Artifact |
| --- | --- | --- |
| DDP 2-step | exit 0；两个 rank 注册；`max_steps=2` reached；生成 `step-00000001.ckpt`、`step-00000002.ckpt` 与 `last.ckpt` | `145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-formal-stage1-ddp-resume-ddp4/011-staged-joint-train/stage_1/stage_1-rvq-8l/metrics.json` |
| resume step 2 -> 3 | exit 0；恢复 optimizer/loop states；`max_steps=3` reached；生成 `step-00000003.ckpt` | `145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-formal-stage1-ddp-resume-ddp4/ddp_resume_step3.log` |

resume run 新增一步的 total/token/RVQ loss 分别为 `16.206238`、`6.756507`、`9.044817`；
`last.ckpt` 大小为 `2342056597` bytes。该 smoke 证明多 payload group 的 pami201 小数据 root
可以完成正式 stage 1 两卡 DDP checkpoint/resume 执行契约，但该 small-root smoke
本身不替代正式 split 的 distributed partition、dev CE、pilot 或长跑验收。后续
1k pilot 的 split/DDP/resume 验收见下节。

## PAMI201 1k Pilot Split, DDP and Resume

32-sample 闭环通过后，2026-07-27 在 pami201 固化 1k pilot 数据根：
`145:/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_pilot_1000_ddp20_20260727`。该 root
包含 `1000` 条样本和 `20` 个 payload group，每组 `50` 条；LongCat tensor 均为
rank-2 integer、`4` 个 codebook，frame 数范围为 `14..43`。root 内无 symlink，
总大小为 `504357694` bytes。

正式 split manifest 为
`145:/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_pilot_1000_ddp20_20260727/manifests/stage1_pilot.json`，
SHA256 为 `ef3f1009bfb1f1c885ec0cfbab6d06875a7678f164865bba89e9011e8a0dc728`。
每个 50-sample payload group 固定分为 train/dev/test `40/5/5`。连续 5 个 epoch 的两卡分片
验证均得到 train rank counts `[400,400]`、dev `[50,50]`、test `[50,50]`，
未出现单 payload group root 曾触发的 rank 空数据问题。

用该 manifest 运行正式 stage 1 train entry，两卡 DDP 2-step 与从 step 2 恢复到
step 3 均 exit `0`：
`145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-stage1-pilot1k-ddp20-resume`。
resume 日志明确包含 `Restoring states...` 和 `Restored all states`；恢复后新增 step 的
total/token/RVQ loss 分别为 `16.085545`、`6.741939`、`9.137355`。

该 pilot 证据固化了 split 指纹，并验证了 payload-group-aware 两卡 partition、
checkpoint 和 resume 执行契约。两步训练和一步恢复的 finite loss 不支持质量或
收敛结论；dev 指标链路的独立验收见下节。

## PAMI201 1k Pilot Dev Validation Smoke

正式 staged train 新增默认关闭的 teacher-forcing validation：从现有 stage speech loader
复制 task weights 和 speech config，只把复制后的 `DatasetConfig.split_label` 改成 dev；训练
loader 保持 train split。token CE 按有效 token 数加权，RVQ 总 CE、逐 codebook CE/top-1
按有效 target frame 数加权；Lightning 在 epoch 结束时同步两卡加权和与计数。sanity 与
optimizer-step interval 结果分别写入 `metrics.json.validation`。

本地验收为 240 tests OK / 1 CUDA skip、basedpyright 0 errors、ruff、compileall、job shell
syntax 和 `git diff --check` 全部通过。远端使用
`experiment=014_stage1_pilot_validation_smoke`、GPU 1/2、batch size 1、完整 dev sanity、
`max_steps=1`、step 1 interval validation；输出为：

`145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-stage1-pilot1k-validation-v1/014-longcat-stable-stage1/pilot-validation-smoke/stage_1-rvq-8l/metrics.json`

两卡均注册成功，进程正常退出，日志未出现 traceback、OOM 或非 finite 指标。manifest 已独立
验证 dev rank counts `[50,50]`；本次两个 validation pass 均遍历该完整 dev split。LongCat
semantic codebook 由 token objective 监督，因此下表 `codebook_0..2` 是 3 个 acoustic RVQ
codebook；dev 有效 target frame 总数为 `4264`。

| Metric | Sanity step 0 | Interval step 1 |
| --- | ---: | ---: |
| token CE | `9.883604` | `9.842715` |
| RVQ CE | `9.151793` | `9.151495` |
| codebook 0 CE | `9.150218` | `9.147419` |
| codebook 1 CE | `9.144267` | `9.144913` |
| codebook 2 CE | `9.160889` | `9.162157` |
| codebook 0 top-1 | `0.000234522` (`1/4264`) | `0.000234522` (`1/4264`) |
| codebook 1 top-1 | `0` | `0` |
| codebook 2 top-1 | `0` | `0` |

每个 acoustic codebook size 为 `8100`，均匀随机 top-1 基线约为 `0.000123457`。
codebook 0 的单次命中不足以支持高于随机的统计结论，另两个 codebook 为 0；因此尚未满足
P1 的“多数 codebook top-1 高于随机”门槛。一步后 token CE 下降约 `0.414%`，RVQ CE
下降约 `0.003%`，也不构成学习或收敛结论。本次只接受 validation split、DDP 加权、指标命名
与 JSON reporting 闭环；下一步先跑 100-step canary，并在 step 50/100 对同一 dev split 复验。

## PAMI201 1k Pilot 100-Step Canary

`experiment=014_stage1_pilot_canary` 使用同一数据根/manifest、GPU 1/2、batch size 1，
从头训练 100 optimizer steps；完整 dev validation 在 step 0/50/100 运行，checkpoint 在
step 50/100 归档并保留 `last.ckpt`。SSH 端到端用时约 133 秒，Lightning fit loop 约 59 秒；
日志无 traceback、OOM 或非 finite 指标。metrics 位于：

`145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-stage1-pilot1k-canary-v1/014-longcat-stable-stage1/pilot-canary/stage_1-rvq-8l/metrics.json`

| Metric | Step 0 | Step 50 | Step 100 |
| --- | ---: | ---: | ---: |
| token CE | `9.883604` | `8.798153` | `8.467776` |
| RVQ CE | `9.151793` | `9.101947` | `9.047078` |
| codebook 0 CE | `9.150218` | `9.078161` | `8.991410` |
| codebook 1 CE | `9.144267` | `9.113574` | `9.083248` |
| codebook 2 CE | `9.160889` | `9.114109` | `9.066575` |
| codebook 0 top-1 | `0.000234522` | `0.000469043` | `0.033302065` (`142/4264`) |
| codebook 1 top-1 | `0` | `0.007035647` | `0.028142588` (`120/4264`) |
| codebook 2 top-1 | `0` | `0` | `0.029315198` (`125/4264`) |

step 100 相对初始 token CE 下降 `14.325%`，RVQ CE 下降 `1.144%`。三个 acoustic codebook
top-1 均已明显高于 `1/8100` 随机基线，满足 P1 的 majority-top1 方向门槛；但 RVQ CE 尚未达到
5% 下降目标 `8.694203`。训练窗口摘要的 total/token/RVQ `last_to_first` 分别为
`0.950962`、`0.916650`、`0.998476`，同样说明当前主要改善来自 token path，acoustic bridge
仍需继续验证。

归档 `step-00000050.ckpt`、`step-00000100.ckpt` 和 `last.ckpt` 均存在，每个约
`2342056597` bytes；TensorBoard event 位于同一输出根的
`tensorboard/014-longcat-stable-stage1/pilot-canary/stage_1-rvq-8l/version_0/`。

## PAMI201 1k Pilot Resume to Step 500

`experiment=014_stage1_pilot_resume_500` 从上述 `last.ckpt` 恢复，日志明确记录
`Restoring states`、两卡注册与 `Restored all states`；checkpoint metadata 的起始
`global_step=100`、Lightning 版本为 `2.6.1`。该 run 在 GPU 1/2 上继续到 step 500，
每 100 steps 遍历完整 dev split。输出位于：

`145:/home/zhuyin/local-runs/s2s-small-data-20260727/train-stage1-pilot1k-resume500-v1/014-longcat-stable-stage1/pilot-resume-500/stage_1-rvq-8l/metrics.json`

| Metric | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| token CE | `8.467776` | `8.273833` | `8.129531` | `8.035213` | `7.946233` |
| RVQ CE | `9.047078` | `8.981592` | `8.941418` | `8.924411` | `8.904623` |
| codebook 0 CE | `8.991410` | `8.874780` | `8.806050` | `8.777058` | `8.749586` |
| codebook 1 CE | `9.083248` | `9.054724` | `9.030916` | `9.021158` | `9.002337` |
| codebook 2 CE | `9.066575` | `9.015273` | `8.987288` | `8.975017` | `8.961946` |
| codebook 0 top-1 | `0.033302065` | `0.038461540` | `0.038461540` | `0.038227018` | `0.038227018` (`163/4264`) |
| codebook 1 top-1 | `0.028142588` | `0.031191370` | `0.031191370` | `0.031191370` | `0.030956848` (`132/4264`) |
| codebook 2 top-1 | `0.029315198` | `0.036585364` | `0.036585364` | `0.036350843` | `0.036350843` (`155/4264`) |

step 500 相对 step 0 的 token CE 下降 `19.602%`，RVQ CE 下降 `2.701%`。所有 interval
指标均为 finite，三个 codebook top-1 持续高于 `1/8100` 随机基线；但 RVQ CE 仍高于
5% 目标 `8.694203`。step 300 到 500 的 RVQ CE 从 `8.941418` 降到 `8.904623`；只按该
局部斜率线性外推，还需约 `1144` steps，此外推只用于设置下一段预算，不作为收敛结论。

run 正常以 `max_steps=500` 退出，`last.ckpt` metadata 为 `global_step=500`；归档
`step-00000200/300/400/500.ckpt` 与 `last.ckpt` 均存在，每个约 `2.2G`，TensorBoard event
约 `500K`。运行期间同机外部评测在所有 GPU 上各占约 `4.8G` 和约 25% utilization，
因此本次 `2:34` fit 时长不用于效率比较。下一步从 step 500 恢复到 step 2000，每 250 steps
复验完整 dev；达到 `8.694203` 后先做 decode/companion manifest，不机械跑满 5k。

## 判定

P0 在 debug-migrated copy 上通过：代码迁移的本地/远端 targeted tests 通过；复旦 `145`
上的 TTS 和 S2ST P0 wrapper 均 exit `0`；训练 metrics、`generation.json` 和 waveform decode
均为 finite；toy smoke 也完成 1 step 并写出 finite metrics。

边界仍然明确：本轮没有修改原 pami202 stable data root；正式根 parquet/fingerprint 审计与
无 duration parse/map-style dataloader probe 已完成，且 probe 没有出现真实音频静默计为
0 秒的问题。1000-sample native token/RVQ 分布审计和 800/100/100 pilot split candidate
也已完成，它们作为原始 debug candidate 记录保留。pami201 上另行固化的
1k/20-group split manifest 已完成两卡 rank 均衡、DDP 2-step、resume 和完整 dev validation
指标链路验收。恢复到 step 500 后三个 acoustic codebook top-1 仍明显高于随机，dev RVQ CE
相对初始下降 `2.701%`，但尚未满足 5% 门槛，因此不允许直接晋级到 native stable P1 长跑。

2026-07-27 的 pami201 32-sample root 进一步证明：当 202 超时时，使用 145 本地运行时、
145 本地 HF/LongCat cache 和 pami201 数据根可以完成 stage 1 TTS/S2ST 2-step smoke，并且
正式 `scripts/train.py` stage 1 可以完成 100-step 小数据 canary；重分片 root 还完成两卡
DDP 2-step 与 resume 到 step 3。后续 1k pilot 已完成正式 split manifest、两卡
rank 均衡、DDP/resume、完整 dev 指标链路，并从 100-step canary 恢复到 step 500。
top-1 已显示学习信号，RVQ CE 单调下降但尚未达到计划门槛；当前仍缺 step-2000 gate、
最多 5k-step pilot、decode 和长跑收敛验收，不支持质量或收敛结论。
