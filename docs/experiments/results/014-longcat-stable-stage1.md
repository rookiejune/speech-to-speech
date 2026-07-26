# 014 LongCat Stable Stage 1 P0 Acceptance

对应 [014 schedule](../schedules/014-longcat-stable-stage1.md)。本文记录 2026-07-25
在 FDU `145` 上完成的 P0 验收和当前边界。状态：**P0 在 debug-migrated copy 上通过**；
正式 stable data root 的 parquet 指纹审计和无 duration parse/map-style dataloader probe
已补齐；split manifest、native token/RVQ 分布和训练验收仍未完成，不能视为正式数据根已进入
stage 1 长跑。

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

## 判定

P0 在 debug-migrated copy 上通过：代码迁移的本地/远端 targeted tests 通过；复旦 `145`
上的 TTS 和 S2ST P0 wrapper 均 exit `0`；训练 metrics、`generation.json` 和 waveform decode
均为 finite；toy smoke 也完成 1 step 并写出 finite metrics。

边界仍然明确：本轮没有修改正式 stable data root；正式根 parquet/fingerprint 审计与
无 duration parse/map-style dataloader probe 已完成，且 probe 没有出现真实音频静默计为
0 秒的问题。但正式 split manifest、native token/RVQ 分布表仍未生成；也没有执行正式根
training step、32-sample 100-step overfit、1k pilot、两卡 DDP 2-step 或 resume。因此当前
P0 只能作为 debug copy acceptance 加正式根抽样 probe，不允许直接晋级到 native stable
P1 长跑。
