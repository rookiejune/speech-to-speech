# 013 FDU codec oracle and staged smoke

目标是在复旦共享环境提交一组明确、可重复的 smoke jobs，覆盖三类 codec oracle 设置，以及 RVQ 与 acoustic=none 两条 stage_0 到 stage_4 路径。

## Codec oracle

| Job | Experiment | Objective | Initialization | Entry |
| --- | --- | --- | --- | --- |
| `jobs/013/01_oracle_flow_codec.sh` | `fdu_oracle_flow_codec_smoke` | Flow | codec | `scripts/codec_oracle.py` |
| `jobs/013/02_oracle_flow_random.sh` | `fdu_oracle_flow_random_smoke` | Flow | random | `scripts/codec_oracle.py` |
| `jobs/013/03_oracle_rvq_codec.sh` | `fdu_oracle_rvq_codec_smoke` | RVQ | codec | `scripts/codec_oracle.py` |

三项都使用 LongCat native codec、32 条 prepared samples、4 秒 hard truncate、2 optimizer steps、LBA enabled，并写入 `013-fdu-codec-oracle-smoke/...`。

## Stage Smoke

| Job range | Experiments | Acoustic path | Entry |
| --- | --- | --- | --- |
| `jobs/013/10_stage_0_smoke.sh` | `fdu_stage_0_smoke` | RVQ | `scripts/overfit.py` |
| `jobs/013/11_stage_1_smoke.sh` – `14_stage_4_smoke.sh` | `fdu_stage_1_smoke` – `fdu_stage_4_smoke` | RVQ | `scripts/train.py` |
| `jobs/013/20_stage_0_acoustic_none_smoke.sh` | `fdu_stage_0_acoustic_none_smoke` | token-only | `scripts/overfit.py` |
| `jobs/013/21_stage_1_acoustic_none_smoke.sh` – `24_stage_4_acoustic_none_smoke.sh` | `fdu_stage_1_acoustic_none_smoke` – `fdu_stage_4_acoustic_none_smoke` | token-only | `scripts/train.py` |

Stage_0 没有 formal train loader，因此使用 fixed-sample overfit entry 验证参数阶段；stage_1 到 stage_4 使用正式 staged joint train entry。全部 stage smoke 使用 Qwen3-0.6B、LongCat native tokens、2 optimizer steps 和 `013-fdu-stage-smoke/...` output 目录。stage_1 到 stage_4 默认两卡 DDP，stage_0 默认单卡。

## Overrides

FDU 物理环境、Qwen checkpoint 默认路径和可选数据 root 由 workspace 侧
`workspace/jobs/fudan/speech_to_speech_env.sh` 统一维护；本 repo 的 `jobs/013/*.sh`
只选择 experiment、entry、device 和可追加的 Hydra overrides：

- `SPEECH_TO_SPEECH_ORACLE_DATA_ROOT` 覆盖 oracle prepared data root。
- `SPEECH_TO_SPEECH_STAGE_DATA_ROOT` 覆盖 stage smoke prepared data root。
- `SPEECH_TO_SPEECH_STAGE_QWEN_ROOT` 覆盖 Qwen checkpoint 路径。
- 每个 wrapper 末尾保留 `"$@"`，提交时可继续追加 Hydra overrides。
