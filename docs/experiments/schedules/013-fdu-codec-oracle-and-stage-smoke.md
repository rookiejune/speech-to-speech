# 013 FDU codec oracle and staged smoke

目标是在复旦共享环境上提交一组明确、可重复的 smoke jobs，覆盖 codec oracle 的三种设置与 stage_0 到 stage_4。

## Codec oracle

提交 jobs/013/01_oracle_flow_codec.sh、jobs/013/02_oracle_flow_random.sh、jobs/013/03_oracle_rvq_codec.sh。三项分别选择：

- fdu_oracle_flow_codec_smoke：Flow objective，codec initialization。
- fdu_oracle_flow_random_smoke：Flow objective，matched random initialization。
- fdu_oracle_rvq_codec_smoke：RVQ objective，codec initialization。

三项都使用 LongCat native codec、32 条 prepared samples、4 秒 hard truncate、2 optimizer steps、LBA enabled，并把 output 写到 013-fdu-codec-oracle-smoke/...。

## Stage smoke

提交 jobs/013/10_stage_0_smoke.sh 到 jobs/013/14_stage_4_smoke.sh。stage_0 没有 formal train loader，因此 fdu_stage_0_smoke 走 scripts/overfit.py fixed-sample entry 验证 stage_0 参数阶段；stage_1 到 stage_4 走正式 scripts/train.py staged joint entry。

五项都使用 Qwen3-0.6B、LongCat native tokens、RVQ acoustic decoder、2 optimizer steps 和 013-fdu-stage-smoke/... output 目录。stage_1 到 stage_4 使用两卡 staged train DDP 策略；stage_0 使用单卡 fixed-sample overfit。speech data root 可用 SPEECH_TO_SPEECH_STAGE_DATA_ROOT 覆盖；oracle data root 可用 SPEECH_TO_SPEECH_ORACLE_DATA_ROOT 覆盖。
