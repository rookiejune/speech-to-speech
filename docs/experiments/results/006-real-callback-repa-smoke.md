# 006 真实 callback 与 REPA smoke 结果

## 环境与数据修复

2026-07-13 在 121 的 A100 GPU 1 上执行。共享盘清理了约 13 GB 可再生 debug cache、临时 checkpoint 和旧 smoke 输出，未删除正式训练输出、模型或数据集。

原 WMT19 LongCat store 是旧 dict contract，`SampleLogger.on_fit_start` 明确失败。使用 `workspace/jobs/prepare_wmt19_tts_longcat.sh` 重新物化 1000 个样本，耗时 59.25 秒。首条 target codes 验证为 `torch.int64 [36, 4]`，取值范围 15 到 8154。

WavLM-base 从 `hf-mirror.com` 下载到共享 Hugging Face cache；正式 smoke 使用离线模式。Qwen3-0.6B 使用 121 本机 snapshot，避免 NAS 权重读取影响测试。

## 训练结果

配置：真实 sample 0、TTS、batch size 1、bf16 mixed、2 step、REPA weight 0.1、WavLM-base layer 9、DiT layer 4。Trainer 正常以 `max_steps=2` 结束。

| objective | step 1 | step 2 |
| --- | ---: | ---: |
| total | 17.9879 | 13.6225 |
| semantic | 15.2812 | 10.9375 |
| flow matching | 2.6082 | 2.5869 |
| REPA | 0.9846 | 0.9814 |

模型包含 991M trainable 参数和 94.4M frozen WavLM 参数；teacher 的 233 个 module 在训练开始时保持 eval。

## Callback 证据

TensorBoard 每个训练 step 都记录了 total/objective loss、flow time、flow/REPA 分项梯度、梯度比值与 cosine、全局 grad norm。`SampleLogger` 写入 `sample/0` 音频；`TextRetentionLogger` 写入基线及每步 NLL、delta 和生成文本；`StageSwitcher` 在 fit start 成功绑定 fixed datamodule。

远端产物位于 `/mnt/pami202/zhuyin/dynamic/debug/speech-to-speech/callback-repa-smoke/wavlm-repa`，完整 stdout/stderr 位于同级 `wavlm-repa.log`。

## 结论与欠账

当前真实 TTS 的在线 WavLM REPA 与全部 callback 可共同运行并完成 optimizer step。两步 smoke 只证明闭环和数值有限，不证明 REPA 改善收敛或音频质量；仍需相同预算的 flow 与 flow + REPA 长程对照。

启动阶段仍缺少结构化 start/done/elapsed 日志；当前依赖远端日志文件和 TensorBoard 定位重型 callback。静态检查另有 `model/acoustic/dit.py` 的 dtype 推断错误，运行时未触发。
