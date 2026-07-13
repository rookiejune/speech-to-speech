# 006 真实 callback 与 REPA smoke

## 目标

- 在 121 A100 上用真实 WMT19 TTS、LongCat、Qwen3-0.6B 和 WavLM-base 验证训练闭环。
- 启用 `repa_weight=0.1`，验证在线 waveform decode、WavLM layer 9 teacher、DiT layer 4 projection 和 backward。
- 每步触发 Outputs、FlowMatching、Grad、GradNorm、Sample、TextRetention callback；fit start 触发 StageSwitcher。

## 验收

- 标准 LongCat store 返回 `[frame, codebook]` 整数 Tensor。
- 2 个真实训练 step 正常结束，semantic、flow matching、REPA 和总 loss 均有限。
- TensorBoard 包含音频、文本、flow histogram、objective、分项梯度和全局梯度范数。
- WavLM 参数冻结并保持 eval。
