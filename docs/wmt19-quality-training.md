# WMT19 Quality Training

## 长跑配置

当前 WMT19 quality 训练使用 `acoustic_loss_weight=0.01`。LoRA 对照实验默认使用
`batch_size=4`，Muon 全量更新实验先使用 `batch_size=1` 和 `learning_rate=1e-5`。

BPE 默认可使用 100k LongCat artifact。配置入口是 `bpe=longcat_100k`，已固定到
`experiment=wmt19_quality_100k_muon` 和
`experiment=wmt19_quality_100k_full_adamw`；artifact 和压缩统计见
[longcat-bpe.md](longcat-bpe.md)。

## 存档间隔

长时间训练的 checkpoint 间隔控制为每 10000 次更新存档一次，避免 500-step 级别的频繁存档在
NAS 上造成明显 I/O 等待。

这个约定只会在下次启动或重启训练时生效。已经启动的训练进程不会因为本地配置或脚本变更自动更新
checkpoint 间隔。

## 2026-06-28 后续卡位

`wmt19-quality-muon-lora-bs4` 使用 GPU1 从头跑 10000 step，用于和 AdamW+LoRA 的
`batch_size=4` 公平对齐。

GPU2 用于从 `wmt19-quality-adamw-lora-bs4-10k/last.ckpt` 续跑到 20000 step，观察当前最好
LoRA 候选是否继续下降或开始过拟合。

GPU3 用于新跑 `wmt19-quality-adamw-full-bs1`，即 no-LoRA 全量更新 + AdamW + `batch_size=1`，
和已经完成的 no-LoRA 全量更新 + Muon 对照。
