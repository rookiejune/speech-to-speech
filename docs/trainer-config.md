# Trainer Config

`trainer` 顶层只表达 Lightning Trainer 自身参数，例如运行名、设备、strategy、
resume checkpoint、日志频率和 UI 开关。训练过程中挂载的 Lightning callbacks 通过
`trainer.callbacks` 指定。

默认 trainer preset 会从 `configs/callback/` 组装 callback：

```yaml
defaults:
  - /callback@callbacks.checkpoint: checkpoint
  - /callback@callbacks.learning_rate_monitor: learning_rate_monitor
  - /callback@callbacks.sample: sample
  - /callback@callbacks.generation: generation
  - _self_
```

因此 callback 可复用配置放在 `configs/callback/*.yaml`，具体 experiment 或 job 用
Hydra override 调整：

```bash
python scripts/train.py experiment=wmt19_quality_100k_muon \
  trainer.callbacks.generation.every_n_steps=1000 \
  trainer.callbacks.generation.acoustic_sampler=diagonal_bpe
```

常用开关：

```yaml
trainer:
  callbacks:
    checkpoint:
      enabled: true
      every_n_steps: 10000
      save_top_k: 2
    sample:
      enabled: true
      every_n_steps: 0
      samples_per_task: 1
      max_audio_samples: 320000
    generation:
      enabled: true
      every_n_steps: 5000
      flow_steps: 32
      chunk_size: null
      left_context_chunks: null
      acoustic_sampler: serial
      max_audio_samples: 320000
```

`trainer.callbacks.generation.enabled=false` 表示不挂 generation logger；
`trainer.callbacks.generation.every_n_steps=null` 也会跳过 generation logger。
默认 `acoustic_sampler=serial` 走 full-sequence acoustic flow，`chunk_size` 和
`left_context_chunks` 保持 `null`。只有切到 chunk/window sampler 时才需要设置
`chunk_size`；`causal_window` 下 `left_context_chunks=null` 表示使用全部左侧 chunks，
显式整数才会限制窗口。`trainer.callbacks.sample` 记录数据/task sample，不触发模型生成。
