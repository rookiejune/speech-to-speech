# 014 LongCat Stable Stage 1

## 目标

用 LongCat native semantic token 建立稳定的 stage 1 训练闭环，先验证语义到声学的可学习
bridge，再决定是否引入 speech BPE。首轮目标是让后续 S2ST joint train 有一个可恢复、可比较、
可止损的 acoustic initialization，而不是一次性把 tokenizer、Qwen 解冻、S2ST 多任务和 BPE
全部叠在一起。

本计划把 stage 1 定义为 frozen-Qwen / codec-derived 条件下的 LongCat acoustic bridge：

- 输入使用 LongCat encoder 产出的 native semantic token。
- 输出使用 LongCat decoder 支持的 RVQ acoustic codebook。
- Qwen backbone 默认冻结或绕过；只训练 speech interface 与 acoustic decoder。
- 所有实验先在 stable native path 上跑通，再把 BPE 作为并行 tokenizer ablation。

## 判断：Stage 1 先不用 BPE

我倾向 stage 1 主线不用 BPE，原因是：

- native token 是 LongCat codec 的原生语义帧接口，能直接定位 bridge / decoder / data 的问题。
- 旧 100k BPE 曾把 1000 条临时数据的 source/target 全压成单个 audio token，说明当前 BPE
  artifact 不能作为训练入口，至少要重训和分布验收后才能用。
- BPE 会同时改变序列长度、对齐、loss 口径、generation contract 和 checkpoint metadata；
  如果 stage 1 一开始就加入 BPE，失败时很难区分是 tokenizer、数据还是 acoustic bridge 问题。
- stage 1 的核心风险是稳定收敛和可 decode，不是极限吞吐。BPE 更适合作为 native stable
  后的效率实验。

因此主线是：native stable 先跑到可恢复、held-out CE 改善、decode finite；BPE 只做 shadow
ablation，达标后才允许进入同等预算比较。

## 路线

### P0：资源与数据合同

1. 在 Python 3.9 / PyTorch 2.8 环境验收官方 LongCat checkpoint：
   `from_pretrained()`、短音频 encode/decode、一步 acoustic forward/backward/optimizer step。
2. 固化 stable data root，不读取 `/tmp` 临时数据；记录 store manifest fingerprint、row index、
   split manifest、LongCat view fingerprint。
3. 对 train/dev/test 统计 native semantic token、source/target frame、RVQ codebook、文本长度、
   parse/span error。parse/span error 必须为 0。
4. 建立 800/100/100 pilot split；完整 train split 未就绪前，只允许 pilot，不下泛化结论。

### P1：Native Stable Stage 1

1. 训练对象：`semantic_audio_embedding`、`semantic_audio_adapter`、必要的 acoustic decoder 参数；
   Qwen frozen 或 bypassed。
2. 冻结边界：LongCat/RVQ 结构性 unused 参数、decoder token embedding、最后 codebook embedding
   与 projection 先保持冻结，除非 P0 证明这些参数确实参与当前路径。
3. 训练预算：
   - 32-sample fixed overfit：100 steps。
   - 1k pilot：最多 5k optimizer steps。
   - stable canary：>=50k train pair 后最多 50k steps。
   - 预算是上限，不是必须跑满的目标。若连续两个 validation interval 的局部改善显著放缓，
     或模型 top-1 与无条件 target 众数基线等价，先检查 condition 使用、参数/optimizer state dtype
     与实际参数更新比例；诊断完成前不继续追加预算。
4. 晋级条件：
   - train/dev RVQ CE finite 且 dev CE 相对初始下降至少 5%。
   - 多数 codebook top-1 accuracy 不仅高于均匀随机基线，也应明确优于各 codebook 的无条件
     众数基线；否则不能据此判断模型利用了条件。
   - teacher-forced waveform decode 100% finite。
   - `last.ckpt`、归档 checkpoint、resume 后 metrics 连续。
   - 两卡 DDP 2-step + resume 通过，rank 间 task schedule 和 loss key 一致。

### P2：BPE Shadow Ablation

只有 native P1 达标后才启动 BPE shadow：

1. 在完整 train split 上重训 speech BPE；禁止复用旧 100k artifact。
2. 限制最大 token span，记录 compression ratio、单 token collapse rate、native-to-BPE span 分布。
3. held-out 验收门槛：
   - 单 token collapse rate 接近 0，任何异常样本写入 manifest。
   - BPE span 与音频时长、native token 数保持单调相关。
   - encode/decode 与 loss mask 不破坏 batch generation contract。
4. 用同一 fixed sample、同一 pilot split、同一 stage 1 参数组跑 native vs BPE A/B：
   - 初始化时间。
   - peak memory。
   - step time / loader wait。
   - dev CE 与 top-1 accuracy。
   - cached generation throughput。
5. 只有 BPE 同等或更好地保持 dev CE / decode finite，并带来明确吞吐或显存收益，才进入 stage 2。

## Stage 2 前置条件

进入 Qwen partial / full joint 前必须同时满足：

- Native stable stage 1 有可恢复 checkpoint 和 companion manifest。
- 至少一个 50k-step canary 表明 dev CE 未发散。
- 两卡 DDP + resume + validation generation 已通过。
- 如果启用 BPE，BPE A/B 必须独立通过，不允许中途替换 native checkpoint tokenizer。
- conclusion 只记录已验证结论；未验证假设留在 todo。

## 产出

- `results/014-longcat-stable-stage1.md`：记录 P0/P1/P2 每个 run 的命令、commit、checkpoint、
  config、数据 fingerprint、metrics 和 stop/go 决策。
- stage 1 companion manifest：绑定 LongCat checkpoint、tokenizer/native-or-BPE、semantic vocab、
  RVQ codebook、decoder config、state-dict prefix whitelist 和 split fingerprint。
- 如果 BPE 失败，保留失败分布和原因，不进入 stage 2。
