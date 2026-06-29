# LongCat BPE 100k

## 目标

记录 WMT19 source+target speech semantic ids 上的 100k LongCat BPE artifact，
并把它作为后续 quality 训练的大词表候选配置。

## 配置

- codec: LongCat semantic ids
- vocab size: 100000
- min frequency: 0
- max token length: none
- codebook sizes: 8192
- corpus: WMT19 train split source 和 target 两侧完整 semantic ids

## 输出

产物写入 `$BPE_CACHE_DIR/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192`。
在 hz 环境中，对应 `BPE_CACHE_DIR=/nfs/yin.zhu/bpe`。
