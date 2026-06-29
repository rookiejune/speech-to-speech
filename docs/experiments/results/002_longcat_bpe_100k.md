# LongCat BPE 100k Result

## Artifact

- actual vocab size: 100000
- artifact dir: `/nfs/yin.zhu/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192`

## Evaluation

| Metric | Value |
| --- | ---: |
| num_sequences | 1061750 |
| original_tokens | 126876136 |
| encoded_tokens | 47744622 |
| mean_original_length | 119.49718483635507 |
| mean_encoded_length | 44.96785684012244 |
| compression_ratio | 0.37630892227045754 |
| compression_gain | 0.6236910777295425 |
| compression_factor | 2.6573911507771495 |

## Conclusion

100k LongCat BPE 将 WMT19 source+target semantic ids 的平均序列长度压到原始长度的
约 `37.63%`，压缩约 `2.66x`。
