# LID Fine-tuning Results (mean ± std across seeds)

| Tokenizer | Init | Seeds | Macro-F1 | Accuracy | HI-F1 | EN-F1 |
|---|---|---|---|---|---|---|
| custom | pretrained | 7 | 0.9458 ± 0.0003 | 0.9552 ± 0.0002 | 0.9683 ± 0.0002 | 0.9233 ± 0.0004 |
| gpt2 | pretrained | 7 | 0.9122 ± 0.0003 | 0.9289 ± 0.0003 | 0.9505 ± 0.0002 | 0.8740 ± 0.0005 |
| custom | scratch | 7 | 0.9360 ± 0.0006 | 0.9469 ± 0.0005 | 0.9625 ± 0.0004 | 0.9095 ± 0.0008 |
| gpt2 | scratch | 7 | 0.8903 ± 0.0006 | 0.9125 ± 0.0005 | 0.9396 ± 0.0004 | 0.8410 ± 0.0009 |

## Custom tokenizer advantage (macro-F1)

- Pretrained: **+0.0336**
- Scratch: **+0.0456**
