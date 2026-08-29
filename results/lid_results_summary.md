# LID Fine-tuning Results (mean ± std across seeds)

| Tokenizer | Init | Seeds | Macro-F1 | Accuracy | HI-F1 | EN-F1 |
|---|---|---|---|---|---|---|
| custom | pretrained | 10 | 0.9458 ± 0.0002 | 0.9552 ± 0.0002 | 0.9683 ± 0.0001 | 0.9233 ± 0.0003 |
| gpt2 | pretrained | 10 | 0.9122 ± 0.0004 | 0.9288 ± 0.0004 | 0.9504 ± 0.0003 | 0.8739 ± 0.0006 |
| custom | scratch | 10 | 0.9360 ± 0.0006 | 0.9469 ± 0.0006 | 0.9624 ± 0.0004 | 0.9095 ± 0.0009 |
| gpt2 | scratch | 10 | 0.8904 ± 0.0007 | 0.9125 ± 0.0006 | 0.9396 ± 0.0005 | 0.8411 ± 0.0010 |

## Custom tokenizer advantage (macro-F1)

- Pretrained: **+0.0336**
- Scratch: **+0.0456**
