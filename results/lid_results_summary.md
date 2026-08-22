# LID Fine-tuning Results (mean ± std across seeds)

| Tokenizer | Init | Seeds | Macro-F1 | Accuracy | HI-F1 | EN-F1 |
|---|---|---|---|---|---|---|
| custom | pretrained | 3 | 0.9457 ± 0.0003 | 0.9550 ± 0.0002 | 0.9683 ± 0.0002 | 0.9231 ± 0.0004 |
| gpt2 | pretrained | 3 | 0.9123 ± 0.0004 | 0.9289 ± 0.0003 | 0.9505 ± 0.0002 | 0.8740 ± 0.0006 |
| custom | scratch | 3 | 0.9359 ± 0.0008 | 0.9468 ± 0.0007 | 0.9624 ± 0.0005 | 0.9094 ± 0.0010 |
| gpt2 | scratch | 3 | 0.8902 ± 0.0004 | 0.9123 ± 0.0003 | 0.9395 ± 0.0002 | 0.8409 ± 0.0007 |

## Custom tokenizer advantage (macro-F1)

- Pretrained: **+0.0334**
- Scratch: **+0.0457**
