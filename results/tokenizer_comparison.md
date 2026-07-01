# Tokenizer Comparison: Custom Hinglish BPE vs GPT-2

Evaluated on **5,000 held-out sentences** from the validation split (never seen during tokenizer training).

| Metric | Custom Tokenizer | GPT-2 Tokenizer |
|---|---|---|
| Avg tokens/sentence | 24.21 | 32.22 |
| Chars per token (compression) | 3.92 | 2.95 |
| Total tokens (full sample) | 121,072 | 161,112 |

**Token count reduction: 24.9%** (mean per-sentence reduction: 22.3%, median: 27.1%)

## Interpretation

GPT-2's tokenizer was trained on English web text, so Hindi/Romanized-Hindi subwords rarely appeared frequently enough to earn BPE merges, causing it to fall back to fragmenting these words into individual characters or short byte sequences. The custom tokenizer, trained directly on code-mixed text, learned merges for common Hindi words and code-mixed patterns, producing meaningfully shorter, more semantically coherent token sequences for the same text.
