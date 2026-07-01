# Tokenizer Reduction by Code-Mixing Density

Sentences were bucketed by an estimated code-mixing ratio (fraction of words matching a common Romanized-Hindi word list — a heuristic, not a ground-truth language-ID label).

| Group | N | Custom avg tok/sent | GPT-2 avg tok/sent | Mean reduction | Median reduction |
|---|---|---|---|---|---|
| low_code_mixing | 586 | 17.11 | 17.86 | 6.2% | 7.1% |
| mid_code_mixing | 1,333 | 27.22 | 30.74 | 11.7% | 11.1% |
| high_code_mixing | 3,081 | 24.27 | 35.60 | 29.9% | 32.2% |

## Interpretation

If reduction increases monotonically from low to high code-mixing groups, this confirms the aggregate result is genuinely driven by code-mixed content rather than a generic vocabulary-size artifact, and explains the variance observed in the unsegmented aggregate comparison.
