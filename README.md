# HingLM — Results

## Overview

This document reports all experimental results for HingLM: a from-scratch
pretraining study investigating whether a BPE tokenizer trained on
Hindi-English code-mixed text achieves better tokenization efficiency than
GPT-2's English-trained tokenizer, and whether this translates to downstream
language modeling performance.

All experiments were run on a single NVIDIA GeForce RTX 3050 Laptop GPU
(4GB VRAM) using a 10.62M parameter decoder-only transformer pretrained
from random initialization on a subset of L3Cube-HingCorpus
(CC BY-NC-SA 4.0).

---

## 1. Dataset

| Split | Raw lines | After cleaning | Tokens (custom tok) | Tokens (GPT-2 tok) |
|---|---|---|---|---|
| Train | 7,542,482 | 5,147,555 | 128,780,032 | 173,007,839 |
| Val | 390,367 | 338,846 | 8,509,011 | 11,199,557 |

**Cleaning drop breakdown (train):**
- Short lines (<10 chars): 86,163 (1.1%)
- Mostly-symbol garbage: 90,686 (1.2%)
- Exact duplicates: 2,218,078 (29.4%)
- Lines kept: 5,147,555 (68.3%)

The high duplicate rate (29.4%) is consistent with Twitter data — retweets,
common short phrases, and bot-generated content account for a significant
fraction of any large-scale social media scrape. This is a known property
of the source corpus and is disclosed as a limitation.

---

## 2. Tokenizer Comparison

### 2.1 Qualitative Example

**Sentence:** *"Yaar aaj bohot busy hai, but I'll definitely call you shaam ko."*

| Tokenizer | Tokens | Count |
|---|---|---|
| Custom (HingLM) | `Yaar` `Ġaaj` `Ġbohot` `Ġbusy` `Ġhai` `,` `Ġbut` `ĠI` `'ll` `Ġdefinitely` `Ġcall` `Ġyou` `Ġshaam` `Ġko` `.` | **15** |
| GPT-2 | `Y` `a` `ar` ` a` `aj` ` b` `oh` `ot` ` busy` ` ha` `i` `,` ` but` ` I` ` will` ` definitely` ` call` ` you` ` sh` `a` `am` ` ko` `.` | **23** |

### 2.2 Aggregate Quantitative Result

Evaluated on **5,000 randomly sampled held-out sentences** from the
validation split.

| Metric | Custom Tokenizer | GPT-2 Tokenizer |
|---|---|---|
| Vocab size | 16,000 | 50,257 |
| Avg tokens/sentence | 24.21 | 32.22 |
| Chars per token | 3.92 | 2.95 |
| Total tokens (5k sample) | 121,050 | 161,100 |
| **Token count reduction** | **—** | **24.9% more tokens** |

Per-sentence reduction: mean=22.3%, median=27.1%, std=18.4%.

### 2.3 Reduction by Code-Mixing Density

| Group | N | Custom tok/sent | GPT-2 tok/sent | Mean reduction | Median reduction |
|---|---|---|---|---|---|
| Low (<5% Hindi words) | 586 | 17.11 | 17.86 | 6.2% | 7.1% |
| Mid (5–20% Hindi words) | 1,333 | 27.22 | 30.74 | 11.7% | 11.1% |
| High (>20% Hindi words) | 3,081 | 24.27 | 35.60 | **29.9%** | **32.2%** |

The reduction scales monotonically with code-mixing density, confirming
the result is driven by code-mixed vocabulary handling, not a generic
vocabulary-size artifact.

### 2.4 Corpus-Level Token Count

| | Custom Tokenizer | GPT-2 Tokenizer | Reduction |
|---|---|---|---|
| Train tokens | 128,780,032 | 173,007,839 | **25.6%** |
| Val tokens | 8,509,011 | 11,199,557 | 24.0% |

---

## 3. Language Model Training

### 3.1 Setup

| Parameter | Value |
|---|---|
| Architecture | Decoder-only transformer (from scratch) |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| MLP | SwiGLU |
| Optimizer | AdamW (fused) |
| LR schedule | Cosine decay with linear warmup (500 steps) |
| Precision | bfloat16 |
| Context length | 256 tokens |
| Effective batch size | 128 (micro_batch=8 × grad_accum=16) |
| Max steps | 10,000 |
| Hardware | RTX 3050 Laptop GPU (4GB VRAM) |

### 3.2 Results

| Metric | Custom Tokenizer Run | GPT-2 Tokenizer Run |
|---|---|---|
| Model parameters | 10.62M | ~24M |
| Vocab size | 16,000 | 50,257 |
| Initial val loss | 9.76 | ~10.8 |
| Final val loss | ~5.0 | ~3.9 |
| Final perplexity | ~148 | ~49 |

### 3.3 Confounds in Downstream Comparison

The GPT-2 run's lower perplexity is confounded by two factors:

**1. Model size.** ~24M vs 10.62M parameters — the larger embedding layer
alone (~19.3M params) gives substantially more capacity.

**2. Cross-vocabulary perplexity incomparability.** Perplexity measures
surprise per token. With 25.6% more tokens representing the same text,
the prediction tasks are fundamentally different. Bits-per-character (BPC)
would be required for a fair comparison.

The downstream numbers should not be interpreted as evidence that GPT-2's
tokenizer produces a better LM. The primary confound-free result is the
tokenizer compression analysis.

---

## 4. Summary of Findings

1. **24.9% token count reduction** on 5,000 held-out sentences vs GPT-2's tokenizer.
2. Effect scales with code-mixing density: **6.2% → 11.7% → 29.9%** (low → mid → high mixing).
3. **25.6% fewer training tokens** at corpus scale — direct efficiency gain for training and inference.
4. Both models trained stably from scratch on consumer hardware (4GB VRAM).
5. Direct perplexity comparison is confounded by model size and cross-vocabulary incomparability.

---

## 5. Limitations

- 29.4% exact-duplicate rate in raw Twitter data
- Code-mixing density estimated via heuristic word list, not ground-truth language-ID
- 10.62M parameters is far below production LLM scale
- Downstream comparison confounds (Section 3.3)
- Single seed per run, no significance testing