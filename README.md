# HingLM — Results

## Overview

This document reports all experimental results for HingLM: a from-scratch
pretraining study investigating whether a BPE tokenizer trained on
Hindi-English code-mixed text achieves better tokenization efficiency than
GPT-2's English-trained tokenizer, and whether this translates to downstream
performance on a real code-mixed task.

All experiments were run using ~10.6M–24M parameter decoder-only transformers pretrained
from random initialization on a subset of L3Cube-HingCorpus (CC BY-NC-SA 4.0),
and fine-tuned on L3Cube-HingLID for downstream language identification.

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

The high duplicate rate (29.4%) is consistent with Twitter data and is
disclosed as a data limitation.

---

## 2. Tokenizer Comparison (Intrinsic)

### 2.1 Qualitative Example

**Sentence:** *"Yaar aaj bohot busy hai, but I'll definitely call you shaam ko."*

| Tokenizer | Count |
|---|---|
| Custom (HingLM) | **15** |
| GPT-2 | **23** |

GPT-2 fragments Hindi words (Yaar→3, aaj→2, bohot→3, shaam→3); the custom
tokenizer keeps each as a single token.

### 2.2 Aggregate Result (5,000 held-out sentences)

| Metric | Custom | GPT-2 |
|---|---|---|
| Vocab size | 16,000 | 50,257 |
| Avg tokens/sentence | 24.21 | 32.22 |
| Chars per token | 3.92 | 2.95 |

**Token count reduction: 24.9%** (per-sentence: mean=22.3%, median=27.1%, std=18.4%).
Equivalently, the custom tokenizer fits ~33% more characters into a fixed
context window (3.92 vs 2.95 chars/token).

### 2.3 Reduction by Code-Mixing Density

| Group | N | Custom tok/sent | GPT-2 tok/sent | Mean reduction |
|---|---|---|---|---|
| Low (<5% Hindi) | 586 | 17.11 | 17.86 | 6.2% |
| Mid (5–20%) | 1,333 | 27.22 | 30.74 | 11.7% |
| High (>20%) | 3,081 | 24.27 | 35.60 | 29.9% |

Reduction scales monotonically with code-mixing density, confirming the
effect is driven by code-mixed vocabulary handling.

### 2.4 Corpus-Level Token Count

| | Custom | GPT-2 | Reduction |
|---|---|---|---|
| Train tokens | 128,780,032 | 173,007,839 | 25.6% |
| Val tokens | 8,509,011 | 11,199,557 | 24.0% |

---

## 3. Pretraining

### 3.1 Setup

| Parameter | Value |
|---|---|
| Architecture | Decoder-only transformer (from scratch) |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| MLP | SwiGLU |
| Optimizer | AdamW (fused) |
| LR schedule | Cosine decay, 500-step warmup |
| Precision | bfloat16 |
| Context length | 256 |
| Effective batch | 128 (micro 8 × accum 16) |
| Max steps | 10,000 |
| Hardware | RTX 3050 Laptop (4GB VRAM) |

### 3.2 Results

| Metric | Custom Tokenizer | GPT-2 Tokenizer |
|---|---|---|
| Parameters | 10.62M | ~24M |
| Vocab size | 16,000 | 50,257 |
| Initial val loss | 9.76 | ~10.8 |
| Final val loss | ~5.0 | ~3.9 |
| Final perplexity | ~148 | ~49 |

### 3.3 Confounds in the Perplexity Comparison

The GPT-2 run's lower perplexity is **not** evidence of a better tokenizer,
due to two confounds:

1. **Model size.** ~24M vs 10.62M params; the larger embedding (~19.3M)
   gives substantially more capacity.
2. **Cross-vocabulary incomparability.** Perplexity is per-token; with 25.6%
   more tokens for the same text, the prediction tasks differ. Bits-per-
   character would be needed for a fair intrinsic comparison.

The clean, confound-free comparison is the downstream evaluation (Section 4).

---

## 4. Downstream Evaluation: Language Identification

### 4.1 Task and Setup

Both pretrained models were fine-tuned on token-level language identification
(L3Cube-HingLID) to test whether the tokenizer's efficiency carries any cost
or benefit on a real task.

| Item | Value |
|---|---|
| Task | Token-level LID (per-word HI vs EN) |
| Train / Val / Test | 31,756 / 6,279 / 6,420 sentences |
| Label distribution | ~72% HI, ~28% EN |
| Fine-tuning | 3 epochs, batch 16, LR 3e-5, AdamW, 10% warmup |
| Metric | Macro-F1 (primary) + per-class F1 |
| Seeds | 3 (1337, 42, 2024), mean ± std |

A 2×2 design ({custom, GPT-2} tokenizer × {pretrained, scratch} init)
separates the tokenizer effect from the pretraining effect. All runs use
identical hyperparameters; only tokenizer and init vary. The classification
model uses the same causal decoder backbone (leftward context only) — a mild
tagging disadvantage shared by both variants, so the comparison stays fair.

### 4.2 Results — Macro-F1 (mean ± std, 3 seeds)

| | Custom tokenizer | GPT-2 tokenizer | Custom advantage |
|---|---|---|---|
| **Pretrained** | **0.9457 ± 0.0003** | 0.9123 ± 0.0004 | **+3.34** |
| **Scratch** | 0.9359 ± 0.0008 | 0.8902 ± 0.0004 | **+4.57** |
| **Pretraining benefit** | +0.98 | +2.21 | |

Per-class F1 (pretrained):

| Tokenizer | HI-F1 | EN-F1 |
|---|---|---|
| Custom | 0.9683 ± 0.0002 | 0.9231 ± 0.0004 |
| GPT-2 | 0.9505 ± 0.0002 | 0.8740 ± 0.0006 |

### 4.3 Findings

1. **The custom tokenizer produces significantly better downstream LID** in
   both conditions: +3.34 (pretrained), +4.57 (scratch) macro-F1. Std <0.001
   is 50–100× smaller than the effect — robust, not noise.
2. **Larger advantage without pretraining** (+4.57 vs +3.34): the tokenizer
   matters most when the model can't lean on pretrained representations.
3. **Tokenizer choice outweighs pretraining** for this task (tokenizer effect
   3.3–4.6 vs pretraining effect 1.0–2.2).
4. **Gap concentrates in the harder minority (EN) class** (~4.9 EN-F1 gap vs
   ~1.8 HI-F1 gap), mechanistically consistent with the tokenizer hypothesis.

Unlike the perplexity comparison, this is not confounded: within each init
condition the models differ only in tokenizer, and macro-F1 is directly
comparable. **The efficiency gain comes not at an accuracy cost, but with a
measurable accuracy improvement.**

---

## 5. Summary of Findings

1. **24.9% token reduction** vs GPT-2 on held-out text (33% more chars/token).
2. Effect **scales with code-mixing density** (6.2% → 29.9%).
3. **25.6% fewer training tokens** at corpus scale.
4. Both models **trained stably from scratch** on 4GB VRAM.
5. **Downstream: +3.34 to +4.57 macro-F1** on LID, robust across 3 seeds —
   efficiency gain comes with an accuracy gain, not a cost.
6. Intrinsic perplexity comparison is confounded and is not used as evidence.

---

## 6. Limitations

- 29.4% exact-duplicate rate in raw Twitter data.
- Code-mixing density (Section 2.3) uses a heuristic word list, not ground-truth LID.
- ~10.6M–24M params — far below production scale.
- Causal (decoder-only) backbone is suboptimal for tagging vs bidirectional encoders; used for both variants to keep the comparison fair.
- Single downstream task (LID); sentiment or other tasks would broaden the claim.
- Perplexity comparison confounded by model size and cross-vocabulary metric issues (Section 3.3).