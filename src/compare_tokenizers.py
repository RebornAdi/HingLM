import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

VAL_CLEAN = "../data/processed/val_clean.txt"
TOKENIZER_DIR = "../tokenizer"
RESULTS_DIR = "../results"


def load_sample_sentences(n_samples, seed=123):
    with open(VAL_CLEAN, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(seed)
    return random.sample(lines, min(n_samples, len(lines)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    args = parser.parse_args()

    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Tokenizer

    custom_tok = ByteLevelBPETokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        os.path.join(TOKENIZER_DIR, "merges.txt"),
    )
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")

    sentences = load_sample_sentences(args.n_samples)
    print(f"Comparing on {len(sentences):,} held-out sentences from val_clean.txt")

    custom_token_counts, gpt2_token_counts, char_counts = [], [], []
    for s in tqdm(sentences, desc="comparing tokenizers"):
        custom_n = len(custom_tok.encode(s).ids)
        gpt2_n = len(gpt2_tok.encode(s))
        custom_token_counts.append(custom_n)
        gpt2_token_counts.append(gpt2_n)
        char_counts.append(len(s))

    custom_arr = np.array(custom_token_counts)
    gpt2_arr = np.array(gpt2_token_counts)
    char_arr = np.array(char_counts)

    custom_avg = custom_arr.mean()
    gpt2_avg = gpt2_arr.mean()
    reduction_pct = (1 - custom_avg / gpt2_avg) * 100

    # chars-per-token = compression efficiency (higher = more compressed)
    custom_chars_per_tok = char_arr.sum() / custom_arr.sum()
    gpt2_chars_per_tok = char_arr.sum() / gpt2_arr.sum()

    # per-sentence reduction, for a distribution view (not just one aggregate number)
    per_sentence_reduction = (1 - custom_arr / np.maximum(gpt2_arr, 1)) * 100

    results = {
        "n_sentences": len(sentences),
        "custom_tokenizer": {
            "avg_tokens_per_sentence": float(custom_avg),
            "total_tokens": int(custom_arr.sum()),
            "chars_per_token": float(custom_chars_per_tok),
        },
        "gpt2_tokenizer": {
            "avg_tokens_per_sentence": float(gpt2_avg),
            "total_tokens": int(gpt2_arr.sum()),
            "chars_per_token": float(gpt2_chars_per_tok),
        },
        "token_count_reduction_pct": float(reduction_pct),
        "per_sentence_reduction_pct": {
            "mean": float(per_sentence_reduction.mean()),
            "median": float(np.median(per_sentence_reduction)),
            "std": float(per_sentence_reduction.std()),
        },
    }

    print("\n=== RESULTS ===")
    print(f"Sentences compared: {results['n_sentences']:,}")
    print(f"Custom tokenizer: avg {custom_avg:.2f} tokens/sentence, "
          f"{custom_chars_per_tok:.2f} chars/token")
    print(f"GPT-2 tokenizer:  avg {gpt2_avg:.2f} tokens/sentence, "
          f"{gpt2_chars_per_tok:.2f} chars/token")
    print(f"\nToken count reduction (custom vs GPT-2): {reduction_pct:.1f}%")
    print(f"Per-sentence reduction: mean={per_sentence_reduction.mean():.1f}%, "
          f"median={np.median(per_sentence_reduction):.1f}%, "
          f"std={per_sentence_reduction.std():.1f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "tokenizer_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results -> {json_path}")

    md_path = os.path.join(RESULTS_DIR, "tokenizer_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tokenizer Comparison: Custom Hinglish BPE vs GPT-2\n\n")
        f.write(f"Evaluated on **{results['n_sentences']:,} held-out sentences** "
                f"from the validation split (never seen during tokenizer training).\n\n")
        f.write("| Metric | Custom Tokenizer | GPT-2 Tokenizer |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Avg tokens/sentence | {custom_avg:.2f} | {gpt2_avg:.2f} |\n")
        f.write(f"| Chars per token (compression) | {custom_chars_per_tok:.2f} | {gpt2_chars_per_tok:.2f} |\n")
        f.write(f"| Total tokens (full sample) | {custom_arr.sum():,} | {gpt2_arr.sum():,} |\n\n")
        f.write(f"**Token count reduction: {reduction_pct:.1f}%** "
                f"(mean per-sentence reduction: {per_sentence_reduction.mean():.1f}%, "
                f"median: {np.median(per_sentence_reduction):.1f}%)\n\n")
        f.write("## Interpretation\n\n")
        f.write("GPT-2's tokenizer was trained on English web text, so Hindi/Romanized-Hindi "
                "subwords rarely appeared frequently enough to earn BPE merges, causing it to "
                "fall back to fragmenting these words into individual characters or short byte "
                "sequences. The custom tokenizer, trained directly on code-mixed text, learned "
                "merges for common Hindi words and code-mixed patterns, producing meaningfully "
                "shorter, more semantically coherent token sequences for the same text.\n")
    print(f"Saved writeup-ready summary -> {md_path}")


if __name__ == "__main__":
    main()