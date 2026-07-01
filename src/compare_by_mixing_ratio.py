import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

VAL_CLEAN = "../data/processed/val_clean.txt"
TOKENIZER_DIR = "../tokenizer"
RESULTS_DIR = "../results"

HINDI_MARKER_WORDS = set("""
hai hain ho hoga hogi the thi tha kya kyun kaise kab kahan kaun kuch
koi sab sabhi ye yeh wo woh is us in un ka ki ke ko se mein me par
pe aur ya nahi nahin mat na bhi tum tumhe tumhara tumhari aap aapka
aapki hum humara humari mera meri mujhe tujhe use unhe iska iski uska
uski jo jab tab agar lekin par kyunki isliye phir abhi kabhi yahan
wahan idhar udhar acha accha theek thik bohot bahut bahot zyada kam
matlab samajh pata chal raha rahi rahe gaya gayi gaye liye diya diya
karo karna kiya kar rha rhi rhe bhai yaar dost ji haan han nai nhi
kuchh wala wali wale hota hoti hote kab tak abhi tak ab toh to
""".split())


def estimate_mixing_ratio(sentence):
    words = sentence.lower().split()
    if not words:
        return 0.0
    hindi_count = sum(1 for w in words if w.strip(".,!?\"'") in HINDI_MARKER_WORDS)
    return hindi_count / len(words)


def run_comparison(sentences, custom_tok, gpt2_tok, label):
    if not sentences:
        return None
    custom_counts, gpt2_counts = [], []
    for s in sentences:
        custom_counts.append(len(custom_tok.encode(s).ids))
        gpt2_counts.append(len(gpt2_tok.encode(s)))
    custom_arr, gpt2_arr = np.array(custom_counts), np.array(gpt2_counts)
    reduction = (1 - custom_arr / np.maximum(gpt2_arr, 1)) * 100
    result = {
        "label": label,
        "n_sentences": len(sentences),
        "custom_avg_tokens": float(custom_arr.mean()),
        "gpt2_avg_tokens": float(gpt2_arr.mean()),
        "mean_reduction_pct": float(reduction.mean()),
        "median_reduction_pct": float(np.median(reduction)),
    }
    print(f"\n[{label}] n={result['n_sentences']:,} | "
          f"custom={result['custom_avg_tokens']:.2f} tok/sent | "
          f"gpt2={result['gpt2_avg_tokens']:.2f} tok/sent | "
          f"reduction: mean={result['mean_reduction_pct']:.1f}% "
          f"median={result['median_reduction_pct']:.1f}%")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--low_threshold", type=float, default=0.05,
                         help="mixing ratio below this = 'low code-mixing' group")
    parser.add_argument("--high_threshold", type=float, default=0.20,
                         help="mixing ratio above this = 'high code-mixing' group")
    args = parser.parse_args()

    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Tokenizer

    custom_tok = ByteLevelBPETokenizer(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        os.path.join(TOKENIZER_DIR, "merges.txt"),
    )
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")

    with open(VAL_CLEAN, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(123)
    sentences = random.sample(lines, min(args.n_samples, len(lines)))

    print(f"Estimating code-mixing ratio for {len(sentences):,} sentences...")
    low_mix, mid_mix, high_mix = [], [], []
    for s in tqdm(sentences, desc="classifying"):
        ratio = estimate_mixing_ratio(s)
        if ratio < args.low_threshold:
            low_mix.append(s)
        elif ratio > args.high_threshold:
            high_mix.append(s)
        else:
            mid_mix.append(s)

    print(f"\nGroup sizes: low-mix={len(low_mix):,}  mid-mix={len(mid_mix):,}  high-mix={len(high_mix):,}")

    print("\nComparing tokenizers within each group...")
    results = []
    for sents, label in [(low_mix, "low_code_mixing"),
                          (mid_mix, "mid_code_mixing"),
                          (high_mix, "high_code_mixing")]:
        r = run_comparison(sents, custom_tok, gpt2_tok, label)
        if r:
            results.append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "mixing_ratio_breakdown.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {json_path}")

    md_path = os.path.join(RESULTS_DIR, "mixing_ratio_breakdown.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tokenizer Reduction by Code-Mixing Density\n\n")
        f.write("Sentences were bucketed by an estimated code-mixing ratio "
                "(fraction of words matching a common Romanized-Hindi word "
                "list — a heuristic, not a ground-truth language-ID label).\n\n")
        f.write("| Group | N | Custom avg tok/sent | GPT-2 avg tok/sent | Mean reduction | Median reduction |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['n_sentences']:,} | {r['custom_avg_tokens']:.2f} | "
                     f"{r['gpt2_avg_tokens']:.2f} | {r['mean_reduction_pct']:.1f}% | "
                     f"{r['median_reduction_pct']:.1f}% |\n")
        f.write("\n## Interpretation\n\n")
        f.write("If reduction increases monotonically from low to high code-mixing groups, "
                "this confirms the aggregate result is genuinely driven by code-mixed content "
                "rather than a generic vocabulary-size artifact, and explains the variance "
                "observed in the unsegmented aggregate comparison.\n")
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()