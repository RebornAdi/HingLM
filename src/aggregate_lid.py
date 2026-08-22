"""
Read results/lid_results.txt and produce mean ± std macro-F1 (and per-class F1)
for each (tokenizer, init) configuration across seeds. Saves a markdown table.
"""

import os
import re
from collections import defaultdict

import numpy as np

RESULTS_DIR = "../results"
IN_PATH = os.path.join(RESULTS_DIR, "lid_results.txt")
OUT_PATH = os.path.join(RESULTS_DIR, "lid_results_summary.md")


def parse_line(line):
    d = {}
    for key in ["tokenizer", "init", "seed", "macro_f1", "acc", "HI_f1", "EN_f1"]:
        m = re.search(rf"{key}=([^\s]+)", line)
        if m:
            d[key] = m.group(1)
    return d


def main():
    groups = defaultdict(lambda: defaultdict(list))
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            if "macro_f1" not in line:
                continue
            d = parse_line(line)
            if "seed" not in d:  # skip old unlabeled runs
                continue
            key = (d["tokenizer"], d["init"])
            for metric in ["macro_f1", "acc", "HI_f1", "EN_f1"]:
                groups[key][metric].append(float(d[metric]))

    lines = ["# LID Fine-tuning Results (mean ± std across seeds)\n"]
    lines.append("| Tokenizer | Init | Seeds | Macro-F1 | Accuracy | HI-F1 | EN-F1 |")
    lines.append("|---|---|---|---|---|---|---|")

    order = [("custom", "pretrained"), ("gpt2", "pretrained"),
             ("custom", "scratch"), ("gpt2", "scratch")]
    for key in order:
        if key not in groups:
            continue
        g = groups[key]
        n = len(g["macro_f1"])
        def ms(metric):
            arr = np.array(g[metric])
            return f"{arr.mean():.4f} ± {arr.std():.4f}"
        lines.append(f"| {key[0]} | {key[1]} | {n} | {ms('macro_f1')} | "
                     f"{ms('acc')} | {ms('HI_f1')} | {ms('EN_f1')} |")

    # advantage summary
    def mean_of(key, metric):
        return np.array(groups[key][metric]).mean() if key in groups else float("nan")

    lines.append("\n## Custom tokenizer advantage (macro-F1)\n")
    pre_adv = mean_of(("custom", "pretrained"), "macro_f1") - mean_of(("gpt2", "pretrained"), "macro_f1")
    scr_adv = mean_of(("custom", "scratch"), "macro_f1") - mean_of(("gpt2", "scratch"), "macro_f1")
    lines.append(f"- Pretrained: **+{pre_adv:.4f}**")
    lines.append(f"- Scratch: **+{scr_adv:.4f}**")

    text = "\n".join(lines) + "\n"
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()