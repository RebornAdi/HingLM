import argparse
import os
import re

import numpy as np
from tqdm import tqdm

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"

TRAIN_SUBSET = os.path.join(RAW_DIR, "hingcorpus_train_subset.txt")
VAL_SUBSET = os.path.join(RAW_DIR, "hingcorpus_val_subset.txt")
TRAIN_CLEAN = os.path.join(PROCESSED_DIR, "train_clean.txt")
VAL_CLEAN = os.path.join(PROCESSED_DIR, "val_clean.txt")

# minimum fraction of a line's non-space characters that must be alphabetic
# (catches scrape garbage like "526  1600 , (R) (R)  5  -" while still
# allowing normal punctuation/emoji-light Hinglish text through)
MIN_ALPHA_RATIO = 0.5


def _is_mostly_garbage(line):
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return True
    alpha_count = sum(1 for c in chars if c.isalpha())
    return (alpha_count / len(chars)) < MIN_ALPHA_RATIO


def _clean_file(src_path, dst_path):
    if not os.path.exists(src_path):
        print(f"NOT FOUND: {src_path}")
        return

    seen_lines = set()
    n_in, n_out, n_garbage, n_dup, n_short = 0, 0, 0, 0, 0

    with open(src_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"cleaning {os.path.basename(src_path)}"):
            n_in += 1
            line = line.strip()
            line = re.sub(r"http\S+|www\.\S+", "", line)   # strip URLs
            line = re.sub(r"\s+", " ", line).strip()        # normalize whitespace

            if len(line) < 10:
                n_short += 1
                continue
            if _is_mostly_garbage(line):
                n_garbage += 1
                continue
            if line in seen_lines:
                n_dup += 1
                continue

            seen_lines.add(line)
            fout.write(line + "\n")
            n_out += 1

    print(f"  {os.path.basename(src_path)}: in={n_in:,} out={n_out:,} "
          f"(dropped: short={n_short:,}, garbage={n_garbage:,}, dup={n_dup:,})")
    print(f"  -> {dst_path}")


def clean_corpus():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    _clean_file(TRAIN_SUBSET, TRAIN_CLEAN)
    _clean_file(VAL_SUBSET, VAL_CLEAN)


def tokenize_corpus(tokenizer_dir, output_dir):
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    os.makedirs(output_dir, exist_ok=True)

    for split_name, src_path in [("train", TRAIN_CLEAN), ("val", VAL_CLEAN)]:
        if not os.path.exists(src_path):
            print(f"NOT FOUND: {src_path} — run `clean` first.")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        ids = []
        for line in tqdm(lines, desc=f"tokenizing {split_name}"):
            encoded = tokenizer.encode(line.strip())
            ids.extend(encoded.ids)
            ids.append(eot_id)

        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(output_dir, f"{split_name}.bin")
        arr.tofile(out_path)
        print(f"{split_name}: {len(arr):,} tokens -> {out_path} ({arr.nbytes / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("clean")

    p_tok = sub.add_parser("tokenize")
    p_tok.add_argument("--tokenizer_dir", default="../tokenizer")
    p_tok.add_argument("--output_dir", default="../data/processed")

    args = parser.parse_args()

    if args.command == "clean":
        clean_corpus()
    elif args.command == "tokenize":
        tokenize_corpus(args.tokenizer_dir, args.output_dir)


if __name__ == "__main__":
    main()