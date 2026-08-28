import argparse
import os
import random

RAW_DIR ="Raw data path"
TRAIN_FULL = os.path.join(RAW_DIR, "hingcorpus_train_full.txt")
VAL_FULL = os.path.join(RAW_DIR, "hingcorpus_val_full.txt")
TRAIN_SUBSET = os.path.join(RAW_DIR, "hingcorpus_train_subset.txt")
VAL_SUBSET = os.path.join(RAW_DIR, "hingcorpus_val_subset.txt")


def _slice_file(src_path, dst_path, target_mb, seed):
    """Randomly sample lines from src_path until ~target_mb is reached, write to dst_path."""
    if not os.path.exists(src_path):
        print(f"NOT FOUND: {src_path}")
        return

    target_bytes = target_mb * 1024 * 1024
    print(f"Counting lines in {src_path} (slow on large files, be patient)...")
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        n_lines = sum(1 for _ in f)
    print(f"  {n_lines:,} lines, {os.path.getsize(src_path) / 1e6:.1f} MB total")

    avg_line_bytes = os.path.getsize(src_path) / max(1, n_lines)
    n_needed = int(target_bytes / avg_line_bytes * 1.05)
    n_needed = min(n_needed, n_lines)

    random.seed(seed)
    keep_indices = set(random.sample(range(n_lines), n_needed))

    written_bytes = 0
    with open(src_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in keep_indices:
                fout.write(line)
                written_bytes += len(line.encode("utf-8"))

    print(f"  -> wrote {n_needed:,} lines, {written_bytes / 1e6:.1f} MB to {dst_path}")


def slice_subset(target_mb):
    """
    Slice train to target_mb, and val to a proportionally smaller size
    (val doesn't need to be huge — fast eval loops matter more than
    a large val set at this project's scale).
    """
    _slice_file(TRAIN_FULL, TRAIN_SUBSET, target_mb, seed=1337)
    val_target_mb = max(10, int(target_mb * 0.05))  # ~5% of train size, min 10MB
    _slice_file(VAL_FULL, VAL_SUBSET, val_target_mb, seed=1338)


def _inspect_file(path, label):
    if not os.path.exists(path):
        print(f"{path} not found — run `slice` first.")
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"\n=== {label}: {path} ===")
    print(f"Total lines: {len(lines):,}")
    print(f"Total size: {os.path.getsize(path) / 1e6:.1f} MB")

    lengths = [len(l) for l in lines]
    print(f"Avg line length (chars): {sum(lengths)/len(lengths):.1f}")
    print(f"Min/Max line length: {min(lengths)} / {max(lengths)}")

    short = sum(1 for l in lines if len(l) < 10)
    print(f"Lines under 10 chars (likely junk): {short:,} ({short/len(lines)*100:.1f}%)")

    has_url = sum(1 for l in lines if "http" in l.lower())
    print(f"Lines containing URLs (need cleaning): {has_url:,} ({has_url/len(lines)*100:.1f}%)")

    print(f"\n--- 15 random samples from {label} — READ THESE ---\n")
    random.seed(42)
    for line in random.sample(lines, min(15, len(lines))):
        print(f"  {line}")


def inspect():
    _inspect_file(TRAIN_SUBSET, "TRAIN")
    _inspect_file(VAL_SUBSET, "VAL")

    print("\n--- Manual checklist (apply to both train and val samples above) ---")
    print("[ ] Text is genuinely code-mixed (not pure Hindi or pure English)")
    print("[ ] Mostly readable, not dominated by garbage/spam/bot text")
    print("[ ] No obviously severe content issues in this sample")
    print("[ ] Encoding looks correct (no mangled characters)")
    print("If any of these fail badly, reconsider cleaning rules before proceeding.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_slice = sub.add_parser("slice")
    p_slice.add_argument("--target_mb", type=int, default=700)
    sub.add_parser("inspect")
    args = parser.parse_args()

    if args.command == "slice":
        slice_subset(args.target_mb)
    elif args.command == "inspect":
        inspect()

if __name__ == "__main__":
    main()
