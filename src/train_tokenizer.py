"""
Train a byte-level BPE tokenizer on your own corpus instead of reusing GPT-2's.
This is a real decision point worth understanding: a tokenizer trained on
code-mixed Hindi-English text will produce more meaningful subword units for
this domain than an English-only tokenizer, which tends to fragment Hindi
(especially Devanagari-script) text into many small/inefficient tokens.

Usage:
    python train_tokenizer.py --input ../data/processed/corpus.txt --vocab_size 16000
"""

import argparse
import os
from tokenizers import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/processed/train_clean.txt",help="Path to cleaned text corpus (.txt) — defaults to the train split")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="../tokenizer")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[args.input],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    )

    tokenizer.save_model(args.out_dir)
    print(f"Tokenizer saved to {args.out_dir} (vocab_size={args.vocab_size})")

    # quick sanity check — encode a code-mixed sentence and show the tokenization
    sample = "Yaar aaj bohot busy hai, but I'll definitely call you shaam ko."
    encoded = tokenizer.encode(sample)
    print(f"\nSample: {sample}")
    print(f"Tokens ({len(encoded.tokens)}): {encoded.tokens}")


if __name__ == "__main__":
    main()
