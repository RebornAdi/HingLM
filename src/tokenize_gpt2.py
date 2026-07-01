import os
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer

PROCESSED_DIR = r"C:\Users\Aditya Atul Deshmukh\Desktop\HingLM\data\processed"
TRAIN_CLEAN = os.path.join(PROCESSED_DIR, "train_clean.txt")
VAL_CLEAN = os.path.join(PROCESSED_DIR, "val_clean.txt")

EOT_TOKEN = "<|endoftext|>"


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    eot_id = tokenizer.encode(EOT_TOKEN)[0]

    for split_name, src_path in [("train", TRAIN_CLEAN), ("val", VAL_CLEAN)]:
        if not os.path.exists(src_path):
            print(f"NOT FOUND: {src_path}")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        ids = []
        for line in tqdm(lines, desc=f"tokenizing {split_name} with GPT-2"):
            encoded = tokenizer.encode(line.strip())
            ids.extend(encoded)
            ids.append(eot_id)

        arr = np.array(ids, dtype=np.uint16)
        out_path = os.path.join(PROCESSED_DIR, f"{split_name}_gpt2.bin")
        arr.tofile(out_path)
        print(f"{split_name}: {len(arr):,} tokens -> {out_path} ({arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()