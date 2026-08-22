"""
Fine-tune the pretrained (or randomly-initialized) models on the L3Cube-HingLID
token-classification task, for both tokenizers, and report macro-F1 + per-class F1.

Runs one configuration per invocation. The four paper conditions × N seeds:
    python finetune_lid.py --tokenizer custom --init pretrained --seed 1337
    python finetune_lid.py --tokenizer custom --init scratch   --seed 42
    python finetune_lid.py --tokenizer gpt2   --init pretrained --seed 2024
    ... etc

All runs use identical hyperparameters (below) — only tokenizer, init, and seed
change, so the comparison is clean.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import ModelConfig, ModelConfigGPT2Tok
from model import GPTForTokenClassification
from lid_data import load_split, IGNORE_INDEX, ID2LABEL

# ---- fixed hyperparameters (identical across all runs) ----
LID_DIR = "../data/mixed"
CKPT_DIR = "checkpoints"
RESULTS_DIR = "../results"
BLOCK_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1

# which pretrained checkpoint to load for each tokenizer
PRETRAINED_CKPT = {
    "custom": os.path.join(CKPT_DIR, "custom_tokenizer_step10000.pt"),
    "gpt2": os.path.join(CKPT_DIR, "gpt2_tokenizer_step10000.pt"),
}


class LIDDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id=0):
    """Pad input_ids and labels to the longest sequence in the batch."""
    maxlen = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        ids = ex["input_ids"]
        labs = ex["labels"]
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)   # padded positions ignored in loss
        attn.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def load_tokenizer(kind):
    if kind == "custom":
        from tokenizers import ByteLevelBPETokenizer
        return ByteLevelBPETokenizer("../tokenizer/vocab.json", "../tokenizer/merges.txt")
    else:
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained("gpt2")


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute macro-F1 and per-class precision/recall/F1 on a split."""
    model.eval()
    stats = {0: [0, 0, 0], 1: [0, 0, 0]}  # per class: [tp, fp, fn]
    for input_ids, labels, attn in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, _ = model(input_ids)
        preds = logits.argmax(-1)
        mask = labels != IGNORE_INDEX
        p = preds[mask].cpu().numpy()
        y = labels[mask].cpu().numpy()
        for cls in (0, 1):
            stats[cls][0] += int(((p == cls) & (y == cls)).sum())  # tp
            stats[cls][1] += int(((p == cls) & (y != cls)).sum())  # fp
            stats[cls][2] += int(((p != cls) & (y == cls)).sum())  # fn

    per_class = {}
    f1s = []
    for cls in (0, 1):
        tp, fp, fn = stats[cls]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[ID2LABEL[cls]] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)

    total_tp = sum(stats[c][0] for c in (0, 1))
    total = total_tp + sum(stats[c][1] for c in (0, 1))
    acc = total_tp / total if total else 0.0

    macro_f1 = sum(f1s) / len(f1s)
    return {"macro_f1": macro_f1, "accuracy": acc, "per_class": per_class}


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["custom", "gpt2"], required=True)
    parser.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = ModelConfigGPT2Tok() if args.tokenizer == "gpt2" else ModelConfig()
    tokenizer = load_tokenizer(args.tokenizer)
    kind = args.tokenizer

    print(f"[seed={seed}] Loading LID data with {kind} tokenizer...")
    train_ex = load_split(os.path.join(LID_DIR, "train.txt"), tokenizer, kind, BLOCK_SIZE)
    val_ex = load_split(os.path.join(LID_DIR, "validation.txt"), tokenizer, kind, BLOCK_SIZE)
    test_ex = load_split(os.path.join(LID_DIR, "test.txt"), tokenizer, kind, BLOCK_SIZE)
    print(f"  train={len(train_ex):,}  val={len(val_ex):,}  test={len(test_ex):,}")

    train_loader = DataLoader(LIDDataset(train_ex), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(LIDDataset(val_ex), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(LIDDataset(test_ex), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = GPTForTokenClassification(mcfg, num_labels=2).to(device)

    if args.init == "pretrained":
        ckpt_path = PRETRAINED_CKPT[kind]
        print(f"Loading pretrained backbone from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_backbone_from_gpt(ckpt["model"])
    else:
        print("Random initialization (no pretraining)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else _null()

    step = 0
    for epoch in range(EPOCHS):
        model.train()
        for input_ids, labels, attn in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val macro-F1={val_metrics['macro_f1']:.4f} acc={val_metrics['accuracy']:.4f}")

    test_metrics = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    print(f"tokenizer={args.tokenizer}  init={args.init}  seed={seed}")
    print(f"macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"accuracy: {test_metrics['accuracy']:.4f}")
    for lab, m in test_metrics["per_class"].items():
        print(f"  {lab}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "lid_results.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"tokenizer={args.tokenizer} init={args.init} seed={seed} "
                f"macro_f1={test_metrics['macro_f1']:.4f} "
                f"acc={test_metrics['accuracy']:.4f} "
                f"HI_f1={test_metrics['per_class']['HI']['f1']:.4f} "
                f"EN_f1={test_metrics['per_class']['EN']['f1']:.4f}\n")
    print(f"\nAppended to {out_path}")


if __name__ == "__main__":
    main()