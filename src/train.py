import argparse
import math
import os
import time

import numpy as np
import torch
import wandb

from config import ModelConfig, ModelConfigGPT2Tok, TrainConfig
from model import GPT


def get_batch(data_path, block_size, batch_size, device):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    if step > cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, data_dir, mcfg, tcfg, device, ctx, data_variant=""):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(
                os.path.join(data_dir, f"{split}{data_variant}.bin"),
                mcfg.block_size, tcfg.micro_batch_size, device
            )
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/processed")
    parser.add_argument("--variant", type=str, default="run",
                        help="label for this run, used in W&B and checkpoint filenames")
    parser.add_argument("--data_variant", type=str, default="",
                        help="suffix for data files, e.g. '_gpt2' for train_gpt2.bin/val_gpt2.bin")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint to resume from")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # select model config based on which tokenizer is being used
    mcfg = ModelConfigGPT2Tok() if args.data_variant == "_gpt2" else ModelConfig()
    tcfg = TrainConfig()
    tcfg.wandb_run_name = args.variant

    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU detected — training will be extremely slow on CPU.")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else _nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    model = GPT(mcfg).to(device)
    model.grad_checkpointing = tcfg.grad_checkpointing
    print(f"Model params: {model.num_params() / 1e6:.2f}M | device={device} | dtype={dtype} | vocab={mcfg.vocab_size}")

    optimizer = model.configure_optimizer(
        tcfg.weight_decay, tcfg.learning_rate, (tcfg.beta1, tcfg.beta2)
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    os.makedirs(tcfg.out_dir, exist_ok=True)

    if not args.no_wandb:
        wandb.init(
            project=tcfg.wandb_project,
            name=tcfg.wandb_run_name,
            config={**vars(mcfg), **vars(tcfg), "data_variant": args.data_variant}
        )

    train_path = os.path.join(args.data_dir, f"train{args.data_variant}.bin")
    t0 = time.time()

    for step in range(start_step, tcfg.max_steps + 1):
        lr = get_lr(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % tcfg.eval_interval == 0:
            losses = estimate_loss(
                model, args.data_dir, mcfg, tcfg, device, ctx, args.data_variant
            )
            elapsed = time.time() - t0
            print(f"step {step}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}, "
                  f"lr {lr:.2e}, elapsed {elapsed/60:.1f}min")
            if not args.no_wandb:
                wandb.log({
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr
                }, step=step)

        if step % tcfg.checkpoint_interval == 0 and step > 0:
            save_full = (
                (step // tcfg.checkpoint_interval) % tcfg.keep_optimizer_in_checkpoint_every == 0
            )
            ckpt = {
                "model": model.state_dict(),
                "step": step,
                "model_config": vars(mcfg),
                "variant": args.variant,
                "data_variant": args.data_variant,
            }
            if save_full:
                ckpt["optimizer"] = optimizer.state_dict()
            path = os.path.join(tcfg.out_dir, f"{args.variant}_step{step}.pt")
            torch.save(ckpt, path)
            print(f"saved checkpoint -> {path} (full={save_full})")

        # ---- gradient accumulation loop ----
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(tcfg.grad_accum_steps):
            x, y = get_batch(
                train_path, mcfg.block_size, tcfg.micro_batch_size, device
            )
            with ctx:
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % tcfg.log_interval == 0:
            print(f"  step {step} | loss {accum_loss:.4f} | lr {lr:.2e}")


if __name__ == "__main__":
    main()