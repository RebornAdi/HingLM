from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False

@dataclass
class ModelConfigGPT2Tok:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False

@dataclass
class TrainConfig:
    # ---- batch / memory ----
    micro_batch_size: int = 8
    grad_accum_steps: int = 16
    block_size: int = 256

    # ---- optimization ----
    max_steps: int = 10000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- precision / memory ----
    dtype: str = "bfloat16"
    grad_checkpointing: bool = True

    # ---- logging / checkpointing ----
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 10
    checkpoint_interval: int = 2000
    keep_optimizer_in_checkpoint_every: int = 5

    out_dir: str = "checkpoints"
    wandb_project: str = "HingLM"
    wandb_run_name: str = "run1"

    seed: int = 1337