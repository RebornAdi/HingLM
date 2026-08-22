"""
Decoder-only transformer, implemented from scratch.
Modern recipe (LLaMA-style): RoPE positional embeddings, RMSNorm, SwiGLU MLP,
no biases. This is the "novelty knob" of the project — swap pieces here
(e.g. RoPE vs learned pos-emb) to run your architecture comparison experiment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm instead of LayerNorm — simpler, no mean-centering, used in LLaMA/Mistral."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0, device=None):
    """Precompute the rotary embedding frequencies (cos/sin) up to max_seq_len."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """
    Apply rotary position embedding to x of shape (B, n_head, T, head_dim).
    Rotates pairs of dimensions by position-dependent angle.
    """
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # flash-attention via PyTorch SDPA — fast and memory-efficient, important at 4GB VRAM
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    """SwiGLU MLP — used in LLaMA/PaLM, generally outperforms plain GELU MLP at same param count."""

    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)  # standard SwiGLU sizing convention
        hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64 for efficiency
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying — saves params, standard practice
        self.tok_emb.weight = self.lm_head.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope_freqs(head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper convention
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.grad_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        n -= self.tok_emb.weight.numel()  # don't double count tied embedding
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)

        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """Separate params into decay/no-decay groups — standard practice (don't decay norms/biases)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=torch.cuda.is_available())

class GPTForTokenClassification(nn.Module):
    """
    Wraps the GPT backbone with a token-classification head for the LID task.
    Reuses the pretrained transformer; adds one linear layer mapping each
    token's final hidden state to num_labels logits.

    Note: the backbone uses causal attention (each token sees only leftward
    context). This is a valid but mildly limiting setup for tagging vs a
    bidirectional encoder — disclosed honestly in the writeup. Both tokenizer
    variants share this constraint, so the comparison stays fair.
    """

    def __init__(self, cfg, num_labels=2):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.classifier = nn.Linear(cfg.n_embd, num_labels)

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope_freqs(head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.grad_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_backbone_from_gpt(self, gpt_state_dict):
        """
        Copy pretrained backbone weights (tok_emb, blocks, norm_f) from a saved
        GPT checkpoint's state_dict. The classifier head stays random.
        lm_head is skipped (not used for token classification).
        """
        own = self.state_dict()
        copied, skipped = 0, 0
        for k, v in gpt_state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                copied += 1
            else:
                skipped += 1
        self.load_state_dict(own)
        print(f"  loaded backbone: {copied} tensors copied, {skipped} skipped (lm_head/classifier)")

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        x = self.drop(self.tok_emb(input_ids))
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)

        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        x = self.norm_f(x)
        logits = self.classifier(x)  # (B, T, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

if __name__ == "__main__":
    from config import ModelConfig

    cfg = ModelConfig()
    model = GPT(cfg)
    print(f"Model params: {model.num_params() / 1e6:.2f}M")
