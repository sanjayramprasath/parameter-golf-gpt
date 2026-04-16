import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CONFIG
# -----------------------------
class CFG:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    seq_len = 2048
    batch_size = 2  # reduced from 6 to fit 2048 seq_len in GPU memory

    vocab_size = 1024

    d_model = 512
    n_heads = 8
    n_layers = 8

    lr = 3e-4
    steps = 200000

    grad_clip = 1.0
    eval_every = 2000
    ckpt_every = 5000

    # TTT config
    ttt_every = 2          # use TTT block every N layers (e.g. layers 0,2,4,6)
    ttt_inner_lr = 1e-2    # inner loop learning rate for TTT
    ttt_inner_steps = 1    # gradient steps taken inside TTT per token

    device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# DATA
# -----------------------------
def load_tokens(path):
    files = sorted(glob.glob(f"{path}/fineweb_train_*.bin"))
    if not files:
        raise FileNotFoundError(f"No shards found in {path}")

    arr = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])
    return torch.from_numpy(arr.astype(np.int64))


class Dataset:
    def __init__(self, tokens):
        self.tokens = tokens
        self.max_i = len(tokens) - CFG.seq_len - 1

    def get_batch(self):
        ix = torch.randint(0, self.max_i, (CFG.batch_size,))
        x = torch.stack([self.tokens[i:i+CFG.seq_len] for i in ix])
        y = torch.stack([self.tokens[i+1:i+CFG.seq_len+1] for i in ix])
        return x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)


# -----------------------------
# STANDARD ATTENTION
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


# -----------------------------
# TTT LAYER
# -----------------------------
# Test-Time Training (TTT) replaces the attention mechanism with a small
# inner model (here: a single linear layer W) that is updated on-the-fly
# for each sequence using a self-supervised reconstruction task.
#
# Intuition:
#   - Standard attention retrieves from a fixed KV cache
#   - TTT instead *learns* a compressed hidden state W per sequence
#   - W is updated via gradient descent on a self-supervised loss (predict v from k)
#   - The updated W is then used to produce the output, like a "smart KV store"
#
# This follows the TTT paper (Sun et al., 2024): https://arxiv.org/abs/2407.04620
# We implement TTT-Linear (the simplest variant).
#
# Causal constraint: token t only uses W updated on tokens 0..t-1
# We achieve this by processing tokens sequentially and accumulating W.

class TTTLinear(nn.Module):
    """
    TTT-Linear layer.

    Maintains a hidden weight matrix W (shape: head_dim x head_dim) per head.
    For each token position t (causally):
      1. Project x -> q, k, v
      2. Self-supervised inner loss: ||W @ k_t - v_t||^2
      3. Take one gradient step on W using k_t, v_t
      4. Output: q_t @ W  (read from updated W)

    W is initialised to zero each forward pass (no state carried across sequences).
    """
    def __init__(self, d_model, n_heads, inner_lr=1e-2, inner_steps=1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable per-head inner LR scale
        self.lr_scale = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)   # (B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        # W shape: (B, H, D, D) — one matrix per head per sequence in batch
        # Initialise to identity * small scale so output isn't zero at step 0
        W = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(T):
            k_t = k[:, t, :, :]   # (B, H, D)
            v_t = v[:, t, :, :]   # (B, H, D)
            q_t = q[:, t, :, :]   # (B, H, D)

            # --- Read: output using current W (before update, for causality) ---
            # out_t[b,h] = W[b,h] @ q_t[b,h]
            out_t = torch.einsum('bhij,bhj->bhi', W, q_t)  # (B, H, D)
            outputs.append(out_t)

            # --- Write: update W with one gradient step on reconstruction loss ---
            # loss = ||W @ k_t - v_t||^2  (per head, per batch)
            # grad_W = 2 * (W @ k_t - v_t) ⊗ k_t
            pred = torch.einsum('bhij,bhj->bhi', W, k_t)   # (B, H, D)
            err  = pred - v_t                                # (B, H, D)
            # Outer product: grad shape (B, H, D, D)
            grad = torch.einsum('bhi,bhj->bhij', err, k_t)

            effective_lr = self.inner_lr * self.lr_scale    # (H, 1, 1) broadcast
            W = W - effective_lr * grad

        # Stack outputs: (B, T, H, D) -> (B, T, C)
        out = torch.stack(outputs, dim=1)           # (B, T, H, D)
        out = out.contiguous().view(B, T, C)

        return self.out_proj(out)


# -----------------------------
# BLOCKS
# -----------------------------
class Block(nn.Module):
    """Standard transformer block with causal self-attention."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TTTBlock(nn.Module):
    """Transformer block with TTT-Linear replacing standard attention."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ttt = TTTLinear(
            d_model, n_heads,
            inner_lr=CFG.ttt_inner_lr,
            inner_steps=CFG.ttt_inner_steps
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.ttt(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------------
# MODEL
# -----------------------------
class GPT(nn.Module):
    """
    GPT with alternating standard attention and TTT-Linear blocks.

    Layer pattern (ttt_every=2):
      0: TTTBlock
      1: Block (standard attention)
      2: TTTBlock
      3: Block
      ...
    """
    def __init__(self):
        super().__init__()

        self.tok_emb = nn.Embedding(CFG.vocab_size, CFG.d_model)
        self.pos_emb = nn.Embedding(CFG.seq_len, CFG.d_model)

        self.blocks = nn.ModuleList([
            TTTBlock(CFG.d_model, CFG.n_heads) if i % CFG.ttt_every == 0
            else Block(CFG.d_model, CFG.n_heads)
            for i in range(CFG.n_layers)
        ])

        self.norm = nn.LayerNorm(CFG.d_model)
        self.head = nn.Linear(CFG.d_model, CFG.vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.tok_emb(x) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        logits = self.head(x)

        if y is None:
            return logits

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )


# -----------------------------
# EVAL
# -----------------------------
@torch.no_grad()
def eval_step(model, data):
    model.eval()
    x, y = data.get_batch()
    loss = model(x, y)
    model.train()
    return loss.item()


# -----------------------------
# TRAIN
# -----------------------------
def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokens = load_tokens(CFG.data_path)
    data = Dataset(tokens)

    model = GPT().to(CFG.device)

    # Count block types
    ttt_count = sum(1 for b in model.blocks if isinstance(b, TTTBlock))
    std_count = CFG.n_layers - ttt_count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {ttt_count} TTT blocks + {std_count} standard blocks")
    print(f"Total params: {total_params:,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    print(f"tokens: {len(tokens)} | device: {CFG.device}")

    model.train()

    for step in range(1, CFG.steps + 1):
        x, y = data.get_batch()

        loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        if step % CFG.eval_every == 0:
            print(f"[eval] step {step} | loss {eval_step(model, data):.4f}")

        if step % CFG.ckpt_every == 0:
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step
            }, f"ckpt_{step}.pt")
            print("checkpoint saved")

    torch.save(model.state_dict(), "model.pt")
    print("final model saved")
    print("final eval:", eval_step(model, data))


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    train()
