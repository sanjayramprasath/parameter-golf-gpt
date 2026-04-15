# Parameter Golf — GPT Training

A tuned baseline for the [modded-nanogpt parameter golf challenge](https://github.com/KellerJordan/modded-nanogpt), built on top of the default `train_gpt.py` starter script.

## Overview

The goal of parameter golf is to train the best possible language model within a strict submission size budget. This script tunes the default baseline with a set of optimizer and scheduler improvements that improve validation BPB without changing model architecture or increasing submission size.

## Changes from Default Baseline

| Parameter | Default | This Repo | Effect |
|---|---|---|---|
| `warmdown_iters` | 900 | **1800** | Smoother LR tail, longer cooldown |
| `matrix_lr` | 0.02 | **0.05** | Faster weight alignment via Muon |
| `scalar_lr` | 0.02 | **0.05** | Matched scalar learning rate |
| `beta2` | 0.95 | **0.99** | Tighter Adam variance estimate |
| `muon_backend_steps` | 3 | **5** | Better orthogonalisation per step |
| `grad_clip_norm` | 0 (off) | **1.0** | Light gradient clipping for stability |
| `muon_momentum_warmup_steps` | 300 | **500** | Slower, smoother momentum ramp |
| `qk_gain_init` | 1.0 | **1.8** | Larger QK init gain aids early training |
| LR warmdown schedule | Linear | **Cosine** | Smoother loss curve at end of training |
| Muon momentum warmup | Linear | **Cosine** | Smoother ease-in from 0.85 → 0.95 |

## Model Architecture (unchanged)

- 8 transformer blocks, width 512
- 8 attention heads, 2 KV heads (GQA), 2× MLP expansion
- Vocab size 1024, sequence length 1024, tied embeddings
- RoPE positional encoding, logit softcap at 30.0

## Usage

```bash
python train_gpt.py
```

All hyperparameters can be overridden via environment variables:

```bash
MATRIX_LR=0.05 WARMDOWN_ITERS=1800 python train_gpt.py
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- `sentencepiece`
- FineWeb 10B dataset preprocessed into `.bin` shards (see modded-nanogpt for preprocessing scripts)

## References

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — original challenge repo
- [Muon optimizer](https://kellerjordan.github.io/posts/muon/) — background on the Muon optimizer used for matrix parameters
