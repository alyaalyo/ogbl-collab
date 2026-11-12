"""Shared decoder, loss, negative sampler and training loop (roles 2 and 3).

Every learned method is  s(u, v) = MLP(z_u * z_v)  trained with BCE on the
training edges against the same number of uniformly sampled node pairs. The
methods differ ONLY in the encoder that produces z:

    features-only MLP  -> FrozenEncoder(x)            (no graph)
    node2vec           -> FrozenEncoder(node2vec_z)   (vectors stay frozen)
    GraphSAGE          -> any nn.Module whose forward() returns z for all nodes,
                          trained end-to-end through this same loop.

Encoder protocol: `encoder()` (no arguments) returns a [num_nodes, d] tensor.
Keep x / adjacency inside the module (buffers or attributes). Full-batch, as in
the OGB reference GraphSAGE for ogbl-collab.

Example for role 3:

    class SAGEEncoder(nn.Module):
        def __init__(self, x, adj_t, ...):
            super().__init__()
            self.register_buffer("x", x); self.adj_t = adj_t; self.convs = ...
        def forward(self):
            h = self.x
            ...
            return h

    result = train_link_predictor(SAGEEncoder(...), LinkDecoder(256), data.train_pos,
                                  data.num_nodes, data.eval_pairs, TrainConfig(lr=1e-3))
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import evaluate_scores
from .utils import Timer


class LinkDecoder(nn.Module):
    """s(u, v) = MLP(z_u * z_v), returns logits. Same shape as the OGB LinkPredictor."""

    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        h = z_u * z_v
        for lin in self.lins[:-1]:
            h = F.relu(lin(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[-1](h).view(-1)


class FrozenEncoder(nn.Module):
    """Fixed node vectors (text features or pre-trained node2vec); nothing is trained."""

    def __init__(self, z: torch.Tensor):
        super().__init__()
        # non-persistent: keeps the [N, d] matrix out of checkpoints (it is input, not a weight)
        self.register_buffer("z", z.float(), persistent=False)

    def forward(self) -> torch.Tensor:
        return self.z


def uniform_negatives(num: int, num_nodes: int, device, generator=None) -> torch.Tensor:
    """`num` node pairs drawn uniformly at random (not filtered against true edges;
    at density ~4e-5 a random pair is almost never an edge)."""
    return torch.randint(0, num_nodes, (num, 2), device=device, generator=generator)


def link_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """BCE: positives -> 1, negatives -> 0, the two halves weighted equally."""
    return (F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)))


@dataclass
class TrainConfig:
    lr: float = 0.01
    batch_size: int = 64 * 1024     # positive pairs per step (+ as many negatives)
    max_epochs: int = 400           # epoch cap, shared by all learned methods
    patience: int = 20              # early stop after this many evals without val gain
    eval_every: int = 1
    select_metric: str = "hits@50"  # model selection on validation only
    eval_batch_size: int = 256 * 1024
    weight_decay: float = 0.0
    device: str = "cpu"
    ckpt_path: Optional[str] = None  # e.g. a Google Drive path on Colab
    ckpt_every: int = 10            # epochs
    resume: bool = True
    log_every: int = 1
    grad_clip: Optional[float] = None  # max grad norm, encoder and decoder clipped separately; None = off


@dataclass
class RunResult:
    best_epoch: int
    valid: dict                     # Hits@{20,50,100} at the best epoch
    test: dict                      # scored once, with the best-on-val weights
    scores: dict                    # split -> logits (CPU tensors), OGB order
    epochs_run: int
    seconds_per_epoch: float
    total_seconds: float
    history: list = field(default_factory=list)
    config: dict = field(default_factory=dict)


@torch.no_grad()
def score_pairs(z: torch.Tensor, decoder: LinkDecoder, pairs: torch.Tensor, batch_size: int) -> torch.Tensor:
    out = []
    for i in range(0, pairs.size(0), batch_size):
        p = pairs[i:i + batch_size].to(z.device)
        out.append(decoder(z[p[:, 0]], z[p[:, 1]]).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def _score_splits(encoder, decoder, eval_pairs, splits, batch_size):
    encoder.eval(); decoder.eval()
    z = encoder()
    return {s: score_pairs(z, decoder, eval_pairs[s], batch_size) for s in splits}


def _cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _has_params(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def train_link_predictor(
    encoder: nn.Module,
    decoder: LinkDecoder,
    train_pos: torch.Tensor,
    num_nodes: int,
    eval_pairs: dict,
    cfg: TrainConfig,
    log: Callable[[str], None] = print,
) -> RunResult:
    device = torch.device(cfg.device)
    encoder, decoder = encoder.to(device), decoder.to(device)
    train_pos = train_pos.to(device)
    trainable_encoder = _has_params(encoder)
    params = list(decoder.parameters()) + (list(encoder.parameters()) if trainable_encoder else [])
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch, best_val, best_epoch, bad_evals = 1, -1.0, 0, 0
    best_state, history, train_seconds = None, [], 0.0

    ckpt = Path(cfg.ckpt_path) if cfg.ckpt_path else None
    if ckpt and cfg.resume and ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(st["encoder"]); decoder.load_state_dict(st["decoder"])
        opt.load_state_dict(st["opt"])
        start_epoch, best_val, best_epoch = st["epoch"] + 1, st["best_val"], st["best_epoch"]
        bad_evals, best_state, history = st["bad_evals"], st["best_state"], st["history"]
        train_seconds = st["train_seconds"]
        torch.set_rng_state(st["rng_cpu"])
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        log(f"resumed from {ckpt} at epoch {start_epoch}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        encoder.train(); decoder.train()
        with Timer() as t:
            total, n = 0.0, 0
            z_fixed = None if trainable_encoder else encoder()
            perm = torch.randperm(train_pos.size(0), device=device)
            for i in range(0, perm.numel(), cfg.batch_size):
                pos = train_pos[perm[i:i + cfg.batch_size]]
                neg = uniform_negatives(pos.size(0), num_nodes, device)
                z = encoder() if trainable_encoder else z_fixed
                loss = link_loss(decoder(z[pos[:, 0]], z[pos[:, 1]]),
                                 decoder(z[neg[:, 0]], z[neg[:, 1]]))
                opt.zero_grad(); loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
                    if trainable_encoder:
                        nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
                opt.step()
                total += loss.item() * pos.size(0); n += pos.size(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
        train_seconds += t.seconds
        rec = {"epoch": epoch, "loss": total / n, "epoch_seconds": round(t.seconds, 3)}

        if epoch % cfg.eval_every == 0:
            val = evaluate_scores(_score_splits(encoder, decoder, eval_pairs,
                                                ("valid_pos", "valid_neg"), cfg.eval_batch_size))["valid"]
            rec["valid"] = val
            if val[cfg.select_metric] > best_val:
                best_val, best_epoch, bad_evals = val[cfg.select_metric], epoch, 0
                best_state = {"encoder": _cpu_state(encoder) if trainable_encoder else None,
                              "decoder": _cpu_state(decoder)}
            else:
                bad_evals += 1
        history.append(rec)
        if epoch % cfg.log_every == 0:
            v = rec.get("valid", {})
            log(f"epoch {epoch:4d}  loss {rec['loss']:.4f}  {t.seconds:6.2f}s  "
                f"val@50 {v.get('hits@50', float('nan')):.4f}  best {best_val:.4f}@{best_epoch}")

        if ckpt and epoch % cfg.ckpt_every == 0:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                        "opt": opt.state_dict(), "best_val": best_val, "best_epoch": best_epoch,
                        "bad_evals": bad_evals, "best_state": best_state, "history": history,
                        "train_seconds": train_seconds, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       ckpt)
        if bad_evals >= cfg.patience:
            log(f"early stop at epoch {epoch} (no val gain for {cfg.patience} evals)")
            break

    # Restore best-on-validation weights; test is scored exactly once, here.
    if best_state["encoder"] is not None:
        encoder.load_state_dict(best_state["encoder"])
    decoder.load_state_dict(best_state["decoder"])
    scores = _score_splits(encoder, decoder, eval_pairs, tuple(eval_pairs), cfg.eval_batch_size)
    metrics = evaluate_scores(scores)
    return RunResult(best_epoch=best_epoch, valid=metrics["valid"], test=metrics["test"], scores=scores,
                     epochs_run=epoch, seconds_per_epoch=train_seconds / max(epoch, 1),
                     total_seconds=train_seconds, history=history, config=asdict(cfg))
