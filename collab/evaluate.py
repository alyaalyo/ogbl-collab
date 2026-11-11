"""Official OGB Hits@K and the shared score-file format.

Score files: scores/{method}_seed{k}_{split}.npy, float32, one score per pair,
in the official OGB order of that split.
  split  in {valid_pos, valid_neg, test_pos, test_neg}
  method e.g. cn, aa, cn_aa, mlp, node2vec, sage; suffix "_valinput" for the
         "val as input" variant. Deterministic methods use seed0.
The bucket analysis only reads these files, so it is method-agnostic.
"""
from pathlib import Path

import numpy as np
import torch

from .utils import SCORES_DIR

KS = (20, 50, 100)
SPLITS = ("valid_pos", "valid_neg", "test_pos", "test_neg")

_evaluators = {}


def hits(pos_scores, neg_scores, ks=KS) -> dict:
    """Hits@K from the official evaluator (strict '>' against the K-th negative)."""
    from ogb.linkproppred import Evaluator

    pos = torch.as_tensor(np.asarray(pos_scores)).view(-1).float()
    neg = torch.as_tensor(np.asarray(neg_scores)).view(-1).float()
    out = {}
    for k in ks:
        ev = _evaluators.setdefault(k, Evaluator("ogbl-collab"))
        ev.K = k
        out[f"hits@{k}"] = float(ev.eval({"y_pred_pos": pos, "y_pred_neg": neg})[f"hits@{k}"])
    return out


def evaluate_scores(scores: dict, ks=KS) -> dict:
    """{"valid_pos":..,"valid_neg":..,"test_pos":..,"test_neg":..} -> {"valid": {...}, "test": {...}}"""
    res = {}
    for part in ("valid", "test"):
        if f"{part}_pos" in scores and f"{part}_neg" in scores:
            res[part] = hits(scores[f"{part}_pos"], scores[f"{part}_neg"], ks)
    return res


def score_path(method: str, seed: int, split: str, root=SCORES_DIR) -> Path:
    assert split in SPLITS, split
    return Path(root) / f"{method}_seed{seed}_{split}.npy"


def save_scores(method: str, seed: int, scores: dict, root=SCORES_DIR) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    for split, s in scores.items():
        s = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
        np.save(score_path(method, seed, split, root), s.astype(np.float32).reshape(-1))


def load_scores(method: str, seed: int, root=SCORES_DIR) -> dict:
    return {s: np.load(score_path(method, seed, s, root)) for s in SPLITS
            if score_path(method, seed, s, root).exists()}
