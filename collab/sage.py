"""GraphSAGE (role 3): 3 layers, 256 units, mean aggregation, input x_v, shared decoder.

The encoder is trained end-to-end through the shared loop in decoder.py
(MLP(z_u * z_v) decoder, BCE against uniform negatives, epoch cap, early stopping on
validation Hits@50, test scored once with the best-on-validation weights).
Full-batch: every optimiser step runs the encoder on the whole input graph A (<= 2017).

    python -m collab.sage --probe                    # 3 epochs, prints s/epoch and a time estimate
    python -m collab.sage --tune                     # lrs from --tune_lrs (default 1e-3,1e-2), seed 0, validation only
    python -m collab.sage --seeds 0                  # one seed
    python -m collab.sage --seeds 0-9 --lr 0.001     # all ten (use --seeds 0-4 for the fallback)

After training, the best-on-validation weights are re-used WITHOUT retraining on the
graph A + 2018 edges ("val as input"); test scores are saved as `sage_valinput`.
Validation scores for that variant are not produced: the 2018 positives are in the input.

Colab: point COLLAB_CKPT / COLLAB_SCORES at Drive (see README); a seed already present in
results/sage.jsonl is skipped, an interrupted seed resumes from its last checkpoint.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .data import load_collab
from .decoder import LinkDecoder, TrainConfig, score_pairs, train_link_predictor
from .evaluate import hits, save_scores
from .mlp import parse_seeds
from .utils import CKPT_DIR, RESULTS_DIR, append_result, get_device, set_seed

REFERENCE_TEST_HITS50 = 48.10   # OGB leaderboard GraphSAGE, test Hits@50 (x100), no val edges as input
REPRO_TOLERANCE = 2.0           # proposal: reproduce within 2 points before running the analysis


class SAGEEncoder(nn.Module):
    """Stack of SAGEConv (mean aggregation) layers; forward() -> [num_nodes, hidden].

    x and the graph are non-persistent buffers: they follow .to(device) but stay out of
    state_dict(), so checkpoints hold only the weights (not a 120 MB copy of x).
    """

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, hidden: int = 256,
                 num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("x", x.float(), persistent=False)
        self.register_buffer("edge_index", edge_index, persistent=False)
        dims = [x.size(1)] + [hidden] * num_layers
        self.convs = nn.ModuleList(SAGEConv(a, b, aggr="mean") for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def set_graph(self, edge_index: torch.Tensor) -> None:
        """Swap the message-passing graph (weights untouched), e.g. for "val as input"."""
        self.edge_index = edge_index.to(self.x.device)

    def forward(self) -> torch.Tensor:
        h = self.x
        for i, conv in enumerate(self.convs):
            h = conv(h, self.edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def _device_name(device) -> str:
    return torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else str(device)


def _build(data, args):
    enc = SAGEEncoder(data.x, data.edge_index, args.hidden, args.layers, args.dropout)
    dec = LinkDecoder(args.hidden, args.hidden, dropout=args.dropout)
    return enc, dec


def _cfg(args, device, ckpt_path, resume, max_epochs=None):
    return TrainConfig(lr=args.lr, batch_size=args.batch_size,
                       max_epochs=max_epochs or args.max_epochs, patience=args.patience,
                       device=str(device), ckpt_path=ckpt_path, resume=resume, log_every=args.log_every,
                       grad_clip=args.grad_clip or None)


def _done_seeds(method: str) -> set:
    path = RESULTS_DIR / f"{method}.jsonl"
    if not path.exists():
        return set()
    return {json.loads(l)["seed"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


@torch.no_grad()
def score_with_graph(encoder, decoder, edge_index, eval_pairs, splits, batch_size=256 * 1024) -> dict:
    """Score `splits` with the trained weights on another input graph (no retraining)."""
    encoder.set_graph(edge_index)
    encoder.eval(); decoder.eval()
    z = encoder()
    return {s: score_pairs(z, decoder, eval_pairs[s], batch_size) for s in splits}


def probe(data, args, device):
    """Short run to time an epoch; decides whether the 10-seed budget fits (proposal: > 1 h -> fallback)."""
    set_seed(0)
    enc, dec = _build(data, args)
    n_epochs = 3
    t0 = time.perf_counter()
    res = train_link_predictor(enc, dec, data.train_pos, data.num_nodes, data.eval_pairs,
                               _cfg(args, device, None, False, n_epochs), log=print)
    wall = time.perf_counter() - t0
    train_epoch = res.seconds_per_epoch
    # wall includes the per-epoch validation pass and the final scoring of all splits
    wall_epoch = wall / n_epochs
    full = wall_epoch * args.max_epochs / 3600
    out = {"device": _device_name(device), "epochs": n_epochs, "train_s_per_epoch": round(train_epoch, 2),
           "wall_s_per_epoch_incl_eval": round(wall_epoch, 2),
           "upper_bound_hours_at_epoch_cap": round(full, 2)}
    print("PROBE", json.dumps(out))
    print(f"A run stops earlier if early stopping triggers; at the cap of {args.max_epochs} epochs it takes <= "
          f"{full:.2f} h. If a typical run is over ~1 h on the T4, use the fallback (--seeds 0-4).")
    append_result("sage_timing", {"method": "sage", "seed": 0, **out,
                                  "hparams": {"lr": args.lr, "dropout": args.dropout, "hidden": args.hidden,
                                              "layers": args.layers, "grad_clip": args.grad_clip, "patience": args.patience}})


def tune(data, args, device):
    """lrs from --tune_lrs on seed 0, chosen by validation Hits@50. Test numbers are not recorded."""
    best = None
    for lr in [float(v) for v in args.tune_lrs.split(",")]:
        args.lr = lr
        set_seed(0)
        enc, dec = _build(data, args)
        res = train_link_predictor(enc, dec, data.train_pos, data.num_nodes, data.eval_pairs,
                                   _cfg(args, device, None, False), log=print)
        print(f"lr={lr}: best epoch {res.best_epoch}  valid {res.valid}")
        append_result("sage_grid", {"method": "sage", "seed": 0, "device": _device_name(device),
                                    "hparams": {"lr": lr, "dropout": args.dropout, "hidden": args.hidden,
                                                "layers": args.layers, "grad_clip": args.grad_clip, "patience": args.patience},
                                    "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
                                    "seconds_per_epoch": round(res.seconds_per_epoch, 3),
                                    "valid": res.valid})
        if best is None or res.valid["hits@50"] > best[0]:
            best = (res.valid["hits@50"], lr)
    print(f"best on validation Hits@50: lr={best[1]} ({best[0]:.4f})  -> use --lr {best[1]}")


def run_seed(data, args, device, seed):
    set_seed(seed)
    enc, dec = _build(data, args)
    ckpt = str(CKPT_DIR / f"{args.method}_seed{seed}.pt")
    print(f"== {args.method} seed {seed} on {_device_name(device)}")
    res = train_link_predictor(enc, dec, data.train_pos, data.num_nodes, data.eval_pairs,
                               _cfg(args, device, ckpt, not args.no_resume), log=print)
    print(f"seed {seed}: best epoch {res.best_epoch}  valid {res.valid}  test {res.test}  "
          f"({res.seconds_per_epoch:.2f}s/epoch)")
    save_scores(args.method, seed, res.scores)
    hparams = {"lr": args.lr, "dropout": args.dropout, "hidden": args.hidden, "layers": args.layers, "grad_clip": args.grad_clip, "patience": args.patience}

    # encoder/decoder now hold the best-on-validation weights: re-score on A + 2018 edges, no retraining
    vi = score_with_graph(enc, dec, data.edge_index_val, data.eval_pairs, ("test_pos", "test_neg"))
    vi_test = hits(vi["test_pos"], vi["test_neg"])
    save_scores(args.method + "_valinput", seed, vi)
    print(f"seed {seed}: val as input, test {vi_test}")

    common = {"seed": seed, "device": _device_name(device), "hparams": hparams, "best_epoch": res.best_epoch,
              "epochs_run": res.epochs_run, "seconds_per_epoch": round(res.seconds_per_epoch, 3),
              "total_seconds": round(res.total_seconds, 1)}
    # the valinput line is written first: a seed counts as done only once the main line exists
    append_result(args.method + "_valinput", {"method": args.method + "_valinput", **common, "test": vi_test})
    append_result(args.method, {"method": args.method, **common, "valid": res.valid, "test": res.test})
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "hparams": hparams},
               CKPT_DIR / f"{args.method}_seed{seed}_final.pt")


def repro_report(method: str) -> None:
    path = RESULTS_DIR / f"{method}.jsonl"
    if not path.exists():
        return
    runs = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            runs[r["seed"]] = r
    vals = 100 * np.array([r["test"]["hits@50"] for r in runs.values()])
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else float("nan")
    ok = abs(mean - REFERENCE_TEST_HITS50) <= REPRO_TOLERANCE
    print(f"test Hits@50 over {len(vals)} seed(s): {mean:.2f} ± {std:.2f}  (reference {REFERENCE_TEST_HITS50}, "
          f"tolerance ±{REPRO_TOLERANCE}) -> {'REPRODUCED' if ok else 'NOT reproduced: do not start the analysis'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="max grad norm, encoder and decoder separately (OGB reference: 1.0); 0 = off")
    ap.add_argument("--tune_lrs", default="0.001,0.01", help="comma-separated lrs for --tune")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64 * 1024)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--method", default="sage", help="name used for score files / results")
    ap.add_argument("--probe", action="store_true", help="3 epochs, time one epoch, write results/sage_timing.jsonl")
    ap.add_argument("--tune", action="store_true", help="lr in {1e-3, 1e-2} on seed 0 (validation only)")
    ap.add_argument("--no_resume", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--force", action="store_true", help="rerun seeds already in results/{method}.jsonl")
    args = ap.parse_args(argv)

    data = load_collab()
    device = get_device(args.device)
    if args.probe:
        return probe(data, args, device)
    if args.tune:
        return tune(data, args, device)
    done = set() if args.force else _done_seeds(args.method)
    for seed in parse_seeds(args.seeds):
        if seed in done:
            print(f"seed {seed} already in results/{args.method}.jsonl, skipping (--force to rerun)")
            continue
        run_seed(data, args, device, seed)
    repro_report(args.method)


if __name__ == "__main__":
    main()
