"""node2vec (Grover & Leskovec, 2016) + the shared decoder on frozen vectors.

Two stages, both seeded:
  1. Skip-gram on biased random walks over the input graph A (<= 2017 edges):
     10 walks of length 40 per author, context window 10, 1 negative per context
     pair (same objective as torch_geometric.nn.Node2Vec). One free 128-d vector
     per author, no text features. Saved to embeddings/.
  2. The vectors are frozen and the shared decoder MLP(z_u * z_v) is trained on
     them exactly like the features-only MLP.

The walker is our own vectorised NumPy implementation (rejection sampling for the
p/q bias, like torch_cluster), so it needs neither pyg-lib nor torch-cluster;
their macOS wheels do not load against Homebrew/uv Python.

    python -m collab.node2vec --p 1 --q 1 --seeds 0                 # one config
    python -m collab.node2vec --grid --seeds 0                      # 9-config p,q grid
    python -m collab.node2vec --p 1 --q 1 --seeds 0-9 --reuse_emb   # seeds for the chosen config
"""
import argparse
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_collab
from .decoder import FrozenEncoder, LinkDecoder, TrainConfig, train_link_predictor
from .evaluate import save_scores
from .mlp import parse_seeds
from .utils import CKPT_DIR, EMB_DIR, Timer, append_result, get_device, set_seed


class Walker:
    """Biased 2nd-order random walks on an undirected graph in CSR form."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, p: float = 1.0, q: float = 1.0, rng=None):
        row, col = edge_index.numpy()
        order = np.lexsort((col, row))                   # sort by row, then col
        self.row, self.col = row[order], col[order]
        self.n = num_nodes
        self.deg = np.bincount(self.row, minlength=num_nodes)
        self.rowptr = np.concatenate([[0], np.cumsum(self.deg)])
        self.keys = self.row.astype(np.int64) * num_nodes + self.col   # sorted -> edge lookup
        self.p, self.q = p, q
        self.rng = rng or np.random.default_rng()

    def _uniform_neighbor(self, cur: np.ndarray) -> np.ndarray:
        d = self.deg[cur]
        off = np.floor(self.rng.random(cur.size) * d).astype(np.int64)
        nxt = self.col[np.minimum(self.rowptr[cur] + off, self.col.size - 1)]
        return np.where(d > 0, nxt, cur)                # isolated author: stay in place

    def _has_edge(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        k = a.astype(np.int64) * self.n + b
        order = np.argsort(k)                            # sorted queries: ~7x faster searchsorted
        i = np.empty_like(order)
        i[order] = np.searchsorted(self.keys, k[order])
        i = np.minimum(i, self.keys.size - 1)
        return self.keys[i] == k

    def walk(self, start: np.ndarray, length: int) -> np.ndarray:
        walks = np.empty((start.size, length), dtype=np.int64)
        walks[:, 0] = start
        walks[:, 1] = self._uniform_neighbor(start)
        unbiased = self.p == 1 and self.q == 1
        max_prob = max(1 / self.p, 1.0, 1 / self.q)
        for t in range(2, length):
            prev, cur = walks[:, t - 2], walks[:, t - 1]
            nxt = self._uniform_neighbor(cur)
            if not unbiased:
                todo = np.arange(start.size)
                while todo.size:
                    x, pv = nxt[todo], prev[todo]
                    prob = np.where(x == pv, 1 / self.p,
                                    np.where(self._has_edge(pv, x), 1.0, 1 / self.q)) / max_prob
                    reject = (self.rng.random(todo.size) >= prob) & (self.deg[cur[todo]] > 0)
                    todo = todo[reject]
                    nxt[todo] = self._uniform_neighbor(cur[todo])
            walks[:, t] = nxt
        return walks


def windows(walks: torch.Tensor, context_size: int) -> torch.Tensor:
    """[W, L] walks -> [W*(L-c+1), c] sub-walks; first column is the centre."""
    L = walks.size(1)
    return torch.cat([walks[:, j:j + context_size] for j in range(L - context_size + 1)], dim=0)


def train_node2vec(edge_index, num_nodes, dim=128, walk_length=40, context_size=10, walks_per_node=10,
                   p=1.0, q=1.0, epochs=1, batch_nodes=256, lr=0.01, seed=0, device="cpu", log=print):
    """Skip-gram with negative sampling.

    Dense embedding + Adam on every device: SparseAdam is unavailable on MPS and on
    CPU its sparse-gradient coalescing makes an epoch take ~100 min. Dense Adam on
    MPS/CUDA takes ~7 min/epoch (M-series) and is used everywhere so runs match.
    """
    rng = np.random.default_rng(seed)
    walker = Walker(edge_index, num_nodes, p, q, rng)
    torch.manual_seed(seed)
    emb = nn.Embedding(num_nodes, dim).to(device)
    opt = torch.optim.Adam(emb.parameters(), lr=lr)
    steps = math.ceil(num_nodes / batch_nodes)
    seconds = 0.0
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(num_nodes)
        total = 0.0
        with Timer() as t:
            for i in range(steps):
                start = np.repeat(perm[i * batch_nodes:(i + 1) * batch_nodes], walks_per_node)
                pos = windows(torch.from_numpy(walker.walk(start, walk_length)), context_size).to(device)
                neg = torch.randint(0, num_nodes, pos.shape, device=device)
                neg[:, 0] = pos[:, 0]
                h = emb(pos[:, :1]); hp = emb(pos[:, 1:]); hn = emb(neg[:, 1:])
                loss = (-F.logsigmoid((h * hp).sum(-1)).mean()
                        - F.logsigmoid(-(h * hn).sum(-1)).mean())
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
                if (i + 1) % 200 == 0:
                    log(f"  node2vec epoch {epoch} step {i + 1}/{steps} loss {total / (i + 1):.4f}")
        seconds += t.seconds
        log(f"node2vec epoch {epoch}: loss {total / steps:.4f}  {t.seconds:.1f}s")
    return emb.weight.detach().cpu(), seconds


def emb_path(p, q, seed, epochs):
    return EMB_DIR / f"node2vec_p{p}_q{q}_e{epochs}_seed{seed}.pt"


def run(data, p, q, seed, args, device):
    set_seed(seed)
    path = emb_path(p, q, seed, args.n2v_epochs)
    if args.reuse_emb and path.exists():
        z, emb_seconds = torch.load(path), None
        print(f"loaded {path}")
    else:
        z, emb_seconds = train_node2vec(data.edge_index, data.num_nodes, dim=args.dim,
                                        walk_length=args.walk_length, context_size=args.context_size,
                                        walks_per_node=args.walks_per_node, p=p, q=q,
                                        epochs=args.n2v_epochs, batch_nodes=args.batch_nodes,
                                        lr=args.n2v_lr, seed=seed, device=str(device))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(z, path)

    set_seed(seed)
    cfg = TrainConfig(lr=args.lr, max_epochs=args.max_epochs, patience=args.patience,
                      device=str(device), log_every=10,
                      ckpt_path=str(CKPT_DIR / f"{args.method}_p{p}_q{q}_seed{seed}.pt"), resume=False)
    res = train_link_predictor(FrozenEncoder(z), LinkDecoder(z.size(1), 256, dropout=args.dropout),
                               data.train_pos, data.num_nodes, data.eval_pairs, cfg)
    print(f"p={p} q={q} seed {seed}: best epoch {res.best_epoch}  valid {res.valid}  test {res.test}")
    if not args.grid:  # grid runs only select (p, q); scores are saved for the chosen config
        save_scores(args.method, seed, res.scores)
    append_result(args.method + ("_grid" if args.grid else ""), {
        "method": args.method, "seed": seed, "device": str(device),
        "hparams": {"p": p, "q": q, "dim": args.dim, "walk_length": args.walk_length,
                    "context_size": args.context_size, "walks_per_node": args.walks_per_node,
                    "n2v_epochs": args.n2v_epochs, "n2v_lr": args.n2v_lr,
                    "lr": args.lr, "dropout": args.dropout},
        "embedding_seconds": None if emb_seconds is None else round(emb_seconds, 1),
        "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
        "seconds_per_epoch": round(res.seconds_per_epoch, 3),
        "valid": res.valid, "test": res.test,
    })
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--grid", action="store_true", help="p, q in {0.5, 1, 2}, select on val Hits@50")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--walk_length", type=int, default=40)
    ap.add_argument("--context_size", type=int, default=10)
    ap.add_argument("--walks_per_node", type=int, default=10)
    ap.add_argument("--n2v_epochs", type=int, default=3)
    ap.add_argument("--n2v_lr", type=float, default=0.01)
    ap.add_argument("--batch_nodes", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01, help="decoder learning rate")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--reuse_emb", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--method", default="node2vec")
    args = ap.parse_args(argv)

    data = load_collab()
    device = get_device(args.device)
    configs = list(itertools.product([0.5, 1.0, 2.0], repeat=2)) if args.grid else [(args.p, args.q)]
    best = None
    for p, q in configs:
        for seed in parse_seeds(args.seeds):
            res = run(data, p, q, seed, args, device)
            if best is None or res.valid["hits@50"] > best[0]:
                best = (res.valid["hits@50"], p, q)
    if args.grid:
        print(f"best on validation Hits@50: p={best[1]} q={best[2]} ({best[0]:.4f})")


if __name__ == "__main__":
    main()
