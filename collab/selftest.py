"""Fast checks of the shared code on toy data (no download, < 1 min on CPU).

    python -m collab.selftest
"""
import tempfile

import numpy as np
import torch

from .decoder import FrozenEncoder, LinkDecoder, TrainConfig, train_link_predictor, uniform_negatives
from .evaluate import hits, load_scores, save_scores
from .node2vec import Walker, windows
from .utils import set_seed


def check_hits():
    neg = np.arange(100, dtype=np.float32)              # 50th-highest negative = 50
    assert hits([1000, 1000], neg)["hits@50"] == 1.0
    assert hits([50, 51], neg)["hits@50"] == 0.5        # strict '>': a tie with the 50th is a miss
    assert hits([-1, -1], neg)["hits@100"] == 0.0


def check_scores_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s = {"valid_pos": torch.rand(5), "test_neg": np.arange(3)}
        save_scores("toy", 3, s, root=d)
        back = load_scores("toy", 3, root=d)
        assert set(back) == {"valid_pos", "test_neg"} and back["test_neg"].dtype == np.float32
        assert np.allclose(back["valid_pos"], s["valid_pos"].numpy())


def toy_graph(n=400, k=8, seed=0):
    """Two-community graph whose node features reveal the community."""
    g = torch.Generator().manual_seed(seed)
    comm = torch.arange(n) % 2
    u = torch.randint(0, n, (n * k,), generator=g)
    v = (u + 2 * torch.randint(1, 20, (n * k,), generator=g)) % n   # same parity = same community
    pairs = torch.unique(torch.sort(torch.stack([u, v], 1), 1).values, dim=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    x = torch.nn.functional.one_hot(comm, 2).float() + 0.05 * torch.randn(n, 2, generator=g)
    return n, x, pairs


def check_decoder_learns():
    set_seed(0)
    n, x, pairs = toy_graph()
    perm = torch.randperm(pairs.size(0))
    tr, va, te = pairs[perm[:-200]], pairs[perm[-200:-100]], pairs[perm[-100:]]
    neg = uniform_negatives(400, n, "cpu")
    neg = neg[(neg[:, 0] % 2) != (neg[:, 1] % 2)]       # cross-community = true non-edges
    ev = {"valid_pos": va, "valid_neg": neg[:100], "test_pos": te, "test_neg": neg[100:200]}
    cfg = TrainConfig(lr=0.01, batch_size=512, max_epochs=30, patience=10, device="cpu", log_every=1000)
    res = train_link_predictor(FrozenEncoder(x), LinkDecoder(2, 32), tr, n, ev, cfg, log=lambda m: None)
    assert res.test["hits@20"] > 0.9, res.test
    assert res.scores["test_pos"].shape == (100,)


def check_walker():
    n, _, pairs = toy_graph()
    ei = torch.cat([pairs.t(), pairs.flip(1).t()], 1)
    for p, q in [(1, 1), (0.5, 2), (2, 0.5)]:
        w = Walker(ei, n, p, q, np.random.default_rng(0))
        W = w.walk(np.repeat(np.arange(n), 3), 12)
        assert w._has_edge(W[:, :-1].ravel(), W[:, 1:].ravel()).all(), (p, q)
    assert windows(torch.from_numpy(W[:2]), 5).shape == (2 * 8, 5)


def main():
    for f in (check_hits, check_scores_roundtrip, check_decoder_learns, check_walker):
        f()
        print(f"ok  {f.__name__}")


if __name__ == "__main__":
    main()
