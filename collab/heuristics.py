"""Deterministic CN and Adamic-Adar on the shared training-only graph.

    python -m collab.heuristics

Only requested pairs are scored, in their original OGB order. Batched sparse
row intersections avoid materialising A**2 or a dense [num_nodes, num_nodes]
matrix. Both methods share the intersections and require no training or GPU.
"""
import argparse
import importlib.metadata
import platform

import numpy as np
from scipy import sparse
import torch

from .data import load_collab
from .evaluate import evaluate_scores, save_scores
from .utils import Timer, append_result


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class HeuristicScorer:
    """CN(u,v) and sum(1/log(deg(w))) over common neighbours w.

    The adjacency is undirected, binary and loop-free even if edge_index
    contains duplicate edges or only one direction of an edge.
    """

    def __init__(self, edge_index, num_nodes):
        edges = _numpy(edge_index)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        adjacency = sparse.csr_matrix(
            (np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])),
            shape=(num_nodes, num_nodes),
        )
        adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        adjacency.data.fill(1)
        adjacency.sort_indices()
        self.adjacency = adjacency
        self.degree = np.diff(adjacency.indptr)
        self.aa_weights = np.zeros(num_nodes, dtype=np.float64)
        mask = self.degree > 1
        self.aa_weights[mask] = 1.0 / np.log(self.degree[mask])

    def score(self, pairs, batch_size=4096):
        """Return float32 score arrays; batch size only changes memory use.

        Degree-0/1 vertices have zero AA weight: they cannot be a common
        neighbour of two distinct endpoints in a simple undirected graph.
        """
        pairs = _numpy(pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [M, 2]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cn = np.empty(len(pairs), dtype=np.float32)
        aa = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            common = self.adjacency[batch[:, 0]].multiply(self.adjacency[batch[:, 1]])
            cn[start:start + len(batch)] = np.asarray(common.sum(axis=1)).ravel()
            aa[start:start + len(batch)] = common @ self.aa_weights
        return {"cn": cn, "aa": aa}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="evaluation pairs per sparse batch (default: 4096)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")

    with Timer() as preparation:
        data = load_collab()
        scorer = HeuristicScorer(data.edge_index, data.num_nodes)
    graph = {
        "num_nodes": data.num_nodes,
        "unique_train_edges": int(scorer.adjacency.nnz // 2),
        "isolated_nodes": int(np.count_nonzero(scorer.degree == 0)),
        "input": "train_only",
    }
    print(f"Graph: {graph}", flush=True)
    scores = {"cn": {}, "aa": {}}
    split_seconds = {}
    with Timer() as scoring:
        for split, pairs in data.eval_pairs.items():
            with Timer() as elapsed:
                values = scorer.score(pairs, args.batch_size)
            split_seconds[split] = round(elapsed.seconds, 3)
            for method, values_for_method in values.items():
                scores[method][split] = values_for_method
            print(f"{split}: {len(pairs):,} pairs, {elapsed.seconds:.2f}s (CN + AA)", flush=True)

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "torch", "torch-geometric", "ogb", "pandas")},
    }
    for method, method_scores in scores.items():
        # Evaluate exactly the float32 arrays that downstream analysis loads.
        metrics = evaluate_scores(method_scores)
        save_scores(method, 0, method_scores)
        append_result(method, {
            "method": method,
            "seed": 0,
            "device": "cpu",
            "hparams": {"batch_size": args.batch_size},
            "graph": graph,
            "split_sizes": {split: len(pairs) for split, pairs in data.eval_pairs.items()},
            "preparation_seconds": round(preparation.seconds, 3),
            "shared_scoring_seconds": round(scoring.seconds, 3),
            "shared_split_seconds": split_seconds,
            "environment": environment,
            **metrics,
        })
        print(f"{method}: valid {metrics['valid']}  test {metrics['test']}", flush=True)


if __name__ == "__main__":
    main()
