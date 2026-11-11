"""Loading ogbl-collab into the tensors every method shares.

Conventions (from the proposal):
  * Input graph A = all training edges (years <= 2017). A pair that appears in
    several years becomes ONE unweighted undirected edge.
  * "val as input" variant: A plus the 2018 validation positives.
  * Evaluation pairs are the official OGB ones, in the official order. Score
    files (see evaluate.py) follow this order, so any method's scores can be
    joined row by row with any other method's.
"""
from dataclasses import dataclass

import torch

from .utils import DATA_DIR


@dataclass
class CollabData:
    num_nodes: int
    x: torch.Tensor               # [N, 128] text features
    train_pos: torch.Tensor       # [E_train, 2] unique undirected pairs, u < v (967,632)
    edge_index: torch.Tensor      # [2, 2*E_train] symmetric, from train_pos (input graph A)
    edge_index_val: torch.Tensor  # [2, ...] symmetric, A + 2018 edges ("val as input")
    eval_pairs: dict              # {"valid_pos","valid_neg","test_pos","test_neg"} -> [M, 2]


def _unique_undirected(pairs: torch.Tensor) -> torch.Tensor:
    pairs = torch.sort(pairs, dim=1).values
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # drop self-loops (none expected)
    return torch.unique(pairs, dim=0)


def _symmetric(pairs: torch.Tensor) -> torch.Tensor:
    return torch.cat([pairs.t(), pairs.flip(1).t()], dim=1)


def load_collab(root=DATA_DIR) -> CollabData:
    from ogb.linkproppred import PygLinkPropPredDataset

    dataset = PygLinkPropPredDataset("ogbl-collab", root=str(root))
    graph = dataset[0]
    split = dataset.get_edge_split()

    train_pos = _unique_undirected(split["train"]["edge"])
    with_val = _unique_undirected(torch.cat([train_pos, split["valid"]["edge"]], dim=0))

    return CollabData(
        num_nodes=graph.num_nodes,
        x=graph.x,
        train_pos=train_pos,
        edge_index=_symmetric(train_pos),
        edge_index_val=_symmetric(with_val),
        eval_pairs={
            "valid_pos": split["valid"]["edge"],
            "valid_neg": split["valid"]["edge_neg"],
            "test_pos": split["test"]["edge"],
            "test_neg": split["test"]["edge_neg"],
        },
    )
