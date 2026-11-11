"""Independent correctness checks, no dataset download: python -m unittest collab.test_heuristics."""
import math
import tempfile
import unittest

import numpy as np
import torch

from .data import _symmetric, _unique_undirected
from .evaluate import evaluate_scores, load_scores, save_scores
from .heuristics import HeuristicScorer


class HeuristicTests(unittest.TestCase):
    def test_known_values_and_graph_normalization(self):
        # N(0) intersect N(1) = {2, 3}; deg(2)=3, deg(3)=2.
        # Vertex 4 has degree 1 and vertex 5 is isolated.
        edges = np.array([[0, 0, 1, 1, 2, 0, 2, 5],
                          [2, 3, 2, 3, 4, 2, 0, 5]])
        scorer = HeuristicScorer(edges, 6)
        # Keep diagonal queries too: the official OGB negatives contain one.
        pairs = np.array([[0, 1], [1, 0], [0, 5], [0, 4], [2, 4], [0, 0]])
        scores = scorer.score(pairs, batch_size=2)
        np.testing.assert_array_equal(scorer.degree, [2, 2, 3, 2, 1, 0])
        np.testing.assert_array_equal(scores["cn"], [2, 2, 0, 1, 0, 2])
        expected = 1 / math.log(3) + 1 / math.log(2)
        np.testing.assert_allclose(scores["aa"],
                                   [expected, expected, 0, 1 / math.log(3), 0, expected], rtol=1e-6)
        self.assertTrue(np.isfinite(scores["aa"]).all())

    def test_random_graph_against_set_reference(self):
        rng = np.random.default_rng(17)
        n = 30
        edges = np.array([(u, v) for u in range(n - 1) for v in range(u + 1, n - 1)
                          if rng.random() < 0.18], dtype=np.int64).T
        neighbours = [set() for _ in range(n)]
        for u, v in edges.T:
            neighbours[u].add(v)
            neighbours[v].add(u)
        pairs = np.array([(u, v) for u in range(n) for v in range(n) if u != v])
        rng.shuffle(pairs)  # score order must follow queries, not graph order
        common = [neighbours[u] & neighbours[v] for u, v in pairs]
        expected_cn = [len(shared) for shared in common]
        expected_aa = [sum(1 / math.log(len(neighbours[w])) for w in sorted(shared))
                       for shared in common]
        scorer = HeuristicScorer(edges, n)
        for batch_size in (1, 37, 4096):
            scores = scorer.score(pairs, batch_size)
            np.testing.assert_array_equal(scores["cn"], expected_cn)
            np.testing.assert_allclose(scores["aa"], expected_aa, rtol=1e-6)
            self.assertEqual(scores["cn"].dtype, np.float32)
            self.assertEqual(scores["aa"].dtype, np.float32)

    def test_empty_graph_and_queries(self):
        scorer = HeuristicScorer(np.empty((2, 0), dtype=np.int64), 3)
        for values in scorer.score(np.array([[0, 1], [1, 2]])).values():
            np.testing.assert_array_equal(values, [0, 0])
        for values in scorer.score(np.empty((0, 2), dtype=np.int64)).values():
            self.assertEqual(values.shape, (0,))
        with self.assertRaises(ValueError):
            scorer.score(np.array([[0, 1]]), batch_size=0)

    def test_shared_graph_helpers(self):
        pairs = torch.tensor([[2, 0], [0, 2], [1, 1], [3, 2]])
        unique = _unique_undirected(pairs)
        self.assertTrue(torch.equal(unique, torch.tensor([[0, 2], [2, 3]])))
        scorer = HeuristicScorer(_symmetric(unique), 5)
        np.testing.assert_array_equal(scorer.degree, [1, 0, 2, 1, 0])
        np.testing.assert_array_equal(scorer.score(torch.tensor([[0, 3]]))["cn"], [1])

    def test_saved_scores_and_official_strict_threshold(self):
        # 100 negative scores all equal 1 (or 1/log(2) for AA).
        # One positive ties, one beats them, one is an isolated pair.
        edges = np.array([[0, 1, 3, 4, 3, 4], [2, 2, 5, 5, 6, 6]])
        scorer = HeuristicScorer(edges, 8)
        positives = scorer.score(np.array([[0, 1], [3, 4], [0, 7]]))
        negatives = scorer.score(np.tile([0, 1], (100, 1)))
        with tempfile.TemporaryDirectory() as root:
            for method in ("cn", "aa"):
                scores = {f"{split}_{kind}": values[method]
                          for split in ("valid", "test")
                          for kind, values in (("pos", positives), ("neg", negatives))}
                save_scores(method, 0, scores, root=root)
                restored = load_scores(method, 0, root=root)
                for split in scores:
                    np.testing.assert_array_equal(restored[split], scores[split])
                metrics = evaluate_scores(restored)
                for split in ("valid", "test"):
                    for k in (20, 50, 100):
                        self.assertAlmostEqual(metrics[split][f"hits@{k}"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
