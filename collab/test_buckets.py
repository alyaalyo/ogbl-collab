"""Protocol tests for CN+AA, test-only valinput and bucket/overlap reports."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import buckets, heuristics
from .data import _symmetric
from .evaluate import hits, load_scores, save_scores


def toy_data():
    # Triangle, a tail and an isolated node. Validation adds an edge to node 4.
    train = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3]])
    pos = torch.tensor([[0, 1], [0, 3], [0, 4], [1, 2]])
    neg = torch.tensor([[3, 4]]).repeat(100, 1)
    return SimpleNamespace(num_nodes=5, train_pos=train, edge_index=_symmetric(train),
                           edge_index_val=_symmetric(torch.cat([train, torch.tensor([[3, 4]])])),
                           eval_pairs={"valid_pos": pos, "valid_neg": neg, "test_pos": pos, "test_neg": neg})


class BucketTests(unittest.TestCase):
    def test_partition_boundaries_and_repeat_orientation(self):
        # Endpoints 0/1 share five neighbours, 2/3 share four, plus an isolated 20.
        edges = [(u, v) for u in (0, 1) for v in range(5, 10)]
        edges += [(u, v) for u in (2, 3) for v in range(10, 14)]
        edges += [(0, 1), (14, 15), (14, 16), (15, 16)]
        pairs = np.array([[0, 1], [1, 0], [2, 3], [14, 15], [20, 0]])
        labels, meta = buckets.make_buckets(np.array(edges).T, 21, pairs)
        np.testing.assert_array_equal(labels["common_neighbors"], [">=5", ">=5", "2-4", "1", "0"])
        np.testing.assert_array_equal(labels["min_degree"], [">5", ">5", "1-5", "1-5", "0"])
        np.testing.assert_array_equal(labels["pair_history"], ["repeat", "repeat", "new", "repeat", "new"])
        np.testing.assert_array_equal(meta["pairs"], pairs)
        for partition in buckets.BUCKETS:
            covered = sum(mask.astype(int) for p, _, mask in buckets.masks(labels) if p == partition)
            np.testing.assert_array_equal(covered, np.ones(len(pairs)))

    def test_global_threshold_ties_empty_buckets_and_weighted_total(self):
        data = toy_data()
        labels, _ = buckets.make_buckets(data.edge_index, 5, data.eval_pairs["test_pos"])
        pos, neg = np.array([50, 51, 100, -1], dtype=np.float32), np.arange(100, dtype=np.float32)
        hit, threshold = buckets.hit_mask(pos, neg)
        self.assertEqual(threshold, 50)
        np.testing.assert_array_equal(hit, [False, True, True, False])
        self.assertEqual(hit.mean(), hits(pos, neg)["hits@50"])
        rows = buckets.bucket_rows("toy", 0, labels, hit, threshold)
        for partition in buckets.BUCKETS:
            selected = [r for r in rows if r["partition"] == partition]
            self.assertEqual(sum(r["size"] for r in selected), len(pos))
            self.assertEqual(sum(r["hits"] for r in selected) / len(pos), hit.mean())
            self.assertTrue(all(r["threshold"] == 50 for r in selected))
        self.assertTrue(any(r["hits50"] is None and r["size"] == 0 for r in rows))

    def test_sample_std_and_intersection_identities(self):
        group = dict(method="toy", partition="all", bucket="all", size=4, share=1)
        result = buckets.aggregate([dict(group, seed=0, hits50=0.25), dict(group, seed=1, hits50=0.75)])[0]
        self.assertEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["std"], np.std([0.25, 0.75], ddof=1))
        left, right = np.array([1, 1, 0, 0], dtype=bool), np.array([1, 0, 1, 0], dtype=bool)
        overlap = buckets.overlap_masks(left, right)
        np.testing.assert_array_equal(sum(v.astype(int) for v in overlap.values()), [1, 1, 1, 1])
        np.testing.assert_array_equal(overlap["left_only"], [False, True, False, False])
        np.testing.assert_array_equal(overlap["both"] | overlap["right_only"], right)

    def test_rank_encoding_matches_lexicographic_evaluation_after_save(self):
        # Tiny AA differences and huge secondary scores must not defeat primary CN.
        cn = {"test_pos": np.array([1, 1, 2, 0], dtype=np.float32), "test_neg": np.ones(100, dtype=np.float32)}
        aa = {"test_pos": np.array([1, np.nextafter(np.float32(1), np.float32(2)), 0, 1e10]),
              "test_neg": np.ones(100, dtype=np.float32)}
        encoded = heuristics.tie_break_scores(cn, aa)
        with tempfile.TemporaryDirectory() as root:
            save_scores("cn_aa", 0, encoded, root)
            saved = load_scores("cn_aa", 0, root)
            mask, _ = buckets.hit_mask(saved["test_pos"], saved["test_neg"])
            np.testing.assert_array_equal(mask, [False, True, True, False])
            self.assertEqual(hits(saved["test_pos"], saved["test_neg"])["hits@50"], 0.5)
        rng = np.random.default_rng(42)
        cn = {"test_pos": rng.integers(0, 6, 50), "test_neg": rng.integers(0, 6, 100)}
        aa = {key: rng.choice([0.0, 0.1, 0.3, 0.7], len(value)) for key, value in cn.items()}
        encoded = heuristics.tie_break_scores(cn, aa)
        threshold_tuple = sorted(zip(cn["test_neg"], aa["test_neg"]))[-50]
        expected = [pair > threshold_tuple for pair in zip(cn["test_pos"], aa["test_pos"])]
        np.testing.assert_array_equal(buckets.hit_mask(encoded["test_pos"], encoded["test_neg"])[0], expected)

    def test_valinput_never_scores_validation(self):
        data = toy_data()
        # If validation is visited, it cannot be scored and this test fails.
        data.eval_pairs["valid_pos"] = None
        data.eval_pairs["valid_neg"] = None
        with tempfile.TemporaryDirectory() as root, patch.object(heuristics, "load_collab", return_value=data), \
                patch.object(heuristics, "append_result") as record, \
                patch.object(heuristics, "save_scores", side_effect=lambda m, s, v: save_scores(m, s, v, root)):
            heuristics.main(["--valinput", "--tie_break"])
            for method in ("cn_valinput", "aa_valinput", "cn_aa_valinput"):
                self.assertEqual(set(load_scores(method, 0, root)), {"test_pos", "test_neg"})
            for call in record.call_args_list:
                self.assertNotIn("valid", call.args[1])
                self.assertEqual(call.args[1]["graph"]["input_edges"], 5)

    def test_bad_scores_stale_metrics_and_sage_gate(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "toy_seed0_test_pos.npy"
            for bad in (np.array([np.nan], dtype=np.float32), np.array([1.0], dtype=np.float64),
                        np.array([[1.0]], dtype=np.float32)):
                np.save(path, bad)
                with self.assertRaises(ValueError):
                    buckets.load_test_scores("toy", 0, root, {"pos": 1, "neg": 100})
            np.save(path, np.ones(1, dtype=np.float32))
            with self.assertRaises(FileNotFoundError):
                buckets.load_test_scores("toy", 0, root, {"pos": 1, "neg": 100})
        with self.assertRaises(ValueError):
            buckets.check_record("toy", 0, {"hits@50": 0.5}, {("toy", 0): {"test": {"hits@50": 0.1}}})
        buckets.check_sage_reproduction("sage", [0.4813])
        buckets.check_sage_reproduction("sage_valinput", [0.5469])
        with self.assertRaises(ValueError):
            buckets.check_sage_reproduction("sage", [0.3])

    def test_cli_reports_missing_methods_and_writes_pair_indices(self):
        data = toy_data()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            score_root, result_root = root / "scores", root / "results"
            result_root.mkdir()
            for method in ("aa", "cn"):
                save_scores(method, 0, {"test_pos": np.array([1, 0, 2, 0], dtype=np.float32),
                                        "test_neg": np.zeros(100, dtype=np.float32)}, score_root)
            with patch.object(buckets, "load_collab", return_value=data):
                buckets.main(["--scores_dir", str(score_root), "--results_dir", str(result_root),
                              "--methods", "aa", "cn", "sage"])
            output = result_root / "buckets"
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn({"method": "sage", "seeds": "-", "status": "missing scores"}, manifest["status"])
            with (output / "aggregate.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["method"] for r in rows}, {"aa", "cn"})
            with np.load(score_root / "buckets" / "aa_seed0__cn_seed0.npz") as membership:
                np.testing.assert_array_equal(membership["both"], [0, 2])
                np.testing.assert_array_equal(membership["neither"], [1, 3])


if __name__ == "__main__":
    unittest.main()
