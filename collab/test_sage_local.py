"""Sparse CPU SAGE must preserve message direction, gradients and analysis gates."""
from dataclasses import replace
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from .buckets import check_sage_reproduction
from .sage import SAGEEncoder, score_with_graph
from .sage_local import refresh_reports, sparse_graph
from . import sage_local
from .evaluate import load_scores, save_scores
from .decoder import LinkDecoder, TrainConfig, train_link_predictor


class LocalSageTests(unittest.TestCase):
    def test_sparse_matches_coo_outputs_and_gradients(self):
        torch.manual_seed(7)
        # Directed graph and an isolated node detect a mistaken transpose or degree handling.
        edges = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 1]])
        x = torch.randn(5, 4)
        coo = SAGEEncoder(x.clone().requires_grad_(), edges, hidden=8)
        csr = SAGEEncoder(x.clone().requires_grad_(), sparse_graph(edges, 5), hidden=8)
        csr.load_state_dict(coo.state_dict())
        actual, expected = csr(), coo()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(csr.x.grad, coo.x.grad, rtol=1e-5, atol=1e-6)
        for a, b in zip(csr.parameters(), coo.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)

    def test_sparse_valinput_matches_coo_and_preserves_weights(self):
        torch.manual_seed(9)
        x = torch.randn(5, 4)
        initial = torch.tensor([[0, 1], [1, 0]])
        expanded = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        enc = SAGEEncoder(x, sparse_graph(initial, 5), hidden=8)
        coo = SAGEEncoder(x, expanded, hidden=8)
        weights = {k: v.clone() for k, v in enc.state_dict().items()}
        coo.load_state_dict(weights)
        decoder = LinkDecoder(8, 8)
        pairs = {"test_pos": torch.tensor([[0, 1], [2, 4]]), "valid_pos": None}
        actual = score_with_graph(enc, decoder, sparse_graph(expanded, 5), pairs, ("test_pos",))
        expected = score_with_graph(coo, decoder, expanded, pairs, ("test_pos",))
        torch.testing.assert_close(actual["test_pos"], expected["test_pos"], rtol=1e-5, atol=1e-6)
        for key, value in enc.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))

    def test_local_names_cannot_bypass_reproduction_gate(self):
        for name, value in (("sage_local", .481), ("sage_local_valinput", .5463)):
            check_sage_reproduction(name, [value])
            with self.assertRaises(ValueError):
                check_sage_reproduction(name, [.3])

    def test_reports_wait_for_both_reproduction_checks(self):
        records = {"sage_local": {"test": {"hits@50": .481}},
                   "sage_local_valinput": {"test": {"hits@50": .3}}}
        with patch("collab.sage_local.subprocess.run") as run:
            self.assertTrue(refresh_reports(records))
            self.assertEqual(run.call_count, 1)  # raw summary only, no buckets
        records["sage_local_valinput"]["test"]["hits@50"] = .5463
        with patch("collab.sage_local.subprocess.run") as run:
            self.assertEqual(refresh_reports(records), [])
            self.assertEqual([c.args[0][2:] for c in run.call_args_list],
                             [["collab.summarize"], ["collab.buckets"], ["collab.buckets", "--valinput"]])

    def test_sparse_training_resumes_exactly(self):
        def run(cfg):
            torch.manual_seed(12)
            x = torch.randn(12, 4)
            nodes = torch.arange(12)
            edges = torch.stack([nodes, (nodes + 1) % 12])
            pairs = edges.t()
            ev = {"valid_pos": pairs[:4], "valid_neg": torch.randint(0, 12, (100, 2)),
                  "test_pos": pairs[4:8], "test_neg": torch.randint(0, 12, (100, 2))}
            return train_link_predictor(SAGEEncoder(x, sparse_graph(edges, 12), hidden=8),
                                         LinkDecoder(8, 8), pairs, 12, ev, cfg, log=lambda message: None)

        cfg = TrainConfig(max_epochs=4, patience=100, batch_size=6, ckpt_every=1, grad_clip=1.0)
        expected = run(cfg)
        with tempfile.TemporaryDirectory() as root:
            cfg = replace(cfg, ckpt_path=str(Path(root) / "sage.pt"))
            run(replace(cfg, max_epochs=2))
            actual = run(cfg)
        self.assertEqual(actual.best_epoch, expected.best_epoch)
        self.assertEqual(actual.epochs_run, expected.epochs_run)
        for split in actual.scores:
            self.assertTrue(torch.equal(actual.scores[split], expected.scores[split]))

    def test_runner_saves_six_scores_then_skips_completed_training(self):
        torch.manual_seed(5)
        nodes = torch.arange(12)
        edges = torch.stack([nodes, (nodes + 1) % 12])
        pairs = edges.t()
        data = SimpleNamespace(num_nodes=12, x=torch.randn(12, 4), edge_index=edges,
                               edge_index_val=torch.cat([edges, edges.flip(0)], dim=1), train_pos=pairs,
                               eval_pairs={"valid_pos": pairs[:4], "test_pos": pairs[4:8],
                                           "valid_neg": torch.randint(0, 12, (100, 2)),
                                           "test_neg": torch.randint(0, 12, (100, 2))})
        with tempfile.TemporaryDirectory() as root, ExitStack() as stack:
            root = Path(root)
            for folder in ("logs", "results", "scores"):
                (root / folder).mkdir()

            def append(name, record):
                with (root / "results" / f"{name}.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

            for attribute, value in (("ROOT", root), ("CKPT_DIR", root / "checkpoints"),
                                     ("RESULTS_DIR", root / "results")):
                stack.enter_context(patch.object(sage_local, attribute, value))
            stack.enter_context(patch.object(sage_local, "append_result", side_effect=append))
            stack.enter_context(patch.object(sage_local, "save_scores", side_effect=lambda m, s, v: save_scores(m, s, v, root / "scores")))
            stack.enter_context(patch.object(sage_local, "load_scores", side_effect=lambda m, s: load_scores(m, s, root / "scores")))
            stack.enter_context(patch.object(sage_local, "TrainConfig", side_effect=lambda **kw: replace(TrainConfig(**kw), max_epochs=2)))
            stack.enter_context(patch.object(sage_local, "refresh_reports", return_value=[]))
            status = root / "logs/sage_local_status.json"
            sage_local.train(data, status)
            self.assertEqual(len(list((root / "scores").glob("*.npy"))), 6)
            self.assertEqual(json.loads(status.read_text())["state"], "complete")
            with patch.object(sage_local, "train_link_predictor", side_effect=AssertionError("must skip")):
                sage_local.train(data, status)
                np.save(root / "scores/sage_local_seed0_test_pos.npy", np.zeros(1, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "malformed"):
                    sage_local.train(data, status)


if __name__ == "__main__":
    unittest.main()
