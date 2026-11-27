"""Checkpoint resumption must preserve the selected model and stopping rule."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from .decoder import FrozenEncoder, LinkDecoder, TrainConfig, train_link_predictor
from .selftest import toy_graph
from .utils import set_seed
from .node2vec import train_node2vec


class ResumeTests(unittest.TestCase):
    def run_model(self, cfg):
        set_seed(17)
        n, x, pairs = toy_graph(n=60, k=3)
        ev = {"valid_pos": pairs[:20], "valid_neg": torch.randint(0, n, (100, 2)),
              "test_pos": pairs[20:40], "test_neg": torch.randint(0, n, (100, 2))}
        return train_link_predictor(FrozenEncoder(x), LinkDecoder(2, 8), pairs[40:], n, ev, cfg,
                                     log=lambda message: None)

    def test_resume_matches_uninterrupted_training(self):
        base = TrainConfig(max_epochs=4, batch_size=64, patience=100, ckpt_every=1, log_every=100)
        reference = self.run_model(base)
        with tempfile.TemporaryDirectory() as root:
            cfg = replace(base, ckpt_path=str(Path(root) / "resume.pt"))
            self.run_model(replace(cfg, max_epochs=2))
            resumed = self.run_model(cfg)
        self.assertEqual(resumed.epochs_run, reference.epochs_run)
        self.assertEqual(resumed.best_epoch, reference.best_epoch)
        for split in reference.scores:
            self.assertTrue(torch.equal(resumed.scores[split], reference.scores[split]))

    def test_resume_after_early_stop_does_not_train_extra_epoch(self):
        # lr=0 guarantees no gain after the first validation check.
        with tempfile.TemporaryDirectory() as root:
            cfg = TrainConfig(lr=0, max_epochs=10, batch_size=64, patience=1, ckpt_every=1,
                              ckpt_path=str(Path(root) / "stopped.pt"), log_every=100)
            first = self.run_model(cfg)
            resumed = self.run_model(cfg)
        self.assertEqual(first.epochs_run, 2)
        self.assertEqual(resumed.epochs_run, 2)
        self.assertEqual(len(first.history), len(resumed.history))
        for split in first.scores:
            self.assertTrue(torch.equal(first.scores[split], resumed.scores[split]))

    def test_node2vec_resumes_inside_epoch_exactly(self):
        # Ring graph; five batches per epoch. Interrupt after the second batch.
        n = 20
        u = torch.arange(n)
        v = (u + 1) % n
        edges = torch.stack([torch.cat([u, v]), torch.cat([v, u])])
        kwargs = dict(dim=8, walk_length=8, context_size=3, walks_per_node=2,
                      epochs=2, batch_nodes=4, seed=7, ckpt_every=2)
        expected, _ = train_node2vec(edges, n, **kwargs, log=lambda message: None)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "skipgram.pt"

            def interrupt(message):
                if message.startswith("node2vec checkpoint"):
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_node2vec(edges, n, **kwargs, ckpt_path=path, log=interrupt)
            actual, _ = train_node2vec(edges, n, **kwargs, ckpt_path=path, resume=True,
                                      log=lambda message: None)
            self.assertTrue(torch.equal(expected, actual))
            complete, _ = train_node2vec(edges, n, **kwargs, ckpt_path=path, resume=True,
                                        log=lambda message: None)
            self.assertTrue(torch.equal(actual, complete))
            with self.assertRaisesRegex(ValueError, "configuration/graph mismatch"):
                train_node2vec(edges.flip(1), n, **kwargs, ckpt_path=path, resume=True,
                               log=lambda message: None)


if __name__ == "__main__":
    unittest.main()
