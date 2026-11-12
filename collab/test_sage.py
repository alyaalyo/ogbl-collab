"""Toy-data checks for collab/sage.py (no download).

    python -m unittest -v collab.test_sage
"""
import unittest
from unittest import mock

import torch

from .decoder import LinkDecoder, TrainConfig, train_link_predictor, uniform_negatives
from .sage import SAGEEncoder, score_with_graph
from .selftest import toy_graph
from .utils import set_seed


def _sym(pairs):
    return torch.cat([pairs.t(), pairs.flip(1).t()], dim=1)


class SageTests(unittest.TestCase):
    def test_shape_and_checkpoint_excludes_inputs(self):
        n, x, pairs = toy_graph()
        enc = SAGEEncoder(x, _sym(pairs), hidden=16, num_layers=3)
        self.assertEqual(enc().shape, (n, 16))
        self.assertEqual(len(enc.convs), 3)
        self.assertFalse({"x", "edge_index"} & set(enc.state_dict()))

    def test_set_graph_changes_output_not_weights(self):
        _, x, pairs = toy_graph()
        enc = SAGEEncoder(x, _sym(pairs[: pairs.size(0) // 2]), hidden=8).eval()
        weights = {k: v.clone() for k, v in enc.state_dict().items()}
        z0 = enc().clone()
        enc.set_graph(_sym(pairs))
        self.assertFalse(torch.allclose(z0, enc()))
        for k, v in enc.state_dict().items():
            self.assertTrue(torch.equal(v, weights[k]))

    def test_trains_and_rescores_on_another_graph(self):
        set_seed(0)
        n, x, pairs = toy_graph()
        perm = torch.randperm(pairs.size(0))
        tr, va, te = pairs[perm[:-200]], pairs[perm[-200:-100]], pairs[perm[-100:]]
        neg = uniform_negatives(400, n, "cpu")
        neg = neg[(neg[:, 0] % 2) != (neg[:, 1] % 2)]
        ev = {"valid_pos": va, "valid_neg": neg[:100], "test_pos": te, "test_neg": neg[100:200]}
        enc, dec = SAGEEncoder(x, _sym(tr), hidden=32), LinkDecoder(32, 32)
        cfg = TrainConfig(lr=0.01, batch_size=512, max_epochs=30, patience=10, device="cpu", log_every=1000)
        res = train_link_predictor(enc, dec, tr, n, ev, cfg, log=lambda m: None)
        self.assertGreater(res.test["hits@20"], 0.9)
        # after training enc/dec hold the best-on-val weights; same graph must reproduce the saved scores
        again = score_with_graph(enc, dec, _sym(tr), ev, ("test_pos", "test_neg"))
        self.assertTrue(torch.allclose(again["test_pos"], res.scores["test_pos"], atol=1e-5))
        other = score_with_graph(enc, dec, _sym(torch.cat([tr, va])), ev, ("test_pos", "test_neg"))
        self.assertEqual(other["test_pos"].shape, (100,))

    def test_grad_clip_is_applied_to_encoder_and_decoder_separately(self):
        set_seed(0)
        n, x, pairs = toy_graph()
        neg = uniform_negatives(300, n, "cpu")
        ev = {"valid_pos": pairs[:50], "valid_neg": neg[:100], "test_pos": pairs[50:100], "test_neg": neg[100:200]}
        enc, dec = SAGEEncoder(x, _sym(pairs), hidden=8), LinkDecoder(8, 8)
        cfg = TrainConfig(lr=0.01, batch_size=100000, max_epochs=2, device="cpu", grad_clip=1.0, log_every=1000)
        with mock.patch("torch.nn.utils.clip_grad_norm_") as clip:
            train_link_predictor(enc, dec, pairs, n, ev, cfg, log=lambda m: None)
        self.assertEqual(clip.call_count, 2 * 2)          # 2 epochs x 1 step x (decoder, encoder)
        self.assertTrue(all(c.args[1] == 1.0 for c in clip.call_args_list))


if __name__ == "__main__":
    unittest.main()
