# Midway check (12 Nov 2026): role 2

Scope of role 2: shared decoder, node2vec, features-only MLP, repository and
environment. Roles 1 (heuristics, buckets) and 3 (GraphSAGE) report separately.

## Done

- **Repository skeleton**: `collab/` package, `Makefile` one-command entry points
  (`make env | check | midway | all`), README with conventions and a Colab recipe.
- **Pinned environment**: `requirements.txt` (direct pins: torch 2.5.1, torch_geometric 2.6.1,
  ogb 1.3.6, numpy 1.26.4, pandas 3.0.6) and `requirements-lock.txt` (full freeze), Python 3.11.
  torch is held below 2.6 on purpose: ogb 1.3.6 calls `torch.load` without
  `weights_only=False`, which breaks under torch ≥ 2.6.
- **Shared data loader** (`collab/data.py`): input graph A = 1,179,052 training
  records → 967,632 unique undirected edges; variant with 2018 edges for "val as input";
  official eval pairs. Role 1 may refine this, but the tensors' meaning is fixed.
- **Evaluator wrapper and score format** (`collab/evaluate.py`): official OGB Hits@20/50/100;
  `scores/{method}_seed{k}_{split}.npy`, split ∈ {valid_pos, valid_neg, test_pos, test_neg}.
- **Shared decoder and training loop** (`collab/decoder.py`), ready for role 3:
  MLP(z_u ⊙ z_v) (3 layers, 256), BCE vs. as many uniform random pairs, epoch cap 400,
  early stopping after 20 evals without val Hits@50 gain, test scored once with the
  best-on-val weights, checkpoint/resume every 10 epochs (point `COLLAB_CKPT` at Google Drive).
  Any encoder whose `forward()` returns `[N, d]` plugs in; trainable ones are trained end-to-end.
- **Features-only MLP** (`collab/mlp.py`), seed 0 run.
- **node2vec** (`collab/node2vec.py`): own vectorised biased walker (p/q by rejection
  sampling, like torch_cluster) + skip-gram with negative sampling, then frozen vectors +
  shared decoder. Walk correctness and the p/q bias are checked in `make check`.
- **Self-tests** (`make check`): strict-">" Hits@K, score round-trip, decoder learns a toy graph, walk validity.

## Preliminary results (seed 0, Hits × 100)

| method | config | val @20 | val @50 | val @100 | test @20 | test @50 | test @100 | time |
|---|---|---|---|---|---|---|---|---|
| features-only MLP | lr 0.01, no dropout | 9.22 | 18.28 | 26.76 | 6.26 | 14.24 | 21.81 | 162 epochs × 14.4 s (CPU, shared with node2vec) |
| node2vec | p=q=1, 1 skip-gram epoch | 32.09 | 41.27 | 47.36 | 27.29 | 35.35 | 40.92 | 5.6 min emb. + 52 dec. epochs × 3.5 s |
| node2vec | p=q=1, 3 skip-gram epochs | 41.03 | 57.18 | 64.15 | 33.30 | **47.72** | 54.29 | 15.6 min emb. + 47 dec. epochs × 3.4 s |

Machine: Apple M-series laptop (MPS for node2vec, CPU for the MLP), not the T4 the final runs target.
OGB leaderboard references: MLP 19.27 ± 1.29, node2vec 48.88 ± 0.54, GraphSAGE 48.10 ± 0.81
(test Hits@50). The leaderboard node2vec concatenates text features with the
embeddings; ours uses the embeddings alone, as in the proposal, so it is not a like-for-like
reproduction target.

Observations so far:
- Skip-gram epochs matter much more than we assumed: 1 → 3 epochs moved test Hits@50 from
  35.4 to 47.7 and the skip-gram loss was still falling (2.80 → 0.82). The number of
  epochs has to join p, q in the grid, and the epoch budget has to be fixed before tuning p, q.
- The features-only MLP (14.2) is below the OGB MLP (19.27). Two known differences: OGB puts
  an MLP encoder on x before the predictor (more capacity), and our run early-stopped at
  epoch 162 while validation was still creeping up (14.4 at epoch 100 → 18.3 at epoch 142).
  Patience 20 is probably too tight for lr 0.01 on this baseline; check before the 10 seeds.
- Text features alone are far weaker than graph structure (14 vs 48 test Hits@50), which is
  what the baseline is for: it separates the effect of text from that of the graph.
- Val–test gap is ~10 points for node2vec (2018 vs 2019 pairs); expected for a temporal split,
  worth a sentence in the report.

## Blockers and risks (role 2)

1. **node2vec cost of the full grid.** One skip-gram epoch is ~5–6 min on MPS. 9 (p, q)
   configs × 3+ epochs + 10 seeds of the chosen config ≈ 4–5 h of GPU time. Plan: fix the
   epoch count first on p=q=1 (1/3/5), then run the p, q grid at that budget, on Colab GPU
   (CUDA, same code path).
2. **pyg-lib / torch-cluster do not load on macOS** with uv/Homebrew Python (the wheels link
   against the python.org framework). Resolved for node2vec with our own walker; role 3's
   GraphSAGE only needs `SAGEConv` from torch_geometric, which works without them.
3. **Dense vs. sparse Adam for skip-gram.** SparseAdam is unavailable on MPS and a CPU epoch with it
   takes ~100 min (sparse-gradient coalescing), so we use dense Adam everywhere (same optimizer on every machine). This differs slightly from
   the reference PyG Node2Vec recipe; to be stated in the report.
4. **MLP stops too early** (see above). Try patience 50 and lr 1e-3 on seed 0 before seeds 0–9;
   if it keeps climbing, the shared epoch cap/patience may need raising for all methods.
5. **Decoder learning rate per method.** The proposal fixes decoder, loss, sampler and epoch cap,
   not the lr. We will tune lr ∈ {1e-3, 1e-2} on validation for each learned method.
6. **Waiting on other roles:** `collab/heuristics.py` (role 1) and `collab/sage.py` (role 3)
   are wired into the Makefile as placeholders; they need to write scores and
   `results/*.jsonl` in the shared format.

## Next (12–19 Nov)

- Skip-gram epoch budget, then the 9-config p, q grid (val Hits@50), then 10 seeds.
- Features-only MLP: lr/dropout check, then seeds 0–9.
- Help role 3 plug GraphSAGE into `train_link_predictor` and run the T4 timing.
- Clean-machine test of `make env && make check` on Colab.
