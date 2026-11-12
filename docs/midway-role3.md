# Midway check (12 Nov 2026): role 3

Scope of role 3: GraphSAGE (3 layers, 256 units, mean aggregation, input x_v, shared decoder),
reproduction check (test Hits@50 within 2 points of 48.10), checkpoints, timing on a T4,
10 (or 5) seeds, re-scoring with 2018 edges without retraining.

## Done

- `collab/sage.py`: `SAGEEncoder` (3 x `SAGEConv`, 256 units, mean aggregation, ReLU between layers,
  full-batch on the input graph A) trained end-to-end through the shared `train_link_predictor`
  with the shared `LinkDecoder`, BCE loss, uniform negatives and epoch cap 400.
- Scores in the shared format: `scores/sage_seed{k}_{split}.npy`.
- "Val as input": the best-on-validation weights are re-used without retraining on A + 2018 edges;
  test scores go to `scores/sage_valinput_seed{k}_test_{pos,neg}.npy`. No validation numbers
  for this variant (the 2018 positives are in the input graph).
- Checkpoints every 10 epochs (point `COLLAB_CKPT` at persistent storage), resume after a
  disconnect; finished seeds are skipped on restart.
- Make targets `sage-probe`, `sage-tune`, `sage`; toy-graph tests in `collab/test_sage.py` (run by `make check`).
- Optional gradient clipping added to the shared loop (`TrainConfig.grad_clip`, off by default,
  so MLP and node2vec are unaffected). `collab/sage.py` turns it on with max norm 1.0 for the
  encoder and the decoder separately, as in the OGB reference example for ogbl-collab.

## Timing (one T4)

| measurement | seconds per epoch | note |
|---|---|---|
| `make sage-probe` (3 epochs), training only | 6.77 | |
| `make sage-probe`, wall clock incl. validation | 7.24 | upper bound 0.8 h for 400 epochs |
| full run, seed 0 (400 epochs) | 7.13 | about 48 min per seed |

A run takes under one hour, so the fallback from the proposal (5 seeds) is not needed.
Ten seeds are about 8 GPU hours at the epoch cap.

## Reproduction (test Hits@50, reference 48.10, tolerance ±2)

Seed 0, lr 1e-3, Hits × 100:

| attempt | change | best epoch | val @50 | test @50 | result |
|---|---|---|---|---|---|
| 1 | shared loop, no clipping, patience 20 | 94 | 35.69 | 29.43 | not reproduced |
| 2 | + gradient clipping 1.0 | 178 | 51.63 | 43.12 | not reproduced |
| 3 | + patience 50 | 365 | 57.17 | **48.13** | **reproduced** |

Final configuration (seed 0): val Hits@20/50/100 = 45.44 / 57.17 / 62.69,
test Hits@20/50/100 = 36.74 / 48.13 / 53.99.
"Val as input" (same weights, A + 2018 edges): test Hits@20/50/100 = 41.57 / 54.69 / 61.89
(expected about 54.63).

Observations:
- Without clipping the loss spiked every few dozen epochs and validation Hits@50 collapsed to
  near zero each time. Clipping removed most spikes.
- Validation Hits@50 moves by 2-5 points between neighbouring epochs, so patience 20 stopped runs
  while the curve was still rising (epochs 114 and 198). The final run reached the epoch cap of 400.
- Learning rate was chosen on validation Hits@50 only (patience 20): without clipping,
  1e-3 gave 41.29 and 1e-2 gave 23.62; with clipping, 1e-3 gave 54.93 and 5e-4 gave 50.82.
  The grid was not repeated with patience 50.
- Patience was raised to 50 after the failed second check. Validation was still rising at the
  stop epoch, but the reproduction criterion itself is on test, so this is stated here explicitly.
- This is one seed. The OGB leaderboard std is 0.81, so a single seed close to the reference does
  not yet show the mean over ten seeds will be.

## Blockers and risks

1. **Patience differs from the shared default.** The proposal fixes patience 20 for every learned
   method; GraphSAGE needs 50 to reproduce. Decision needed from the team: use 50 for all
   learned methods, or keep 50 for GraphSAGE only and say so in the report. Role 2 already
   saw the same effect on the features-only MLP.
2. **Shared files touched.** `collab/decoder.py` (optional `grad_clip`, default off) and
   `collab/summarize.py` (skips `*_grid`/`*_timing` logs and records without test metrics, no
   validation columns for `_valinput` rows). Roles 1 and 2 should review.
3. **Test is computed in every run.** The shared loop scores test at the end of each run, including
   tuning runs. `--tune` records only validation metrics, but the number is computed.
4. **Seed variance unknown** until the remaining nine seeds are done.

## Next

- Seeds 1-9 with the same configuration (`make sage SEEDS=1-9`, patience 50 is set in the Makefile).
- Hand the score files to role 1 for bucket analysis, including `sage_valinput`.
- Decide with role 2 on the shared patience.
