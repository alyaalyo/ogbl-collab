# Local reproduction queue (27 Nov 2026)

The team score files are unavailable locally. This queue reproduces the published
seed-0 configurations to generate independent per-pair scores for bucket analysis.
Results are named `mlp_local` and `node2vec_local`; the teammates' `mlp` and
`node2vec_e3` records are preserved. These runs do not replace the planned ten-seed
final experiments or GraphSAGE's missing score files.

## Fixed configurations

Both decoders use hidden size 256, dropout 0, learning rate 0.01, positive batch
size 65,536, epoch cap 400 and early-stopping patience 20. Selection is on
validation Hits@50; test is scored with the selected weights after training.
Evaluation batch size is 32,768 to reduce peak RAM use. The CPU runs use two
Torch threads. No hyperparameter search is performed.

- MLP uses only the original 128-dimensional text features.
- Node2vec uses frozen 128-dimensional embeddings, p=q=1, three skip-gram epochs,
  ten walks of length 40 per node, context size 10, 256 nodes per batch, dense
  Adam with learning rate 0.01, and the repository's original negative sampler.

## Start or resume

With the existing Python 3.11 environment, in PowerShell:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
.\.venv\Scripts\python.exe -u -m collab.local_pipeline
```

The queue runs MLP, refreshes bucket reports and the summary table, runs
node2vec, then refreshes the reports again. Completed runs are skipped if their
four score files exist. It refuses configuration mismatches or completed
records with missing score files. A process lock prevents two queue instances.

The completed launch attached the queue to an MLP process already
running in the background. `--wait_pid` is only for attaching to an existing
`collab.reproduce` process; it should be omitted when restarting after a failure.
The queue stops and records an error if its attached MLP exits without completing.
It never commits or publishes results automatically.

Individual runs are available as `python -m collab.reproduce --model mlp` and
`--model node2vec`. Use the queue for normal operation; do not launch a duplicate
individual run while that same model is already training.

## Outputs and progress

| Artifact | Location |
|---|---|
| Overall queue state | `logs/local_pipeline_status.json` |
| Latest model progress | `logs/{mlp,node2vec}_local_status.json` |
| Per-model console output/errors | `logs/{mlp,node2vec}_local.{stdout,stderr}.log` |
| Training histories | `logs/{mlp,node2vec}_local_history.json` |
| Metric records | `results/{mlp,node2vec}_local.jsonl` |
| Scores, official pair order | `scores/{mlp,node2vec}_local_seed0_{split}.npy` |
| Final decoder weights | `checkpoints/{mlp,node2vec}_local_seed0_final.pt` |
| Final node2vec embeddings | `embeddings/node2vec_local_seed0.pt` |
| Updated bucket tables | `results/buckets/` |

Read progress without interrupting training:

```powershell
Get-Content logs/local_pipeline_status.json
Get-Content logs/mlp_local.stdout.log -Tail 5
Get-Content logs/node2vec_local.stdout.log -Tail 5
```

The computer must remain powered on and awake for the queue to progress.
Keep the working checkout on the reproduction branch while it runs: each queued
stage imports code from this checkout when that stage starts.

## Checkpoints and validation

Decoder checkpoints are saved every epoch. Skip-gram checkpoints are saved
every 100 batches and at epoch boundaries, using a temporary file and atomic
replacement. They include embeddings, optimizer state, the node permutation,
position inside the epoch, Torch/NumPy RNG state, and a configuration/graph
fingerprint. Resume continues from the last saved batch without starting the
whole embedding stage again. Checkpoint files can be several hundred MB.

The shared decoder now also respects an already-triggered early stop when
resuming: it restores/scores the chosen weights without an extra training epoch.
Tests verify exact equality with uninterrupted decoder training, resumption
after early stopping, and resumption inside a node2vec epoch, including completed
checkpoint reuse and rejection of a mismatched graph.

## Timing evidence and current limits

Early MLP epochs on this Intel i5-8300H take roughly 16-17 seconds of training
time each, plus validation/checkpoint overhead. The 400-epoch cap is roughly two
hours at this rate; early stopping can finish sooner.

The CPU node2vec probe ran three full-size batches in 8.10 seconds on one Torch
thread while MLP was active. Extrapolating 922 batches/epoch to three epochs gives
about 2.07 hours for embeddings, excluding decoder training. This is a rough
estimate affected by warm-up, threading, checkpoint overhead and concurrent
workload, not an observed full-run duration. The sequential queue uses two threads.
The probe never scores validation/test and discards its embeddings.

## Completed local runs

Both runs and report generation completed on 27 November 2026. Each model has
four score arrays, selected decoder weights, a training history and a metric
record. The queue status is `complete`.

| Method | Seed | Best / completed decoder epoch | Validation Hits@50 (%) | Test Hits@50 (%) |
|---|---:|---:|---:|---:|
| mlp_local | 0 | 122 / 142 | 16.67 | 12.62 |
| node2vec_local | 0 | 10 / 30 | 59.15 | 49.21 |

Both decoders stopped after 20 validation checks without improvement. Recorded
MLP training time was 38.17 minutes. Node2vec embeddings took 56.71 minutes for
three epochs, followed by 5.06 minutes of decoder training. Decoder training
times exclude validation, final inference and checkpoint I/O; they are not
end-to-end durations. The observed embedding duration supersedes the earlier
short-probe estimate.

These are independent local runs: the existing team records remain 14.24% for
MLP and 47.72% for node2vec_e3 on test Hits@50. The differing results are retained
under distinct names; matching seed/configuration does not establish identical
outputs across execution environments. No cause for the differences is inferred
from these two runs alone, and there is no multi-seed uncertainty estimate.

The saved scores were checked against the official evaluator and exact inference
from the saved weights in official pair order. The audit also checks validation
selection, early stopping, bucket totals, source hashes and intersection indices;
see [audit record](../results/local_reproduction_audit.json),
[bucket tables](../results/buckets/summary.md) and
[interpretation](analysis-role1.md#local-learned-model-comparison).
Score arrays and weights remain local in the ignored artifact directories;
the committed tables and audit hashes do not replace those files for teammates.
