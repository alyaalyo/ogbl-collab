# Local GraphSAGE reproduction

The teammate's GraphSAGE aggregate metrics are available, but its weights and
per-pair scores are not. `collab.sage_local` trains an independent seed-0 model
on CPU and saves `sage_local` / `sage_local_valinput` artifacts. It preserves
the original `sage` records and does not claim to reproduce their exact scores.

## Configuration and sparse computation

The configuration is the successful role-3 run documented in
[midway-role3.md](midway-role3.md): three mean-aggregation GraphSAGE layers,
hidden size 256, dropout 0, learning rate 0.001, gradient clipping 1.0 separately
for encoder/decoder, maximum 400 epochs, patience 50, positive batch size 65,536.
The shared decoder, loss, negative sampler and full-graph training loop are used.
Patience 50 is the documented role-3 departure from the proposal's shared default
20; this local run adopts it without a new hyperparameter search.

The input graph uses PyTorch's native sparse CSR representation instead of
materialising per-edge feature messages. Rows are destination nodes and columns
are source nodes. Architecture and unweighted mean aggregation are unchanged.
Tests compare outputs, feature/parameter gradients and valinput scores against
the original COO representation, including a directed graph and isolated node.
Floating-point summation order can differ; cross-backend bitwise training
identity is not promised. Interrupted-versus-uninterrupted CSR training is
tested for exact score equality.

Evaluation batches are 32,768 for lower memory use. Two CPU threads are used.
No GPU environment changes are required.

## Start, resume and monitor

In PowerShell with the existing environment:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
.\.venv\Scripts\python.exe -u -m collab.sage_local
```

The command resumes after the last completed epoch, or verifies the six scores
and refreshes reports when already complete. A process lock prevents duplicate
local GraphSAGE jobs. Configuration and data hashes must match saved artifacts.
Checkpoint weights, optimizer, validation history and RNG states are saved
every epoch. There is no automatic Git commit, push or PR update.

The current background launch writes:

- `logs/sage_local_status.json`: stage, latest epoch and final gate outcome;
- `logs/sage_local.stdout.log` / `.stderr.log`: console output and errors;
- `logs/sage_local_history.json`: completed validation history;
- `checkpoints/sage_local_seed0.pt`: resumable checkpoint;
- `checkpoints/sage_local_seed0_final.pt`: weights selected on validation;
- `checkpoints/sage_local_seed0_config.json`: fixed settings and data hashes;
- `scores/sage_local_seed0_{valid,test}_{pos,neg}.npy`: four baseline scores;
- `scores/sage_local_valinput_seed0_test_{pos,neg}.npy`: two valinput scores;
- `results/sage_local.jsonl` / `sage_local_valinput.jsonl`: final metrics.

```powershell
Get-Content logs/sage_local_status.json
Get-Content logs/sage_local.stdout.log -Tail 5
```

Keep the computer powered on and awake, and the checkout on `sage-local` while
training/report generation runs. If interrupted, rerun the command above.

## Timing and current state

A two-step training-only CPU probe measured 6.85 and 6.79 seconds per optimizer
step. There are 15 steps per epoch: the rough extrapolation is 102 seconds per
epoch, or 11.37 hours at 400 epochs before validation/checkpoint overhead.
Early stopping may shorten the run; later epochs and other workloads can change
the speed. This is an estimate, not a measured full-run duration. The probe did
not score validation/test and its weights were discarded. To repeat a timing
probe when no local SAGE job is running, use `python -m collab.sage_local --probe`.

The first full epoch completed in 96.80 seconds of training, approximately
99.7 seconds including validation/checkpoint work, and its checkpoint was saved.
At submission, the full seed-0 job is running; final quality metrics are
pending. Existing MLP/node2vec results are already complete. Inspect the status
file for live progress rather than inferring completion from this document.

## Reproduction gate and next analysis

The final selected weights score validation/test on the training graph and only
test on the training-plus-2018 graph, without retraining. Raw metrics and all
six scores are saved even if reproduction fails. The summary table is refreshed.

Both criteria must pass before the automatic queue updates bucket comparisons:

- baseline test Hits@50: 48.10 +/- 2 percentage points;
- valinput test Hits@50: 54.63 +/- 2 percentage points.

The bucket generator also applies its gate to the local method names, so using
`sage_local` cannot bypass the check. If either criterion fails, status reports
`state: complete`, `reproduction_passed: false`, `analysis: withheld` and the
reason. That means training finished, not that the reproduction succeeded.
Existing bucket reports remain unchanged until a permitted analysis is run.
The runner does not tune or retry against test results automatically.

If both pass, baseline and valinput bucket tables, AA intersections and input
manifests are regenerated. Then verify the saved weights/scores, review the
AA-versus-GraphSAGE disagreements and add the observed conclusions to the role-1
report. One local seed does not complete the planned final multi-seed study.
