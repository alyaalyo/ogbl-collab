"""CPU seed-0 GraphSAGE reproduction with native sparse adjacency and saved scores.

python -m collab.sage_local --probe   # two training-only steps, discard weights
python -m collab.sage_local           # train/resume, score both graphs, gate analysis
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time

from filelock import FileLock
import numpy as np
import psutil
import torch

from .buckets import check_sage_reproduction
from .data import load_collab
from .decoder import LinkDecoder, TrainConfig, link_loss, train_link_predictor, uniform_negatives
from .evaluate import evaluate_scores, load_scores, save_scores
from .reproduce import write_status
from .sage import SAGEEncoder, score_with_graph
from .utils import CKPT_DIR, RESULTS_DIR, ROOT, append_result, set_seed

METHOD = "sage_local"
SETTINGS = dict(seed=0, lr=0.001, dropout=0.0, hidden=256, layers=3,
                grad_clip=1.0, patience=50, max_epochs=400, batch_size=65536,
                eval_batch_size=32768, device="cpu", torch_threads=2,
                adjacency="native_torch_csr")


def sparse_graph(edge_index, num_nodes):
    """Incoming-neighbour CSR for the shared unique, unweighted graph.

    Sparse SAGEConv expects rows=targets and columns=sources. Coalescing/sorting
    changes summation order, so equivalence to COO is numerical, not bitwise.
    """
    return torch.sparse_coo_tensor(
        edge_index.flip(0), torch.ones(edge_index.shape[1], device=edge_index.device),
        (num_nodes, num_nodes)).coalesce().to_sparse_csr()


def build(data):
    return (SAGEEncoder(data.x, sparse_graph(data.edge_index, data.num_nodes),
                        hidden=SETTINGS["hidden"], num_layers=SETTINGS["layers"], dropout=0.0),
            LinkDecoder(SETTINGS["hidden"], SETTINGS["hidden"], dropout=0.0))


def probe(data):
    """No evaluation or checkpoint publication: time two real optimizer steps."""
    set_seed(0)
    encoder, decoder = build(data)
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=SETTINGS["lr"])
    permutation = torch.randperm(len(data.train_pos))
    times = []
    for step in range(2):
        started = time.perf_counter()
        pos = data.train_pos[permutation[step * 65536:(step + 1) * 65536]]
        neg = uniform_negatives(len(pos), data.num_nodes, "cpu")
        z = encoder()
        loss = link_loss(decoder(z[pos[:, 0]], z[pos[:, 1]]), decoder(z[neg[:, 0]], z[neg[:, 1]]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        times.append(time.perf_counter() - started)
        print(f"probe step {step + 1}: {times[-1]:.2f}s", flush=True)
    epoch_seconds = sum(times) / len(times) * math.ceil(len(data.train_pos) / 65536)
    result = dict(settings=SETTINGS, step_seconds=times, estimated_epoch_seconds=epoch_seconds,
                  estimated_400_epochs_hours=epoch_seconds * 400 / 3600,
                  rss_gib=psutil.Process().memory_info().rss / 2**30,
                  note="Two training-only steps; no validation/test evaluation; weights discarded. "
                       "Estimate excludes evaluation/checkpoint overhead; RSS is not peak memory.")
    (ROOT / "logs/sage_local_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def refresh_reports(records):
    """Record raw metrics even if reproduction fails; withhold bucket analysis."""
    subprocess.run([sys.executable, "-m", "collab.summarize"], cwd=ROOT, check=True)
    failures = []
    for method in (METHOD, METHOD + "_valinput"):
        try:
            check_sage_reproduction(method, [records[method]["test"]["hits@50"]])
        except ValueError as error:
            failures.append(str(error))
    if failures:
        return failures
    for flags in ([], ["--valinput"]):
        subprocess.run([sys.executable, "-m", "collab.buckets", *flags], cwd=ROOT, check=True)
    return []


def train(data, status_path):
    signature = dict(settings=SETTINGS, input_sha256={
        name: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        for name, tensor in {"x": data.x, "train": data.edge_index, "valinput": data.edge_index_val,
                             **data.eval_pairs}.items()})
    config_path = CKPT_DIR / "sage_local_seed0_config.json"
    ckpt_path = CKPT_DIR / "sage_local_seed0.pt"
    final_path = CKPT_DIR / "sage_local_seed0_final.pt"
    records = {}
    for method in (METHOD, METHOD + "_valinput"):
        path = RESULTS_DIR / f"{method}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record["seed"] == 0:
                        records[method] = record
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Local SAGE configuration/data mismatch; refusing to mix artifacts")
    elif ckpt_path.exists() or final_path.exists() or records:
        raise ValueError("Existing local SAGE artifacts have no configuration manifest")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    if METHOD in records:
        if METHOD + "_valinput" not in records or not final_path.exists():
            raise ValueError("Completed SAGE record has incomplete artifacts")
        for method, expected in ((METHOD, set(data.eval_pairs)),
                                 (METHOD + "_valinput", {"test_pos", "test_neg"})):
            scores = load_scores(method, 0)
            if set(scores) != expected or any(
                    values.shape != (len(data.eval_pairs[split]),) or values.dtype != np.float32
                    or not np.isfinite(values).all() for split, values in scores.items()):
                raise ValueError("Completed SAGE scores are missing or malformed")
            if evaluate_scores(scores) != {
                    key: records[method][key] for key in ("valid", "test") if key in records[method]}:
                raise ValueError("Completed SAGE scores are missing or disagree with recorded metrics")
        print("Local SAGE already complete; refreshing reports only", flush=True)
    else:
        set_seed(0)
        encoder, decoder = build(data)
        started = time.perf_counter()

        def progress(message):
            print(message, flush=True)
            write_status(status_path, "training", settings=SETTINGS, progress=message,
                         process_seconds=round(time.perf_counter() - started, 1))

        cfg = TrainConfig(lr=.001, batch_size=65536, max_epochs=400, patience=50,
                          eval_batch_size=32768, device="cpu", grad_clip=1.0,
                          ckpt_path=str(ckpt_path), ckpt_every=1, resume=True, log_every=1)
        progress("Starting/resuming GraphSAGE training")
        result = train_link_predictor(encoder, decoder, data.train_pos, data.num_nodes,
                                      data.eval_pairs, cfg, log=progress)
        save_scores(METHOD, 0, result.scores)
        torch.save(dict(encoder=encoder.state_dict(), decoder=decoder.state_dict(),
                        signature=signature, best_epoch=result.best_epoch), final_path)
        (ROOT / "logs/sage_local_history.json").write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
        write_status(status_path, "scoring_valinput", best_epoch=result.best_epoch)
        vi = score_with_graph(encoder, decoder, sparse_graph(data.edge_index_val, data.num_nodes),
                              data.eval_pairs, ("test_pos", "test_neg"), batch_size=32768)
        save_scores(METHOD + "_valinput", 0, vi)
        common = dict(seed=0, device="cpu", hparams=SETTINGS, best_epoch=result.best_epoch,
                      epochs_run=result.epochs_run, seconds_per_epoch=round(result.seconds_per_epoch, 3),
                      total_seconds=round(result.total_seconds, 3),
                      environment=dict(python=platform.python_version(), platform=platform.platform(),
                                       torch_threads=torch.get_num_threads(),
                                       packages={name: importlib.metadata.version(name)
                                                 for name in ("torch", "numpy", "ogb", "torch-geometric")}))
        records = {METHOD: dict(method=METHOD, **common, valid=result.valid, test=result.test),
                   METHOD + "_valinput": dict(method=METHOD + "_valinput", **common, **evaluate_scores(vi))}
        # Main record marks completion only after the secondary scores and record exist.
        for method in (METHOD + "_valinput", METHOD):
            append_result(method, records[method])
    write_status(status_path, "checking_reproduction")
    failures = refresh_reports(records)
    write_status(status_path, "complete", reproduction_passed=not failures,
                 analysis="withheld" if failures else "updated", reproduction_failures=failures,
                 results={method: record["test"] for method, record in records.items()})
    print("Training complete; " + ("bucket analysis withheld: " + "; ".join(failures)
                                   if failures else "both reproduction checks passed; bucket reports updated"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "logs/sage_local_status.json"
    with FileLock(str(ROOT / "logs/sage_local.lock"), timeout=0):
        try:
            if not args.probe:
                write_status(status_path, "loading", settings=SETTINGS)
            data = load_collab()
            if args.probe:
                return probe(data)
            train(data, status_path)
        except Exception as error:
            if not args.probe:
                write_status(status_path, "failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
