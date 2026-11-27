"""Reproduce the published seed-0 MLP/node2vec configurations with separate artifacts.

    python -m collab.reproduce --model mlp
    python -m collab.reproduce --model node2vec

No hyperparameter search: lr=.01, dropout=0, hidden=256, 400 epochs, patience=20,
batch size 65536. Saves a checkpoint every epoch and resumes after interruption.
Uses smaller evaluation batches for a memory-constrained computer. Test is
scored only once the validation-selected weights have been restored.
Node2vec uses p=q=1, three skip-gram epochs and a checkpoint every 100 batches.
"""
import argparse
import importlib.metadata
import json
import platform
import time

import torch

from .data import load_collab
from .decoder import FrozenEncoder, LinkDecoder, TrainConfig, train_link_predictor
from .evaluate import load_scores, save_scores
from .node2vec import train_node2vec
from .utils import CKPT_DIR, EMB_DIR, RESULTS_DIR, ROOT, append_result, set_seed

SETTINGS = dict(seed=0, lr=0.01, dropout=0.0, hidden=256, max_epochs=400,
                patience=20, batch_size=65536, eval_batch_size=32768, device="cpu")


def write_status(path, state, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(state=state, updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                                    method=path.stem.removesuffix("_status"), **fields), indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "node2vec"), default="mlp")
    args = parser.parse_args(argv)
    method = args.model + "_local"
    settings = dict(SETTINGS)
    if args.model == "node2vec":
        settings.update(p=1.0, q=1.0, dim=128, walk_length=40, context_size=10,
                        walks_per_node=10, n2v_epochs=3, n2v_lr=0.01, batch_nodes=256)
    status_path = ROOT / "logs" / f"{method}_status.json"
    config_path = CKPT_DIR / f"{method}_seed0_config.json"
    ckpt_path = CKPT_DIR / f"{method}_seed0.pt"
    result_path = RESULTS_DIR / f"{method}.jsonl"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Existing local run has different settings; refusing to mix checkpoints")
    elif ckpt_path.exists() or result_path.exists():
        raise ValueError("Existing local artifacts have no configuration manifest")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    if result_path.exists():
        records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        finished = [r for r in records if r["seed"] == 0]
        if finished:
            if set(load_scores(method, 0)) != {"valid_pos", "valid_neg", "test_pos", "test_neg"}:
                raise ValueError("Completed run has missing scores; do not silently retrain")
            print(f"Completed {method} seed0 already exists; skipping training", flush=True)
            write_status(status_path, "complete", result=finished[-1])
            return
    write_status(status_path, "loading", settings=settings)
    started = time.perf_counter()

    def progress(message):
        print(message, flush=True)
        write_status(status_path, "training", settings=settings, progress=message,
                     process_seconds=round(time.perf_counter() - started, 1))

    try:
        data = load_collab()
        set_seed(0)
        z, embedding_seconds = data.x, None
        if args.model == "node2vec":
            z, embedding_seconds = train_node2vec(
                data.edge_index, data.num_nodes, p=1, q=1, epochs=3, batch_nodes=256,
                device="cpu", seed=0, ckpt_path=CKPT_DIR / "node2vec_local_skipgram.pt",
                resume=True, ckpt_every=100, log=progress)
            EMB_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(z, EMB_DIR / "node2vec_local_seed0.pt")
        set_seed(0)
        encoder, decoder = FrozenEncoder(z), LinkDecoder(128, 256, dropout=0.0)
        cfg = TrainConfig(lr=0.01, batch_size=65536, max_epochs=400, patience=20,
                          eval_batch_size=32768, device="cpu", ckpt_path=str(ckpt_path),
                          ckpt_every=1, resume=True, log_every=1)
        progress("Starting/resuming decoder training")
        result = train_link_predictor(encoder, decoder, data.train_pos, data.num_nodes,
                                      data.eval_pairs, cfg, log=progress)
        save_scores(method, 0, result.scores)
        torch.save({"decoder": decoder.state_dict(), "settings": settings,
                    "best_epoch": result.best_epoch}, CKPT_DIR / f"{method}_seed0_final.pt")
        history_path = ROOT / "logs" / f"{method}_history.json"
        history_path.write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
        record = dict(method=method, seed=0, device="cpu", hparams=settings,
                      best_epoch=result.best_epoch, epochs_run=result.epochs_run,
                      seconds_per_epoch=round(result.seconds_per_epoch, 3),
                      total_seconds=round(result.total_seconds, 3), embedding_seconds=embedding_seconds,
                      valid=result.valid, test=result.test,
                      environment=dict(python=platform.python_version(), platform=platform.platform(),
                                       torch_threads=torch.get_num_threads(),
                                       packages={name: importlib.metadata.version(name)
                                                 for name in ("torch", "numpy", "ogb", "torch-geometric")}))
        append_result(method, record)
        write_status(status_path, "complete", result=record)
        print(json.dumps(record, indent=2), flush=True)
    except Exception as error:
        write_status(status_path, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
