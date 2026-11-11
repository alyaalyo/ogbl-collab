"""Features-only baseline: the shared decoder on x_u * x_v, no graph.

    python -m collab.mlp --seeds 0            # one seed
    python -m collab.mlp --seeds 0-9          # all ten
"""
import argparse

from .data import load_collab
from .decoder import FrozenEncoder, LinkDecoder, TrainConfig, train_link_predictor
from .evaluate import save_scores
from .utils import CKPT_DIR, append_result, get_device, set_seed


def parse_seeds(s: str) -> list:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64 * 1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--method", default="mlp", help="name used for score files / results")
    ap.add_argument("--no_save_scores", action="store_true")
    args = ap.parse_args(argv)

    data = load_collab()
    device = get_device(args.device)
    for seed in parse_seeds(args.seeds):
        set_seed(seed)
        print(f"== {args.method} seed {seed} on {device}")
        cfg = TrainConfig(lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
                          patience=args.patience, device=str(device),
                          ckpt_path=str(CKPT_DIR / f"{args.method}_seed{seed}.pt"), resume=False,
                          log_every=10)
        res = train_link_predictor(FrozenEncoder(data.x),
                                   LinkDecoder(data.x.size(1), args.hidden, dropout=args.dropout),
                                   data.train_pos, data.num_nodes, data.eval_pairs, cfg)
        print(f"seed {seed}: best epoch {res.best_epoch}  valid {res.valid}  test {res.test}  "
              f"({res.seconds_per_epoch:.2f}s/epoch)")
        if not args.no_save_scores:
            save_scores(args.method, seed, res.scores)
        append_result(args.method, {
            "method": args.method, "seed": seed, "device": str(device),
            "hparams": {"lr": args.lr, "dropout": args.dropout, "hidden": args.hidden},
            "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
            "seconds_per_epoch": round(res.seconds_per_epoch, 3), "total_seconds": round(res.total_seconds, 1),
            "valid": res.valid, "test": res.test,
        })


if __name__ == "__main__":
    main()
