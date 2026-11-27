"""Run/resume local MLP, then node2vec, refreshing reports after each model.

    python -m collab.local_pipeline

--wait_pid attaches to an already running collab.reproduce process instead of
starting another MLP. The pipeline never publishes commits or PRs automatically.
"""
import argparse
import json
import os
import subprocess
import sys

import psutil
from filelock import FileLock

from .utils import ROOT
from .reproduce import write_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait_pid", type=int)
    args = parser.parse_args(argv)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "local_pipeline_status.json"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8",
               PYTHONWARNINGS="ignore::FutureWarning")

    def run(module, *arguments, log_name):
        write_status(status_path, "running", stage=" ".join((module, *arguments)))
        with (log_dir / f"{log_name}.stdout.log").open("a", encoding="utf-8") as out, \
                (log_dir / f"{log_name}.stderr.log").open("a", encoding="utf-8") as err:
            subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=ROOT,
                           env=env, stdout=out, stderr=err, check=True)

    with FileLock(str(log_dir / "local_pipeline.lock"), timeout=0):
        try:
            if args.wait_pid is not None:
                write_status(status_path, "waiting", stage="mlp", pid=args.wait_pid)
                try:
                    process = psutil.Process(args.wait_pid)
                    if "collab.reproduce" not in process.cmdline():
                        raise ValueError("Refusing to attach to a process other than collab.reproduce")
                    process.wait()
                except psutil.NoSuchProcess:
                    pass
                state = json.loads((log_dir / "mlp_local_status.json").read_text(encoding="utf-8"))
                if state["state"] != "complete":
                    raise RuntimeError("MLP did not finish successfully; inspect its status/log before resuming")
            else:
                run("collab.reproduce", "--model", "mlp", log_name="mlp_local")
            run("collab.buckets", log_name="local_reports")
            run("collab.summarize", log_name="local_reports")
            run("collab.reproduce", "--model", "node2vec", log_name="node2vec_local")
            run("collab.buckets", log_name="local_reports")
            run("collab.summarize", log_name="local_reports")
            write_status(status_path, "complete", models=["mlp_local", "node2vec_local"])
        except Exception as error:
            write_status(status_path, "failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
