"""The unattended queue must attach without duplicate training and stop on failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from . import local_pipeline


class LocalPipelineTests(unittest.TestCase):
    def test_attached_mlp_then_node2vec_and_reports(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "logs").mkdir()
            (root / "logs/mlp_local_status.json").write_text('{"state": "complete"}')
            process = MagicMock()
            process.cmdline.return_value = ["python", "-m", "collab.reproduce"]
            with patch.object(local_pipeline, "ROOT", root), \
                    patch.object(local_pipeline.psutil, "Process", return_value=process), \
                    patch.object(local_pipeline.subprocess, "run") as run:
                local_pipeline.main(["--wait_pid", "1234"])
            process.wait.assert_called_once()
            commands = [call.args[0][3:] for call in run.call_args_list]
            self.assertEqual(commands, [["collab.buckets"], ["collab.summarize"],
                                        ["collab.reproduce", "--model", "node2vec"],
                                        ["collab.buckets"], ["collab.summarize"]])
            status = json.loads((root / "logs/local_pipeline_status.json").read_text())
            self.assertEqual(status["state"], "complete")

    def test_failed_attached_mlp_never_starts_node2vec(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "logs").mkdir()
            (root / "logs/mlp_local_status.json").write_text('{"state": "failed"}')
            process = MagicMock()
            process.cmdline.return_value = ["python", "-m", "collab.reproduce"]
            with patch.object(local_pipeline, "ROOT", root), \
                    patch.object(local_pipeline.psutil, "Process", return_value=process), \
                    patch.object(local_pipeline.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "MLP did not finish"):
                    local_pipeline.main(["--wait_pid", "1234"])
                run.assert_not_called()
            status = json.loads((root / "logs/local_pipeline_status.json").read_text())
            self.assertEqual(status["state"], "failed")


if __name__ == "__main__":
    unittest.main()
