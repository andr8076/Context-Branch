"""Offline harness tests for the real-inference comparator, using fake output only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoint_stress as cs


class CheckpointStressHarnessTests(unittest.TestCase):
    def test_two_arms_identical_except_checkpoint_threshold_and_output(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"

            def simulate_main(args):
                seen.append(args)
                folder = Path(args[args.index("--output") + 1])
                folder.mkdir(parents=True)
                enabled = float(args[args.index("--recursive-checkpoint-threshold") + 1]) > 0
                result = ({
                    "verified_cases": 1, "exact_final": True,
                    "error_type": "none", "recursive_checkpoints": 2,
                    "authority_docs_viewed": 5, "total_actions": 25
                } if enabled else {
                    "verified_cases": 0, "exact_final": False,
                    "error_type": "context_overflow_confirmed",
                    "recursive_checkpoints": 0, "authority_docs_viewed": 5,
                    "total_actions": 18
                })
                (folder / "results.json").write_text(json.dumps({"trials": [result]}))
                return 0

            with patch.object(cs.benchmark, "main", simulate_main):
                code = cs.run(["--output", str(output), "--context-limit", "4096"])
            self.assertEqual(code, 0)
            self.assertEqual(len(seen), 2)
            before = list(seen[0])
            after = list(seen[1])
            for args in (before, after):
                args[args.index("--recursive-checkpoint-threshold") + 1] = "SAME"
                args[args.index("--output") + 1] = "SAME"
            self.assertEqual(before, after)
            summary = json.loads((output / "comparison.json").read_text())
            self.assertIn("SUCCESSFUL DEMONSTRATION", summary["verdict"])
            self.assertEqual(summary["off"]["authority_docs_viewed"], 5)
            self.assertEqual(summary["on"]["authority_docs_viewed"], 5)

    def test_no_checkpoint_is_explicitly_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"

            def simulate_main(args):
                folder = Path(args[args.index("--output") + 1])
                folder.mkdir(parents=True)
                (folder / "results.json").write_text(json.dumps({"trials": [{
                    "verified_cases": 1, "exact_final": True, "error_type": "none",
                    "recursive_checkpoints": 0
                }]}))
                return 0

            with patch.object(cs.benchmark, "main", simulate_main):
                self.assertEqual(cs.run(["--output", str(output)]), 0)
            summary = json.loads((output / "comparison.json").read_text())
            self.assertIn("INCONCLUSIVE", summary["verdict"])

    def test_control_failure_with_completed_checkpoint_is_reported_as_robustness_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"

            def simulate_main(args):
                folder = Path(args[args.index("--output") + 1])
                folder.mkdir(parents=True)
                enabled = float(args[args.index("--recursive-checkpoint-threshold") + 1]) > 0
                result = ({
                    "verified_cases": 1, "exact_final": True,
                    "error_type": "none", "final_status": "success",
                    "recursive_checkpoints": 3,
                    "recursive_checkpoint_blocked": 0,
                } if enabled else {
                    "verified_cases": 0, "exact_final": False,
                    "error_type": "none", "final_status": "incorrect_answer",
                    "recursive_checkpoints": 0,
                    "recursive_checkpoint_blocked": 0,
                })
                (folder / "results.json").write_text(json.dumps({"trials": [result]}))
                return 0

            with patch.object(cs.benchmark, "main", simulate_main):
                self.assertEqual(cs.run(["--output", str(output)]), 0)
            summary = json.loads((output / "comparison.json").read_text())
            self.assertIn("ROBUSTNESS WIN", summary["verdict"])

    def test_checkpoint_block_after_control_overflow_is_reported_as_safe_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"

            def simulate_main(args):
                folder = Path(args[args.index("--output") + 1])
                folder.mkdir(parents=True)
                enabled = float(args[args.index("--recursive-checkpoint-threshold") + 1]) > 0
                result = ({
                    "verified_cases": 0, "exact_final": False,
                    "error_type": "context_checkpoint_blocked", "final_status": "not_attempted",
                    "recursive_checkpoints": 4,
                    "recursive_checkpoint_blocked": 1,
                } if enabled else {
                    "verified_cases": 0, "exact_final": False,
                    "error_type": "context_overflow_confirmed", "final_status": "not_attempted",
                    "recursive_checkpoints": 0,
                    "recursive_checkpoint_blocked": 0,
                })
                (folder / "results.json").write_text(json.dumps({"trials": [result]}))
                return 0

            with patch.object(cs.benchmark, "main", simulate_main):
                self.assertEqual(cs.run(["--output", str(output)]), 0)
            summary = json.loads((output / "comparison.json").read_text())
            self.assertIn("SAFE CHECKPOINT BLOCK", summary["verdict"])

    def test_empty_output_directory_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "existing.txt").write_text("keep")
            with self.assertRaises(SystemExit):
                cs.run(["--output", str(target)])


if __name__ == "__main__":
    unittest.main()

