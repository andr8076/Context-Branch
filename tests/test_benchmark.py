"""Offline protocol/scoring smoke test. These are NOT model-performance results."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import benchmark as bm


class FakeJan:
    def __init__(self, max_chars=999999):
        self.max_chars = max_chars
        self.requested_limits = []

    def __call__(self, method, url, cfg, payload=None):
        if method == "GET":
            return {"data": [{"id": "fake-local-model"}]}
        self.requested_limits.append(payload["max_tokens"])
        messages = payload["messages"]
        raw = json.dumps(messages, ensure_ascii=False)
        if len(raw) > self.max_chars:
            raise RuntimeError("Simulated context capacity exceeded")
        last = messages[-1]["content"]
        if "FINAL REPORT" in last:
            mission = re.search(r'MISSION_ID=([A-Z0-9-]+)', raw).group(1)
            control = re.search(r'CONTROL_CODE=([A-Z0-9-]+)', raw).group(1)
            seals = {}
            for message in messages:
                c = message.get("content") or ""
                if c.startswith("Isolated lookup finished."):
                    found = re.search(r'"batch": "(batch_\d+)", "status": "success", "result": "(S[A-Z2-9]+)"', c)
                    if found:
                        seals[found.group(1)] = found.group(2)
                if message["role"] == "user" and c.startswith("INDEPENDENT ARCHIVE LOOKUP"):
                    label = re.search(r'INDEPENDENT ARCHIVE LOOKUP: (batch_\d+)', c).group(1)
                    target = re.search(r'EXACTLY (REF-\d+-\d+)', c).group(1)
                    match = re.search(re.escape(target) + r' \| seal=(S[A-Z2-9]+)', c)
                    seals[label] = match.group(1)
            answer = json.dumps({"mission_id": mission, "control_code": control, "seals": seals})
        elif last.startswith("Isolated lookup finished."):
            answer = 'READY'
        else:
            assert "INDEPENDENT ARCHIVE LOOKUP" in last
            target = re.search(r'EXACTLY (REF-\d+-\d+)', last).group(1)
            found = re.search(re.escape(target) + r' \| seal=(S[A-Z2-9]+)', last)
            answer = json.dumps({"status":"success","result":found.group(1), "what_changed":"Nothing","next_step":"Use this seal in the final report","caveats":""})
        return {"choices":[{"message":{"role":"assistant","content":answer}}], "usage":{"prompt_tokens":len(raw)//4, "completion_tokens":len(answer)//4}}


class BenchmarkTests(unittest.TestCase):
    def cfg(self):
        c = dict(bm.bc.DEFAULTS)
        c.update({"model": "fake-local-model", "max_branch_depth": 0, "max_agent_turns": 1, "save_transcripts": False, "max_output_tokens": 1024})
        return c

    def test_scenarios_are_deterministic_and_bounded(self):
        a = bm.create_puzzle(42, 5, 65)
        b = bm.create_puzzle(42, 5, 65)
        self.assertEqual(a, b)
        self.assertEqual(len(a["tasks"]), 5)
        self.assertTrue(all(len(prompt) < 5100 for _, prompt, _ in a["tasks"]))
        self.assertNotIn("password", a["brief"].lower())

    def test_both_modes_pass_when_model_returns_correct_answers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', FakeJan()):
            for mode in ("linear", "branch"):
                trial = bm.run_trial(mode, 77, 3, 15, self.cfg(), "fake-local-model", 8192, Path(tmp) / mode)
                self.assertFalse(trial["error"], trial["error"])
                self.assertTrue(trial["exact_final"], (mode, trial["final_parsed"]))
                self.assertEqual(trial["subtask_accuracy"], 3)
                self.assertTrue(trial["token_counts_exact_from_api"])
                self.assertEqual(trial["api_calls"], 4 if mode == "linear" else 7)


    def test_cli_default_output_cap_and_override(self):
        self.assertEqual(bm.arguments(["--dry-run"]).max_output_tokens, 1024)
        self.assertEqual(bm.arguments(["--dry-run", "--max-output-tokens", "1536"]).max_output_tokens, 1536)

    def test_output_cap_reaches_every_api_call_in_both_modes(self):
        fake = FakeJan()
        cfg = self.cfg()
        cfg["max_output_tokens"] = 1024
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, "http_json", fake):
            linear = bm.run_trial("linear", 77, 3, 15, cfg, "fake-local-model", 8192, Path(tmp) / "linear")
            branch = bm.run_trial("branch", 77, 3, 15, cfg, "fake-local-model", 8192, Path(tmp) / "branch")
        self.assertTrue(linear["exact_final"])
        self.assertTrue(branch["exact_final"])
        self.assertEqual(len(fake.requested_limits), 11)
        self.assertEqual(set(fake.requested_limits), {1024})
        for trial in (linear, branch):
            self.assertEqual(trial["max_output_tokens"], 1024)
            self.assertTrue(all(call["requested_max_tokens"] == 1024 for call in trial["calls"]))

    def test_overflow_changes_linear_not_branch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', FakeJan(max_chars=15500)):
            linear = bm.run_trial("linear", 91, 5, 50, self.cfg(), "fake-local-model", 4096, Path(tmp) / 'linear')
            branch = bm.run_trial("branch", 91, 5, 50, self.cfg(), "fake-local-model", 4096, Path(tmp) / 'branch')
            self.assertIn("Simulated context", linear["error"])
            self.assertTrue(branch["exact_final"], branch["error"])


class SummarizationJan(FakeJan):
    """Deterministic fake AI which can compact memory and use it later."""
    def __init__(self, max_chars=999999):
        super().__init__(max_chars=max_chars)
        self.messages_seen = []

    def __call__(self, method, url, cfg, payload=None):
        if method == "GET":
            return super().__call__(method, url, cfg, payload)
        self.messages_seen.append(json.loads(json.dumps(payload["messages"])))
        last = payload["messages"][-1]["content"]
        if ("PERIODIC CONTEXT COMPACTION" not in last and "FINAL REPORT" not in last):
            return super().__call__(method, url, cfg, payload)
        raw = json.dumps(payload["messages"], ensure_ascii=False)
        if len(raw) > self.max_chars:
            raise RuntimeError("Simulated context capacity exceeded")
        seals = {}
        for message in payload["messages"]:
            c = message.get("content") or ""
            if c.startswith("COMPACT MEMORY OF COMPLETED LOOKUPS:"):
                for label, seal in re.findall(r'\b(batch_\d+)\s*=\s*(S[A-Z2-9]+)', c):
                    seals[label] = seal
            if c.startswith("Isolated lookup finished."):
                found = re.search(r'"batch": "(batch_\d+)", "status": "success", "result": "(S[A-Z2-9]+)"', c)
                if found:
                    seals[found.group(1)] = found.group(2)
            if message["role"] == "user" and c.startswith("INDEPENDENT ARCHIVE LOOKUP"):
                label = re.search(r'INDEPENDENT ARCHIVE LOOKUP: (batch_\d+)', c).group(1)
                target = re.search(r'EXACTLY (REF-\d+-\d+)', c).group(1)
                match = re.search(re.escape(target) + r' \| seal=(S[A-Z2-9]+)', c)
                seals[label] = match.group(1)
        if "PERIODIC CONTEXT COMPACTION" in last:
            answer = "\n".join(f"{label}={seal}" for label, seal in sorted(seals.items()))
        else:
            mission = re.search(r'MISSION_ID=([A-Z0-9-]+)', raw).group(1)
            control = re.search(r'CONTROL_CODE=([A-Z0-9-]+)', raw).group(1)
            answer = json.dumps({"mission_id": mission, "control_code": control, "seals": seals})
        self.requested_limits.append(payload["max_tokens"])
        return {"choices": [{"message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(raw) // 4, "completion_tokens": len(answer) // 4}}


class ThirdCompetitorTests(unittest.TestCase):
    def cfg(self):
        c = dict(bm.bc.DEFAULTS)
        c.update({"model": "fake-local-model", "max_branch_depth": 0,
                  "max_agent_turns": 1, "save_transcripts": False,
                  "max_output_tokens": 1024})
        return c

    def test_three_modes_same_answers_and_output_limit(self):
        fake = SummarizationJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            trials = [bm.run_trial(mode, 77, 5, 15, self.cfg(), "fake-local-model", 8192,
                                   Path(temp) / mode)
                      for mode in ("linear", "summarized", "branch")]
        self.assertTrue(all(t["exact_final"] for t in trials), [t["error"] for t in trials])
        self.assertEqual([t["subtask_accuracy"] for t in trials], [5, 5, 5])
        self.assertEqual([t["api_calls"] for t in trials], [6, 8, 11])
        self.assertEqual(trials[1]["summaries_generated"], 2)
        self.assertTrue(all(t["max_output_tokens"] == 1024 for t in trials))
        self.assertEqual(set(fake.requested_limits), {1024})

    def test_summary_discards_old_archive_rows_not_original_instructions(self):
        fake = SummarizationJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            outcome = bm.run_trial("summarized", 77, 3, 15, self.cfg(),
                                   "fake-local-model", 8192, Path(temp))
        self.assertTrue(outcome["exact_final"])
        self.assertEqual(outcome["summaries_generated"], 1)
        messages = fake.messages_seen[-2]  # The batch_3 lookup, after compaction.
        self.assertEqual(len(messages), 4)
        joined = json.dumps(messages)
        self.assertIn("MISSION_ID=", joined)
        self.assertIn("CONTROL_CODE=", joined)
        self.assertIn("COMPACT MEMORY OF COMPLETED LOOKUPS", joined)
        self.assertIn("INDEPENDENT ARCHIVE LOOKUP: batch_3", joined)
        self.assertNotIn("INDEPENDENT ARCHIVE LOOKUP: batch_1", joined)
        self.assertNotIn("INDEPENDENT ARCHIVE LOOKUP: batch_2", joined)

    def test_summary_does_not_get_ground_truth(self):
        original = bm.create_puzzle(77, 3, 15)
        fake = SummarizationJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            bm.run_trial("summarized", 77, 3, 15, self.cfg(),
                         "fake-local-model", 8192, Path(temp))
        summary_call = next(msg for msg in fake.messages_seen if "PERIODIC CONTEXT COMPACTION" in msg[-1]["content"])
        self.assertEqual(len(summary_call), 7)  # 2 immutable + 2 lookup pairs + request
        self.assertNotIn("expected_answers", json.dumps(summary_call))
        self.assertNotIn("target_row", json.dumps(summary_call))
        self.assertIn(original["tasks"][0][2], json.dumps(summary_call))

    def test_oversized_summary_is_recorded_as_failure(self):
        class VerboseJan(SummarizationJan):
            def __call__(self, method, url, cfg, payload=None):
                if method == "POST" and "PERIODIC CONTEXT COMPACTION" in payload["messages"][-1]["content"]:
                    self.messages_seen.append(payload["messages"])
                    return {"choices": [{"message": {"role": "assistant", "content": "X" * 2000},
                                         "finish_reason": "stop"}]}
                return super().__call__(method, url, cfg, payload)
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", VerboseJan()):
            outcome = bm.run_trial("summarized", 77, 3, 15, self.cfg(),
                                   "fake-local-model", 8192, Path(temp),
                                   summary_every=2, summary_max_chars=400)
        self.assertEqual(outcome["error_type"], "summary_failed")
        self.assertEqual(outcome["final_status"], "not_attempted")
        self.assertEqual(outcome["failed_phase"], "summary:after:batch_2")

    def test_smoke_cli_writes_all_three_modes(self):
        import contextlib
        import io
        fake = SummarizationJan()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "smoke"
            with patch.object(bm.bc, "http_json", fake), \
                 patch.object(bm.bc, "config", return_value=self.cfg()), \
                 patch.object(bm.bc, "chosen_model", return_value="fake-local-model"), \
                 contextlib.redirect_stdout(io.StringIO()):
                exit_code = bm.main(["--smoke", "--output", str(folder)])
            self.assertEqual(exit_code, 0)
            data = json.loads((folder / "results.json").read_text())
            modes = {trial["mode"] for trial in data["trials"]}
            self.assertEqual(modes, {"linear", "summarized", "branch"})
            self.assertEqual(len(data["trials"]), 3)
            self.assertEqual(data["metadata"]["scoring_version"], 3)
            self.assertEqual(next(t for t in data["trials"] if t["mode"] == "summarized")["summaries_generated"], 1)
            self.assertIn("summarized", (folder / "SUMMARY.md").read_text())
            self.assertIn("summaries_generated", (folder / "results.csv").read_text())

    def test_lost_fact_in_summary_is_not_silently_restored(self):
        class ForgetfulJan(SummarizationJan):
            def __call__(self, method, url, cfg, payload=None):
                response = super().__call__(method, url, cfg, payload)
                if method == "POST" and "PERIODIC CONTEXT COMPACTION" in payload["messages"][-1]["content"]:
                    text = response["choices"][0]["message"]["content"]
                    response["choices"][0]["message"]["content"] = "\n".join(
                        line for line in text.splitlines() if not line.startswith("batch_2="))
                return response
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", ForgetfulJan()):
            trial = bm.run_trial("summarized", 77, 3, 15, self.cfg(),
                                 "fake-local-model", 8192, Path(temp))
        self.assertEqual(trial["summaries_generated"], 1)
        self.assertEqual(trial["final_status"], "incorrect_answer")
        self.assertEqual(trial["final_seals_correct"], 2)
        self.assertFalse(trial["exact_final"])
        self.assertEqual(trial["any_stage_seals_correct"], 3)

    def test_summary_cadence_and_cli(self):
        args = bm.arguments(["--dry-run", "--summary-every", "3", "--summary-max-chars", "1200"])
        self.assertEqual((args.summary_every, args.summary_max_chars), (3, 1200))
        self.assertEqual(bm.arguments(["--smoke"]).tasks, [3])
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", SummarizationJan()):
            outcome = bm.run_trial("summarized", 77, 6, 15, self.cfg(),
                                   "fake-local-model", 8192, Path(temp), summary_every=3)
        self.assertTrue(outcome["exact_final"])
        self.assertEqual(outcome["summaries_generated"], 1)


class ScoringV2Tests(unittest.TestCase):
    def setUp(self):
        self.expected = {"batch_1": "SABCDEFGH", "batch_2": "S23456789"}
        self.mission = "M-TEST"
        self.control = "C-TEST"

    def final(self, seals=None):
        return json.dumps({"mission_id": self.mission, "control_code": self.control,
                           "seals": seals if seals is not None else self.expected})

    def score(self, text, **kwargs):
        return bm.score_final_report(text, self.expected, self.mission, self.control, kwargs)

    def test_valid_complete_json_or_fence_is_strict_success(self):
        for text in (self.final(), "```json\n" + self.final() + "\n```"):
            result = self.score(text, finish_reason="stop")
            self.assertEqual(result["final_status"], "success")
            self.assertTrue(result["exact_final"])
            self.assertEqual(result["visible_seals_correct"], 2)
            self.assertEqual(result["final_seals_correct"], 2)

    def test_complete_but_wrong_value_is_incorrect_answer(self):
        result = self.score(self.final({"batch_1": "SABCDEFGH", "batch_2": "SZZZZZZZZ"}))
        self.assertEqual(result["final_status"], "incorrect_answer")
        self.assertEqual(result["final_seals_correct"], 1)
        self.assertFalse(result["exact_final"])

    def test_complete_but_missing_field_is_different_failure(self):
        result = self.score(json.dumps({"mission_id": self.mission, "seals": self.expected}))
        self.assertEqual(result["final_status"], "missing_fields")
        self.assertIn("control_code", result["final_missing_fields"])
        self.assertEqual(result["final_seals_correct"], 2)
        self.assertFalse(result["exact_final"])

    def test_partial_json_can_recover_only_fully_visible_values(self):
        text = ('```json\n{"mission_id": "M-TEST", "control_code": "C-TEST", '
                '"seals": {"batch_1": "SABCDEFGH", "batch_2": "S234')
        result = self.score(text, finish_reason="length", completion_tokens=350, requested_max_tokens=350)
        self.assertEqual(result["final_status"], "output_truncated_confirmed")
        self.assertFalse(result["final_json_valid"])
        self.assertFalse(result["exact_final"])
        self.assertEqual(result["final_seals_correct"], 0)
        self.assertEqual(result["visible_seals_correct"], 1)
        self.assertEqual(result["visible_seals_count"], 1)
        self.assertTrue(result["visible_mission_ok"])
        self.assertTrue(result["visible_control_ok"])
        self.assertFalse(result["mission_ok"])

    def test_unknown_finish_reason_uses_suspected_not_confirmed(self):
        result = self.score('{"mission_id": "M-TEST",',
                            completion_tokens=350, requested_max_tokens=350)
        self.assertEqual(result["final_status"], "output_truncated_suspected")
        self.assertEqual(result["final_output_truncation"], "suspected")

    def test_incomplete_not_near_limit_is_malformed(self):
        result = self.score('{"mission_id": "M-TEST",',
                            completion_tokens=12, requested_max_tokens=1024)
        self.assertEqual(result["final_status"], "malformed_json")
        self.assertEqual(result["final_output_truncation"], "none")

    def test_partial_json_must_not_steal_random_nested_fields(self):
        text = '{"seals":{"batch_1":"SABCDEFGH", "batch_2":"S23456'
        result = self.score(text)
        self.assertEqual(result["visible_seals_correct"], 1)
        self.assertFalse(result["exact_final"])

    def test_error_classes_do_not_invent_context_or_timeout(self):
        self.assertEqual(bm.classify_exception(RuntimeError("Simulated context capacity exceeded")),
                         'context_overflow_confirmed')
        self.assertEqual(bm.classify_exception(TimeoutError("request timed out")), 'timeout')
        self.assertEqual(bm.classify_exception(RuntimeError("Cannot reach Jan API (or request timed out). Start Jan")),
                         'connection_or_timeout_unknown')
        self.assertEqual(bm.classify_exception(RuntimeError("HTTP 502")), 'api_or_runtime_error')
        self.assertEqual(bm.classify_exception(bm.bc.LocalAPIError(400, "context budget exceeded", "context_overflow_confirmed")),
                         'context_overflow_confirmed')

    def test_lookup_recovers_plain_seal_without_looking_into_archive(self):
        self.assertEqual(bm.extract_lookup_answer('seal: SABCDEFGH'), 'SABCDEFGH')
        self.assertEqual(bm.extract_lookup_answer('SABCDEFGH'), 'SABCDEFGH')
        self.assertEqual(bm.extract_lookup_answer('random REF-01-0001 | seal=SABCDEFGH'), '')

    def test_full_final_may_recover_missing_direct_lookup(self):
        class UnstructuredJan(FakeJan):
            def __call__(self, method, url, cfg, payload=None):
                if method == 'POST' and 'INDEPENDENT ARCHIVE LOOKUP' in payload['messages'][-1]['content']:
                    self.requested_limits.append(payload['max_tokens'])
                    return {"choices": [{"message": {"role": "assistant", "content": "READY"},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 25, "completion_tokens": 2}}
                return super().__call__(method, url, cfg, payload)

        c = dict(bm.bc.DEFAULTS)
        c.update({"model": "fake-local-model", "max_branch_depth": 0, "max_agent_turns": 1,
                  "save_transcripts": False, "max_output_tokens": 1024})
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', UnstructuredJan()):
            result = bm.run_trial('linear', 77, 2, 15, c, 'fake-local-model', 8192, Path(tmp))
        self.assertEqual(result['subtask_accuracy'], 0)
        self.assertEqual(result['final_recovered_subtasks'], 2)
        self.assertEqual(result['any_stage_seals_correct'], 2)
        self.assertEqual(result['final_seals_correct'], 2)
        self.assertTrue(result['exact_final'])

    def test_exception_is_not_scored_as_wrong_json_and_store_restores(self):
        original = bm.bc.DATA
        c = dict(bm.bc.DEFAULTS)
        c.update({"model": "fake-local-model", "max_branch_depth": 0,
                  "max_agent_turns": 1, "save_transcripts": False, "max_output_tokens": 1024})
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', FakeJan(max_chars=15)):
            result = bm.run_trial('linear', 77, 3, 15, c, 'fake-local-model', 8192, Path(tmp))
        self.assertEqual(result['final_status'], 'not_attempted')
        self.assertEqual(result['error_type'], 'context_overflow_confirmed')
        self.assertTrue(result['failed_phase'].startswith('lookup:'))
        self.assertEqual(result['solved_subtasks'], 0)
        self.assertEqual(bm.bc.DATA, original)

    def test_csv_summary_reports_new_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', FakeJan()):
            c = dict(bm.bc.DEFAULTS)
            c.update({"model": "fake-local-model", "max_branch_depth": 0, "max_agent_turns": 1,
                      "save_transcripts": False, "max_output_tokens": 1024})
            trial = bm.run_trial('linear', 77, 2, 15, c, 'fake-local-model', 8192, Path(tmp))
            args = bm.arguments(['--smoke'])
            bm.write_files(Path(tmp), [trial], c, args)
            summary = (Path(tmp) / 'SUMMARY.md').read_text()
            csv_file = (Path(tmp) / 'results.csv').read_text()
            self.assertIn('scoring v3', summary)
            self.assertIn('Direct lookups correct / attempted', summary)
            self.assertIn('final_status', csv_file)
            self.assertIn('visible_seals_correct', csv_file)


if __name__ == '__main__':
    unittest.main()
