"""Regression tests for the actual MCP runner; no live-model efficacy claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import branch_context as b
import branch_runtime as runtime


def final(text="Done"):
    return {"content": json.dumps({"status": "success", "result": text,
                                   "what_changed": "Nothing", "next_step": "", "caveats": ""})}


def tool_call(identifier, name, **arguments):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.project = self.base / "project"
        self.project.mkdir()
        self.cfg = dict(b.DEFAULTS, model="test-model", max_branch_depth=0,
                        read_only_project_root=str(self.project), save_transcripts=True)
        for name, value in (("DATA", self.base / "branches"), ("CONFIG", self.base / "config.json")):
            p = patch.object(b, name, value)
            p.start()
            self.addCleanup(p.stop)

    def assert_protocol(self, messages):
        normalized, dropped = runtime.normalize_conversation(messages, b.SYSTEM)
        self.assertEqual(dropped, 0)
        self.assertEqual(normalized, messages)

    def test_repeated_reads_remain_bounded_and_multiple_resumes_keep_valid_exchanges(self):
        (self.project / "evidence.txt").write_text("Matching evidence record, preserved verbatim.\n" * 160)
        sent = []

        def agent(cfg, model, messages, tools=None):
            sent.append((copy.deepcopy(messages), copy.deepcopy(tools), cfg["max_output_tokens"]))
            if len(sent) <= 5:
                return {"content": "", "tool_calls": [
                    tool_call(f"read-{len(sent)}-{i}", "read_project_file", path="evidence.txt")
                    for i in range(3)]}
            return final()

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Inspect the evidence")
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["usage"]["checkpoints"], 0)
            for question in ("Explain your result", "Continue"):
                result = b.run_branch(self.cfg, question, existing_id=result["branch_id"])
                self.assertEqual(result["status"], "success")
                self.assert_protocol(b.load_branch(result["branch_id"])["transcript"])
        for messages, tools, cap in sent:
            self.assert_protocol(messages)
            self.assertLessEqual(runtime.estimate_prompt_tokens(messages, tools) + cap + 256, 8192)
        self.assertEqual(result["usage"]["model_calls"], 8)
        self.assertIn("model_reported", result["verification"])

    def test_unique_evidence_cannot_fit_stops_before_oversized_request_without_dropping_it(self):
        self.cfg.update(context_limit_tokens=2048, max_output_tokens=256)
        for name in ("a.txt", "b.txt"):
            (self.project / name).write_text((name + " unique evidence content.\n") * 160)

        def agent(cfg, model, messages, tools=None):
            return {"content": "", "tool_calls": [
                tool_call("a", "read_project_file", path="a.txt"),
                tool_call("b", "read_project_file", path="b.txt")]}

        with patch.object(b, "chat", side_effect=agent) as chat:
            result = b.run_branch(self.cfg, "Compare the two sources")
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stop_reason"], "context_budget")
        messages = b.load_branch(result["branch_id"])["transcript"]
        self.assert_protocol(messages)
        observations = [m["content"] for m in messages if m["role"] == "tool"]
        self.assertEqual(len(observations), 2)
        self.assertTrue(any("a.txt unique evidence" in text for text in observations))
        self.assertTrue(any("b.txt unique evidence" in text for text in observations))

    def test_model_call_budget_is_shared_by_children_and_survives_resume(self):
        self.cfg.update(max_branch_depth=2, read_only_project_root="", max_total_model_calls=3)
        calls = []

        def agent(cfg, model, messages, tools=None):
            calls.append(copy.deepcopy(messages))
            if not tools:
                return final("Leaf complete")
            return {"content": "", "tool_calls": [
                tool_call(f"delegate-{len(calls)}", "solve_nested_branch", task=f"Independent subtask {len(calls)}")]}

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Root investigation")
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stop_reason"], "run_call_limit")
        self.assertEqual(result["usage"]["branches"], 3)
        with patch.object(b, "chat", side_effect=AssertionError("No fresh allowance on resume")):
            resumed = b.run_branch(self.cfg, "Continue", existing_id=result["branch_id"])
        self.assertEqual(resumed["stop_reason"], "run_call_limit")
        self.assertEqual(resumed["usage"]["model_calls"], 3)
        for path in b.DATA.glob("*.json"):
            record = json.loads(path.read_text())
            self.assertEqual(record["root_id"], result["branch_id"])
            self.assert_protocol(record["transcript"])

    def test_branch_budget_keeps_completed_sibling_result_and_closes_unexecuted_call(self):
        self.cfg.update(max_branch_depth=1, read_only_project_root="", max_total_branches=2)
        calls = 0

        def agent(cfg, model, messages, tools=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"content": "", "tool_calls": [
                    tool_call("child-1", "solve_nested_branch", task="First independent task"),
                    tool_call("child-2", "solve_nested_branch", task="Second independent task")]}
            return final("First sibling really completed")

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Root")
        self.assertEqual(calls, 2)
        self.assertEqual(result["stop_reason"], "run_branch_limit")
        self.assertEqual(result["usage"]["branches"], 2)
        transcript = b.load_branch(result["branch_id"])["transcript"]
        self.assert_protocol(transcript)
        self.assertIn("First sibling really completed", transcript[-2]["content"])
        self.assertEqual(transcript[-1]["tool_call_id"], "child-2")
        self.assertIn("Not executed", transcript[-1]["content"])

    def test_duplicate_sibling_tasks_do_not_create_another_branch(self):
        self.cfg.update(max_branch_depth=1, read_only_project_root="")
        calls = 0

        def agent(cfg, model, messages, tools=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"content": "", "tool_calls": [
                    tool_call("one", "solve_nested_branch", task="Find a detail"),
                    tool_call("two", "solve_nested_branch", task=" FIND  A DETAIL ")]}
            return final()

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Root")
        self.assertEqual(result["status"], "success")
        self.assertEqual(calls, 3)
        self.assertEqual(result["usage"]["branches"], 2)
        self.assertEqual(result["usage"]["duplicate_delegations_blocked"], 1)

    def test_ancestor_task_cannot_be_redelegated_with_changed_context(self):
        self.cfg.update(max_branch_depth=2, read_only_project_root="")
        calls = 0

        def agent(cfg, model, messages, tools=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"content": "", "tool_calls": [tool_call(
                    "cycle", "solve_nested_branch", task=" ROOT ", relevant_context="Different context")]}
            self.assertIn("ancestor", messages[-1]["content"])
            return final()

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Root")
        self.assertEqual(result["usage"]["branches"], 1)
        self.assertEqual(calls, 2)

    def test_explicit_parent_id_shares_existing_ledger_and_records_child_link(self):
        self.cfg.update(max_branch_depth=2, max_total_model_calls=2)
        with patch.object(b, "chat", return_value=final()):
            parent = b.run_branch(self.cfg, "Parent")
            child = b.run_branch(self.cfg, "Child", parent_id=parent["branch_id"])
        self.assertEqual(child["usage"]["model_calls"], 2)
        self.assertEqual(child["usage"]["root_id"], parent["branch_id"])
        self.assertIn(child["branch_id"], b.load_branch(parent["branch_id"])["subbranches"])
        refreshed = b.format_parent(b.load_branch(parent["branch_id"]), self.cfg)
        self.assertEqual(refreshed["usage"]["model_calls"], 2)
        with patch.object(b, "chat", side_effect=AssertionError("Ledger must persist")):
            resumed = b.run_branch(self.cfg, "Continue child", existing_id=child["branch_id"])
        self.assertEqual(resumed["stop_reason"], "run_call_limit")

    def test_empty_reply_recovery_is_short_and_charged_to_shared_budget(self):
        caps = []

        def agent(cfg, model, messages, tools=None):
            caps.append(cfg["max_output_tokens"])
            return {"content": ""} if len(caps) == 1 else final()

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Answer")
        self.assertEqual(result["status"], "success")
        self.assertEqual(caps, [700, 256])
        self.assertEqual(result["usage"]["model_calls"], 2)
        self.assertEqual(result["usage"]["no_action_retries"], 1)
        self.assertFalse(result["usage"]["token_counts_exact_from_api"])

    def test_empty_reply_cannot_retry_past_shared_call_limit(self):
        self.cfg["max_total_model_calls"] = 1
        with patch.object(b, "chat", return_value={"content": ""}) as chat:
            result = b.run_branch(self.cfg, "Answer")
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stop_reason"], "run_call_limit")

    def test_exhausted_recovery_does_not_claim_success(self):
        with patch.object(b, "chat", return_value={"content": ""}) as chat:
            result = b.run_branch(self.cfg, "Answer")
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stop_reason"], "no_usable_response")

    def test_forced_final_cannot_bypass_the_shared_call_limit(self):
        self.cfg.update(max_agent_turns=1, max_total_model_calls=1)
        (self.project / "small.txt").write_text("Actual observed fact")
        reply = {"content": "", "tool_calls": [tool_call("read", "read_project_file", path="small.txt")]}
        with patch.object(b, "chat", return_value=reply) as chat:
            result = b.run_branch(self.cfg, "Read a fact")
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["stop_reason"], "run_call_limit")
        transcript = b.load_branch(result["branch_id"])["transcript"]
        self.assert_protocol(transcript)
        self.assertTrue(any("Actual observed fact" in m.get("content", "")
                            for m in transcript if m["role"] == "tool"))

    def test_runtime_records_real_api_usage_and_ignores_claimed_external_changes(self):
        answer = final()
        data = json.loads(answer["content"])
        data["what_changed"] = "Repaired the external service"
        answer["content"] = json.dumps(data)
        reply = b.ChatMessage(answer, {"prompt_tokens": 200, "completion_tokens": 50}, "stop")
        with patch.object(b, "chat", return_value=reply):
            result = b.run_branch(self.cfg, "Explain the service")
        self.assertEqual(result["usage"]["total_tokens"], 250)
        self.assertTrue(result["usage"]["token_counts_exact_from_api"])
        self.assertEqual(result["what_changed"], "Nothing")
        self.assertIn("not verified", result["caveats"])
        row = b.load_branch(result["branch_id"])["runtime"]["calls"][0]
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual(row["completion_tokens"], 50)

    def test_large_child_handoff_stays_valid_json_and_keeps_status_and_budget(self):
        with patch.object(b, "chat", return_value=final()):
            child = b.run_branch(self.cfg, "Child")
        child.update(result="Detailed evidence " * 300, next_step="Next step " * 50,
                     caveats="Uncertainty " * 30, subbranches=["a" * 32] * 100)
        handoff = b.nested_handoff(child)
        self.assertLessEqual(len(handoff), 4500)
        parsed = json.loads(handoff)
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["usage"]["model_calls"], 1)
        self.assertEqual(parsed["subbranch_count"], 100)

    def test_legacy_orphan_transcript_is_repaired_before_resume(self):
        with patch.object(b, "chat", return_value=final()):
            initial = b.run_branch(self.cfg, "Original")
        record = b.load_branch(initial["branch_id"])
        record["transcript"] = [{"role": "tool", "tool_call_id": "lost", "content": "Old orphan"},
                                {"role": "assistant", "content": "Old result"}]
        b.save_branch(record)

        def agent(cfg, model, messages, tools=None):
            self.assert_protocol(messages)
            self.assertIn("legacy", messages[-1]["content"])
            return final()

        with patch.object(b, "chat", agent):
            result = b.run_branch(self.cfg, "Follow up", existing_id=initial["branch_id"])
        self.assertEqual(result["status"], "success")
        self.assertGreater(b.load_branch(result["branch_id"])["runtime"]["repaired_transcript_fragments"], 0)

    def test_health_with_explicit_model_checks_api_and_rejects_missing_model(self):
        with patch.object(b, "config", return_value=self.cfg):
            with patch.object(b, "http_json", side_effect=RuntimeError("Offline")) as api:
                with self.assertRaisesRegex(RuntimeError, "Offline"):
                    b.dispatch_tool("branch_health", {})
                self.assertEqual(api.call_count, 1)
            with patch.object(b, "http_json", return_value={"data": [{"id": "other-model"}]}):
                with self.assertRaisesRegex(RuntimeError, "not advertised"):
                    b.dispatch_tool("branch_health", {})
            with patch.object(b, "http_json", return_value={"data": [{"id": "test-model"}]}):
                health = b.dispatch_tool("branch_health", {})
                self.assertTrue(health["ok"])
                self.assertIn("generation_not_tested", health["check"])

    def test_ranges_can_reach_sources_beyond_the_first_page(self):
        (self.project / "long.txt").write_text("\n".join(f"Evidence {i}" for i in range(1, 151)))
        page = b.read_project(self.cfg, "long.txt", start_line=90, max_lines=3)
        self.assertIn("90: Evidence 90", page)
        self.assertIn("92: Evidence 92", page)
        self.assertNotIn("Evidence 89", page)
        self.assertIn("start_line=93", page)
        with self.assertRaises(ValueError):
            b.read_project(self.cfg, "long.txt", start_line=0)

    def test_api_failure_keeps_call_reservation_in_persisted_ledger(self):
        with patch.object(b, "chat", side_effect=RuntimeError("Inference failed")):
            with self.assertRaisesRegex(RuntimeError, "Inference failed"):
                b.run_branch(self.cfg, "Answer")
        record = json.loads(next(b.DATA.glob("*.json")).read_text())
        ledger = json.loads(b.budget_path(record["root_id"]).read_text())
        self.assertEqual(ledger["state"]["model_calls"], 1)
        self.assertGreater(ledger["state"]["total_tokens"], 0)
        self.assertEqual(ledger["state"]["estimated_usage_calls"], 1)

    def test_new_config_defaults_are_compatible_and_invalid_budgets_rejected(self):
        b.CONFIG.write_text('{"model":"test-model"}')
        self.assertEqual(b.config()["max_total_model_calls"], 24)
        for key, value in (("max_total_model_calls", True), ("max_total_branches", 0),
                           ("max_total_tokens", -1), ("no_action_retries", 3),
                           ("context_limit_tokens", 1024)):
            b.CONFIG.write_text(json.dumps({key: value, "max_output_tokens": 1024}))
            with self.assertRaises(ValueError, msg=key):
                b.config()


class BudgetAndConversationTests(unittest.TestCase):
    def test_token_usage_reservation_and_reconciliation(self):
        cfg = dict(b.DEFAULTS, max_total_tokens=2000)
        budget = runtime.RunBudget("root", cfg)
        request = budget.before_call(1000, 200)
        self.assertEqual(budget.state["total_tokens"], 1200)
        budget.after_call(request, {"prompt_tokens": 900, "completion_tokens": 100})
        self.assertEqual(budget.state["total_tokens"], 1000)
        self.assertTrue(budget.public()["token_counts_exact_from_api"])
        with self.assertRaises(runtime.RunLimitExceeded) as caught:
            budget.before_call(1000, 200)
        self.assertEqual(caught.exception.kind, "run_token_limit")
        self.assertEqual(budget.state["model_calls"], 1)

    def test_cumulative_active_time_excludes_idle_between_resumes(self):
        cfg = dict(b.DEFAULTS, max_total_seconds=5)
        clock = [10.0]
        with patch.object(runtime.time, "monotonic", side_effect=lambda: clock[0]):
            budget = runtime.RunBudget("root", cfg)
            clock[0] = 13.0
            saved = budget.snapshot()
            clock[0] = 1000.0
            resumed = runtime.RunBudget("root", cfg, saved)
            self.assertAlmostEqual(resumed.remaining_seconds(), 2)
            clock[0] = 1002.0
            with self.assertRaises(runtime.RunLimitExceeded) as caught:
                resumed.before_call(100, 100)
        self.assertEqual(caught.exception.kind, "run_time_limit")

    def test_tools_are_included_in_prompt_budget_and_api_calibration_is_retained(self):
        messages = [{"role": "user", "content": "Short"}]
        tools = [{"description": "Long tool schema " * 100}]
        self.assertGreater(runtime.estimate_prompt_tokens(messages, tools), runtime.estimate_prompt_tokens(messages))
        budget = runtime.RunBudget("root", b.DEFAULTS)
        estimated = budget.predicted_prompt(messages, tools)
        request = budget.before_call(estimated, 200)
        budget.after_call(request, {"prompt_tokens": estimated * 2, "completion_tokens": 100})
        self.assertGreaterEqual(budget.predicted_prompt(messages, tools), estimated * 2)

    def test_deduplication_preserves_changed_versions_and_latest_observation_order(self):
        messages = [{"role": "system", "content": "Instructions"},
                    {"role": "user", "content": "Original task"}]
        for i, text in enumerate(("A", "B", "A")):
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [tool_call(str(i), "read_project_file", path="source.txt")]},
                {"role": "tool", "tool_call_id": str(i), "content": text}])
        compact, removed = runtime.deduplicate_reads(messages)
        self.assertEqual(removed, 1)
        self.assertEqual([m["content"] for m in compact if m["role"] == "tool"], ["B", "A"])
        normalized, dropped = runtime.normalize_conversation(compact, "Instructions")
        self.assertEqual(dropped, 0)
        self.assertEqual(normalized, compact)
        self.assertEqual(compact[1]["content"], "Original task")

    def test_multicall_deduplication_never_discards_child_results(self):
        calls = [tool_call("r1", "read_project_file", path="a"),
                 tool_call("r2", "read_project_file", path="a"),
                 tool_call("c1", "solve_nested_branch", task="Child")]
        messages = [{"role": "system", "content": "Instructions"},
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    *[{"role": "tool", "tool_call_id": call["id"], "content": "Observed text"} for call in calls]]
        compact, removed = runtime.deduplicate_reads(messages)
        self.assertEqual(removed, 1)
        self.assertEqual([call["id"] for call in compact[1]["tool_calls"]], ["r2", "c1"])
        self.assertEqual(runtime.normalize_conversation(compact, "Instructions")[1], 0)


if __name__ == "__main__":
    unittest.main()
