"""Offline tests for the interactive trap simulator, NOT measured AI intelligence."""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark as bm
import trap_benchmark as tb


class DeliberateFakeJan:
    """Scripted, deterministic JSON-choice agent that reads the simulator text.

    It's an integration test double. It never reads the puzzle oracle; it derives
    the proposed repair from actual documents shown in the conversation.
    """
    def __init__(self, max_chars=999999):
        self.max_chars = max_chars
        self.messages_seen = []
        self.limits = []

    def __call__(self, method, url, cfg, payload=None):
        if method == "GET":
            return {"data": [{"id": "fake-model"}]}
        messages = payload["messages"]
        self.messages_seen.append(json.loads(json.dumps(messages)))
        self.limits.append(payload["max_tokens"])
        serialized = json.dumps(messages, ensure_ascii=False)
        if len(serialized) > self.max_chars:
            raise RuntimeError("Simulated context capacity exceeded")
        last = messages[-1]["content"]
        if "PERIODIC CONTEXT COMPACTION for simulated repair" in last:
            answer = "\n".join(f"{k}={v}" for k, v in sorted(self._found_seals(messages).items()))
        elif "FINAL REPORT" in last:
            mission = re.search(r'MISSION_ID=([A-Z0-9-]+)', serialized).group(1)
            control = re.search(r'CONTROL_CODE=([A-Z0-9-]+)', serialized).group(1)
            answer = json.dumps({"mission_id": mission, "control_code": control,
                                 "seals": self._found_seals(messages)})
        elif last.startswith("Isolated repair finished."):
            answer = "READY"
        else:
            # Work only on the most recent incident, to avoid counting old docs.
            start = max(i for i, m in enumerate(messages)
                        if "SIMULATED REPAIR INCIDENT batch_" in m.get("content", ""))
            fragment = messages[start:]
            visited = [a["path"] for m in fragment if m["role"] == "assistant"
                       for a in [tb.parse_action(m["content"])]
                       if a and a["action"] == "inspect"]
            initial = ("guide/index", "guide/policy", "registry/releases")
            if any(p not in visited for p in initial):
                answer = json.dumps({"action": "inspect", "path": next(p for p in initial if p not in visited)})
            else:
                answer = self._next_action(fragment, visited)
        return {"choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(serialized)//4, "completion_tokens": len(answer)//4}}

    def _next_action(self, fragment, visited):
        index = next(m["content"] for m in fragment
                     if "SIMULATOR RESULT:\nEvidence cabinet" in m.get("content", ""))
        for stage, success_text, action_name in (
                ("index", "INDEX REBUILT:", "rebuild_index"),
                ("escrow", "ESCROW UNLOCKED:", "unlock_escrow")):
            if f"{stage}/shard-a" not in index:
                continue
            for path in (f"{stage}/shard-a", f"{stage}/shard-b"):
                if path not in visited:
                    return json.dumps({"action": "inspect", "path": path})
            if not any(success_text in m.get("content", "") for m in fragment):
                return json.dumps({"action": action_name, "token": self._derive_nested(fragment, stage)})
        for path in ("recovery/shard-a", "recovery/shard-b"):
            if path not in visited:
                return json.dumps({"action": "inspect", "path": path})
        if not any("RECOVERY COMPLETE:" in m.get("content", "") for m in fragment):
            return json.dumps({"action": "recover", "token": self._derive_recovery(fragment)})
        for path in ("registry/approvals", "network/aliases", "network/routes", "security/certificates"):
            if path not in visited:
                return json.dumps({"action": "inspect", "path": path})
        attempts = sum(bool(tb.parse_action(m["content"]) and
                         tb.parse_action(m["content"])["action"] == "repair")
                       for m in fragment if m["role"] == "assistant")
        failures = sum("SIMULATOR RESULT:\nREJECTED:" in m.get("content", "") for m in fragment)
        if attempts <= failures:
            return json.dumps({"action": "repair", **self._derive_repair(fragment)})
        return json.dumps({"action": "verify"})

    def _derive_nested(self, fragment, stage):
        index = next(m["content"] for m in fragment
                     if "SIMULATOR RESULT:\nEvidence cabinet" in m.get("content", ""))
        level = 2 if stage == "index" else 1
        rows = re.findall(
            r'STEP [12] \(nested depth ([12])\):.*?Match run=(RUN-[A-Z2-9]+).*?'
            r'join fragments in order ([AB]{2})', index, re.S)
        nonce, order = next((nonce, order) for depth, nonce, order in rows if int(depth) == level)
        fragments = {}
        for msg in fragment:
            content = msg.get("content", "")
            if not content.startswith(f"SIMULATOR RESULT:\n{stage.upper()} SHARD"):
                continue
            for run, slot, state, value in re.findall(
                    r'run=(RUN-[A-Z2-9]+) slot=([AB]) state=(CURRENT|ARCHIVED|STALE|REVOKED) '
                    r'fragment=([A-Z2-9]+)', content):
                if run == nonce and state == "CURRENT":
                    fragments[slot] = value
        return "".join(fragments[c] for c in order)

    def _derive_recovery(self, fragment):
        index = next(m["content"] for m in fragment if "SIMULATOR RESULT:\nEvidence cabinet" in m.get("content", ""))
        nonce = re.search(r'MANDATORY RECOVERY:.*?Match run=(RUN-[A-Z2-9]+)', index, re.S).group(1)
        order = re.search(r'MANDATORY RECOVERY:.*?join fragments in order ([AB]{2})', index, re.S).group(1)
        fragments = {}
        for m in fragment:
            content = m.get("content", "")
            if "SIMULATOR RESULT:\nRECOVERY SHARD" not in content:
                continue
            for run, slot, state, value in re.findall(
                r'run=(RUN-[A-Z2-9]+) slot=([AB]) state=(CURRENT|ARCHIVED|STALE|REVOKED) fragment=([A-Z2-9]+)', content):
                if run == nonce and state == "CURRENT":
                    fragments[slot] = value
        return "".join(fragments[slot] for slot in order)

    def _found_seals(self, messages):
        seals = {}
        for idx, m in enumerate(messages):
            c = m.get("content", "")
            if "COMPACT MEMORY OF COMPLETED LOOKUPS:" in c:
                seals.update(dict(re.findall(r'\b(batch_\d+)=(S[A-Z2-9]{8})\b', c)))
            if c.startswith("Isolated repair finished."):
                match = re.search(r'"batch": "(batch_\d+)", "status": "verified", "seal": "(S[A-Z2-9]{8})"', c)
                if match:
                    seals[match.group(1)] = match.group(2)
            if "SIMULATOR RESULT:\nVERIFIED:" in c:
                match = re.search(r'seal=(S[A-Z2-9]{8})', c)
                label = None
                for earlier in reversed(messages[:idx]):
                    candidate = re.search(r'SIMULATED REPAIR INCIDENT (batch_\d+)', earlier.get("content", ""))
                    if candidate:
                        label = candidate.group(1)
                        break
                if label and match:
                    seals[label] = match.group(1)
        return seals

    def _derive_repair(self, fragment):
        docs = {}
        for m in fragment:
            content = m.get("content", "")
            if not content.startswith("SIMULATOR RESULT:\n"):
                continue
            raw = content.removeprefix("SIMULATOR RESULT:\n").split("\nContinue the incident;")[0]
            for name, prefix in (("releases", "RELEASE INVENTORY"), ("approvals", "SIGNED-RELEASE LEDGER"),
                                 ("aliases", "ALIAS GRAPH"), ("routes", "ROUTE INVENTORY"),
                                 ("certificates", "CURRENT CERTIFICATE VALIDATION")):
                if raw.startswith(prefix):
                    docs[name] = raw
        releases = re.findall(r'rank=(\d+) id=(REL-[A-Z2-9]+) state=(\w+) primary=(\S+) fallback=(\S+) zone=(\S+)', docs["releases"])
        approvals = dict(re.findall(r'\b(REL-[A-Z2-9]+) (SIGNED|REVOKED|PENDING)\b', docs["approvals"]))
        rank, release, _, primary, fallback, zone = max((int(rank), rev, status, primary, fallback, zone)
                   for rank, rev, status, primary, fallback, zone in releases
                   if status == "ACTIVE" and approvals.get(rev) == "SIGNED")
        aliases = dict(re.findall(r'\b([A-Z]+-[A-Z2-9]+) -> ([A-Z]+-[A-Z2-9]+)', docs["aliases"]))
        routes = re.findall(r'\b(ROUTE-[A-Z2-9]+) node=(\S+) zone=(\S+) state=(\S+) cert=(\S+)', docs["routes"])
        certs = dict(re.findall(r'\b(CERT-[A-Z2-9]+) (EXPIRED|VALID)\b', docs["certificates"]))
        def resolve(alias):
            while alias in aliases:
                alias = aliases[alias]
            return alias
        for candidate in (primary, fallback):
            node = resolve(candidate)
            for rid, rnode, rzone, status, cert in routes:
                if node == rnode and rzone == zone and status == "LIVE" and certs.get(cert) == "VALID":
                    return {"release": release, "route": rid, "certificate": cert}
        return {"release": "NOT-FOUND", "route": "NOT-FOUND", "certificate": "NOT-FOUND"}


class TrapBenchmarkTests(unittest.TestCase):
    def cfg(self):
        c = dict(bm.bc.DEFAULTS)
        c.update({"model": "fake-model", "max_output_tokens": 1024, "save_transcripts": False})
        return c

    def test_reproducible_no_oracle_in_prompt(self):
        a = tb.create_trap_puzzle(123, 4, 40, 3)
        b = tb.create_trap_puzzle(123, 4, 40, 3)
        self.assertEqual(a, b)
        self.assertEqual(len(a["cases"]), 4)
        for case in a["cases"]:
            self.assertNotIn(case["seal"], json.dumps(case["docs"]))
            self.assertNotIn(case["expected_repair"]["release"], case["intro"])
            self.assertTrue(len(case["docs"]["logs/diagnostics"]) > 2000)

    def test_wrong_repair_generates_large_detour_and_no_verify_seal(self):
        case = tb.create_trap_puzzle(42, 1, 60, 3)["cases"][0]
        sim = tb.RepairSimulator(case)
        sim.step({"action": "inspect", "path": "recovery/shard-a"})
        sim.step({"action": "inspect", "path": "recovery/shard-b"})
        sim.step({"action": "recover", "token": case["recovery_token"]})
        for doc in ("registry/releases", "registry/approvals", "network/aliases",
                    "network/routes", "security/certificates"):
            sim.step({"action": "inspect", "path": doc})
        reply = sim.step({"action": "repair", "release": "REL-BOGUS", "route": "ROUTE-BOGUS", "certificate": "CERT-BOGUS"})
        self.assertIn("REJECTED", reply)
        self.assertGreater(len(reply), 3000)
        self.assertEqual(sim.wrong_repairs, 1)
        self.assertGreater(sim.trap_chars, 3000)
        self.assertIn("VERIFY FAILED", sim.step({"action": "verify"}))
        self.assertFalse(sim.verified)

    def test_primary_route_expired_fallback_route_necessary(self):
        case = tb.create_trap_puzzle(22, 1)["cases"][0]
        docs = case["docs"]
        routes = re.findall(r'(ROUTE-[A-Z2-9]+) node=\S+ zone=\S+ state=LIVE cert=(CERT-[A-Z2-9]+)', docs["network/routes"])
        certs = dict(re.findall(r'(CERT-[A-Z2-9]+) (VALID|EXPIRED)', docs["security/certificates"]))
        self.assertEqual(sum(certs[cert] == "EXPIRED" for _, cert in routes), 1)
        sim = tb.RepairSimulator(case)
        repair = case["expected_repair"]
        self.assertIn("REPAIR BLOCKED", sim.step({"action": "repair", **repair}))
        sim.step({"action": "inspect", "path": "recovery/shard-a"})
        sim.step({"action": "inspect", "path": "recovery/shard-b"})
        self.assertIn("RECOVERY COMPLETE", sim.step({"action": "recover", "token": case["recovery_token"]}))
        for doc in ("registry/releases", "registry/approvals", "network/aliases",
                    "network/routes", "security/certificates"):
            sim.step({"action": "inspect", "path": doc})
        self.assertIn("APPLIED", sim.step({"action": "repair", **repair}))
        self.assertFalse(sim.verified)
        self.assertIn(case["seal"], sim.step({"action": "verify"}))
        self.assertTrue(sim.verified)

    def test_pressure_is_deterministic_and_preserves_underlying_repair(self):
        baseline = tb.create_trap_puzzle(928, 2, nested_depth=2, pressure_records=0)
        pressured = tb.create_trap_puzzle(928, 2, nested_depth=2,
                                         pressure_records=10, pressure_detail_chars=80)
        self.assertEqual(pressured, tb.create_trap_puzzle(
            928, 2, nested_depth=2, pressure_records=10, pressure_detail_chars=80))
        for plain, noisy in zip(baseline["cases"], pressured["cases"]):
            self.assertEqual(noisy["expected_repair"], plain["expected_repair"])
            self.assertEqual(noisy["seal"], plain["seal"])
            self.assertEqual(noisy["recovery_token"], plain["recovery_token"])
            self.assertEqual(noisy["nested"], plain["nested"])
            self.assertEqual(len(noisy["pressure_deltas"]), 11)
            self.assertTrue(all(value > 0 for value in noisy["pressure_deltas"].values()))
            self.assertNotIn(noisy["seal"], json.dumps(noisy["docs"]))
            self.assertGreater(sum(noisy["pressure_deltas"].values()), 15000)
            for path, increase in noisy["pressure_deltas"].items():
                self.assertEqual(len(noisy["docs"][path]) - len(plain["docs"][path]), increase)
            for doc in ("registry/releases", "registry/approvals", "network/aliases",
                        "network/routes", "security/certificates"):
                self.assertIn("audit=", noisy["docs"][doc])

    def test_pressure_requires_authority_investigation_before_repair(self):
        case = tb.create_trap_puzzle(928, 1, nested_depth=2,
                                     pressure_records=4, pressure_detail_chars=64)["cases"][0]
        sim = tb.RepairSimulator(case)
        for stage, path, action in (("index", "index", "rebuild_index"),
                                    ("escrow", "escrow", "unlock_escrow")):
            for slot in ("a", "b"):
                sim.step({"action": "inspect", "path": f"{path}/shard-{slot}"})
            self.assertIn("UNLOCKED" if stage == "escrow" else "REBUILT", sim.step({
                "action": action, "token": case["nested"][stage]["token"]}))
        for slot in ("a", "b"):
            sim.step({"action": "inspect", "path": f"recovery/shard-{slot}"})
        self.assertIn("RECOVERY COMPLETE", sim.step({
            "action": "recover", "token": case["recovery_token"]}))
        self.assertIn("REPAIR BLOCKED: all five", sim.step({
            "action": "repair", **case["expected_repair"]}))
        self.assertEqual(sim.authority_blocked, 1)
        for path in ("registry/releases", "registry/approvals", "network/aliases",
                     "network/routes", "security/certificates"):
            sim.step({"action": "inspect", "path": path})
        self.assertIn("APPLIED", sim.step({"action": "repair", **case["expected_repair"]}))
        self.assertIn("VERIFIED", sim.step({"action": "verify"}))
        self.assertEqual(len(sim.authority_views), 5)
        self.assertEqual(len(sim.pressure_views), 11)
        self.assertEqual(sim.pressure_evidence_chars, sum(case["pressure_deltas"].values()))
        self.assertGreater(sim.authority_evidence_chars, 500)
        self.assertFalse(sim.trap_views)

    def test_three_modes_complete_with_pressure_and_export_metrics(self):
        fake = DeliberateFakeJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            outputs = [tb.run_trap_trial(
                mode, 481, 2, self.cfg(), "fake-model", 8192, Path(temp), bm,
                nested_depth=2, nested_log_lines=6, recovery_log_lines=12,
                pressure_records=4, pressure_detail_chars=32)
                for mode in ("linear", "summarized", "branch")]
        for result in outputs:
            self.assertEqual(result["verified_cases"], 2, result["case_breakdown"])
            self.assertTrue(result["exact_final"], result.get("error"))
            self.assertEqual(result["pressure_records"], 4)
            self.assertEqual(result["authority_docs_viewed"], 10)
            self.assertEqual(result["pressure_evidence_chars"], result["pressure_generated_chars"])
            self.assertGreater(result["pressure_evidence_chars"], 5000)
            self.assertEqual(result["authority_blocked"], 0)
            self.assertEqual(result["wrong_repairs"], 0)

    def test_pressure_settings_cli_dry_run_warning_and_validation(self):
        with contextlib.redirect_stdout(io.StringIO()) as sink:
            status = bm.main(["--scenario", "repair", "--tasks", "1", "--dry-run",
                              "--pressure-records", "30", "--pressure-detail-chars", "96"])
        self.assertEqual(status, 0)
        self.assertIn("Required evidence per incident", sink.getvalue())
        self.assertIn("WARNING", sink.getvalue())
        for args in (("--pressure-records", "-1"), ("--pressure-records", "161"),
                     ("--pressure-detail-chars", "257")):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bm.arguments(["--scenario", "repair", args[0], args[1]])

    def test_parser_rejects_unstructured_or_extra_text(self):
        self.assertEqual(tb.parse_action('hello'), None)
        self.assertEqual(tb.parse_action('{"action":"repair"}\n{"action":"verify"}'), None)
        self.assertEqual(tb.parse_action('{"action":"inspect","path":"guide/policy"}')['path'], "guide/policy")

    def test_parser_normalizes_qwen_target_alias(self):
        action = tb.parse_action('{"action":"inspect","target":"guide/index"}')
        self.assertEqual(action['path'], "guide/index")
        self.assertEqual(action['_schema_alias'], "target_to_path")
        action = tb.parse_action(
            '{"action":"repair","target":"REL-1234","route":"ROUTE-5678",'
            '"certificate":"CERT-9012"}')
        self.assertEqual(action['release'], "REL-1234")
        self.assertEqual(action['_schema_alias'], "target_to_release")

    def test_all_modes_finish_with_evidence_based_fake_agent(self):
        fake = DeliberateFakeJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            outputs = [tb.run_trap_trial(mode, 77, 3, self.cfg(), "fake-model", 8192,
                                         Path(temp) / mode, bm, 2, 1800, 25, 2, 14)
                       for mode in ("linear", "summarized", "branch")]
        for r in outputs:
            self.assertEqual(r["verified_cases"], 3, (r["mode"], r["error"], r["case_breakdown"]))
            self.assertTrue(r["exact_final"], (r["mode"], r["error"], r.get("final_parsed")))
            self.assertEqual(r["total_actions"], 36)
            self.assertEqual(r["recovery_completed"], 3)
            self.assertEqual(r["recovery_shard_views"], 6)
            self.assertGreater(r["forced_evidence_chars"], 10000)
            self.assertEqual(r["wrong_repairs"], 0)
        self.assertEqual(outputs[1]["summaries_generated"], 1)
        self.assertTrue(all(v == 1024 for v in fake.limits))

    def test_repair_summary_empty_then_retry_succeeds(self):
        # Regression: hidden reasoning previously consumed all summary output
        # tokens, halting an otherwise correctly solved repair run.
        scripted = DeliberateFakeJan()
        summary_seen = 0
        seen = []

        def fake(method, url, cfg, payload=None):
            nonlocal summary_seen
            if payload:
                seen.append(payload)
                if "PERIODIC CONTEXT COMPACTION for simulated repair" in payload["messages"][-1]["content"]:
                    summary_seen += 1
                    if summary_seen == 1:
                        return {"choices": [{"message": {"content": "", "reasoning_content": "hidden"},
                                             "finish_reason": "length"}],
                                "usage": {"prompt_tokens": 800, "completion_tokens": 2048,
                                          "completion_tokens_details": {"reasoning_tokens": 2048}}}
            return scripted(method, url, cfg, payload)

        cfg = self.cfg()
        cfg["_summary_output_tokens"] = 2048
        cfg["_summary_retries"] = 1
        cfg["_report_reasoning_effort"] = "none"
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, "http_json", fake):
            result = tb.run_trap_trial("summarized", 17, 3, cfg, "fake-model", 8192, Path(tmp), bm,
                                       summary_every=2, nested_depth=0)
        self.assertEqual(result["verified_cases"], 3)
        self.assertTrue(result["exact_final"], result["error"])
        self.assertEqual(result["summaries_generated"], 1)
        self.assertEqual(result["summary_attempts"], 2)
        self.assertEqual(result["summary_retry_recoveries"], 1)
        summary_api_calls = [c for c in seen
                             if "PERIODIC CONTEXT COMPACTION" in c["messages"][-1]["content"]]
        self.assertEqual(len(summary_api_calls), 2)
        self.assertTrue(all(c["max_tokens"] == 2048 for c in summary_api_calls))
        self.assertTrue(all(c.get("reasoning_effort") == "none" for c in summary_api_calls))

    def test_branch_isolation_and_summary_compaction(self):
        fake = DeliberateFakeJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            tb.run_trap_trial("summarized", 17, 3, self.cfg(), "fake-model", 8192, Path(temp), bm, 2)
        incident_three = next(msgs for msgs in fake.messages_seen
                              if any("INCIDENT TO SOLVE: SIMULATED REPAIR INCIDENT batch_3" in m.get("content", "") for m in msgs))
        combined = json.dumps(incident_three)
        self.assertIn("COMPACT MEMORY OF COMPLETED LOOKUPS", combined)
        self.assertNotIn("INCIDENT TO SOLVE: SIMULATED REPAIR INCIDENT batch_1", combined)
        self.assertNotIn("RELEASE INVENTORY", combined)
        self.assertIn("MISSION_ID", combined)

    def test_context_pressure_linear_vs_branch(self):
        fake = DeliberateFakeJan(max_chars=36000)
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            linear = tb.run_trap_trial("linear", 17, 3, self.cfg(), "fake-model", 8192, Path(temp) / "a", bm)
            branch = tb.run_trap_trial("branch", 17, 3, self.cfg(), "fake-model", 8192, Path(temp) / "b", bm)
        self.assertTrue(linear["error"])
        self.assertTrue(linear["input_over_limit_risk"] or linear["error_type"] == "context_overflow_confirmed")
        self.assertTrue(branch["exact_final"], branch["error"])


    def test_agent_recovers_after_baited_wrong_repair(self):
        class MisledOnceJan(DeliberateFakeJan):
            def __init__(self):
                super().__init__()
                self.misled_cases = set()
            def __call__(self, method, url, cfg, payload=None):
                result = super().__call__(method, url, cfg, payload)
                if method != "POST":
                    return result
                messages = payload["messages"]
                last_case = next((re.search(r"SIMULATED REPAIR INCIDENT (batch_\d+)", m.get("content", ""))
                                  for m in reversed(messages)
                                  if "SIMULATED REPAIR INCIDENT batch_" in m.get("content", "")), None)
                proposed = tb.parse_action(result["choices"][0]["message"]["content"])
                if last_case and proposed and proposed["action"] == "repair" and last_case.group(1) not in self.misled_cases:
                    self.misled_cases.add(last_case.group(1))
                    result["choices"][0]["message"]["content"] = json.dumps({
                        "action": "repair", "release": "REL-FAKE", "route": "ROUTE-FAKE", "certificate": "CERT-FAKE"})
                return result
        fake = MisledOnceJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake):
            result = tb.run_trap_trial("branch", 222, 1, self.cfg(), "fake-model", 8192,
                                       Path(temp), bm, trap_log_lines=50, max_actions=15)
        self.assertTrue(result["exact_final"], result.get("error"))
        self.assertEqual(result["recovery_completed"], 1)
        self.assertEqual(result["wrong_repairs"], 1)
        self.assertGreater(result["trap_waste_chars"], 3500)
        self.assertEqual(result["verified_cases"], 1)
        self.assertEqual(result["total_actions"], 13)

    def test_mandatory_evidence_unlocks_correctly_and_wrong_run_is_rejected(self):
        for seed in range(20, 40):
            case = tb.create_trap_puzzle(seed, 1, recovery_log_lines=5)["cases"][0]
            sim = tb.RepairSimulator(case)
            self.assertIn("LEDGER LOCKED", sim.step({"action": "inspect", "path": "registry/approvals"}))
            self.assertIn("REPAIR BLOCKED", sim.step({"action": "repair", **case["expected_repair"]}))
            self.assertIn("RECOVERY REJECTED", sim.step({"action": "recover", "token": case["recovery_token"]}))
            self.assertIn("RECOVERY SHARD A", sim.step({"action": "inspect", "path": "recovery/shard-a"}))
            self.assertIn("RECOVERY REJECTED", sim.step({"action": "recover", "token": case["recovery_token"]}))
            self.assertIn("RECOVERY SHARD B", sim.step({"action": "inspect", "path": "recovery/shard-b"}))
            self.assertIn("RECOVERY REJECTED", sim.step({"action": "recover", "token": "WRONG"}))
            self.assertIn("RECOVERY COMPLETE", sim.step({"action": "recover", "token": case["recovery_token"]}))
            self.assertIn("SIGNED-RELEASE LEDGER", sim.step({"action": "inspect", "path": "registry/approvals"}))
            for doc in ("registry/releases", "network/aliases",
                        "network/routes", "security/certificates"):
                sim.step({"action": "inspect", "path": doc})
            self.assertIn("APPLIED", sim.step({"action": "repair", **case["expected_repair"]}))
            self.assertIn("VERIFIED", sim.step({"action": "verify"}))
            self.assertEqual(sim.recovery_failures, 3)
            self.assertEqual(sim.recovery_blocked, 2)
            self.assertEqual(len(sim.recovery_shards_viewed), 2)
            self.assertGreater(sim.forced_evidence_chars, 500)
            self.assertEqual(sim.wrong_repairs, 0)
            self.assertEqual(sim.trap_chars, 0)

    def test_authoritative_evidence_always_required_even_without_pressure(self):
        """A correct repair proposal is not evidence of a valid investigation."""
        mandatory = ("registry/releases", "registry/approvals", "network/aliases",
                     "network/routes", "security/certificates")
        for records in (0, 1, 4):
            case = tb.create_trap_puzzle(121, 1, recovery_log_lines=4,
                                         pressure_records=records)["cases"][0]
            sim = tb.RepairSimulator(case)
            for path in ("recovery/shard-a", "recovery/shard-b"):
                sim.step({"action": "inspect", "path": path})
            self.assertIn("RECOVERY COMPLETE", sim.step(
                {"action": "recover", "token": case["recovery_token"]}))
            self.assertIn("EVIDENCE CHECK", case["docs"]["guide/policy"])
            for count, path in enumerate(mandatory):
                self.assertIn("REPAIR BLOCKED", sim.step({
                    "action": "repair", **case["expected_repair"]}))
                self.assertEqual(len(sim.authority_views), count)
                self.assertFalse(sim.applied)
                sim.step({"action": "inspect", "path": path})
            self.assertEqual(sim.authority_views, set(mandatory))
            self.assertIn("APPLIED", sim.step({
                "action": "repair", **case["expected_repair"]}))
            self.assertIn("VERIFIED", sim.step({"action": "verify"}))
            self.assertEqual(sim.authority_blocked, len(mandatory))
            self.assertTrue(sim.verified)

    def test_policy_does_not_allow_correct_guess_after_only_four_documents(self):
        case = tb.create_trap_puzzle(170, 1, pressure_records=0)["cases"][0]
        sim = tb.RepairSimulator(case)
        for path in ("recovery/shard-a", "recovery/shard-b"):
            sim.step({"action": "inspect", "path": path})
        sim.step({"action": "recover", "token": case["recovery_token"]})
        for path in ("registry/releases", "registry/approvals", "network/aliases",
                     "network/routes"):
            sim.step({"action": "inspect", "path": path})
        self.assertEqual(len(sim.authority_views), 4)
        result = sim.step({"action": "repair", **case["expected_repair"]})
        self.assertIn("REPAIR BLOCKED", result)
        self.assertFalse(sim.applied)
        self.assertEqual(sim.wrong_repairs, 0)
        self.assertEqual(sim.authority_blocked, 1)

    def test_shards_have_current_and_misleading_records(self):
        for seed in range(15):
            case = tb.create_trap_puzzle(seed, 1, recovery_log_lines=30)["cases"][0]
            fake = DeliberateFakeJan()
            sim = tb.RepairSimulator(case)
            outputs = [sim.step({"action": "inspect", "path": p}) for p in
                       ("guide/index", "recovery/shard-a", "recovery/shard-b")]
            token = fake._derive_recovery([{"content": "SIMULATOR RESULT:\n" + line} for line in outputs])
            self.assertEqual(token, case["recovery_token"])
            self.assertNotIn(token, outputs[0])
            self.assertTrue(all("state=ARCHIVED" in o and "state=CURRENT" in o for o in outputs[1:]))

    def test_nested_depth_two_cannot_be_bypassed(self):
        case = tb.create_trap_puzzle(837, 1, nested_depth=2, nested_log_lines=12)["cases"][0]
        sim = tb.RepairSimulator(case)
        self.assertIn("LOCKED", sim.step({"action": "inspect", "path": "recovery/shard-a"}))
        self.assertIn("LOCKED", sim.step({"action": "inspect", "path": "escrow/shard-a"}))
        self.assertIn("REPAIR BLOCKED", sim.step({"action": "repair", **case["expected_repair"]}))
        self.assertIn("REJECTED", sim.step({"action": "recover", "token": case["recovery_token"]}))
        self.assertIn("REJECTED", sim.step({"action": "unlock_escrow", "token": case["nested"]["escrow"]["token"]}))
        self.assertIn("REJECTED", sim.step({"action": "rebuild_index", "token": case["nested"]["index"]["token"]}))
        for path in ("index/shard-a", "index/shard-b"):
            self.assertIn("INDEX SHARD", sim.step({"action": "inspect", "path": path}))
        self.assertIn("REJECTED", sim.step({"action": "rebuild_index", "token": "WRONG"}))
        self.assertIn("INDEX REBUILT", sim.step({"action": "rebuild_index", "token": case["nested"]["index"]["token"]}))
        for path in ("escrow/shard-a", "escrow/shard-b"):
            self.assertIn("ESCROW SHARD", sim.step({"action": "inspect", "path": path}))
        self.assertIn("ESCROW UNLOCKED", sim.step({"action": "unlock_escrow", "token": case["nested"]["escrow"]["token"]}))
        for path in ("recovery/shard-a", "recovery/shard-b"):
            self.assertIn("RECOVERY SHARD", sim.step({"action": "inspect", "path": path}))
        self.assertIn("RECOVERY COMPLETE", sim.step({"action": "recover", "token": case["recovery_token"]}))
        for doc in ("registry/releases", "registry/approvals", "network/aliases",
                    "network/routes", "security/certificates"):
            sim.step({"action": "inspect", "path": doc})
        self.assertIn("APPLIED", sim.step({"action": "repair", **case["expected_repair"]}))
        self.assertIn("VERIFIED", sim.step({"action": "verify"}))
        self.assertEqual(sim.nested_completed, {"index", "escrow"})
        self.assertEqual(sum(len(v) for v in sim.nested_views.values()), 4)
        self.assertGreater(sim.nested_evidence_chars, 1000)
        self.assertGreaterEqual(sim.nested_blocked, 3)

    def test_depth_one_only_requires_escrow(self):
        case = tb.create_trap_puzzle(9, 1, nested_depth=1, nested_log_lines=3)["cases"][0]
        sim = tb.RepairSimulator(case)
        self.assertNotIn("index", case["nested"])
        self.assertNotIn("index/shard-a", case["docs"])
        self.assertIn("LOCKED", sim.step({"action": "inspect", "path": "recovery/shard-a"}))
        for name in ("escrow/shard-a", "escrow/shard-b"):
            sim.step({"action": "inspect", "path": name})
        self.assertIn("ESCROW UNLOCKED", sim.step({"action": "unlock_escrow", "token": case["nested"]["escrow"]["token"]}))
        for name in ("recovery/shard-a", "recovery/shard-b"):
            sim.step({"action": "inspect", "path": name})
        self.assertIn("RECOVERY COMPLETE", sim.step({"action": "recover", "token": case["recovery_token"]}))

    def test_nested_puzzles_deterministic_and_hidden(self):
        a = tb.create_trap_puzzle(77, 2, nested_depth=2, nested_log_lines=5)
        self.assertEqual(a, tb.create_trap_puzzle(77, 2, nested_depth=2, nested_log_lines=5))
        for case in a["cases"]:
            for stage in ("index", "escrow"):
                self.assertNotIn(case["nested"][stage]["token"], json.dumps(case["docs"]))
                self.assertNotIn(case["nested"][stage]["token"], case["intro"])
            self.assertNotIn(case["recovery_token"], json.dumps(case["docs"]))

    def test_nested_three_way_completes_with_evidence_based_agent(self):
        cfg = self.cfg()
        for mode in ("linear", "summarized", "branch"):
            fake = DeliberateFakeJan()
            with patch.object(bm.bc, "http_json", fake):
                result = tb.run_trap_trial(mode, 42, 2, cfg, "fake-model", 8192, Path("/tmp"), bm,
                                           summary_every=1, nested_depth=2, nested_log_lines=8)
            self.assertTrue(result["exact_final"], (mode, result["error"], result["case_breakdown"]))
            self.assertEqual(result["verified_cases"], 2)
            self.assertTrue(result["compiled_report_complete"])
            self.assertEqual(result["compiled_verified_cases"], 2)
            self.assertEqual(result["nested_index_completed"], 2)
            self.assertEqual(result["nested_escrow_completed"], 2)
            self.assertEqual(result["nested_shard_views"], 8)
            self.assertGreater(result["nested_evidence_chars"], 1000)
            self.assertTrue(all(x["nested_completed"] == ["escrow", "index"] for x in result["case_breakdown"]))

    def test_cli_creates_three_way_repair_report(self):
        fake = DeliberateFakeJan()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, "http_json", fake), \
             patch.object(bm.bc, "config", return_value=self.cfg()), \
             patch.object(bm.bc, "chosen_model", return_value="fake-model"), \
             contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp) / "repair-results"
            rc = bm.main(["--scenario", "repair", "--smoke", "--output", str(folder)])
            self.assertEqual(rc, 0)
            summary = (folder / "SUMMARY.md").read_text()
            rows = json.loads((folder / "results.json").read_text())
            self.assertEqual(rows["metadata"]["scenario"], "repair")
            self.assertEqual(rows["metadata"]["benchmark_version"], "1.7.0")
            self.assertEqual(rows["metadata"]["pressure_records"], 4)
            self.assertEqual(len(rows["trials"]), 3)
            self.assertIn("Repair traps: observable investigation behavior", summary)
            self.assertIn("trap_waste_chars", (folder / "results.csv").read_text())
            self.assertIn("forced_evidence_chars", (folder / "results.csv").read_text())
            self.assertIn("Mandatory recovery investigation", summary)
            self.assertIn("Required-evidence context pressure", summary)
            self.assertIn("pressure_evidence_chars", (folder / "results.csv").read_text())
            self.assertTrue(all(t["pressure_evidence_chars"] > 0 for t in rows["trials"]))
            self.assertTrue(all(t["recovery_completed"] == 1 for t in rows["trials"]))
            self.assertTrue(all(t["recovery_shard_views"] == 2 for t in rows["trials"]))


if __name__ == "__main__":
    unittest.main()

