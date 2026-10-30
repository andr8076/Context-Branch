import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

SCRIPT = Path(__file__).resolve().parents[1] / "branch_context.py"
spec = importlib.util.spec_from_file_location("branch_context", SCRIPT)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class FakeJanHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, fmt, *args):
        pass

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self.send_json({"data": [{"id": "mock-model"}]})
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        FakeJanHandler.requests.append(request)
        messages = request["messages"]
        if any(m.get("role") == "tool" for m in messages):
            content = '{"status":"success","result":"Read tool response","what_changed":"Nothing","next_step":"Use finding","caveats":""}'
            return self.send_json({"choices":[{"message":{"role":"assistant", "content":content}}]})
        if 'Read the recipe' in messages[-1].get("content", ""):
            return self.send_json({"choices":[{"message":{"role":"assistant", "content":"", "tool_calls":[{"id":"call01", "type":"function", "function":{"name":"read_project_file", "arguments":"{\"path\":\"recipe.txt\"}"}}]}}]})
        content = '{"status":"success","result":"Make glaze using powdered sugar and milk","what_changed":"Nothing","next_step":"Mix it","caveats":"Not physically prepared"}'
        return self.send_json({"choices": [{"message": {"role":"assistant", "content":content}}]})


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeJanHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old = (b.DATA, b.CONFIG)
        b.DATA = self.base / "data" / "branches"
        b.CONFIG = self.base / "config.json"
        self.addCleanup(self.restore)
        self.cfg = dict(b.DEFAULTS)
        self.cfg.update({"api_base_url":f"http://127.0.0.1:{self.server.server_port}/v1", "model":"auto"})
        b.CONFIG.write_text(json.dumps(self.cfg))
        FakeJanHandler.requests.clear()

    def restore(self):
        b.DATA, b.CONFIG = self.old

    def test_http_400_context_error_is_specific_and_redacted(self):
        body = b'{"error":{"message":"the request exceeds the available context size; private-secret-123"}}'
        error = HTTPError("http://127.0.0.1:18089/v1/chat/completions", 400,
                          "Bad Request", {}, BytesIO(body))
        with patch.object(b, "urlopen", side_effect=error):
            with self.assertRaises(b.LocalAPIError) as caught:
                b.http_json("POST", self.cfg["api_base_url"] + "/chat/completions", self.cfg, {"messages": []})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.error_kind, "context_overflow_confirmed")
        self.assertIn("context budget exceeded", str(caught.exception))
        self.assertNotIn("private-secret-123", str(caught.exception))
        self.assertNotIn("Jan API", str(caught.exception))

    def test_http_400_unknown_does_not_echo_prompt(self):
        body = b'{"error":{"message":"illegal prompt: private-secret-123"}}'
        error = HTTPError("http://127.0.0.1:18089/v1/chat/completions", 400,
                          "Bad Request", {}, BytesIO(body))
        with patch.object(b, "urlopen", side_effect=error):
            with self.assertRaises(b.LocalAPIError) as caught:
                b.http_json("POST", self.cfg["api_base_url"] + "/chat/completions", self.cfg, {"messages": []})
        self.assertEqual(caught.exception.error_kind, "api_or_runtime_error")
        self.assertNotIn("private-secret-123", str(caught.exception))

    def test_isolation_storage_and_resume(self):
        a = b.dispatch_tool("branch_solve", {"task":"Create glaze", "relevant_context":"Cake is ready"})
        self.assertEqual(a["status"], "success")
        self.assertEqual(a["what_changed"], "Nothing")
        self.assertTrue((b.DATA / (a["branch_id"] + '.json')).is_file())
        self.assertNotIn("transcript", b.load_branch(a["branch_id"]))
        follow = b.dispatch_tool("branch_resume", {"branch_id":a["branch_id"], "question":"Make it vegan"})
        self.assertEqual(follow["branch_id"], a["branch_id"])
        self.assertIn("PREVIOUS BRANCH RESULT", FakeJanHandler.requests[-1]["messages"][-1]["content"])
        self.assertNotIn("Cake is ready", FakeJanHandler.requests[-1]["messages"][-1]["content"])
        self.assertEqual(len(b.dispatch_tool("branch_list", {})["branches"]), 1)
        self.assertEqual(b.dispatch_tool("branch_health", {})["model"], "mock-model")
        if os.name == "posix":
            self.assertEqual((b.DATA / (a["branch_id"] + '.json')).stat().st_mode & 0o777, 0o600)
            self.assertEqual(b.DATA.stat().st_mode & 0o777, 0o700)

    def test_project_file_tool(self):
        project = self.base / "source"
        project.mkdir()
        (project / "recipe.txt").write_text("milk + sugar")
        self.cfg["read_only_project_root"] = str(project)
        b.CONFIG.write_text(json.dumps(self.cfg))
        result = b.dispatch_tool("branch_solve", {"task":"Read the recipe"})
        self.assertEqual(result["status"], "success")
        self.assertTrue(any(m.get("role") == "tool" for m in FakeJanHandler.requests[-1]["messages"]))
        self.assertIn("milk + sugar", FakeJanHandler.requests[-1]["messages"][-1]["content"])
        self.assertIn("recipe.txt:1", b.search_project(self.cfg, "milk"))
        with self.assertRaises(ValueError):
            b.safe_file(project, "../private.txt")

    def test_state_files_are_atomic_and_corrupt_json_is_rejected(self):
        target = b.DATA / "branch-state.json"
        b.ensure_store()
        b.atomic_json_write(target, {"status": "ok", "value": 1})
        self.assertTrue(target.is_file())
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["status"], "ok")
        target.write_text('{"broken": ', encoding='utf-8')
        with self.assertRaises(ValueError):
            b.load_json_if_exists(target)
        self.assertEqual(target.exists(), True)

    def test_cleanup_prunes_stale_parent_and_child_links(self):
        parent_id = "a" * 32
        child_id = "b" * 32
        parent = {"id": parent_id, "parent_id": None, "subbranches": [child_id, "c" * 32], "status": "success", "task": "Parent", "updated": "2025-01-01T00:00:00Z"}
        child = {"id": child_id, "parent_id": "missing" * 4, "subbranches": [], "status": "success", "task": "Child", "updated": "2025-01-01T00:00:00Z"}
        b.save_branch(parent)
        b.save_branch(child)
        stale_budget = b.DATA / "budgets" / "stale.json"
        stale_budget.parent.mkdir(parents=True, exist_ok=True)
        stale_budget.write_text('{"root_id": "garbage"}', encoding='utf-8')
        report = b.cleanup_orphaned_state()
        self.assertEqual(report["pruned_subbranches"], 1)
        self.assertEqual(report["cleared_parent_links"], 1)
        refreshed_parent = b.load_branch(parent_id)
        self.assertEqual(refreshed_parent["subbranches"], [child_id])
        refreshed_child = b.load_branch(child_id)
        self.assertIsNone(refreshed_child["parent_id"])
        self.assertFalse(stale_budget.exists())

    def test_cleanup_repairs_malformed_mixed_branch_links(self):
        broken_id = "d" * 32
        malformed = {
            "id": broken_id,
            "parent_id": ["not", "an", "id"],
            "subbranches": {"unexpected": "mapping"},
            "status": "blocked",
            "task": "Interrupted branch",
        }
        b.atomic_json_write(b.branch_path(broken_id), malformed)

        report = b.cleanup_orphaned_state()

        self.assertEqual(report["cleared_parent_links"], 1)
        self.assertEqual(report["pruned_subbranches"], 1)
        repaired = b.load_branch(broken_id)
        self.assertIsNone(repaired["parent_id"])
        self.assertEqual(repaired["subbranches"], [])

    def test_invalid_endpoints_and_protocol(self):
        self.cfg["api_base_url"] = "https://remote.example/v1"
        b.CONFIG.write_text(json.dumps(self.cfg))
        with self.assertRaises(ValueError):
            b.config()
        init = b.handle_rpc({"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{"protocolVersion":"2025-03-26"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "jan-branch-context")
        tools = b.handle_rpc({"jsonrpc":"2.0", "id":2, "method":"tools/list"})
        self.assertEqual(len(tools["result"]["tools"]), 6)
        self.assertIsNone(b.handle_rpc({"jsonrpc":"2.0", "method":"notifications/initialized"}))

    def test_nested_branch_retrieval(self):
        a = b.dispatch_tool("branch_solve", {"task":"First task"})
        nested = b.dispatch_tool("branch_solve", {"task":"Second task", "parent_id":a["branch_id"]})
        self.assertEqual(b.load_branch(nested["branch_id"])["depth"], 1)
        self.assertEqual(nested["parent_id"], a["branch_id"])


if __name__ == '__main__':
    unittest.main()
