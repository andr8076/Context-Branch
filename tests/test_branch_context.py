import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

    def test_invalid_endpoints_and_protocol(self):
        self.cfg["api_base_url"] = "https://remote.example/v1"
        b.CONFIG.write_text(json.dumps(self.cfg))
        with self.assertRaises(ValueError):
            b.config()
        init = b.handle_rpc({"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{"protocolVersion":"2025-03-26"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "jan-branch-context")
        tools = b.handle_rpc({"jsonrpc":"2.0", "id":2, "method":"tools/list"})
        self.assertEqual(len(tools["result"]["tools"]), 5)
        self.assertIsNone(b.handle_rpc({"jsonrpc":"2.0", "method":"notifications/initialized"}))

    def test_nested_branch_retrieval(self):
        a = b.dispatch_tool("branch_solve", {"task":"First task"})
        nested = b.dispatch_tool("branch_solve", {"task":"Second task", "parent_id":a["branch_id"]})
        self.assertEqual(b.load_branch(nested["branch_id"])["depth"], 1)
        self.assertEqual(nested["parent_id"], a["branch_id"])


if __name__ == '__main__':
    unittest.main()
