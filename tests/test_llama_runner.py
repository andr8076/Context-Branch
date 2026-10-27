"""Integration tests exercise real owned subprocesses, but use a FAKE inference server."""
import json
import socket
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark
from llama_runner import LocalLlamaServer, ServerError

FAKE_SERVER = r'''#!/usr/bin/env python3
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p = argparse.ArgumentParser()
p.add_argument('--host')
p.add_argument('--port', type=int)
p.add_argument('--model')
p.add_argument('--ctx-size')
p.add_argument('--n-gpu-layers')
p.add_argument('--parallel')
p.add_argument('--alias')
p.add_argument('--no-webui', action='store_true')
p.add_argument('--exit-on-start', action='store_true')
a = p.parse_args()
if a.exit_on_start:
    sys.exit(42)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/health':
            body = {'status':'ok'}
        elif self.path == '/v1/models':
            body = {'data':[{'id':a.alias}]}
        else:
            self.send_error(404); return
        data=json.dumps(body).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length',str(len(data))); self.end_headers();self.wfile.write(data)
    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_error(404);return
        size=int(self.headers.get('Content-Length','0'))
        payload=json.loads(self.rfile.read(size))
        data=json.dumps({'choices':[{'message':{'role':'assistant','content':'{}'},'finish_reason':'stop'}],
                         'usage':{'prompt_tokens':32, 'completion_tokens':3}}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
'''


def unused_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class DedicatedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / 'llama-server-mock'
        self.binary.write_text(FAKE_SERVER, encoding='utf-8')
        self.binary.chmod(0o700)
        self.model = self.root / 'tiny.gguf'
        self.model.write_bytes(b'not real gguf, mock does not read it')
        self.port = unused_port()

    def runner(self, **kwargs):
        return LocalLlamaServer(str(self.binary), str(self.model), port=self.port,
                                context_size=8192, timeout=12, **kwargs)

    def test_start_stop_and_fresh_process(self):
        r = self.runner(log_dir=self.root / 'logs')
        try:
            a = r.start()
            self.assertEqual(a['status'], 'verified')
            self.assertEqual(a['model'], 'branch-benchmark')
            self.assertIsNone(r.process.poll())
            old_pid = a['pid']
            self.assertEqual(r.stop()['status'], 'stopped')
            b = r.start()
            self.assertNotEqual(old_pid, b['pid'])
            self.assertEqual(r.start_count, 2)
            self.assertEqual(len(list((self.root / 'logs').glob('*.log'))), 2)
        finally:
            r.stop()
        self.assertIsNone(r.process)

    def test_refuses_existing_server_without_killing_it(self):
        r1 = self.runner()
        r2 = self.runner()
        try:
            r1.start()
            with self.assertRaisesRegex(ServerError, 'already in use'):
                r2.start()
            self.assertIsNone(r1.process.poll())
            self.assertIsNone(r2.process)
        finally:
            r1.stop()
            r2.stop()

    def test_failed_start_cleans_up(self):
        r = self.runner(extra_args=['--exit-on-start'], log_dir=self.root / 'logs')
        with self.assertRaisesRegex(ServerError, 'exited with code 42'):
            r.start()
        self.assertIsNone(r.process)
        self.assertIsNone(r.log_handle)

    def test_binary_and_model_validation(self):
        invalid = LocalLlamaServer('definitely-not-present-llama-server-xyz', str(self.model))
        with self.assertRaisesRegex(ServerError, 'not found'):
            invalid.preflight()
        invalid = LocalLlamaServer(str(self.binary), str(self.root/'absent.gguf'))
        with self.assertRaisesRegex(ServerError, 'cannot be read'):
            invalid.preflight()

    def test_cli_full_three_mode_smoke_and_pids_change(self):
        destination = self.root / 'results'
        rc = benchmark.main(['--engine', 'llama', '--llama-server', str(self.binary),
                             '--llama-model', str(self.model), '--llama-port', str(self.port),
                             '--llama-startup-timeout', '12', '--scenario', 'archive', '--smoke',
                             '--output', str(destination)])
        self.assertEqual(rc, 0)
        obj = json.loads((destination/'results.json').read_text())
        self.assertEqual(obj['metadata']['engine'], 'llama')
        self.assertEqual(len(obj['trials']), 3)
        self.assertEqual(len(obj['resets']), 3)
        pids = [e['pid'] for e in obj['resets']]
        self.assertEqual(len(set(pids)), 3)
        self.assertTrue(all(e['status']=='verified' for e in obj['resets']))
        self.assertEqual(len(list((destination/'server_logs').glob('*.log'))), 3)
        self.assertEqual({row['mode'] for row in obj['trials']}, {'linear','summarized','branch'})
        self.assertTrue(all(row['reset_status']=='verified' for row in obj['trials']))
        with socket.socket() as sock:
            self.assertNotEqual(sock.connect_ex(('127.0.0.1', self.port)), 0)

    def test_dry_preflight_never_starts_process(self):
        rc = benchmark.main(['--engine','llama','--llama-server',str(self.binary),
                              '--llama-model',str(self.model),'--llama-port',str(self.port),
                              '--check-llama-server'])
        self.assertEqual(rc, 0)
        with socket.socket() as sock:
            self.assertNotEqual(sock.connect_ex(('127.0.0.1',self.port)), 0)


if __name__ == '__main__': unittest.main()
