#!/usr/bin/env python3
"""Own and verify an independent llama-server for local benchmark trials.

Never discovers or sends signals to Jan workers or servers not started here.
Only the subprocess group created by this object can be terminated.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class ServerError(RuntimeError):
    pass


class LocalLlamaServer:
    def __init__(self, binary: str, model_path: str, port=18089, context_size=8192,
                 gpu_layers=99, timeout=180.0, extra_args=(), log_dir=None):
        self.binary = binary
        self.model_path = Path(model_path).expanduser()
        self.port = port
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.timeout = timeout
        self.extra_args = list(extra_args)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.process = None
        self.log_handle = None
        self.start_count = 0
        self.last_pid = None
        self.model_id = None
        self.base_url = f"http://127.0.0.1:{port}"
        self._opener = build_opener(ProxyHandler({}))

    def preflight(self):
        exe = shutil.which(self.binary)
        if not exe or not Path(exe).is_file() or not os.access(exe, os.X_OK):
            raise ServerError(f"llama-server executable not found or not executable: {self.binary}. Install llama-cpp and an AMD backend, or use --llama-server /path/to/llama-server")
        if not self.model_path.is_file() or not os.access(self.model_path, os.R_OK):
            raise ServerError(f"GGUF model file cannot be read: {self.model_path}. Use --llama-model /absolute/path/to/model.gguf")
        if self.model_path.suffix.lower() != '.gguf':
            raise ServerError("--llama-model must point to a GGUF file")
        if not (1024 <= self.context_size <= 1048576 and 1 <= self.port <= 65535 and 0 <= self.gpu_layers <= 999):
            raise ServerError("Invalid llama.cpp context size, port, or GPU layer count")
        if not (10 <= self.timeout <= 1200):
            raise ServerError("--llama-startup-timeout must be between 10 and 1200 seconds")
        if any(not isinstance(arg, str) or not arg or '\x00' in arg for arg in self.extra_args):
            raise ServerError("Invalid --llama-arg")
        return {"binary": str(Path(exe).resolve()), "model": str(self.model_path.resolve()),
                "base_url": self.base_url, "context_size": self.context_size, "gpu_layers": self.gpu_layers}

    def _ensure_port_free(self):
        # Safe refusal: never connect to an existing service and assume it is ours.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", self.port))
            except OSError as exc:
                raise ServerError(f"Local port {self.port} is already in use; choose --llama-port another number. No existing process was stopped.") from exc

    def _json_get(self, path, timeout=2.5):
        req = Request(self.base_url + path, headers={"Accept": "application/json"})
        with self._opener.open(req, timeout=timeout) as response:
            return json.load(response)

    def start(self):
        if self.process is not None:
            raise ServerError("This manager already owns a server. Stop it before starting again.")
        spec = self.preflight()
        self._ensure_port_free()
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"llama-server-{self.start_count + 1:04d}.log"
            self.log_handle = log_path.open('w', encoding='utf-8')
        else:
            self.log_handle = open(os.devnull, 'w', encoding='utf-8')
            log_path = None
        command = [spec['binary'], '--host', '127.0.0.1', '--port', str(self.port),
                   '--model', spec['model'], '--ctx-size', str(self.context_size),
                   '--n-gpu-layers', str(self.gpu_layers), '--parallel', '1',
                   '--alias', 'branch-benchmark', '--no-webui', *self.extra_args]
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=self.log_handle, stderr=subprocess.STDOUT,
                                            start_new_session=True, close_fds=True)
            self.last_pid = self.process.pid
            self.start_count += 1
            while time.monotonic() - started < self.timeout:
                if self.process.poll() is not None:
                    raise ServerError(f"llama-server exited with code {self.process.returncode} during startup. See {log_path or 'server log'}")
                try:
                    health = self._json_get('/health')
                    if health.get('status') == 'ok':
                        models = self._json_get('/v1/models')
                        items = models.get('data', [])
                        ids = [m.get('id') for m in items if isinstance(m, dict) and isinstance(m.get('id'), str)]
                        if len(ids) != 1:
                            raise ServerError(f"Expected one model in dedicated llama-server; got {len(ids)}")
                        self.model_id = ids[0]
                        return {'status': 'verified', 'method': 'fresh_llama_process',
                                'elapsed_seconds': round(time.monotonic() - started, 3),
                                'pid': self.last_pid, 'model': self.model_id,
                                'log_file': str(log_path) if log_path else None}
                except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
                    pass  # still loading; retry without treating 503 as failure
                time.sleep(0.35)
            raise ServerError(f"llama-server did not become healthy within {self.timeout:g}s. See {log_path or 'server log'}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        proc = self.process
        self.process = None
        self.model_id = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    if os.name == 'posix':
                        os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        if os.name == 'posix':
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                        proc.wait(timeout=8)
                else:
                    proc.wait(timeout=1)
            except ProcessLookupError:
                proc.wait(timeout=2)
            finally:
                if self.log_handle:
                    self.log_handle.close()
                    self.log_handle = None
        elif self.log_handle:
            self.log_handle.close()
            self.log_handle = None
        return {'status': 'stopped', 'method': 'owned_process_stop'}
