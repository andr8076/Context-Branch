#!/usr/bin/env python3
"""Safely unload/reload one locally-hosted Jan llama.cpp router model.

Does not terminate Jan Desktop or the API process. A reset must be verified or
benchmarking stops; a failed reset is not a clean independent trial.
"""
from __future__ import annotations

import re
import os
import subprocess
import time
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError


class ResetError(RuntimeError):
    """The requested reset was not verified. Do not continue the benchmark."""


class RouterHttpError(ResetError):
    """An HTTP failure, with no response body or credential in diagnostics."""

    def __init__(self, status, method, url):
        self.status = status
        super().__init__(f'Jan router returned HTTP {status} to {method} {url}')


def validate_router_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or not parsed.port or parsed.path not in ('', '/') or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise ResetError('Jan router URL must be an HTTP localhost address with port, e.g. http://127.0.0.1:12345')
    return url.rstrip('/')


def api_json(method: str, url: str, body=None, timeout=5, api_key=None):
    """Request only a local endpoint; never use proxy settings for router secrets."""
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ResetError('Refusing a non-local router request')
    payload = None if body is None else json.dumps(body).encode('utf-8')
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = Request(url, data=payload, method=method, headers=headers)
    # Explicitly disable system proxies, even ones that would ignore NO_PROXY.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as res:
            data = json.load(res)
    except HTTPError as ex:
        raise RouterHttpError(ex.code, method, url) from None
    except (URLError, OSError, ValueError) as ex:
        raise ResetError(f'Jan router request failed ({type(ex).__name__}) at {method} {url}') from None
    if not isinstance(data, dict):
        raise ResetError('Jan router returned something other than a JSON object')
    return data


def _request(method, url, body=None, timeout=5, api_key=None):
    # Preserve compatibility with no-key clients and offline test doubles.
    if api_key:
        return api_json(method, url, body, timeout=timeout, api_key=api_key)
    return api_json(method, url, body, timeout=timeout)


def match_model(jan_model: str, candidates: list):
    """Require one unambiguous, exact model ID or basename match."""
    needle = jan_model.casefold()
    direct = [m for m in candidates if isinstance(m, dict) and str(m.get('id', '')).casefold() == needle]
    if len(direct) == 1:
        return direct[0]
    basename = Path(jan_model).name.casefold()
    matches = [m for m in candidates if isinstance(m, dict) and Path(str(m.get('id', ''))).name.casefold() == basename]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None
    # Jan's gateway sometimes adds .gguf where the internal preset omits it.
    stripped = basename[:-5] if basename.endswith('.gguf') else basename
    matches = [m for m in candidates if isinstance(m, dict) and
               Path(str(m.get('id', ''))).name.casefold().removesuffix('.gguf') == stripped]
    return matches[0] if len(matches) == 1 else None


def router_models(url, timeout=5, api_key=None):
    data = _request('GET', url + '/models', timeout=timeout, api_key=api_key)
    rows = data.get('data')
    if not isinstance(rows, list):
        raise ResetError('Router /models does not expose a data array; not a compatible llama.cpp router')
    return rows


def status_of(model):
    status = model.get('status') or {}
    if isinstance(status, dict):
        return str(status.get('value') or '').lower()
    if isinstance(status, str):
        return status.lower()
    return ''


def _jan_worker_ancestor(pid, max_steps=8):
    visited = set()
    while pid > 1 and pid not in visited and len(visited) < max_steps:
        visited.add(pid)
        try:
            raw = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0', 1)[0]
            if Path(os.fsdecode(raw)).name == 'jan-llama-worker':
                return True
            lines = (Path('/proc') / str(pid) / 'status').read_text(errors='replace')
            match = re.search(r'^PPid:\s*(\d+)', lines, re.M)
            pid = int(match.group(1)) if match else 0
        except (OSError, ValueError):
            break
    return False


def _ports_owned_by_worker_ss():
    """Find Jan-owned listening ports using ss, when PID details are visible."""
    try:
        res = subprocess.run(['ss', '-H', '-lntp'], capture_output=True, text=True,
                             timeout=6, check=True)
    except (OSError, subprocess.SubprocessError):
        return set()
    ports = set()
    for line in res.stdout.splitlines():
        pids = re.findall(r'pid=(\d+)', line)
        if not pids or not any(_jan_worker_ancestor(int(pid)) for pid in pids):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        local = fields[3]
        match = re.search(r':(\d+)$', local)
        if match and 0 < int(match.group(1)) < 65536:
            ports.add(int(match.group(1)))
    return ports


def _worker_pids():
    """List worker and descendant PIDs, without reading environment or secrets."""
    proc = Path('/proc')
    workers = set()
    pids = []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return workers, pids
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        pids.append(pid)
        try:
            raw = (entry / 'cmdline').read_bytes().split(b'\0', 1)[0]
            if Path(os.fsdecode(raw)).name == 'jan-llama-worker':
                workers.add(pid)
        except (OSError, ValueError):
            continue
    related = set(workers)
    if workers:
        for pid in pids:
            if pid not in related and _jan_worker_ancestor(pid):
                related.add(pid)
    return workers, related


def _tcp_listen_inode_ports():
    """Map LISTEN socket inode to port in our network namespace."""
    ports = {}
    for file in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = Path(file).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            row = line.split()
            if len(row) < 10 or row[3] != '0A':  # TCP_LISTEN
                continue
            try:
                port = int(row[1].rsplit(':', 1)[1], 16)
                inode = row[9]
            except (ValueError, IndexError):
                continue
            if 0 < port < 65536:
                ports[inode] = port
    return ports


def _ports_owned_by_worker_proc():
    """Find worker-owned listening sockets via /proc/PID/fd.

    This works when `ss -p` omits PID/owner details (e.g. Flatpak workers).
    Only PIDs descended from a Jan worker are inspected. No HTTP POST here.
    """
    workers, related = _worker_pids()
    inode_ports = _tcp_listen_inode_ports()
    ports = set()
    for pid in related:
        fds = Path('/proc') / str(pid) / 'fd'
        try:
            names = list(fds.iterdir())
        except OSError:
            continue
        for fd in names:
            try:
                link = os.readlink(fd)
            except OSError:
                continue
            match = re.fullmatch(r'socket:\[(\d+)\]', link)
            if match and match.group(1) in inode_ports:
                ports.add(inode_ports[match.group(1)])
    return workers, ports



def _known_jan_log_paths(preset_paths):
    """Only inspect Jan's documented data/log paths or a confirmed worker preset.

    Flatpak uses a different XDG data directory from native Jan. Data may also
    have been relocated by the user; in that case the worker preset is best.
    """
    paths = []
    for preset in preset_paths:
        path = Path(preset)
        if path.name == 'router.preset.ini' and path.parent.name == 'llamacpp':
            paths.append(path.parent.parent / 'logs' / 'app.log')
    home = Path.home()
    paths.extend((
        home / '.var/app/ai.jan.Jan/data/Jan/data/logs/app.log',
        home / '.local/share/Jan/data/logs/app.log',
    ))
    return list(dict.fromkeys(paths))


def _log_keys_for_port(preset_paths, port):
    """Read Jan router keys from verified-port startup records, never log them.

    Only called after a port is proven owned by the Jan worker. A current port
    must match exactly; stale startup records for other ports are disregarded.
    """
    if not port:
        return []
    keys = []
    for log in _known_jan_log_paths(preset_paths):
        try:
            with log.open('rb') as file:
                file.seek(0, os.SEEK_END)
                length = file.tell()
                # Only a small tail, not arbitrary files or the whole history.
                file.seek(max(0, length - 2_097_152))
                lines = file.read().decode('utf-8', errors='replace').splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if 'Router argv:' not in line:
                continue
            raw = line.split('Router argv:', 1)[1].strip()
            try:
                args, _ = json.JSONDecoder().raw_decode(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                continue
            key = None
            recorded_port = None
            for i, item in enumerate(args):
                if item == '--port' and i + 1 < len(args):
                    recorded_port = args[i + 1]
                elif item.startswith('--port='):
                    recorded_port = item.split('=', 1)[1]
                if item == '--api-key' and i + 1 < len(args):
                    key = args[i + 1]
                elif item.startswith('--api-key='):
                    key = item.split('=', 1)[1]
            if str(port) == recorded_port and key and key not in keys:
                keys.append(key)
                if len(keys) >= 8:
                    break
    return keys


def _jan_router_secret_candidates(port=None):
    """Try private router keys locally; never print or persist candidates.

    Explicit opt-in JAN_ROUTER_API_KEY wins over automatically discovered keys.
    Only Jan-worker descendants and the documented Jan log locations are read.
    """
    _, related = _worker_pids()
    keys = []
    presets = []

    def append(key):
        if key and key not in keys and len(keys) < 12:
            keys.append(key)

    append(os.environ.get('JAN_ROUTER_API_KEY'))
    for pid in sorted(related):
        base = Path('/proc') / str(pid)
        try:
            argv = [os.fsdecode(item) for item in (base / 'cmdline').read_bytes().split(b'\0') if item]
            for i, item in enumerate(argv):
                if item == '--api-key' and i + 1 < len(argv):
                    append(argv[i + 1])
                elif item.startswith('--api-key='):
                    append(item.partition('=')[2])
                elif item in ('--preset', '--models-preset') and i + 1 < len(argv):
                    presets.append(argv[i + 1])
                elif item.startswith(('--preset=', '--models-preset=')):
                    presets.append(item.partition('=')[2])
        except (OSError, ValueError):
            pass
        try:
            env = (base / 'environ').read_bytes().split(b'\0')
            for item in env:
                if item.startswith((b'LLAMA_API_KEY=', b'LLAMA_ARG_API_KEY=')):
                    append(os.fsdecode(item.partition(b'=')[2]))
        except (OSError, ValueError):
            pass
    for key in _log_keys_for_port(presets, port):
        append(key)
    return keys


def _probe_models(url, timeout=2, key=None):
    if key:
        return router_models(url, timeout=timeout, api_key=key)
    return router_models(url, timeout=timeout)


def _probe_with_auth(url, timeout=2, gateway_key=None):
    """GET only. Try private process credentials solely for Jan-owned ports."""
    try:
        return _probe_models(url, timeout), None
    except RouterHttpError as ex:
        if ex.status not in (401, 403):
            raise
    candidates = _jan_router_secret_candidates(port=urlparse(url).port)
    # The Jan public API key may be identical to the internal router key.
    # It is a *candidate*, not proof: a distinct private key is possible.
    # Only try it against endpoints already verified as Jan-worker-owned.
    if gateway_key and gateway_key not in candidates:
        priority = 1 if os.environ.get('JAN_ROUTER_API_KEY') else 0
        candidates.insert(priority, gateway_key)
    for key in candidates:
        try:
            return _probe_models(url, timeout, key=key), key
        except RouterHttpError as ex:
            if ex.status not in (401, 403):
                raise
    detail = ('no candidate keys found in worker arguments, environment or Jan logs'
              if not candidates else f'{len(candidates)} candidate key(s) tried and rejected')
    raise ResetError(f'Jan router at {url} requires authentication (HTTP 401/403); {detail}. '
                     'Jan may use separate credentials for its public API and internal router; '
                     'if the public API key is rejected, a known router key can be supplied locally '
                     'via JAN_ROUTER_API_KEY; no keys were printed or sent to other hosts.')



def discover_router(jan_model: str, include_auth=False, gateway_key=None):
    """Probe only loopback ports verified as belonging to the Jan worker.

    If model IDs differ between Jan's public gateway and the internal router,
    report their IDs so the mismatch can be fixed deliberately—not guessed.
    """
    ss_ports = _ports_owned_by_worker_ss()
    workers, proc_ports = _ports_owned_by_worker_proc()
    ports = ss_ports | proc_ports
    matches = []
    no_model = []
    failures = []
    for port in sorted(ports):
        url = f'http://127.0.0.1:{port}'
        try:
            models, auth_key = _probe_with_auth(url, timeout=2, gateway_key=gateway_key)
            if match_model(jan_model, models) is not None:
                matches.append((url, auth_key))
            else:
                ids = [str(m.get('id', '')) for m in models if isinstance(m, dict)]
                no_model.append((port, ids[:8]))
        except RouterHttpError as ex:
            failures.append(f'{port}: HTTP {ex.status}')
        except ResetError as ex:
            # Only expose a safe summary. Never include credentials or response bodies.
            failure = str(ex)
            if 'requires authentication' in failure:
                info = 'no key found' if 'no candidate keys found' in failure else ('candidate keys rejected' if 'tried and rejected' in failure else 'auth failed')
                failures.append(f'{port}: authentication required (HTTP 401/403, {info})')
            else:
                failures.append(f'{port}: {failure[:180]}')
    if len(matches) == 1:
        return matches[0] if include_auth else matches[0][0]
    details = (f'Found {len(matches)} matching Jan router endpoints; '
               f'worker PIDs={sorted(workers)[:8]}, '
               f'ss ports={sorted(ss_ports)}, /proc ports={sorted(proc_ports)}. ')
    if no_model:
        details += 'Router responded but did not list the selected model: ' + repr(no_model[:3]) + '. '
    if failures:
        details += f'Other Jan-owned ports without a compatible /models response: {failures}. '
    details += ('No models were unloaded. A 401 after a verified Jan port may also indicate a Jan router authentication bug. '
                'If necessary, use --jan-router-url http://127.0.0.1:PORT '
                'after independently verifying that it is the llama.cpp router.')
    raise ResetError(details)


def available_memory_mb():
    """Diagnostic only; OS memory may change for unrelated reasons."""
    try:
        match = re.search(r'^MemAvailable:\s*(\d+)\s+kB', Path('/proc/meminfo').read_text(), re.M)
        return round(int(match.group(1)) / 1024) if match else None
    except (OSError, ValueError):
        return None


class JanModelReset:
    def __init__(self, jan_model, jan_cfg, router_url=None, cooldown=8.0, timeout=180.0):
        self.jan_model = jan_model
        self.cfg = jan_cfg
        self.cooldown = cooldown
        self.timeout = timeout
        self.router_url = validate_router_url(router_url) if router_url else None
        self.router_model = None
        self.router_key = None  # volatile; never saved in benchmark results

    def check(self):
        if self.router_url is None:
            self.router_url, self.router_key = discover_router(
                self.jan_model, include_auth=True, gateway_key=self.cfg.get('api_key'))
        else:
            # Only auto-read private keys when explicit endpoint belongs to Jan.
            ss = _ports_owned_by_worker_ss()
            _, proc = _ports_owned_by_worker_proc()
            port = urlparse(self.router_url).port
            if port in ss | proc:
                _, self.router_key = _probe_with_auth(
                    self.router_url, gateway_key=self.cfg.get('api_key'))
            elif os.environ.get('JAN_ROUTER_API_KEY'):
                self.router_key = os.environ['JAN_ROUTER_API_KEY']
        info = match_model(self.jan_model, _probe_models(self.router_url, timeout=5, key=self.router_key))
        if info is None:
            raise ResetError('The selected model is not unambiguously available in this Jan router')
        self.router_model = str(info['id'])
        status = status_of(info)
        if not status:
            raise ResetError('Jan router did not report a model status; cannot verify a complete reset')
        return {'router_url': self.router_url, 'router_model': self.router_model, 'router_status': status}

    def _poll(self, desired):
        deadline = time.monotonic() + self.timeout
        last = ''
        while time.monotonic() < deadline:
            item = match_model(self.router_model, _probe_models(self.router_url, timeout=5, key=self.router_key))
            if item is None:
                raise ResetError('The model disappeared from Jan router while resetting')
            last = status_of(item)
            if not last:
                raise ResetError('Jan router stopped reporting model status')
            if desired == 'unloaded' and last in ('unloaded', 'not-loaded', 'unavailable', 'idle'):
                return last
            if desired == 'loaded' and last == 'loaded':
                return last
            if last in ('error', 'failed'):
                raise ResetError(f'Jan router reported {last} while expecting {desired}')
            time.sleep(1.0)
        raise ResetError(f'Timed out waiting for model {desired}; last status={last!r}')

    def _jan_api_ready(self):
        # Health-check Jan's documented GET /v1/models without generating tokens
        # (and without warming the next trial's KV cache).
        url = self.cfg['api_base_url'].rstrip('/') + '/models'
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                data = api_json('GET', url, timeout=self.cfg['request_timeout_seconds'],
                                api_key=self.cfg['api_key'])
                if match_model(self.jan_model, data.get('data') or []) is not None:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise ResetError('Jan API did not confirm the selected model after reload')

    def reset(self):
        info = self.check()
        started = time.perf_counter()
        record = {'method': 'model_unload_reload', 'router_model': self.router_model,
                  'status_before': info['router_status'], 'status': 'started',
                  'memory_available_before_mb': available_memory_mb()}
        try:
            if info['router_status'] in ('unloaded', 'not-loaded', 'unavailable', 'idle'):
                # The model is already unloaded; do not require the router to
                # unload something that has no active KV/model allocation.
                record['status_after_unload'] = info['router_status']
            else:
                unload = _request('POST', self.router_url + '/models/unload',
                                  {'model': self.router_model}, timeout=self.timeout, api_key=self.router_key)
                if unload.get('success') is not True:
                    raise ResetError('Jan router did not confirm model unload')
                record['status_after_unload'] = self._poll('unloaded')
            record['memory_available_after_unload_mb'] = available_memory_mb()
            if self.cooldown:
                time.sleep(self.cooldown)
            load = _request('POST', self.router_url + '/models/load',
                            {'model': self.router_model}, timeout=self.timeout, api_key=self.router_key)
            if load.get('success') is not True:
                raise ResetError('Jan router did not confirm model reload')
            record['status_after_reload'] = self._poll('loaded')
            self._jan_api_ready()
            record['status'] = 'verified'
            record['memory_available_after_reload_mb'] = available_memory_mb()
            return record
        except Exception as ex:
            record['status'] = 'failed'
            if isinstance(ex, ResetError):
                ex.reset_record = record
                raise
            error = ResetError(f'Could not verify Jan reset: {type(ex).__name__}')
            error.reset_record = record
            raise error from ex
        finally:
            record['elapsed_seconds'] = round(time.perf_counter() - started, 3)
