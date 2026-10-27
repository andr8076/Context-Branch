"""Offline tests: reset protocol and benchmark isolation. No Jan process is killed."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jan_reset as jr
import benchmark as bm
from test_benchmark import SummarizationJan


class FakeRouter:
    def __init__(self, status='loaded'):
        self.status = status
        self.calls = []

    def api(self, method, url, body=None, timeout=5):
        self.calls.append((method, url, body))
        if method == 'POST' and url.endswith('/models/unload'):
            self.status = 'unloaded'
            return {'success': True}
        if method == 'POST' and url.endswith('/models/load'):
            self.status = 'loaded'
            return {'success': True}
        return {'data': [{'id': '/models/Qwen3.5-9B-Q4_K_M.gguf',
                          'status': {'value': self.status}}]}


class JanResetTests(unittest.TestCase):
    def cfg(self):
        return dict(bm.bc.DEFAULTS, model='fake-local-model')

    def test_url_rejects_non_local_and_unexpected_paths(self):
        for url in ('http://example.com:1234', 'https://127.0.0.1:1234',
                    'http://127.0.0.1:1337/v1', 'http://localhost:1337/?q=1'):
            with self.assertRaises(jr.ResetError, msg=url):
                jr.validate_router_url(url)
        self.assertEqual(jr.validate_router_url('http://127.0.0.1:1234/'), 'http://127.0.0.1:1234')

    def test_unambiguous_model_name_or_basename(self):
        names = [{'id': '/opt/models/Qwen3.5-9B-Q4_K_M.gguf'}, {'id': 'other'}]
        self.assertEqual(jr.match_model('Qwen3.5-9B-Q4_K_M.gguf', names), names[0])
        self.assertIsNone(jr.match_model('missing', names))
        self.assertIsNone(jr.match_model('Qwen3.5-9B-Q4_K_M.gguf', names + [names[0]]))

    def test_no_model_unload_on_preflight(self):
        fake = FakeRouter()
        with patch.object(jr, 'api_json', fake.api):
            reset = jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', self.cfg(),
                                     router_url='http://127.0.0.1:1234', cooldown=0)
            info = reset.check()
        self.assertEqual(info['router_status'], 'loaded')
        self.assertTrue(all(method == 'GET' for method, _, _ in fake.calls))

    def test_reset_unloads_and_reloads_and_checks_health(self):
        fake = FakeRouter()
        with patch.object(jr, 'api_json', fake.api), \
             patch.object(jr.JanModelReset, '_jan_api_ready', return_value=None):
            reset = jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', self.cfg(),
                                     router_url='http://127.0.0.1:1234', cooldown=0)
            result = reset.reset()
        actions = [url.split('/')[-1] for method, url, _ in fake.calls if method == 'POST']
        self.assertEqual(actions, ['unload', 'load'])
        self.assertEqual(result['status_after_unload'], 'unloaded')
        self.assertEqual(result['status_after_reload'], 'loaded')
        self.assertEqual(result['status'], 'verified')

    def test_already_unloaded_skips_unload_request(self):
        fake = FakeRouter(status='unloaded')
        with patch.object(jr, 'api_json', fake.api), \
             patch.object(jr.JanModelReset, '_jan_api_ready', return_value=None):
            reset = jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', self.cfg(),
                                     router_url='http://127.0.0.1:1234', cooldown=0)
            result = reset.reset()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual([url.rsplit('/',1)[-1] for method,url,_ in fake.calls if method=='POST'], ['load'])

    def test_failed_reload_must_be_reported_not_silently_reused(self):
        fake = FakeRouter()
        def broken(method, url, body=None, timeout=5):
            if method == 'POST' and url.endswith('/models/load'):
                return {'success': False}
            return fake.api(method, url, body, timeout)
        with patch.object(jr, 'api_json', broken):
            reset = jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', self.cfg(),
                                     router_url='http://127.0.0.1:1234', cooldown=0)
            with self.assertRaises(jr.ResetError) as caught:
                reset.reset()
        self.assertEqual(caught.exception.reset_record['status'], 'failed')
        self.assertIn('elapsed_seconds', caught.exception.reset_record)

    def test_autodiscovery_probes_only_jan_worker_ports(self):
        result = SimpleNamespace(stdout='LISTEN 0 4096 127.0.0.1:16010 0.0.0.0:* users:(("jan-llama-worker",pid=42,fd=3))\n'
                                        'LISTEN 0 4096 127.0.0.1:1337 0.0.0.0:* users:(("python",pid=100,fd=8))\n')
        with patch.object(jr.subprocess, 'run', return_value=result), \
             patch.object(jr, '_jan_worker_ancestor', side_effect=lambda pid: pid == 42), \
             patch.object(jr, 'router_models', return_value=[{'id': 'Qwen3.5-9B-Q4_K_M.gguf'}]):
            found = jr.discover_router('Qwen3.5-9B-Q4_K_M.gguf')
        self.assertEqual(found, 'http://127.0.0.1:16010')

    def test_proc_net_parser_reads_listening_port_and_inode(self):
        example = ('sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n'
                   '0: 0100007F:3E8A 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 712345 1\n'
                   '1: 0100007F:0539 00000000:0000 01 00000000:00000000 00:00000000 00000000 1000 0 987654 1\n')
        with patch.object(jr.Path, 'read_text', return_value=example):
            ports = jr._tcp_listen_inode_ports()
        self.assertEqual(ports, {'712345': 16010})

    def test_proc_fallback_discovers_worker_router_if_ss_pid_hidden(self):
        with patch.object(jr, '_ports_owned_by_worker_ss', return_value=set()), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({42}, {16010})), \
             patch.object(jr, 'router_models', return_value=[{'id': 'Qwen3.5-9B-Q4_K_M.gguf'}]):
            self.assertEqual(jr.discover_router('Qwen3.5-9B-Q4_K_M.gguf'),
                             'http://127.0.0.1:16010')

    def test_router_model_mismatch_error_explains_what_was_found(self):
        with patch.object(jr, '_ports_owned_by_worker_ss', return_value=set()), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({42}, {16010})), \
             patch.object(jr, 'router_models', return_value=[{'id': 'different-model'}]):
            with self.assertRaises(jr.ResetError) as caught:
                jr.discover_router('Qwen3.5-9B-Q4_K_M.gguf')
        self.assertIn('different-model', str(caught.exception))
        self.assertIn('16010', str(caught.exception))

    def test_cli_flags(self):
        args = bm.arguments(['--dry-run', '--restart-jan-between-trials', '--cooldown-seconds', '2'])
        self.assertTrue(args.restart_jan_between_trials)
        self.assertEqual(args.cooldown_seconds, 2)
        self.assertTrue(bm.arguments(['--reuse-jan']).reuse_jan)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            bm.arguments(['--reuse-jan', '--restart-jan-between-trials'])

    def test_real_benchmark_loop_calls_reset_before_each_mode(self):
        class FakeResetter:
            calls = 0
            def __init__(self, *a, **kw): pass
            def check(self): return {'router_url':'http://127.0.0.1:1234', 'router_model':'fake-local-model', 'router_status':'loaded'}
            def reset(self):
                type(self).calls += 1
                return {'status':'verified', 'method':'model_unload_reload', 'elapsed_seconds':2.3,
                        'memory_available_before_mb':24000, 'memory_available_after_unload_mb':30000,
                        'memory_available_after_reload_mb':24000}
        FakeResetter.calls = 0
        cfg = self.cfg()
        cfg['max_output_tokens'] = 1024
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, 'http_json', SummarizationJan()), \
             patch.object(bm.bc, 'config', return_value=cfg), \
             patch.object(bm.bc, 'chosen_model', return_value='fake-local-model'), \
             patch.object(bm.jan_reset, 'JanModelReset', FakeResetter), \
             contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp) / 'out'
            rc = bm.main(['--smoke', '--restart-jan-between-trials', '--output', str(folder)])
            self.assertEqual(rc, 0)
            report = json.loads((folder/'results.json').read_text())
            self.assertEqual(report['metadata']['jan_reset_policy'], 'each_trial')
            self.assertEqual(len(report['resets']), 3)
            self.assertEqual(FakeResetter.calls, 3)
            self.assertTrue(all(r['reset_before_trial'] and r['reset_seconds'] == 2.3 for r in report['trials']))
            self.assertIn('Reset policy: **every trial**', (folder/'SUMMARY.md').read_text())
            self.assertIn('reset_memory_after_unload_mb', (folder/'results.csv').read_text())

    def test_reset_failure_stops_and_keeps_finished_trial(self):
        class FailingReset:
            calls = 0
            def __init__(self, *a, **kw): pass
            def check(self): return {'router_url':'http://127.0.0.1:1234', 'router_model':'fake-local-model', 'router_status':'loaded'}
            def reset(self):
                type(self).calls += 1
                if type(self).calls == 2:
                    error = jr.ResetError('simulated reload failure')
                    error.reset_record = {'status': 'failed', 'elapsed_seconds': 0.8}
                    raise error
                return {'status':'verified', 'method':'model_unload_reload', 'elapsed_seconds':1.1}
        FailingReset.calls = 0
        cfg = self.cfg()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, 'http_json', SummarizationJan()), \
             patch.object(bm.bc, 'config', return_value=cfg), \
             patch.object(bm.bc, 'chosen_model', return_value='fake-local-model'), \
             patch.object(bm.jan_reset, 'JanModelReset', FailingReset), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            folder = Path(temp) / 'out'
            rc = bm.main(['--smoke', '--restart-jan-between-trials', '--output', str(folder)])
            self.assertEqual(rc, 3)
            report = json.loads((folder/'results.json').read_text())
            self.assertEqual(len(report['trials']), 1)
            self.assertEqual(len(report['resets']), 2)
            self.assertEqual(report['resets'][1]['status'], 'failed')

    def test_authentication_failure_is_classified_not_hidden_as_port_failure(self):
        with patch.object(jr, '_probe_models', side_effect=jr.RouterHttpError(401, 'GET', 'http://127.0.0.1:41225/models')), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[]):
            with self.assertRaises(jr.ResetError) as caught:
                jr._probe_with_auth('http://127.0.0.1:41225')
        self.assertIn('requires authentication', str(caught.exception))
        self.assertIn('JAN_ROUTER_API_KEY', str(caught.exception))

    def test_api_key_is_used_for_model_list_and_reset_but_never_in_results(self):
        secret = 'SECRET_KEY_DO_NOT_REPORT'
        status = {'value': 'loaded'}
        requests = []

        def authorized_api(method, url, body=None, timeout=5, api_key=None):
            requests.append((method, url, api_key))
            if api_key != secret:
                raise jr.RouterHttpError(401, method, url)
            if url.endswith('/models/unload'):
                status['value'] = 'unloaded'
                return {'success': True}
            if url.endswith('/models/load'):
                status['value'] = 'loaded'
                return {'success': True}
            return {'data': [{'id': 'Qwen3.5-9B-Q4_K_M', 'status': dict(status)}]}

        with patch.object(jr, 'api_json', side_effect=authorized_api), \
             patch.object(jr, '_ports_owned_by_worker_ss', return_value={41225}), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({123}, {41225})), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[secret]), \
             patch.object(jr.JanModelReset, '_jan_api_ready', return_value=None):
            reset = jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', self.cfg(),
                                     router_url='http://127.0.0.1:41225', cooldown=0)
            info = reset.check()
            result = reset.reset()
        self.assertEqual(info['router_model'], 'Qwen3.5-9B-Q4_K_M')
        self.assertEqual(result['status'], 'verified')
        self.assertTrue(any(key is None for _, _, key in requests))
        self.assertTrue(any(key == secret for _, _, key in requests))
        self.assertNotIn(secret, repr(info) + repr(result))
        self.assertEqual([method for method, url, key in requests if method == 'POST'], ['POST', 'POST'])

    def test_autodiscovery_returns_authenticated_router_without_exposing_secret(self):
        secret = 'SECRET_KEY_DO_NOT_REPORT'
        def modeled(url, timeout=2, key=None):
            if key != secret:
                raise jr.RouterHttpError(401, 'GET', url + '/models')
            return [{'id': '/models/Qwen3.5-9B-Q4_K_M.gguf', 'status': {'value': 'loaded'}}]
        with patch.object(jr, '_ports_owned_by_worker_ss', return_value={41225}), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({123}, {41225})), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[secret]), \
             patch.object(jr, '_probe_models', side_effect=modeled):
            discovered = jr.discover_router('Qwen3.5-9B-Q4_K_M.gguf', include_auth=True)
        self.assertEqual(discovered, ('http://127.0.0.1:41225', secret))
        self.assertEqual(jr.validate_router_url(discovered[0]), discovered[0])

    def test_model_id_extension_match_requires_uniqueness(self):
        one = [{'id': 'Qwen3.5-9B-Q4_K_M'}]
        self.assertEqual(jr.match_model('Qwen3.5-9B-Q4_K_M.gguf', one), one[0])
        self.assertIsNone(jr.match_model('Qwen3.5-9B-Q4_K_M.gguf', one + one))

    def test_jan_log_credentials_require_exact_matching_port(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / 'logs').mkdir()
            (data / 'llamacpp').mkdir()
            preset = data / 'llamacpp' / 'router.preset.ini'
            # The parser never returns a key from a different or stale router port.
            lines = [
                'INFO Router argv: ' + json.dumps(['--port', '12345', '--api-key', 'WRONG']),
                'INFO Router argv: ' + json.dumps(['--models-preset', str(preset), '--port', '41225', '--api-key', 'RIGHT']),
            ]
            (data / 'logs' / 'app.log').write_text('\n'.join(lines))
            self.assertEqual(jr._log_keys_for_port([str(preset)], 41225), ['RIGHT'])
            self.assertEqual(jr._log_keys_for_port([str(preset)], 9999), [])

    def test_flatpak_default_jan_log_is_in_search_paths(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(jr.Path, 'home', return_value=Path(temp)):
            paths = jr._known_jan_log_paths([])
        self.assertIn(Path(temp) / '.var/app/ai.jan.Jan/data/Jan/data/logs/app.log', paths)

    def test_flatpak_log_key_is_found_by_exact_router_port(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            log = home / '.var/app/ai.jan.Jan/data/Jan/data/logs/app.log'
            log.parent.mkdir(parents=True)
            log.write_text('\n'.join([
                'INFO Router argv: ' + json.dumps(['--port', '1234', '--api-key', 'WRONG']),
                'INFO Router argv: ' + json.dumps(['--port', '41225', '--api-key', 'FLATPAK_SECRET']),
            ]))
            with patch.object(jr.Path, 'home', return_value=home):
                self.assertEqual(jr._log_keys_for_port([], 41225), ['FLATPAK_SECRET'])
                self.assertEqual(jr._log_keys_for_port([], 55555), [])

    def test_explicit_router_key_is_prioritized_over_discovered_stale_keys(self):
        with patch.dict(jr.os.environ, {'JAN_ROUTER_API_KEY': 'USER_KEY'}), \
             patch.object(jr, '_worker_pids', return_value=(set(), set())), \
             patch.object(jr, '_log_keys_for_port', return_value=['OLD_KEY']):
            self.assertEqual(jr._jan_router_secret_candidates(port=41225), ['USER_KEY', 'OLD_KEY'])

    def test_authentication_error_avoids_exposing_secret(self):
        with patch.object(jr, '_probe_models', side_effect=jr.RouterHttpError(401, 'GET', 'http://127.0.0.1:41225/models')), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=['TOP_SECRET']):
            with self.assertRaises(jr.ResetError) as caught:
                jr._probe_with_auth('http://127.0.0.1:41225')
        self.assertIn('1 candidate key(s) tried and rejected', str(caught.exception))
        self.assertNotIn('TOP_SECRET', str(caught.exception))

    def test_benchmark_gateway_key_can_authenticate_router_without_disclosure(self):
        secret = 'API_KEY_ALREADY_IN_CONFIG'
        requests = []
        def api(method, url, body=None, timeout=5, api_key=None):
            requests.append((method, api_key))
            if api_key != secret:
                raise jr.RouterHttpError(401, method, url)
            return {'data': [{'id':'Qwen3.5-9B-Q4_K_M.gguf',
                              'status': {'value':'loaded'}}]}
        cfg = self.cfg()
        cfg['api_key'] = secret
        with patch.object(jr, 'api_json', side_effect=api), \
             patch.object(jr, '_ports_owned_by_worker_ss', return_value={41225}), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({792102}, {41225})), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[]):
            info = jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', cfg).check()
        self.assertEqual(info['router_status'], 'loaded')
        self.assertEqual(info['router_url'], 'http://127.0.0.1:41225')
        self.assertNotIn(secret, repr(info))
        self.assertTrue(all(method == 'GET' for method, _ in requests))

    def test_distinct_router_key_failure_does_not_unload_or_reveal_gateway_key(self):
        secret = 'WRONG_GATEWAY_KEY'
        calls = []
        def reject(method, url, body=None, timeout=5, api_key=None):
            calls.append(method)
            raise jr.RouterHttpError(401, method, url)
        cfg = self.cfg()
        cfg['api_key'] = secret
        with patch.object(jr, 'api_json', side_effect=reject), \
             patch.object(jr, '_ports_owned_by_worker_ss', return_value={41225}), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({792102}, {41225})), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[]):
            with self.assertRaises(jr.ResetError) as caught:
                jr.JanModelReset('Qwen3.5-9B-Q4_K_M.gguf', cfg).check()
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn('candidate keys rejected', str(caught.exception))
        self.assertTrue(all(method == 'GET' for method in calls))

    def test_explicit_router_key_keeps_priority_over_gateway_key(self):
        secret = 'ROUTER_SECRET'
        requests = []
        def probe(url, timeout=2, key=None):
            requests.append(key)
            if key != secret:
                raise jr.RouterHttpError(401, 'GET', url + '/models')
            return [{'id': 'my_model'}]
        with patch.dict(jr.os.environ, {'JAN_ROUTER_API_KEY':secret}), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[secret]), \
             patch.object(jr, '_probe_models', side_effect=probe):
            _, key = jr._probe_with_auth('http://127.0.0.1:41225', gateway_key='PUBLIC_KEY')
        self.assertEqual(key, secret)
        self.assertEqual(requests, [None, secret])

    def test_discovery_missing_auth_reports_401_not_bare_port(self):
        with patch.object(jr, '_ports_owned_by_worker_ss', return_value={41225}), \
             patch.object(jr, '_ports_owned_by_worker_proc', return_value=({123}, {41225})), \
             patch.object(jr, '_probe_models', side_effect=jr.RouterHttpError(401, 'GET', 'http://127.0.0.1:41225/models')), \
             patch.object(jr, '_jan_router_secret_candidates', return_value=[]):
            with self.assertRaises(jr.ResetError) as caught:
                jr.discover_router('Qwen3.5-9B-Q4_K_M.gguf')
        self.assertIn('authentication required', str(caught.exception))
        self.assertIn('41225', str(caught.exception))

if __name__ == '__main__':
    unittest.main()
