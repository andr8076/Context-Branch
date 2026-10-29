"""Controller regression tests; these are not local-model performance results."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark as bm
import trap_benchmark as tb
from test_trap_benchmark import DeliberateFakeJan


class NestedFake(DeliberateFakeJan):
    def __call__(self, method, url, cfg, payload=None):
        if method == 'POST' and 'ADAPTIVE CHILD CONTEXT' in payload['messages'][0]['content']:
            msgs = payload['messages']
            level = int(re.search(r'level=(\d+)', msgs[0]['content']).group(1))
            path = 'adaptive/blocker' if level == 1 else f'adaptive/blocker/{level}'
            actions = [tb.parse_action(m['content']) for m in msgs if m['role'] == 'assistant']
            if not any(a and a.get('path') == path for a in actions):
                action = {'action': 'inspect', 'path': path}
            else:
                evidence = next(m['content'] for m in reversed(msgs)
                                if m['role'] == 'user' and 'reconciliation_token=' in m['content'])
                action = {'action': 'resolve_blocker', 'level': level,
                          'token': re.search(r'reconciliation_token=(CHECK-[A-Z2-9]+)', evidence).group(1)}
            answer = json.dumps(action)
            self.messages_seen.append(json.loads(json.dumps(msgs)))
            self.limits.append(payload['max_tokens'])
            return {'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': len(json.dumps(msgs)) // 4, 'completion_tokens': len(answer) // 4}}
        return super().__call__(method, url, cfg, payload)


class AdaptiveGuardTests(unittest.TestCase):
    def run_case(self, fake=None, **overrides):
        cfg = {**bm.bc.DEFAULTS, 'api_base_url': 'http://fake/v1', 'max_output_tokens': 1024,
               'timeout_seconds': 30, '_autonomous_max_tokens': 100000,
               '_adaptive_surprise_levels': 2, '_context_limit': 8192}
        cfg.update(overrides)
        fake = fake or NestedFake()
        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, 'http_json', fake):
            result = tb.run_trap_trial('adaptive', 17, 1, cfg, 'mock', cfg['_context_limit'],
                                      Path(temp), bm, nested_depth=0, recovery_log_lines=0,
                                      nested_log_lines=0, pressure_records=0, max_actions=48)
        return result, fake

    def test_nested_return_finishes_both_enclosing_tasks(self):
        result, fake = self.run_case()
        self.assertEqual(result['verified_cases'], 1, result['error'])
        self.assertEqual(result['adaptive_child_resolutions'], 2)
        self.assertEqual(result['adaptive_resumes'], 2)
        events = result['adaptive_events'][0]['events']
        self.assertEqual([e['depth'] for e in events], [0, 1, 2])
        self.assertTrue(all(e['completed'] for e in events[1:]))
        self.assertTrue(events[0]['post_resume_trace'])
        nested_inputs = [m for m in fake.messages_seen if 'level=2' in m[0]['content']]
        secret = re.findall(r'reconciliation_token=(CHECK-[A-Z2-9]+)', json.dumps(nested_inputs))[-1]
        resumed = [m for m in fake.messages_seen if 'NESTED BRANCH RESUME' in json.dumps(m)
                   and 'level=1' in m[0]['content']]
        self.assertTrue(resumed)
        self.assertIn('level-1 blocker remains active', json.dumps(resumed))
        self.assertNotIn(secret, json.dumps(resumed))

    def test_depth_stop_is_explicit_and_not_success(self):
        result, _ = self.run_case(_adaptive_max_depth=1)
        self.assertEqual(result['verified_cases'], 0)
        self.assertEqual(result['error_type'], 'branch_depth_limit')
        self.assertTrue(all(e['status'] != 'running' for e in result['adaptive_events'][0]['events']))

    def test_call_guard_includes_requests_without_usage(self):
        class NoUsage(NestedFake):
            def __call__(self, *args, **kwargs):
                response = super().__call__(*args, **kwargs)
                response.pop('usage', None)
                return response
        result, _ = self.run_case(NoUsage(), _adaptive_max_calls=1)
        self.assertEqual(result['error_type'], 'model_call_budget_exhausted')
        self.assertEqual(result['verified_cases'], 0)

    def test_capacity_stop_preserves_evidence(self):
        result, fake = self.run_case(_context_limit=1100)
        self.assertEqual(result['error_type'], 'context_capacity_exceeded_estimate')
        self.assertEqual(result['adaptive_decisions'], 0)

    def test_reserves_generation_against_shared_work_budget(self):
        result, _ = self.run_case(_adaptive_max_tokens=1100)
        self.assertEqual(result['error_type'], 'token_budget_exhausted')
        self.assertEqual(result['adaptive_decisions'], 0)

    def test_retry_inherits_output_and_is_counted(self):
        class EmptyOnce(NestedFake):
            empty_sent = False

            def __call__(self, method, url, cfg, payload=None):
                if method == 'POST' and not self.empty_sent:
                    self.empty_sent = True
                    self.limits.append(payload['max_tokens'])
                    return {'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}],
                            'usage': {'prompt_tokens': 100, 'completion_tokens': 1024}}
                return super().__call__(method, url, cfg, payload)

        result, fake = self.run_case(EmptyOnce(), _no_action_retry_output_tokens=256,
                                     _no_action_retry_reasoning_effort='none')
        self.assertEqual(result['verified_cases'], 1, result['error'])
        self.assertEqual(fake.limits[:2], [1024, 1024])
        self.assertGreater(result['no_action_retry_total_tokens'], 0)

    def test_child_transport_failure_preserves_stop_status(self):
        class BrokenChild(NestedFake):
            def __call__(self, method, url, cfg, payload=None):
                if method == 'POST' and 'ADAPTIVE CHILD CONTEXT' in payload['messages'][0]['content']:
                    raise TimeoutError('test timeout')
                return super().__call__(method, url, cfg, payload)

        result, _ = self.run_case(BrokenChild())
        self.assertEqual(result['error_type'], 'timeout')
        self.assertEqual(result['adaptive_branch_failures'], 1)
        self.assertFalse(any(e['status'] == 'running' for e in result['adaptive_events'][0]['events']))
        self.assertEqual(result['adaptive_resumes'], 0)

    def test_refused_actions_consume_work(self):
        class Delegator(NestedFake):
            def __call__(self, method, url, cfg, payload=None):
                if method == 'POST' and 'ADAPTIVE CHILD CONTEXT' in payload['messages'][0]['content']:
                    return {'choices': [{'message': {'content': '{"action":"delegate"}'}, 'finish_reason': 'stop'}]}
                return super().__call__(method, url, cfg, payload)

        result, _ = self.run_case(Delegator(), _adaptive_max_tokens=200000)
        self.assertEqual(result['verified_cases'], 0)
        self.assertGreater(result['out_of_scope_actions'], 0)
        self.assertEqual(result['error_type'], 'adaptive_scope_loop')
        events = result['adaptive_events'][0]['events']
        self.assertEqual(events[1]['status'], 'adaptive_scope_loop')

    def test_child_count_guard(self):
        result, _ = self.run_case(_adaptive_max_children=1)
        self.assertEqual(result['error_type'], 'branch_count_limit')

    def test_wrong_outer_token_does_not_create_nested_blocker(self):
        puzzle = tb.create_trap_puzzle(
            17, 1, 0, 1, 0, 0, 0, 0, 48, adaptive_surprise=True)
        case = puzzle['cases'][0]
        case['adaptive_surprise_levels'] = 2
        case['adaptive_nested_tokens'] = {2: 'CHECK-NESTED'}
        case['docs']['adaptive/blocker/2'] = 'state=CURRENT reconciliation_token=CHECK-NESTED'
        sim = tb.RepairSimulator(case)
        sim.step({'action': 'repair', 'release': 'x', 'route': 'x', 'certificate': 'x'})
        sim.step({'action': 'inspect', 'path': 'adaptive/blocker'})
        text = sim.step({'action': 'resolve_blocker', 'level': 1, 'token': 'CHECK-WRONG'})
        self.assertIn('REJECTED', text)
        self.assertEqual(sim.adaptive_nested_triggered, set())
        self.assertEqual(sim.surprise_resolution_failures, 1)
        self.assertEqual(sim.adaptive_nested_resolution_failures, 0)
        self.assertEqual(sim.history[-1]['label'], 'adaptive_blocker_rejected')

    def test_failed_child_does_not_count_as_parent_resume(self):
        class ImmediateFailure(NestedFake):
            def __call__(self, method, url, cfg, payload=None):
                if method == 'POST' and 'ADAPTIVE CHILD CONTEXT' in payload['messages'][0]['content']:
                    answer = '{"action":"delegate"}'
                else:
                    answer = '{"action":"repair","release":"REL-X","route":"ROUTE-X","certificate":"CERT-X"}'
                return {'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}

        result, _ = self.run_case(ImmediateFailure(), _adaptive_surprise_levels=1,
                                  _adaptive_max_tokens=10000)
        self.assertEqual(result['error_type'], 'adaptive_scope_loop')
        self.assertTrue(result['partial_failure'])
        self.assertEqual(result['adaptive_resumes'], 0)
        self.assertEqual(result['adaptive_branch_failures'], 1)
        self.assertFalse(result['adaptive_events'][0]['events'][0]['resumed'])

    def test_nested_failure_never_returns_a_success_receipt(self):
        class StuckNested(NestedFake):
            def __call__(self, method, url, cfg, payload=None):
                if method == 'POST' and 'level=2' in payload['messages'][0]['content']:
                    return {'choices': [{'message': {'content': '{"action":"delegate"}'}, 'finish_reason': 'stop'}]}
                return super().__call__(method, url, cfg, payload)

        result, fake = self.run_case(StuckNested(), _adaptive_max_tokens=200000)
        self.assertEqual(result['error_type'], 'nested_child_failed')
        self.assertEqual(result['adaptive_branch_failures'], 2)
        self.assertEqual(result['adaptive_resumes'], 0)
        self.assertNotIn('NESTED BRANCH RESUME', json.dumps(fake.messages_seen))

    def test_three_nested_returns(self):
        result, _ = self.run_case(_adaptive_surprise_levels=3)
        self.assertEqual(result['verified_cases'], 1, result['error'])
        self.assertEqual(result['adaptive_child_resolutions'], 3)

    def test_corrupted_resume_is_measured_before_inference(self):
        original = tb.request_investigation_action

        def corrupt(cfg, model, context, *args, **kwargs):
            if any(m['content'].startswith('ADAPTIVE BRANCH RESUME') for m in context):
                context.insert(1, {'role': 'user', 'content': 'LEAKED PRIVATE TRANSCRIPT'})
            return original(cfg, model, context, *args, **kwargs)

        with patch.object(tb, 'request_investigation_action', corrupt):
            result, fake = self.run_case()
        self.assertEqual(result['error_type'], 'context_isolation_violation')
        self.assertEqual(result['adaptive_child_transcript_leaks'], 1)
        self.assertNotIn('LEAKED PRIVATE TRANSCRIPT', json.dumps(fake.messages_seen))


if __name__ == '__main__':
    unittest.main()
