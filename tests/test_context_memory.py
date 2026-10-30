"""Context-management regressions using deterministic doubles, not model benchmarks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark as bm
import branch_context as bc
import context_memory as cm
import recursive_benchmark as rb
import trap_benchmark as tb
from branch_runtime import RunBudget, RunLimitExceeded
from test_branch_runtime import final, tool_call
from test_recursive_benchmark import RecursiveFakeJan


class MemoryTests(unittest.TestCase):
    def test_archive_ranges_round_trip_and_do_not_alias_live_messages(self):
        original = [{'role': 'user', 'content': 'unique evidence Ω' * 400}]
        archives = []
        ref = cm.archive_snapshot(archives, original)
        original[0]['content'] = 'changed'
        expected = json.dumps(archives[0], ensure_ascii=False, indent=2)
        recovered = ''
        for start in range(0, len(expected), 500):
            recovered += cm.read_archive(archives, ref, start, 500).split('\n', 1)[1].rsplit('\n', 1)[0]
        self.assertEqual(recovered, expected)
        for ref in ('archive/0', 'archive/-1', '../../secret', 'archive/2'):
            with self.assertRaises(ValueError):
                cm.read_archive(archives, ref)

    def test_runtime_compaction_archives_exact_evidence_and_charges_budget(self):
        cfg = dict(bc.DEFAULTS, context_limit_tokens=8192)
        messages = [{'role': 'system', 'content': bc.SYSTEM}, {'role': 'user', 'content': 'Find key K.'},
                    {'role': 'user', 'content': 'SOURCE alpha line 7: K=42. ' + 'noise ' * 1600},
                    {'role': 'assistant', 'content': 'Still investigating; no result verified.'}]
        original = copy.deepcopy(messages)
        budget = RunBudget('r', cfg)
        metrics = {'model_calls': 0, 'checkpoints': 0, 'calls': []}
        archives = []
        reply = bc.ChatMessage({'content': 'Goal: find K. Observed alpha:7 K=42; verification pending.'},
                               {'prompt_tokens': 3000, 'completion_tokens': 30}, 'stop')
        with patch.object(bc, 'chat', return_value=reply):
            self.assertTrue(bc.compact_working_memory(cfg, 'm', messages, None, budget, metrics, archives))
        self.assertEqual(archives[0], original)
        self.assertEqual(messages[:2], original[:2])
        self.assertNotIn('noise noise', json.dumps(messages))
        self.assertIn('verification pending', json.dumps(messages))
        self.assertEqual(budget.state['model_calls'], 1)
        self.assertEqual(budget.state['total_tokens'], 3030)

    def test_invalid_memory_retains_original_context(self):
        cfg = dict(bc.DEFAULTS)
        messages = [{'role': 'system', 'content': bc.SYSTEM}, {'role': 'user', 'content': 'Goal'},
                    {'role': 'user', 'content': 'evidence ' * 1200}, {'role': 'assistant', 'content': 'Pending'}]
        original = copy.deepcopy(messages)
        for content, stop in (('', 'stop'), ('partial', 'length'), ('partial', 'max_tokens'), ('x' * 2000, 'stop')):
            archives = []
            reply = bc.ChatMessage({'content': content}, {}, stop)
            with patch.object(bc, 'chat', return_value=reply), self.assertRaises(RunLimitExceeded):
                bc.compact_working_memory(cfg, 'm', messages, None, RunBudget('r', cfg),
                    {'model_calls': 0, 'checkpoints': 0, 'calls': []}, archives)
            self.assertEqual(messages, original)
            self.assertEqual(archives, [])

    def test_calibrated_predictions_never_shrink_below_observed_ratio(self):
        rows = [{'prompt_est_tokens': 100, 'prompt_tokens': 180}]
        self.assertEqual(cm.predicted_tokens([], lambda m: 200, rows), 360)

    def test_early_runtime_read_is_isolated_and_child_shares_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = 'UNIQUE-SOURCE-DETAIL K=42\n' * 170
            (root / 'evidence.txt').write_text(source)
            cfg = dict(bc.DEFAULTS, model='m', context_limit_tokens=6000,
                       max_branch_depth=1, read_only_project_root=temp,
                       context_memory=False, save_transcripts=True)
            seen = []
            def agent(cfg, model, messages, tools=None):
                seen.append(copy.deepcopy(messages))
                if not any(m['role'] == 'tool' for m in messages):
                    return {'tool_calls': [tool_call('read', 'read_project_file', path='evidence.txt', max_lines=200)]}
                return final('K=42, evidence.txt:1; model-reported')
            with patch.object(bc, 'chat', agent):
                result = bc.run_branch(cfg, 'Find K', relevant_context='Background. ' * 170,
                                       data_dir=root / 'branches')
            self.assertEqual(result['status'], 'success', result)
            self.assertEqual(result['usage']['branches'], 2)
            parent = bc.load_branch(result['branch_id'], root / 'branches')
            self.assertEqual(parent['runtime']['early_branches'], 1)
            self.assertNotIn('UNIQUE-SOURCE-DETAIL', json.dumps(parent['transcript']))
            child = bc.load_branch(parent['subbranches'][0], root / 'branches')
            self.assertIn('UNIQUE-SOURCE-DETAIL', json.dumps(child['transcript']))
            self.assertEqual(result['usage']['model_calls'], 4)

    def test_active_summary_runs_inside_first_case_and_archive_is_retrievable(self):
        cfg = dict(bc.DEFAULTS, _context_memory_threshold=.4, _memory_max_chars=1800,
                   _summary_output_tokens=512, max_output_tokens=256)
        actions = iter([{'action': 'inspect', 'path': 'logs/diagnostics'},
                        {'action': 'inspect', 'path': 'archive/1'},
                        {'action': 'inspect', 'path': 'guide/index'}])
        seen = []
        def api(method, url, cfg, payload=None):
            msgs = payload['messages']; seen.append(copy.deepcopy(msgs))
            if 'Compress the observed conversation' in msgs[-1]['content']:
                answer = 'Current incident incomplete. Diagnostic log is historical; no repair verified. Inspect authoritative sources.'
            elif 'FINAL REPORT' in msgs[-1]['content']:
                answer = '{}'
            else:
                answer = json.dumps(next(actions, {'action': 'verify'}))
            return {'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': bm.approximate_tokens(msgs), 'completion_tokens': 40}}
        with tempfile.TemporaryDirectory() as temp, patch.object(bc, 'http_json', api):
            result = tb.run_trap_trial('summarized', 17, 1, cfg, 'm', 8192, Path(temp), bm,
                                      trap_log_lines=100, max_actions=3)
            archives = json.loads((Path(temp) / 'context_archives.json').read_text())['parent']
        self.assertGreater(result['summaries_generated'], 0, result['error'])
        self.assertEqual(result['summaries'][0]['during_batch'], 'batch_1')
        self.assertIn('NON-AUTHORITATIVE HISTORICAL LOG', json.dumps(archives))
        self.assertTrue(any('archive/1 chars' in json.dumps(m) for m in seen))
        self.assertEqual(result['verified_cases'], 0)

    def test_early_benchmark_branch_runs_before_parent_reads_shards(self):
        fake = RecursiveFakeJan()
        cfg = dict(bc.DEFAULTS, _early_branch_threshold=.1, _context_memory_threshold=0,
                   _adaptive_max_tokens=300000, _adaptive_max_calls=96, max_output_tokens=512)
        with tempfile.TemporaryDirectory() as temp, patch.object(bc, 'http_json', fake):
            result = tb.run_trap_trial('adaptive', 17, 1, cfg, 'm', 8192, Path(temp), bm,
                                      nested_depth=2, nested_log_lines=8,
                                      recovery_log_lines=8, max_actions=8)
        self.assertEqual(result['adaptive_early_branches'], 1, result['error'])
        self.assertEqual(result['nested_index_completed'], 1)
        parents = [m for m in fake.messages_seen if 'RECURSIVE STAGE=' not in m[0]['content']]
        self.assertTrue(any('EARLY BRANCH VERIFIED' in json.dumps(m) for m in parents))
        self.assertNotIn('INDEX SHARD A:', json.dumps(parents))
        self.assertIn('INDEX SHARD A:', json.dumps(fake.messages_seen))

    def test_disabled_memory_does_not_call_model(self):
        cfg = dict(bc.DEFAULTS, context_memory=False)
        with patch.object(bc, 'chat') as api:
            self.assertFalse(bc.compact_working_memory(cfg, 'm', [{}] * 10, None,
                RunBudget('r', cfg), {}, []))
        api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
