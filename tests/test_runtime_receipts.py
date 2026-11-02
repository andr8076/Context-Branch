"""Exercise real runtime handoffs with scripted model responses, without a GPU."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import branch_context as bc
from branch_runtime import RunBudget, normalize_conversation
from receipt_archive import ReceiptArchive
from test_branch_runtime import final, tool_call


class RuntimeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'branches'
        (self.root / 'config.ini').write_text('retry_delay=17')
        self.cfg = dict(bc.DEFAULTS, model='test', read_only_project_root=str(self.root),
                        save_transcripts=False, context_memory=False, early_branching=False)

    def run_script(self, child_reply=None):
        replies = iter([
            {'tool_calls': [tool_call('delegate', 'solve_nested_branch', task='Inspect delay configuration')]},
            {'tool_calls': [tool_call('read', 'read_project_file', path='config.ini')]},
            child_reply or final('Configuration inspected; exact value available in evidence.'),
            final('Ready for follow-up'),
        ])
        self.seen = []
        def chat(cfg, model, messages, tools=None):
            self.seen.append(copy.deepcopy(messages))
            self.assertEqual(normalize_conversation(messages, bc.SYSTEM)[1], 0)
            return next(replies)
        with patch.object(bc, 'chat', chat):
            result = bc.run_branch(self.cfg, 'Prepare retry calculation', data_dir=self.data)
        self.parent = bc.load_branch(result['branch_id'], self.data)
        self.child = bc.load_branch(self.parent['subbranches'][0], self.data)
        return result

    def test_resume_after_child_release_recovers_omitted_value(self):
        result = self.run_script()
        self.assertEqual(result['usage']['model_calls'], 4)
        self.assertNotIn('transcript', self.child)
        self.assertNotIn('retry_delay=17', json.dumps(self.seen[-1]))
        receipt = json.loads(self.child['receipt'].split('\n', 1)[1])
        self.assertEqual(receipt['verification'], 'model_reported')
        store = ReceiptArchive(self.data / 'receipts')
        self.assertIn('retry_delay=17', store.retrieve(receipt['archive'], 3)['text'])
        turns = []
        def resumed(cfg, model, messages, tools=None):
            turns.append(copy.deepcopy(messages))
            if len(turns) == 1:
                self.assertIn('read_child_evidence', json.dumps(tools))
                return {'tool_calls': [tool_call('recall', 'read_child_evidence',
                    archive=receipt['archive'], message_index=3)]}
            evidence = json.loads(messages[-1]['content'])
            self.assertEqual(evidence['status'], 'historical_observation_not_current_verification')
            value = int(evidence['text'].split('retry_delay=')[1].strip())
            return final(str(value * 2))
        with patch.object(bc, 'chat', resumed):
            result = bc.run_branch(self.cfg, 'Compute two delays', existing_id=result['branch_id'], data_dir=self.data)
        self.assertEqual(result['result'], '34')
        self.assertEqual(result['usage']['model_calls'], 6)
        self.assertGreater(result['usage']['total_tokens'], 0)

    def test_partial_child_stays_partial_in_receipt(self):
        reply = final('Still missing a second source')
        body = json.loads(reply['content'])
        body.update(status='partial', caveats='Unconfirmed')
        reply['content'] = json.dumps(body)
        self.run_script(reply)
        receipt = json.loads(self.child['receipt'].split('\n', 1)[1])
        self.assertEqual(receipt['outcome'], 'partial')
        self.assertIn('Unconfirmed', receipt['unresolved'])

    def test_storage_failure_retains_evidence_and_reports_failure(self):
        replace = os.replace
        def fail_archive(source, target):
            if Path(target).parent.name == 'receipts':
                raise OSError('disk unavailable')
            return replace(source, target)
        with patch('receipt_archive.os.replace', side_effect=fail_archive):
            self.run_script()
        self.assertEqual(self.child['stop_reason'], 'receipt_handoff_failed')
        self.assertIn('retry_delay=17', json.dumps(self.child['transcript']))
        self.assertNotIn('receipt', self.child)
        self.assertIn('Child receipt handoff failed', json.dumps(self.seen[-1]))

    def test_budget_refusal_retains_child(self):
        original = ReceiptArchive.handoff
        def refuse(store, parent, child, receipt, **kwargs):
            return original(store, parent, child, receipt, fits=lambda candidate: False)
        with patch.object(ReceiptArchive, 'handoff', refuse):
            self.run_script()
        self.assertEqual(self.child['stop_reason'], 'receipt_handoff_failed')
        self.assertIn('retry_delay=17', json.dumps(self.child['transcript']))
        self.assertFalse(list((self.data / 'receipts').glob('*.json')))

    def test_oversized_serialized_receipt_retains_child(self):
        # JSON escaping can exceed the whole-receipt limit despite field caps.
        self.run_script(final('\x01' * 1000))
        self.assertEqual(self.child['stop_reason'], 'receipt_handoff_failed')
        self.assertIn('retry_delay=17', json.dumps(self.child['transcript']))

    def test_retrieval_refuses_unowned_archive_and_over_budget_range(self):
        result = self.run_script()
        receipt = json.loads(self.child['receipt'].split('\n', 1)[1])
        predict = RunBudget.predicted_prompt
        def pressure(budget, messages, tools=None):
            if messages[-1].get('role') == 'tool' and 'historical_observation_not_current_verification' in messages[-1].get('content', ''):
                return self.cfg['context_limit_tokens']
            return predict(budget, messages, tools)
        for identifier, error in [('0' * 32, 'not returned'), (receipt['archive'], 'does not fit')]:
            turns = []
            def agent(cfg, model, messages, tools=None):
                turns.append(copy.deepcopy(messages))
                if len(turns) == 1:
                    return {'tool_calls': [tool_call('recall', 'read_child_evidence',
                        archive=identifier, message_index=3)]}
                self.assertIn(error, messages[-1]['content'])
                self.assertNotIn('retry_delay=17', messages[-1]['content'])
                return final('Evidence unavailable; cannot calculate')
            with patch.object(bc, 'chat', agent), patch.object(RunBudget, 'predicted_prompt', pressure):
                bc.run_branch(self.cfg, 'Recall evidence', existing_id=result['branch_id'], data_dir=self.data)

    def test_receipt_budget_checks_actual_parent_tool_exchange(self):
        predict = RunBudget.predicted_prompt
        checked = []
        def pressure(budget, messages, tools=None):
            if messages[-1].get('content', '').startswith('CHILD RECEIPT:'):
                checked.append(copy.deepcopy(messages))
                return self.cfg['context_limit_tokens']
            return predict(budget, messages, tools)
        with patch.object(RunBudget, 'predicted_prompt', pressure):
            self.run_script()
        self.assertTrue(checked)
        self.assertEqual(checked[0][-1]['role'], 'tool')
        self.assertEqual(checked[0][-1]['tool_call_id'], 'delegate')
        self.assertEqual(self.child['stop_reason'], 'receipt_handoff_failed')
        self.assertIn('retry_delay=17', json.dumps(self.child['transcript']))


if __name__ == '__main__':
    unittest.main()
