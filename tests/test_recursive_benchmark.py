"""Controlled recursive context isolation tests; fake agent, never a Qwen result."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark as bm
import trap_benchmark as tb
from test_trap_benchmark import DeliberateFakeJan


class RecursiveFakeJan(DeliberateFakeJan):
    """A model double that chooses actions only from visible stage documents."""
    def __call__(self, method, url, cfg, payload=None):
        if method == 'GET':
            return super().__call__(method, url, cfg, payload)
        msgs = payload['messages']
        match = re.search(r'RECURSIVE STAGE=(index|escrow|recovery|repair)', msgs[0]['content'])
        if not match:
            return super().__call__(method, url, cfg, payload)
        stage = match.group(1)
        if len(json.dumps(msgs, ensure_ascii=False)) > self.max_chars:
            raise RuntimeError('Simulated context capacity exceeded')
        self.messages_seen.append(json.loads(json.dumps(msgs)))
        self.limits.append(payload['max_tokens'])
        already = [tb.parse_action(m['content']) for m in msgs if m['role']=='assistant']
        read = [a['path'] for a in already if a and a['action']=='inspect']
        if 'guide/index' not in read:
            action = {'action':'inspect','path':'guide/index'}
        elif stage in ('index','escrow','recovery'):
            for shard in (stage+'/shard-a', stage+'/shard-b'):
                if shard not in read:
                    action={'action':'inspect','path':shard};break
            else:
                if stage in ('index','escrow'):
                    token=self._derive_nested(msgs,stage)
                    action={'action':'rebuild_index' if stage=='index' else 'unlock_escrow','token':token}
                else:
                    action={'action':'recover','token':self._derive_recovery(msgs)}
        else:
            required=('guide/policy','registry/releases','registry/approvals',
                      'network/aliases','network/routes','security/certificates')
            for doc in required:
                if doc not in read:
                    action={'action':'inspect','path':doc};break
            else:
                if any(a and a['action']=='repair' for a in already):
                    action={'action':'verify'}
                else:
                    action={'action':'repair', **self._derive_repair(msgs)}
        answer=json.dumps(action)
        return {'choices':[{'message':{'role':'assistant','content':answer},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':len(json.dumps(msgs))//4,'completion_tokens':len(answer)//4}}


class RecursiveContextTests(unittest.TestCase):
    def cfg(self):
        c=dict(bm.bc.DEFAULTS)
        c.update({'max_output_tokens':1024, 'max_branch_depth':0,
                  'max_agent_turns':1,'save_transcripts':False, '_recursive_max_depth':3})
        return c

    def test_nested_stages_share_simulator_not_conversations(self):
        fake=RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as temp,patch.object(bm.bc,'http_json',fake):
            trial=tb.run_trap_trial('recursive',17,1,self.cfg(),'mock',8192,
                                    Path(temp),bm,nested_depth=2,
                                    recovery_log_lines=8,nested_log_lines=8,
                                    pressure_records=0)
        self.assertEqual(trial['verified_cases'],1,trial['case_breakdown'])
        self.assertTrue(trial['exact_final'],trial['error'])
        self.assertEqual(trial['recursive_branch_count'],3)
        self.assertEqual(trial['recursive_context_count'],4)
        self.assertEqual(trial['recursive_max_depth'],3)
        self.assertEqual(trial['recursive_stage_successes'],4)
        self.assertEqual(trial['nested_index_completed'],1)
        self.assertEqual(trial['nested_escrow_completed'],1)
        self.assertEqual(trial['recovery_completed'],1)
        self.assertEqual(trial['wrong_repairs'],0)
        stages=trial['recursive_events'][0]['stages']
        self.assertEqual([s['stage'] for s in stages],['repair','recovery','escrow','index'])
        self.assertEqual([s['depth'] for s in stages],[0,1,2,3])
        self.assertEqual(sum(s['model_actions'] for s in stages),trial['total_actions'])
        # No parent ever receives a child transcript, shard fragments or secret token.
        model_inputs=[m for msgs in fake.messages_seen for m in msgs]
        root=[msgs for msgs in fake.messages_seen
              if 'RECURSIVE STAGE=repair' in msgs[0]['content']]
        self.assertTrue(root)
        joined=json.dumps(root)
        self.assertIn('CHILD RESULT: recovery stage completed',joined)
        self.assertNotIn('RECOVERY SHARD A:',joined)
        self.assertNotIn('INDEX SHARD A:',joined)
        self.assertIn('branch_recursive_3_index',[c['phase'] for c in trial['calls']])
        self.assertIn('branch_recursive_0_repair',[c['phase'] for c in trial['calls']])

    def test_recursive_survives_single_incident_context_pressure(self):
        # Genuine prompts from each stage, but a scripted agent and simplified
        # character/token cap: not empirical Qwen performance.
        outcome = {}
        for mode in ('branch','recursive'):
            fake = RecursiveFakeJan(max_chars=8192*4)
            with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc,'http_json',fake):
                outcome[mode] = tb.run_trap_trial(
                    mode, 17, 1, self.cfg(), 'mock', 8192, Path(temp), bm,
                    nested_depth=2, recovery_log_lines=36, nested_log_lines=14,
                    pressure_records=4, max_actions=32)
        self.assertEqual(outcome['branch']['verified_cases'],0)
        self.assertEqual(outcome['branch']['error_type'],'context_overflow_confirmed')
        self.assertEqual(outcome['recursive']['verified_cases'],1, outcome['recursive']['error'])
        self.assertTrue(outcome['recursive']['exact_final'])
        self.assertLess(outcome['recursive']['peak_branch_prompt_tokens'],8192)

    def test_shared_action_budget_prevents_free_recursion(self):
        fake=RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as temp,patch.object(bm.bc,'http_json',fake):
            result=tb.run_trap_trial('recursive',17,1,self.cfg(),'mock',8192,
                                     Path(temp),bm,nested_depth=2,max_actions=12,
                                     recovery_log_lines=8,nested_log_lines=8)
        self.assertLessEqual(result['total_actions'],12)
        self.assertEqual(result['verified_cases'],0)
        self.assertGreater(result['recursive_stage_failures'],0)
        self.assertFalse(result['exact_final'])

    def test_depth_cap_is_explicit_and_no_hidden_oracle_solution(self):
        cfg=self.cfg();cfg['_recursive_max_depth']=1
        fake=RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as temp,patch.object(bm.bc,'http_json',fake):
            result=tb.run_trap_trial('recursive',17,1,cfg,'mock',8192,
                                     Path(temp),bm,nested_depth=2)
        self.assertEqual(result['total_actions'],0)
        self.assertEqual(result['verified_cases'],0)
        self.assertEqual(result['recursive_events'][0]['stages'][-1]['status'],'depth_limit')

    def test_empty_investigation_action_is_measured_not_fabricated(self):
        fake = RecursiveFakeJan()
        injected = False

        def maybe_empty(method, url, cfg, payload=None):
            nonlocal injected
            if (payload and not injected
                    and payload['messages'][0]['content'].startswith('RECURSIVE STAGE=')):
                injected = True
                return {'choices':[{'message':{'content':'', 'reasoning_content':'hidden reasoning'},
                                    'finish_reason':'length'}],
                        'usage':{'prompt_tokens':250, 'completion_tokens':1024,
                                 'completion_tokens_details':{'reasoning_tokens':1024}}}
            return fake(method, url, cfg, payload)

        with tempfile.TemporaryDirectory() as temp, patch.object(bm.bc, 'http_json', maybe_empty):
            trial = tb.run_trap_trial(
                'recursive', 17, 1, self.cfg(), 'mock', 8192, Path(temp), bm,
                nested_depth=2, recovery_log_lines=8, nested_log_lines=8)
        self.assertTrue(injected)
        self.assertEqual(trial['no_action_investigation_calls'], 1)
        self.assertEqual(trial['empty_investigation_replies'], 1)
        self.assertEqual(trial['invalid_actions'], 0)  # recovered before simulator action
        self.assertEqual(trial['no_action_retry_attempts'], 1)
        self.assertEqual(trial['no_action_retry_recoveries'], 1)
        self.assertEqual(trial['no_action_retry_failures'], 0)
        self.assertEqual(trial['verified_cases'], 1)
        self.assertTrue(trial['exact_final'])

    def test_wrong_stage_actions_are_rejected_without_environment_changes(self):
        fake = RecursiveFakeJan()
        wrong = [False]
        def repeat_child(method, url, cfg, payload=None):
            if payload and not wrong[0] and 'RECURSIVE STAGE=escrow' in payload['messages'][0]['content']:
                wrong[0] = True
                answer = json.dumps({'action': 'rebuild_index', 'token': 'WRONG'})
                return {'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 70, 'completion_tokens': 30}}
            return fake(method, url, cfg, payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', repeat_child):
            trial = tb.run_trap_trial('recursive', 17, 1, self.cfg(), 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2, max_actions=32,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertTrue(wrong[0])
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        self.assertEqual(trial['out_of_scope_actions'], 1)
        self.assertEqual(trial['nested_index_attempts'], 1)  # no repeat was applied
        self.assertEqual(trial['case_breakdown'][0]['stage_scope_rejected'], 1)
        escrow = next(e for e in trial['recursive_events'][0]['stages'] if e['stage'] == 'escrow')
        self.assertEqual(escrow['rejected_out_of_scope'], 1)

    def test_scoped_handoffs_do_not_expose_child_documents(self):
        fake = RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', fake):
            trial = tb.run_trap_trial('recursive', 17, 1, self.cfg(), 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        root = json.dumps([msgs for msgs in fake.messages_seen
                           if 'RECURSIVE STAGE=repair' in msgs[0]['content']])
        self.assertIn('registry/approvals unlocked', root)
        for forbidden in ('index/shard-a', 'escrow/shard-a', 'recovery/shard-a',
                          'INDEX SHARD A:', 'RECOVERY SHARD A:'):
            self.assertNotIn(forbidden, root)
        for event in trial['recursive_events'][0]['stages']:
            if event['handoff']:
                self.assertTrue(event['handoff']['verified'])
                self.assertNotIn('token', event['handoff'])

    def test_exhausted_no_action_retry_counts_one_simulator_action_per_slot(self):
        def silent(method, url, cfg, payload=None):
            return {'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 1024}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', silent):
            trial = tb.run_trap_trial('branch', 17, 1, self.cfg(), 'mock', 8192,
                                      Path(tmp), bm, nested_depth=0, max_actions=12)
        self.assertEqual(trial['verified_cases'], 0)
        self.assertEqual(trial['total_actions'], 12)
        self.assertEqual(trial['invalid_actions'], 12)
        self.assertEqual(trial['no_action_investigation_calls'], 24)
        self.assertEqual(trial['no_action_retry_attempts'], 12)
        self.assertEqual(trial['no_action_retry_failures'], 12)

    def test_no_action_recovery_does_not_reduce_action_budget_for_linear_or_branch(self):
        for mode in ('linear', 'branch'):
            fake = RecursiveFakeJan()
            empty = [False]
            def empty_first(method, url, cfg, payload=None):
                if payload and not empty[0] and any('SIMULATED REPAIR INCIDENT' in m.get('content','')
                                                     for m in payload['messages']):
                    empty[0] = True
                    return {'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}],
                            'usage': {'prompt_tokens': 180, 'completion_tokens': 1024}}
                return fake(method, url, cfg, payload)
            with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', empty_first):
                trial = tb.run_trap_trial(mode, 17, 1, self.cfg(), 'mock', 8192,
                                          Path(tmp), bm, nested_depth=0, max_actions=20,
                                          recovery_log_lines=8)
            self.assertTrue(empty[0])
            self.assertEqual(trial['verified_cases'], 1, trial['error'])
            self.assertEqual(trial['no_action_retry_attempts'], 1)
            self.assertEqual(trial['no_action_retry_recoveries'], 1)
            self.assertEqual(trial['invalid_actions'], 0)
            self.assertEqual(trial['total_actions'], 12)

    def test_cli_opt_in_and_archive_rejected(self):
        args=bm.arguments(['--scenario','repair','--tasks','1','--repeats','1',
                          '--modes','linear','summarized','branch','recursive'])
        self.assertEqual(args.modes,['linear','summarized','branch','recursive'])
        self.assertEqual(args.recursive_max_depth,3)
        with self.assertRaises(SystemExit):
            bm.arguments(['--modes','recursive','--scenario','archive'])


if __name__=='__main__':
    unittest.main()