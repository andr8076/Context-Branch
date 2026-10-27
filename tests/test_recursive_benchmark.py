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

    def test_stage_start_only_initial_inspections_are_lightweight(self):
        cfg = self.cfg()
        cfg.update({'_stage_start_inspects': 2, '_stage_start_output_tokens': 192,
                    '_stage_start_reasoning_effort': 'none'})
        fake = RecursiveFakeJan()
        observed = []
        def spy(method, url, conf, payload=None):
            if payload and 'RECURSIVE STAGE=' in payload['messages'][0]['content']:
                stage = re.search(r'RECURSIVE STAGE=(\w+)', payload['messages'][0]['content']).group(1)
                observed.append((stage, payload['max_tokens'], payload.get('reasoning_effort')))
            return fake(method, url, conf, payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', spy):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        self.assertEqual(trial['stage_start_calls'], 8)
        self.assertEqual(trial['stage_start_empty_visible_calls'], 0)
        self.assertGreater(trial['stage_start_total_tokens'], 0)
        self.assertGreater(trial['stage_start_inference_seconds'], -1)
        for stage in ('index', 'escrow', 'recovery', 'repair'):
            calls = [pair[1:] for pair in observed if pair[0] == stage]
            self.assertGreater(len(calls), 2, stage)
            self.assertEqual(calls[:2], [(192, 'none'), (192, 'none')], stage)
            self.assertTrue(all(c == (1024, None) for c in calls[2:]), stage)
        for ev in trial['recursive_events'][0]['stages']:
            self.assertEqual(ev['stage_start_inspections'], 2)
            self.assertFalse(ev['stage_start_disabled'])

    def test_stage_start_opt_out_is_identical_to_normal_reasoning(self):
        cfg = self.cfg()
        cfg.update({'_stage_start_inspects': 0, '_stage_start_output_tokens': 192,
                    '_stage_start_reasoning_effort': 'none'})
        fake = RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', fake):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        self.assertEqual(trial['stage_start_calls'], 0)
        self.assertTrue(all(v == 1024 for v in fake.limits
                            if isinstance(v, int) and v != 2048))

    def test_noninspect_action_disables_stage_start_before_complex_work(self):
        cfg = self.cfg()
        cfg.update({'_stage_start_inspects': 2, '_stage_start_output_tokens': 192,
                    '_stage_start_reasoning_effort': 'none'})
        fake = RecursiveFakeJan()
        injected = [False]
        def early_action(method, url, conf, payload=None):
            if payload and not injected[0] and 'RECURSIVE STAGE=index' in payload['messages'][0]['content']:
                previously_read_guide = any('{"action": "inspect", "path": "guide/index"}' in
                                            m.get('content', '') for m in payload['messages'])
                if previously_read_guide:
                    injected[0] = True
                    content = json.dumps({'action':'rebuild_index', 'token':'WRONG'})
                    return {'choices':[{'message':{'content':content}, 'finish_reason':'stop'}],
                            'usage':{'prompt_tokens':40,'completion_tokens':15}}
            return fake(method, url, conf, payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', early_action):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertTrue(injected[0])
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        index = next(e for e in trial['recursive_events'][0]['stages'] if e['stage'] == 'index')
        self.assertTrue(index['stage_start_disabled'])
        self.assertEqual(index['stage_start_inspections'], 1)
        self.assertEqual(trial['stage_start_calls'], 8)

    def test_stage_start_empty_reply_uses_bounded_recovery(self):
        cfg = self.cfg()
        cfg.update({'_stage_start_inspects': 2, '_stage_start_output_tokens': 192,
                    '_stage_start_reasoning_effort': 'none',
                    '_no_action_retry_output_tokens': 256,
                    '_no_action_retry_reasoning_effort': 'none'})
        fake = RecursiveFakeJan()
        injected = [False]
        def empty_first(method, url, conf, payload=None):
            if (payload and not injected[0] and
                    'RECURSIVE STAGE=index' in payload['messages'][0]['content']):
                injected[0] = True
                return {'choices':[{'message':{'content':''}, 'finish_reason':'length'}],
                        'usage':{'prompt_tokens':50,'completion_tokens':192}}
            return fake(method, url, conf, payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', empty_first):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertTrue(injected[0])
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        self.assertEqual(trial['no_action_retry_recoveries'], 1)
        self.assertEqual(trial['no_action_retry_attempts'], 1)
        self.assertEqual(trial['stage_start_empty_visible_calls'], 1)
        self.assertEqual(trial['stage_start_calls'], 8)
        self.assertEqual(trial['invalid_actions'], 0)

    def test_context_checkpoint_retains_real_observations_and_completes(self):
        cfg = self.cfg()
        cfg.update({'_recursive_checkpoint_threshold': 0.50,
                    '_recursive_checkpoint_max': 8,
                    '_stage_start_inspects': 2,
                    '_stage_start_output_tokens': 256})
        fake = RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', fake):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        self.assertTrue(trial['exact_final'], trial['error'])
        self.assertGreater(trial['recursive_checkpoints'], 0, trial['recursive_events'])
        self.assertEqual(trial['recursive_checkpoint_blocked'], 0)
        cases = trial['recursive_events'][0]['stages']
        self.assertEqual(sum(e['context_checkpoints'] for e in cases), trial['recursive_checkpoints'])
        self.assertTrue(any(e['checkpoint_details'] for e in cases))
        self.assertTrue(all('after_est_tokens' in x for e in cases for x in e['checkpoint_details']))
        # The parent must never receive child shard transcripts after checkpoints.
        parent_messages = [msgs for msgs in fake.messages_seen
                           if 'RECURSIVE STAGE=repair' in msgs[0]['content']]
        self.assertTrue(parent_messages)
        joined = json.dumps(parent_messages)
        self.assertNotIn('RECOVERY SHARD A:', joined)
        self.assertNotIn('INDEX SHARD A:', joined)

    def test_failed_recursive_inference_keeps_partial_stage_history(self):
        cfg = self.cfg()
        fake = RecursiveFakeJan()
        interrupted = [False]
        def fail_in_repair(method, url, conf, payload=None):
            if payload and 'RECURSIVE STAGE=repair' in payload['messages'][0]['content']:
                interrupted[0] = True
                raise RuntimeError('request exceeds context capacity')
            return fake(method, url, conf, payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', fail_in_repair):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertTrue(interrupted[0])
        self.assertEqual(trial['error_type'], 'context_overflow_confirmed')
        self.assertTrue(trial['partial_failure'])
        self.assertEqual(trial['interrupted_batch'], 'batch_1')
        self.assertGreater(trial['total_actions'], 0)
        self.assertEqual(trial['total_actions'], trial['case_breakdown'][0]['actions'])
        self.assertEqual(len(trial['case_breakdown'][0]['events']), trial['total_actions'])
        self.assertEqual(trial['recursive_stage_successes'], 3)
        self.assertEqual(trial['recursive_context_count'], 4)
        self.assertEqual(trial['recursive_stage_failures'], 1)
        self.assertEqual(trial['nested_index_completed'], 1)
        self.assertEqual(trial['nested_escrow_completed'], 1)
        self.assertEqual(trial['recovery_completed'], 1)
        self.assertEqual(trial['verified_cases'], 0)
        self.assertFalse(trial['exact_final'])
        self.assertEqual(trial['final_status'], 'not_attempted')

    def test_failed_ordinary_branch_preserves_nonzero_actions(self):
        fake = RecursiveFakeJan()
        calls = [0]
        def fail_after_inspection(method, url, conf, payload=None):
            if payload and any('SIMULATED REPAIR INCIDENT' in m.get('content','')
                               for m in payload['messages']):
                calls[0] += 1
                if calls[0] == 4:
                    raise RuntimeError('request exceeds context capacity')
            return fake(method, url, conf, payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', fail_after_inspection):
            trial = tb.run_trap_trial('branch', 17, 1, self.cfg(), 'mock', 8192,
                                      Path(tmp), bm, nested_depth=0,
                                      recovery_log_lines=8)
        self.assertEqual(trial['error_type'], 'context_overflow_confirmed')
        self.assertTrue(trial['partial_failure'])
        self.assertEqual(trial['total_actions'], 3, trial['case_breakdown'])
        self.assertEqual(len(trial['case_breakdown'][0]['events']), 3)
        self.assertEqual(trial['recursive_checkpoints'], 0)

    def test_checkpoint_opt_out_has_no_context_rebuilds(self):
        cfg = self.cfg()
        cfg['_recursive_checkpoint_threshold'] = 0
        cfg['_recursive_checkpoint_max'] = 8
        fake = RecursiveFakeJan()
        with tempfile.TemporaryDirectory() as tmp, patch.object(bm.bc, 'http_json', fake):
            trial = tb.run_trap_trial('recursive', 17, 1, cfg, 'mock', 8192,
                                      Path(tmp), bm, nested_depth=2,
                                      recovery_log_lines=8, nested_log_lines=8)
        self.assertEqual(trial['verified_cases'], 1, trial['error'])
        self.assertEqual(trial['recursive_checkpoints'], 0)

    def test_checkpoint_threshold_argument_validation(self):
        args = bm.arguments(['--scenario','repair','--recursive-checkpoint-threshold','0.76',
                             '--recursive-checkpoint-max','5'])
        self.assertEqual(args.recursive_checkpoint_threshold, 0.76)
        self.assertEqual(args.recursive_checkpoint_max, 5)
        self.assertEqual(bm.arguments(['--recursive-checkpoint-threshold','0']).recursive_checkpoint_threshold, 0)
        for bad_args in (['--recursive-checkpoint-threshold','0.25'],
                         ['--recursive-checkpoint-threshold','0.98'],
                         ['--recursive-checkpoint-max','99']):
            with self.assertRaises(SystemExit):
                bm.arguments(bad_args)

    def test_cli_opt_in_and_archive_rejected(self):
        args=bm.arguments(['--scenario','repair','--tasks','1','--repeats','1',
                          '--modes','linear','summarized','branch','recursive'])
        self.assertEqual(args.modes,['linear','summarized','branch','recursive'])
        self.assertEqual(args.recursive_max_depth,3)
        with self.assertRaises(SystemExit):
            bm.arguments(['--modes','recursive','--scenario','archive'])


if __name__=='__main__':
    unittest.main()