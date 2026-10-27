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
        self.assertEqual(trial['recursive_branch_count'],4)
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

    def test_cli_opt_in_and_archive_rejected(self):
        args=bm.arguments(['--scenario','repair','--tasks','1','--repeats','1',
                          '--modes','linear','summarized','branch','recursive'])
        self.assertEqual(args.modes,['linear','summarized','branch','recursive'])
        self.assertEqual(args.recursive_max_depth,3)
        with self.assertRaises(SystemExit):
            bm.arguments(['--modes','recursive','--scenario','archive'])


if __name__=='__main__':
    unittest.main()