import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import benchmark as bm
import trap_benchmark as tb
from test_recursive_benchmark import RecursiveFakeJan
from context_memory import context_budget
from benchmark import arguments


class AccountingTests(unittest.TestCase):
    def test_child_calibration_cannot_block_parent(self):
        calls = [{'budget_scope': 'child', 'prompt_tokens': 3000, 'prompt_est_tokens': 1000},
                 {'budget_scope': 'parent', 'prompt_tokens': 1100, 'prompt_est_tokens': 1000}]
        parent = context_budget([], lambda _: 2000, calls, 'parent', 1024, 4096)
        child = context_budget([], lambda _: 2000, calls, 'child', 1024, 4096)
        self.assertTrue(parent['fits'])
        self.assertFalse(child['fits'])
        self.assertEqual(parent['required_tokens'], 3224)

    def test_reserve_boundary_and_unseen_scope(self):
        check = context_budget([], lambda _: 3072, [], 'new', 1024, 4096)
        self.assertTrue(check['fits'])
        self.assertEqual(check['calibration_samples'], 0)
        self.assertFalse(context_budget([], lambda _: 3073, [], 'new', 1024, 4096)['fits'])

    def test_cli_comparison_defaults_to_shared_workload(self):
        self.assertEqual(arguments(['--scenario', 'repair']).surprise_workload, 'shared')

    def test_shared_modes_generate_identical_cases(self):
        generated = []
        original = tb.create_trap_puzzle
        def capture(*args, **kwargs):
            puzzle = original(*args, **kwargs)
            generated.append(puzzle)
            return puzzle
        cfg = dict(bm.bc.DEFAULTS, _surprise_workload='shared')
        with tempfile.TemporaryDirectory() as temp, patch.object(tb, 'create_trap_puzzle', capture), \
                patch.object(bm.bc, 'http_json', RecursiveFakeJan()):
            for mode in ('linear', 'summarized', 'adaptive'):
                tb.run_trap_trial(mode, 17, 1, cfg, 'fake', 8192, Path(temp) / mode,
                                  bm, nested_depth=2, max_actions=2)
        self.assertEqual(generated[0]['cases'], generated[1]['cases'])
        self.assertEqual(generated[1]['cases'], generated[2]['cases'])
