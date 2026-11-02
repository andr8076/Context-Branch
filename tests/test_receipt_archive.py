import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch
from receipt_archive import ReceiptArchive


class ReceiptLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.parent = [{'role': 'system', 'content': 'Preserve the mission'},
                       {'role': 'user', 'content': 'Investigate and continue'}]
        self.child = [{'role': 'user', 'content': 'Source config.ini: retry_delay=17'},
                      {'role': 'assistant', 'content': 'Connection investigation finished'}]
        self.receipt = dict(outcome='Connection investigation completed',
            findings='Connection is configured', sources='config.ini; archive message 0',
            changes='None', unresolved='Retry timing not evaluated')

    def test_resume_then_recall_omitted_fact_after_restart(self):
        original = copy.deepcopy(self.parent)
        with tempfile.TemporaryDirectory() as folder:
            archive = ReceiptArchive(folder)
            identifier = archive.handoff(self.parent, self.child, self.receipt, fits=lambda m: True)
            self.assertEqual(self.child, [])
            self.assertEqual(self.parent[:2], original)
            self.assertNotIn('retry_delay=17', json.dumps(self.parent))
            self.assertIn('model_reported', self.parent[-1]['content'])
            self.parent.append({'role': 'user', 'content': 'Now calculate two retry delays'})
            restarted = ReceiptArchive(folder)
            detail = restarted.recall(self.parent, identifier, 0, fits=lambda m: True)
            delay = int(detail['text'].split('retry_delay=')[1])
            self.assertEqual(delay * 2, 34)
            self.assertEqual(len(self.parent), 5)
            self.assertNotIn('Connection investigation finished', self.parent[-1]['content'])

    def test_failures_retain_both_histories(self):
        for failure in ('budget', 'verification', 'storage'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                parent, child = copy.deepcopy(self.parent), copy.deepcopy(self.child)
                archive = ReceiptArchive(folder)
                with patch('receipt_archive.os.replace', side_effect=OSError('disk failure')) if failure == 'storage' else nullcontext():
                    with self.assertRaises((ValueError, OSError)):
                        archive.handoff(parent, child, self.receipt,
                            fits=lambda m: failure != 'budget',
                            verifier=(lambda r, e: False) if failure == 'verification' else None)
                self.assertEqual(parent, self.parent)
                self.assertEqual(child, self.child)

    def test_recall_rejects_oversized_context_and_bad_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = ReceiptArchive(folder)
            identifier = archive.handoff(self.parent, self.child, self.receipt, fits=lambda m: True)
            snapshot = copy.deepcopy(self.parent)
            with self.assertRaises(ValueError):
                archive.recall(self.parent, identifier, 0, fits=lambda m: False)
            self.assertEqual(self.parent, snapshot)
            with self.assertRaises(ValueError):
                archive.retrieve('../escape', 0)

    def test_checked_outcome_keeps_uncertainty_in_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = ReceiptArchive(folder)
            archive.handoff(self.parent, self.child, self.receipt, fits=lambda m: True,
                verifier=lambda r, evidence: 'retry_delay=17' in evidence[0]['content'])
            self.assertIn('application_verified', self.parent[-1]['content'])
            self.assertIn('Retry timing not evaluated', self.parent[-1]['content'])
