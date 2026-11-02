"""Durable child handoffs, independent of model providers and task domains.

The caller generates the receipt and owns inference/tool budgets. Archiving is
not verification. Only an application-supplied verifier may certify an outcome.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import uuid


class ReceiptArchive:
    def __init__(self, directory, max_receipt_chars=2400):
        self.directory = Path(directory)
        self.max_receipt_chars = max_receipt_chars

    def handoff(self, parent, child, receipt, *, fits, verifier=None):
        """Persist evidence before releasing child history; retain it on failure.

        fits(messages) must account for the parent's context and output reserve.
        verifier(receipt, evidence) is trusted application code, never model input.
        Parent mutation during the synchronous handoff is not supported.
        """
        if parent is child or not child:
            raise ValueError('A nonempty, separate child history is required')
        required = {'outcome', 'findings', 'sources', 'changes', 'unresolved'}
        if set(receipt) != required or any(not isinstance(receipt[k], str) for k in required):
            raise ValueError('Receipt requires outcome, findings, sources, changes, unresolved text')
        if not receipt['outcome'].strip():
            raise ValueError('Receipt outcome must not be empty')
        evidence = copy.deepcopy(child)
        report = copy.deepcopy(receipt)
        if len(json.dumps(report, ensure_ascii=False)) > self.max_receipt_chars:
            raise ValueError('Receipt exceeds size limit; child history retained')
        verified = False
        if verifier is not None:
            if verifier(copy.deepcopy(report), copy.deepcopy(evidence)) is not True:
                raise ValueError('Outcome verification failed; child history retained')
            verified = True
        identifier = uuid.uuid4().hex
        record = {'version': 1, 'id': identifier, 'receipt': report,
                  'verification': 'application_verified' if verified else 'model_reported',
                  'messages': evidence}
        note = {'role': 'user', 'content': 'CHILD RECEIPT:\n' + json.dumps({
            'archive': identifier, 'message_count': len(evidence),
            'verification': record['verification'], **report}, ensure_ascii=False)}
        candidate = copy.deepcopy(parent) + [note]
        if not fits(copy.deepcopy(candidate)):
            raise ValueError('Receipt does not fit parent budget; child history retained')
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=self.directory, delete=False) as stream:
                temporary = stream.name
                json.dump(record, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / (identifier + '.json'))
            temporary = None
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                os.unlink(temporary)
        parent[:] = candidate
        child.clear()
        return identifier

    def retrieve(self, identifier, message_index, start=0, limit=1200):
        """Read one bounded exact message fragment, including after process restart."""
        if not isinstance(identifier, str) or len(identifier) != 32 or any(
                c not in '0123456789abcdef' for c in identifier):
            raise ValueError('Invalid archive identifier')
        if (type(message_index) is not int or message_index < 0 or type(start) is not int
                or start < 0 or type(limit) is not int or not 1 <= limit <= 3000):
            raise ValueError('Invalid evidence range')
        record = json.loads((self.directory / (identifier + '.json')).read_text(encoding='utf-8'))
        message = record['messages'][message_index]
        text = message.get('content', '')
        if not isinstance(text, str):
            text = json.dumps(message, ensure_ascii=False)
        end = min(start + limit, len(text))
        if start > len(text):
            raise ValueError('Range starts beyond the evidence')
        return {'archive': identifier, 'message_index': message_index,
                'role': message.get('role'), 'start': start, 'end': end,
                'total_chars': len(text), 'text': text[start:end],
                'next_start': end if end < len(text) else None,
                'status': 'historical_observation_not_current_verification'}

    def recall(self, parent, identifier, message_index, *, fits, start=0, limit=1200):
        fragment = self.retrieve(identifier, message_index, start, limit)
        candidate = copy.deepcopy(parent) + [{'role': 'user', 'content':
            'ARCHIVED EVIDENCE:\n' + json.dumps(fragment, ensure_ascii=False)}]
        if not fits(copy.deepcopy(candidate)):
            raise ValueError('Evidence range does not fit parent budget')
        parent[:] = candidate
        return fragment
