"""Replay recorded simulator actions without loading a model or invoking inference.

This checks recorded environment outcomes, not model reasoning or context savings.
"""
import argparse
import json
from pathlib import Path
from trap_benchmark import RepairSimulator, create_trap_puzzle


def replay(document):
    metadata = document['metadata']
    reports = []
    for trial in document['trials']:
        surprise = metadata.get('surprise_workload', 'legacy')
        enabled = surprise == 'shared' or (surprise == 'legacy' and trial['mode'] == 'adaptive')
        if metadata.get('adaptive_surprise_levels', 1) != 1:
            raise ValueError('Replay currently supports one surprise level only')
        puzzle = create_trap_puzzle(trial['seed'], trial['tasks'],
            **{key: metadata[key] for key in ('trap_log_lines', 'trap_density',
               'recovery_log_lines', 'nested_depth', 'nested_log_lines',
               'pressure_records', 'pressure_detail_chars')}, adaptive_surprise=enabled)
        for record, case in zip(trial['case_breakdown'], puzzle['cases']):
            sim = RepairSimulator(case)
            mismatches = []
            for index, event in enumerate(record['events']):
                if event['label'] in ('archive_read', 'archive_read_rejected',
                                      'completed_stage_receipt', 'stage_scope_rejected'):
                    raise ValueError('Controller-only event requires a controller replay fixture: '
                                     + event['label'])
                sim.step(event['action'])
                actual = sim.history[-1]['label']
                if actual != event['label']:
                    mismatches.append({'action_index': index, 'expected': event['label'], 'actual': actual})
            reports.append({'mode': trial['mode'], 'seed': trial['seed'], 'case': record['batch'],
                            'actions': sim.actions, 'verified': sim.verified,
                            'recorded_verified': record['verified'], 'mismatches': mismatches,
                            'matches': not mismatches and sim.verified == record['verified']})
    if not reports:
        raise ValueError('No recorded case actions to replay')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    reports = replay(json.loads(args.results.read_text()))
    print(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(row['matches'] for row in reports) else 1)
