"""Controlled nested repair branches with stage permissions and verified handoffs.

Not autonomous: the benchmark selects dependencies. The same RepairSimulator
persists across private branch conversations; no oracle or child logs pass up.
"""
from __future__ import annotations
import re
import json

STAGE_GOALS = {
    'index': ('rebuild_index', 'Rebuild CURRENT index from the two index shards.'),
    'escrow': ('unlock_escrow', 'Unlock escrow from CURRENT escrow shards.'),
    'recovery': ('recover', 'Recover approvals using CURRENT recovery shards.'),
    'repair': ('verify', 'Select the authorized route, repair it and verify.'),
}
AUTHORITY_PATHS = ('guide/policy', 'registry/releases', 'registry/approvals',
                   'network/aliases', 'network/routes', 'security/certificates')
STAGE_PATHS = {
    'index': ('guide/index', 'index/shard-a', 'index/shard-b'),
    'escrow': ('guide/index', 'escrow/shard-a', 'escrow/shard-b'),
    'recovery': ('guide/index', 'recovery/shard-a', 'recovery/shard-b'),
    'repair': ('guide/index', *AUTHORITY_PATHS),
}
STAGE_ACTIONS = {
    'index': ('inspect', 'rebuild_index'),
    'escrow': ('inspect', 'unlock_escrow'),
    'recovery': ('inspect', 'recover'),
    'repair': ('inspect', 'repair', 'verify'),
}


def stage_done(stage, sim):
    if stage in ('index', 'escrow'):
        return stage in sim.nested_completed
    return sim.recovery_completed if stage == 'recovery' else sim.verified


def prerequisite(stage, nested_depth):
    if stage == 'repair':
        return 'recovery'
    if stage == 'recovery' and nested_depth >= 1:
        return 'escrow'
    if stage == 'escrow' and nested_depth >= 2:
        return 'index'
    return None


def verified_handoff(child, sim):
    """Only public state transitions, never solved fragments or child history."""
    if not stage_done(child, sim):
        raise ValueError('Cannot hand off an unverified stage')
    return {
        'index': 'index rebuilt; escrow shards unlocked',
        'escrow': 'escrow unlocked; recovery shards unlocked',
        'recovery': 'approval ledger recovered; registry/approvals unlocked',
    }[child]


def stage_guide(stage, case):
    """Filter only the matching *visible* guide/index section, not the oracle."""
    lines = case['docs']['guide/index'].splitlines()
    if stage == 'index':
        relevant = [line for line in lines if line.startswith('STEP ') and 'index/shard-a' in line]
    elif stage == 'escrow':
        relevant = [line for line in lines if line.startswith('STEP ') and 'escrow/shard-a' in line]
    elif stage == 'recovery':
        relevant = [line for line in lines if line.startswith('MANDATORY RECOVERY:')]
    else:
        relevant = ['Authorization already VERIFIED; inspect policy and five authoritative documents.']
    if not relevant:
        raise ValueError(f'Missing published guide for stage={stage}')
    return ('Evidence cabinet: stage-scoped excerpt from guide/index.\n'
            'Available paths: ' + ', '.join(STAGE_PATHS[stage]) + '\n' + '\n'.join(relevant))


def stage_policy(case):
    """Filter completed prerequisite directions, retain actual repair rules."""
    return '\n'.join(line for line in case['docs']['guide/policy'].splitlines()
                     if re.match(r'^[1-7]\. ', line))


def stage_instruction(stage, depth):
    commands = {
        'index': '{"action":"rebuild_index","token":"<joined CURRENT fragments>"}',
        'escrow': '{"action":"unlock_escrow","token":"<joined CURRENT fragments>"}',
        'recovery': '{"action":"recover","token":"<joined CURRENT fragments>"}',
        'repair': '{"action":"repair","release":"REL-...","route":"ROUTE-...","certificate":"CERT-..."} and then {"action":"verify"}',
    }
    return (f'RECURSIVE STAGE={stage} DEPTH={depth}. '
            f'YOUR ONLY OBJECTIVE: {STAGE_GOALS[stage][1]} '
            'All prerequisite children have already updated the shared simulator. '
            'NEVER repeat child actions or inspect child documents. '
            f'ALLOWED DOCUMENT PATHS ONLY: {", ".join(STAGE_PATHS[stage])}. '
            f'ALLOWED ACTIONS ONLY: {", ".join(STAGE_ACTIONS[stage])}. '
            'First inspect guide/index; then inspect ONLY documents for your stage. '
            f'Complete your stage with {commands[stage]}. '
            'Every response MUST be exactly one JSON simulator action, no analysis or markdown. '
            'Other-stage actions are rejected and cost an action. '
            'Evidence may be misleading: use matching CURRENT run and listed join order. '
            'Only the simulator can verify success.')


def checkpoint_context(base_context, inspected, last_change):
    """Rebuild from exact previously observed simulator output, de-duplicated.

    Preserve the normal SIMULATOR RESULT framing so the agent can reliably
    identify separate original evidence documents. Re-inspections are not
    replayed; each path appears once in the new conversation.
    """
    context = [dict(m) for m in base_context]
    context.append({'role': 'user', 'content':
                    'CONTEXT CHECKPOINT: previous simulator reads below were '
                    'completed in this same stage. Do not repeat them.'})
    for path, observation in inspected.items():
        context.extend((
            {'role': 'assistant', 'content': json.dumps({'action': 'inspect', 'path': path})},
            {'role': 'user', 'content': 'SIMULATOR RESULT:\n' + observation},
        ))
    if last_change is not None:
        action, observation = last_change
        context.extend((
            {'role': 'assistant', 'content': json.dumps(action)},
            {'role': 'user', 'content': 'SIMULATOR RESULT:\n' + observation},
        ))
    context.append({'role': 'user', 'content':
                    'Continue using the above evidence; it was actually observed '
                    'and the shared environment is unchanged. One permitted JSON action.'})
    return context


def estimated_prompt_tokens(bm, messages, last_prompt_tokens=None, last_prompt_estimate=None):
    """Conservative prompt-size prediction calibrated by last API token count."""
    estimate = bm.approximate_tokens(messages)
    if last_prompt_tokens is not None and last_prompt_estimate is not None:
        estimate = max(estimate, last_prompt_tokens +
                       max(0, estimate - last_prompt_estimate))
    return estimate


def solve_recursive_case(sim, case, cfg, model, trace, bm, max_actions, max_depth=3, context_limit=None):
    """Return (success, events, counters); shared action cap across all stages."""
    from trap_benchmark import request_investigation_action
    events = []
    counters = {'no_actions': 0, 'retries': 0, 'recoveries': 0, 'retry_failures': 0}
    # Mutable references are retained by Trace if inference fails halfway through a stage.
    trace.active_recursive_events = events
    trace.active_recursive_counters = counters
    checkpoint_threshold = cfg.get('_recursive_checkpoint_threshold', 0)
    checkpoint_max = cfg.get('_recursive_checkpoint_max', 4)
    context_limit = context_limit or cfg.get('_context_limit', 8192)

    def solve(stage, depth, parent):
        event = {'stage': stage, 'depth': depth, 'parent': parent,
                 'completed': False, 'model_actions': 0, 'status': 'blocked',
                 'rejected_out_of_scope': 0, 'handoff': None,
                 'stage_start_inspections': 0, 'stage_start_disabled': False,
                 'context_checkpoints': 0, 'checkpoint_details': [],
                 'context_checkpoint_blocked': False}
        events.append(event)
        if depth > max_depth:
            event['status'] = 'depth_limit'
            return False
        child = prerequisite(stage, case['nested_depth'])
        if child and not solve(child, depth + 1, stage):
            event['status'] = 'child_blocked'
            return False
        context = [{'role': 'system', 'content': stage_instruction(stage, depth)},
                   {'role': 'user', 'content': f'INCIDENT: {case["name"]}. Solve ONLY stage={stage}.'}]
        if child:
            receipt = verified_handoff(child, sim)
            event['handoff'] = {'stage': child, 'verified': True, 'state_change': receipt}
            context.append({'role': 'user', 'content':
                            f'CHILD RESULT: {child} stage completed and verified by the simulator. '
                            f'SHARED ENVIRONMENT STATE: {receipt}. '
                            f'Now complete ONLY {stage}. Do not repeat the child investigation. '
                            f'You may inspect ONLY {", ".join(STAGE_PATHS[stage])}.'})
        original_context = [dict(m) for m in context]
        inspected = {}  # path -> exact last simulator response, not case['docs'] or oracle
        last_change = None
        last_prompt_tokens = None
        last_prompt_estimate = None
        # Stage opening only requires the guide and first evidence lookup.
        # Ask the MODEL for these actions with a lightweight inference request.
        # All later evidence interpretation and repairs keep full reasoning.
        stage_start_limit = cfg.get('_stage_start_inspects', 0)
        while sim.actions < max_actions and not stage_done(stage, sim):
            starting = (event['stage_start_inspections'] < stage_start_limit
                        and not event['stage_start_disabled'])
            phase = (f'branch_recursive_{depth}_{stage}_stage_start' if starting
                     else f'branch_recursive_{depth}_{stage}')
            budget = (cfg.get('_stage_start_output_tokens', cfg['max_output_tokens'])
                      if starting else cfg['max_output_tokens'])
            predicted = estimated_prompt_tokens(
                bm, context, last_prompt_tokens, last_prompt_estimate)
            before_tokens = predicted
            if (checkpoint_threshold and event['context_checkpoints'] < checkpoint_max
                    and predicted + budget + 256 >= int(context_limit * checkpoint_threshold)
                    and (inspected or last_change is not None)):
                shorter = checkpoint_context(original_context, inspected, last_change)
                after_tokens = bm.approximate_tokens(shorter)
                # Do not loop on a checkpoint that cannot actually reduce history.
                if after_tokens + budget + 256 >= context_limit:
                    event['context_checkpoint_blocked'] = True
                    raise RuntimeError('Context checkpoint cannot fit unmodified observed evidence')
                if after_tokens + 64 < predicted:
                    context = shorter
                    event['context_checkpoints'] += 1
                    event['checkpoint_details'].append(
                        {'before_est_tokens': before_tokens,
                         'after_est_tokens': after_tokens,
                         'documents_preserved': list(inspected),
                         'retained_change': last_change is not None})
                    last_prompt_tokens = None
                    last_prompt_estimate = None
                    # Following a checkpoint, the branch still has the same environment.
            request_estimate = bm.approximate_tokens(context)
            try:
                action, answer, info = request_investigation_action(
                    cfg, model, context, trace, bm, phase=phase,
                    max_retries=cfg.get('_no_action_retries', 1))
            except Exception:
                event['status'] = 'inference_error'
                raise
            call = trace.calls[-1] if trace.calls else {}
            last_prompt_tokens = call.get('prompt_tokens')
            last_prompt_estimate = request_estimate
            counters['no_actions'] += info['no_actions']
            counters['retries'] += info['retries']
            counters['recoveries'] += info['recovered']
            counters['retry_failures'] += info['failed']
            context.append({'role': 'assistant', 'content': answer})
            in_scope_inspect = (action is not None and action['action'] == 'inspect'
                                and action.get('path') in STAGE_PATHS[stage])
            if starting:
                if in_scope_inspect:
                    event['stage_start_inspections'] += 1
                else:
                    # Do not apply the cheap policy to complex or invalid actions.
                    # The action still gets handled normally by the simulator.
                    event['stage_start_disabled'] = True
            if action and (action['action'] not in STAGE_ACTIONS[stage] or
                           (action['action'] == 'inspect' and action.get('path') not in STAGE_PATHS[stage])):
                event['rejected_out_of_scope'] += 1
                observation = sim.reject_out_of_scope(
                    stage, action, STAGE_PATHS[stage], STAGE_ACTIONS[stage])
            else:
                observation = sim.step(action)
                if action and action.get('action') == 'inspect':
                    if action['path'] == 'guide/index':
                        observation = stage_guide(stage, case)
                    elif action['path'] == 'guide/policy' and stage == 'repair':
                        observation = stage_policy(case)
            event['model_actions'] += 1
            if in_scope_inspect:
                inspected[action['path']] = observation
            elif action is not None and not (action['action'] == 'inspect'):
                last_change = (action, observation)
            context.append({'role': 'user', 'content':
                            'SIMULATOR RESULT:\n' + observation +
                            f'\nContinue ONLY {stage}. Output one permitted JSON action.'})
        event['completed'] = stage_done(stage, sim)
        event['status'] = 'verified' if event['completed'] else 'action_budget_exhausted'
        return event['completed']

    success = solve('repair', 0, None)
    return success, events, counters
