"""Controlled recursive delegation for the virtual repair benchmark.

This is deliberately NOT autonomous branch selection. A known dependency chain
is delegated by the harness to test whether nested context isolation helps.
Only the one shared RepairSimulator is allowed to change environment state.
Neither oracle secrets nor child histories are sent to ancestor conversations.
"""
from __future__ import annotations

from dataclasses import dataclass


STAGE_GOALS = {
    'index': ('rebuild_index', 'Rebuild the CURRENT index from both index shards.'),
    'escrow': ('unlock_escrow', 'Unlock escrow from both CURRENT escrow shards.'),
    'recovery': ('recover', 'Recover the locked approvals ledger from both CURRENT recovery shards.'),
    'repair': ('verify', 'Find the valid release, route and certificate; apply the repair and verify.'),
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


def solve_recursive_case(sim, case, cfg, model, trace, bm, max_actions, max_depth=3):
    """Return (success, events) while every model action counts against ONE cap.

    Each child has a separate API message list. A parent's messages contain
    only a short statement that the child *actually achieved its simulator goal*.
    The simulator is shared, never cloned or reset between nested stages.
    """
    from trap_benchmark import ACTION_SYSTEM, parse_action

    events = []

    def solve(stage, depth, parent_stage):
        event = {'stage': stage, 'depth': depth, 'parent': parent_stage,
                 'completed': False, 'model_actions': 0, 'status': 'blocked'}
        events.append(event)
        if depth > max_depth:
            event['status'] = 'depth_limit'
            return False
        child = prerequisite(stage, case['nested_depth'])
        if child and not solve(child, depth + 1, stage):
            event['status'] = 'child_blocked'
            return False
        # Never give hidden solution tokens, expected repairs or child transcripts.
        instruction = (
            f'RECURSIVE STAGE={stage} DEPTH={depth}. ' + ACTION_SYSTEM + ' ' +
            STAGE_GOALS[stage][1] + ' Work ONLY on this stage; do not repeat '
            'completed child stages. First inspect guide/index. Every response '
            'must contain exactly one simulator JSON action.'
        )
        context = [{'role': 'system', 'content': instruction},
                   {'role': 'user', 'content': case['intro']}]
        if child:
            # Known successful outcome obtained from the actual shared simulator.
            context.append({'role': 'user', 'content':
                            f'CHILD RESULT: {child} stage completed and verified by the simulator. '
                            'The shared simulator state is already updated. You have not '
                            'received the child history. Continue with your own stage.'})
        while sim.actions < max_actions and not stage_done(stage, sim):
            trace.phase = f'branch_recursive_{depth}_{stage}'
            answer = bm.call_parent(cfg, model, context)
            context.append({'role': 'assistant', 'content': answer})
            action = parse_action(answer)
            observation = sim.step(action)
            event['model_actions'] += 1
            context.append({'role': 'user', 'content':
                            'SIMULATOR RESULT:\n' + observation +
                            '\nContinue this stage. Output one JSON action only.'})
        event['completed'] = stage_done(stage, sim)
        event['status'] = 'verified' if event['completed'] else 'action_budget_exhausted'
        return event['completed']

    success = solve('repair', 0, None)
    return success, events