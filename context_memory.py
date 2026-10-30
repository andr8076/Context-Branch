"""Bounded working memory with exact, retrievable historical snapshots.

Summaries are model reports, never verified facts. Archives contain only observed
messages and are not automatically replayed into the active conversation.
"""
import copy
import json
import math

class MemoryCompactionError(ValueError):
    error_kind = 'summary_failed'


MEMORY_REQUEST = (
    "Compress the observed conversation into working memory, at most {max_chars} characters. "
    "Preserve the current goal, exact identifiers, constraints, verified outcomes and their "
    "evidence references, unresolved dependencies, failed attempts worth avoiding, and next action. "
    "For unfinished investigations preserve relevant evidence and source paths, not just completed tasks. "
    "Distinguish observed facts from model guesses. Never invent success or missing evidence. "
    "Return only the memory text, no analysis."
)


def predicted_tokens(messages, estimate, calls=()):
    """Use a conservative observed calibration, consistently across comparisons."""
    ratio = max([1.0] + [r['prompt_tokens'] / max(1, r['prompt_est_tokens'])
                for r in calls if isinstance(r.get('prompt_tokens'), int)
                and r.get('prompt_est_tokens')])
    return math.ceil(estimate(messages) * ratio)


def archive_snapshot(archives, messages):
    archives.append(copy.deepcopy(messages))
    return f"archive/{len(archives)}"


def archive_note(archives):
    return (f"Historical observations are archived as archive/1 through archive/{len(archives)}. "
            "Retrieve a numbered archive in small ranges if memory omits a necessary detail. "
            "Archived observations may be stale; memory is model-reported, not independent verification.")


def read_archive(archives, path, start=0, limit=2400):
    if type(start) is not int or start < 0 or type(limit) is not int or not 1 <= limit <= 3000:
        raise ValueError('Invalid archive range')
    pieces = str(path).split('/')
    if len(pieces) != 2 or pieces[0] != 'archive' or not pieces[1].isdigit():
        raise ValueError('Invalid archive reference')
    index = int(pieces[1]) - 1
    if not 0 <= index < len(archives):
        raise ValueError('Unknown archive')
    body = json.dumps(archives[index], ensure_ascii=False, indent=2)
    end = min(len(body), start + limit)
    return f"{path} chars {start}:{end} of {len(body)}\n{body[start:end]}\nnext_start={end if end < len(body) else 'end'}"


def benchmark_memory(cfg, model, messages, preserved, trace, bm, archives,
                     context_limit, incoming_chars=0, before_request=None):
    """Pressure-aware model compaction, before an observation enters the prompt."""
    threshold = cfg.get('_context_memory_threshold', 0)
    if not threshold or len(messages) <= len(preserved) + 1:
        return None
    if messages[-1].get("content", "").startswith("Historical observations are archived"):
        return None
    estimate = predicted_tokens(messages, bm.approximate_tokens, trace.calls)
    reserve = max(cfg['max_output_tokens'], cfg.get('_summary_output_tokens', 2048))
    incoming = predicted_tokens([{'role': 'user', 'content': 'x' * incoming_chars}],
                                bm.approximate_tokens, trace.calls) if incoming_chars else 0
    if estimate + reserve + incoming + 256 < context_limit * threshold:
        return None
    def guard(prompt):
        if predicted_tokens(prompt, bm.approximate_tokens, trace.calls) + reserve + 256 > context_limit:
            raise MemoryCompactionError('Working-memory summary cannot fit; original evidence retained')
        if before_request:
            before_request(prompt)
    try:
        shorter, summary = bm.compact_history(cfg, model, messages, preserved[:2], trace,
            cfg.get('_memory_max_chars', 1800), summary_request=MEMORY_REQUEST, before_request=guard)
    except ValueError as exc:
        raise MemoryCompactionError(str(exc)) from exc
    # Additional immutable incident instructions survive every mid-case compaction.
    shorter[2:2] = [dict(m) for m in preserved[2:]]
    shorter.append({'role': 'user', 'content': archive_note(archives + [None]) +
        ' Use inspect with path=archive/N and start=<character offset> to retrieve 2400 characters.'})
    if predicted_tokens(shorter, bm.approximate_tokens, trace.calls) >= estimate:
        # No destructive replacement and no automatic retry loop.
        raise MemoryCompactionError('Working-memory summary did not reduce context; original evidence retained')
    reference = archive_snapshot(archives, messages)
    messages[:] = shorter
    return {'archive': reference, 'chars': len(summary), 'text': summary,
            'before_est_tokens': estimate,
            'after_est_tokens': predicted_tokens(shorter, bm.approximate_tokens, trace.calls)}


def archive_observation(sim, archives, action):
    """A retrieval consumes one ordinary action and appears in the audit trail."""
    sim.actions += 1
    label = 'archive_read'
    try:
        observation = read_archive(archives, action['path'], action.get('start', 0))
    except ValueError as exc:
        observation = str(exc)
        label = 'archive_read_rejected'
        sim.invalid_actions += 1
    sim.history.append({'action': dict(action), 'label': label,
                        'observation_chars': len(observation)})
    return observation
