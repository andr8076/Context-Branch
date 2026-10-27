#!/usr/bin/env python3
"""Reproducible Jan benchmark: archive lookups or interactive gated repair traps.

Three controlled context modes: linear, periodic summaries and isolated branches.
No autonomous branch selection is implied. Offline tests mock Jan, not its reasoning.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import gc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import branch_context as bc
import jan_reset
import llama_runner

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "benchmarks"
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

def resolve_llama_model_path(cli_model=None, project_root=None):
    """Find the GGUF path without storing machine-specific paths in Git.

    Priority: explicit --llama-model > CONTEXT_BRANCH_MODEL environment
    variable > ignored local_model.json MODEL_PATH. Jan does not use this.
    """
    if cli_model is not None:
        return cli_model
    env_model = os.environ.get("CONTEXT_BRANCH_MODEL", "").strip()
    if env_model:
        return env_model
    settings = Path(project_root or ROOT) / "local_model.json"
    if not settings.is_file():
        return None
    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {settings.name}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("MODEL_PATH"), str):
        raise ValueError('local_model.json must contain a string "MODEL_PATH" value')
    return data["MODEL_PATH"].strip() or None



def code(rng, n=8):
    return "".join(rng.choices(ALPHABET, k=n))


def create_puzzle(seed: int, n_tasks: int, n_records: int):
    rng = random.Random(seed)
    mission = "M-" + code(rng, 7)
    control = "C-" + code(rng, 10)
    side_tasks, targets = [], {}
    for task_number in range(1, n_tasks + 1):
        task_name = f"batch_{task_number}"
        target_row = rng.randrange(n_records)
        entries = []
        for row in range(n_records):
            record_id = f"REF-{task_number:02d}-{row:04d}"
            seal = "S" + code(rng, 8)
            # Distractors in every row are deliberately similar-looking.
            entries.append(f"{record_id} | seal={seal} | lot={code(rng,5)} | bay={rng.randint(11,89)} | checked=YES")
            if row == target_row:
                targets[task_name] = {"ref": record_id, "seal": seal, "row": row}
        target = targets[task_name]["ref"]
        side_prompt = (
            f"INDEPENDENT ARCHIVE LOOKUP: {task_name}\n"
            f"Find the row whose reference ID is EXACTLY {target}.\n"
            "Extract that row's SEAL. Do not combine it with other rows.\n"
            'Answer ONLY one JSON object with keys: status="success", '
            'result="the exact seal value", what_changed="Nothing", '
            'next_step="Use this seal in the final report", caveats="".\n'
            "ARCHIVE:\n" + "\n".join(entries)
        )
        if len(side_prompt) > 5100:
            raise ValueError(f"Archive prompt for {task_name} is {len(side_prompt)} chars. Reduce --records (maximum branch task is ~5500 chars).")
        side_tasks.append((task_name, side_prompt, targets[task_name]["seal"]))
    return {
        "mission": mission, "control": control, "tasks": side_tasks,
        "targets": targets,
        "brief": (
            "You are an audit assistant collecting seals for a final report. "
            "You will receive independent archive lookup tasks one at a time. "
            "Preserve these ORIGINAL INSTRUCTIONS even after doing all lookups.\n"
            f"MISSION_ID={mission}\nCONTROL_CODE={control}\n"
            "Final answer must be ONLY JSON, with these keys: mission_id, control_code, seals. "
            "'seals' must be a JSON object mapping each batch_N to its seal. "
            "Keep the seal exact; do not guess missing values. "
            "Wait for the explicit FINAL REPORT request before giving the report."
        )
    }


def approximate_tokens(messages):
    # This is a rough fallback, *not* the model's true tokenizer count.
    return math.ceil(len(json.dumps(messages, ensure_ascii=False)) / 4)


@dataclass
class Trace:
    mode: str
    api: object
    calls: list = field(default_factory=list)
    summary_events: list = field(default_factory=list)
    phase: str = "parent"

    def __call__(self, method, url, cfg, payload=None):
        if method != "POST" or not url.endswith("/chat/completions"):
            return self.api(method, url, cfg, payload)
        messages = payload.get("messages") or []
        phase = self.phase
        started = time.perf_counter()
        row = {
            "phase": phase, "prompt_est_tokens": approximate_tokens(messages),
            "prompt_chars": len(json.dumps(messages, ensure_ascii=False)),
            "elapsed_seconds": None, "prompt_tokens": None,
            "completion_tokens": None, "completion_est_tokens": None,
            "requested_max_tokens": payload.get("max_tokens"), "finish_reason": None, "error": None,
            "requested_reasoning_effort": payload.get("reasoning_effort"),
            "visible_output_chars": None, "reasoning_output_chars": None,
            "reasoning_tokens": None, "empty_visible_output": None,
        }
        try:
            response = self.api(method, url, cfg, payload)
            usage = response.get("usage") or {}
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["completion_tokens"] = usage.get("completion_tokens")
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            row["finish_reason"] = choice.get("finish_reason")
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
            reason = message.get("reasoning_content") or ""
            if isinstance(reason, list):
                reason = "".join(str(x.get("text", "")) for x in reason if isinstance(x, dict))
            row["visible_output_chars"] = len(str(content))
            row["reasoning_output_chars"] = len(str(reason))
            # Some servers report reasoning tokens but omit the reasoning text.
            details = usage.get("completion_tokens_details") or {}
            row["reasoning_tokens"] = details.get("reasoning_tokens") if isinstance(details, dict) else None
            row["empty_visible_output"] = not bool(str(content).strip())
            row["completion_est_tokens"] = approximate_tokens(message)
            return response
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
            raise
        finally:
            row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.calls.append(row)

    def totals(self):
        peak_main = max((r["prompt_tokens"] or r["prompt_est_tokens"] for r in self.calls if r["phase"].startswith("parent")), default=0)
        peak_branch = max((r["prompt_tokens"] or r["prompt_est_tokens"] for r in self.calls if r["phase"].startswith("branch")), default=0)
        total_prompt = sum((r["prompt_tokens"] if r["prompt_tokens"] is not None else r["prompt_est_tokens"]) for r in self.calls)
        total_completion = sum((r["completion_tokens"] if r["completion_tokens"] is not None else (r["completion_est_tokens"] or 0)) for r in self.calls)
        actual_usage = len(self.calls) > 0 and all(r["prompt_tokens"] is not None and r["completion_tokens"] is not None for r in self.calls)
        action_retry_rows = [r for r in self.calls if r["phase"].endswith("_retry") and
                             (r["phase"].startswith("branch_recursive_") or
                              r["phase"] in ("branch_investigate_retry", "parent_investigate_retry"))]
        action_retry_prompt = sum(
            r["prompt_tokens"] if r["prompt_tokens"] is not None else r["prompt_est_tokens"]
            for r in action_retry_rows)
        action_retry_completion = sum(
            r["completion_tokens"] if r["completion_tokens"] is not None else (r["completion_est_tokens"] or 0)
            for r in action_retry_rows)
        return {
            "no_action_retry_inference_seconds": round(sum(r["elapsed_seconds"] or 0 for r in action_retry_rows), 3),
            "no_action_retry_prompt_tokens": action_retry_prompt,
            "no_action_retry_completion_tokens": action_retry_completion,
            "no_action_retry_total_tokens": action_retry_prompt + action_retry_completion,
            "no_action_retry_token_counts_exact_from_api": all(
                r["prompt_tokens"] is not None and r["completion_tokens"] is not None for r in action_retry_rows),
            "no_action_retry_empty_visible_calls": sum(r["empty_visible_output"] is True for r in action_retry_rows),
            "peak_main_prompt_tokens": peak_main, "peak_branch_prompt_tokens": peak_branch,
            "total_prompt_tokens": total_prompt, "total_completion_tokens": total_completion,
            "total_processed_tokens": total_prompt + total_completion,
            "token_counts_exact_from_api": actual_usage,
            "api_calls": len(self.calls), "api_error_calls": sum(bool(r["error"]) for r in self.calls),
            "length_stops": sum(r["finish_reason"] == "length" for r in self.calls),
            "empty_visible_calls": sum(r["empty_visible_output"] is True for r in self.calls),
            "reported_reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in self.calls),
            "reasoning_only_calls": sum(r["empty_visible_output"] is True and
                ((r["reasoning_output_chars"] or 0) > 0 or (r["reasoning_tokens"] or 0) > 0)
                for r in self.calls),
            "summary_attempts": sum(event["attempts"] for event in self.summary_events),
            "summary_retries": sum(event["attempts"] - 1 for event in self.summary_events),
            "summary_retry_recoveries": sum(bool(event["recovered_on_retry"]) for event in self.summary_events),
            "summary_failures": sum(event["status"] != "success" for event in self.summary_events),
            "summary_events": list(self.summary_events),
        }


def extract_json(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    decoder = json.JSONDecoder()
    for pos in range(len(text)):
        if text[pos] != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[pos:])
            if isinstance(result, dict):
                return result
        except ValueError:
            pass
    return {}



def parse_complete_json(text):
    """Accept valid JSON (including a surrounding ```json fence) only.

    A complete nested JSON object from prose is not a valid final report.
    """
    value = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", value, flags=re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    try:
        obj = json.loads(value)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def visible_json_string(text, field):
    """Extract an entire quoted string; never infer unfinished values."""
    pattern = r'"' + re.escape(field) + r'"\s*:\s*("(?:\\.|[^"\\])*")'
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except ValueError:
        return None


def visible_partial_report(text, expected):
    """Diagnostics only: fully emitted values from incomplete final JSON.

    Not proof of a valid report. Only search inside the seals object, and only
    count complete, quoted values for expected batch keys.
    """
    value = str(text or "")
    mission = visible_json_string(value, 'mission_id')
    control = visible_json_string(value, 'control_code')
    seals = {}
    marker = re.search(r'"seals"\s*:\s*\{', value)
    if marker:
        remainder = value[marker.end():]
        # Constrain to the start of the seals object; do not inspect other JSON.
        end = remainder.find('}')
        if end >= 0:
            remainder = remainder[:end]
        for name in expected:
            found = visible_json_string(remainder, name)
            if isinstance(found, str):
                seals[name] = found
    return {'mission_id': mission, 'control_code': control, 'seals': seals}


def output_was_cut_off(reason, output_tokens, limit, text, complete):
    """Separate definitive Jan stop reasons from output-cap circumstantial evidence."""
    reason = str(reason or "").lower()
    if reason in ('length', 'max_tokens'):
        return 'confirmed'
    if complete is not None:
        return 'none'
    if isinstance(output_tokens, (float, int)) and isinstance(limit, (float, int)) and limit > 0:
        if output_tokens >= limit - 3:
            return 'suspected'
    # An unterminated JSON document alone does NOT establish the cause.
    return 'none'


def score_final_report(text, expected, mission, control, call=None):
    """Pure evaluator, reusable for tests and post-run audits.

    strict accuracy requires complete JSON. Evidence visible before truncation
    is recorded independently, never silently upgraded to a strict pass.
    """
    parsed = parse_complete_json(text)
    evidence = visible_partial_report(text, expected)
    if parsed is not None:
        # A complete, valid report supersedes regex evidence.
        evidence = {'mission_id': parsed.get('mission_id'),
                    'control_code': parsed.get('control_code'),
                    'seals': parsed.get('seals') if isinstance(parsed.get('seals'), dict) else {}}
    seals = evidence['seals']
    visible_correct = sum(seals.get(name) == answer for name, answer in expected.items())
    fields = set(parsed) if parsed is not None else set()
    required = {'mission_id', 'control_code', 'seals'}
    missing = sorted(required - fields) if parsed is not None else []
    if parsed is not None and not isinstance(parsed.get('seals'), dict):
        missing.append('seals (not an object)')
    call = call or {}
    cut_off = output_was_cut_off(call.get('finish_reason'), call.get('completion_tokens'),
                                 call.get('requested_max_tokens'), text, parsed)
    if parsed is None:
        if cut_off == 'confirmed':
            status = 'output_truncated_confirmed'
        elif cut_off == 'suspected':
            status = 'output_truncated_suspected'
        else:
            status = 'malformed_json'
    elif missing:
        status = 'missing_fields'
    elif evidence['mission_id'] != mission or evidence['control_code'] != control or visible_correct != len(expected):
        status = 'incorrect_answer'
    else:
        status = 'success'
    return {
        'final_status': status,
        'final_json_valid': parsed is not None,
        'final_parsed': parsed if parsed is not None else {},
        'final_missing_fields': missing,
        'final_stop_reason': call.get('finish_reason'),
        'final_output_truncation': cut_off,
        'mission_ok': bool(parsed is not None and evidence['mission_id'] == mission),
        'control_ok': bool(parsed is not None and evidence['control_code'] == control),
        'final_seals_correct': visible_correct if parsed is not None else 0,
        'visible_mission_ok': evidence['mission_id'] == mission,
        'visible_control_ok': evidence['control_code'] == control,
        'visible_seals_correct': visible_correct,
        'visible_seals_count': len(seals),
        'visible_seals': seals,
        'exact_final': status == 'success',
    }


def extract_lookup_answer(text):
    """Score explicit answers even if the model formats one outside JSON."""
    report = parse_complete_json(text) or extract_json(text)
    if isinstance(report, dict):
        for key in ('result', 'seal'):
            value = report.get(key)
            if isinstance(value, str) and re.fullmatch(r'S[A-HJ-NP-Z2-9]{8}', value.strip()):
                return value.strip()
    # Some models respond with only the requested seal or `seal: VALUE`.
    plain = str(text or '').strip().strip('`').strip()
    match = re.fullmatch(r'(?:seal|result)\s*[:=]\s*(S[A-HJ-NP-Z2-9]{8})|(?P<plain>S[A-HJ-NP-Z2-9]{8})', plain, re.I)
    if match:
        return match.group(1) or match.group('plain')
    return ''


def classify_exception(exc):
    """Never claim definite context overflow/timeout from ambiguous errors."""
    if getattr(exc, 'error_kind', None) == 'context_overflow_confirmed':
        return 'context_overflow_confirmed'
    message = str(exc).lower()
    if any(s in message for s in ('context length', 'context window', 'context capacity', 'context overflow',
                                    'maximum context', 'prompt is too long', 'n_ctx', 'tokens exceed')):
        return 'context_overflow_confirmed'
    if isinstance(exc, TimeoutError) or ('timed out' in message and 'cannot reach' not in message):
        return 'timeout'
    if ('cannot reach jan api (or request timed out)' in message or
        'cannot reach local inference api (or request timed out)' in message):
        return 'connection_or_timeout_unknown'
    return 'api_or_runtime_error'


def call_parent(cfg, model, messages, *, reporting=False, final=False, summary=False,
                no_action_retry=False):
    # In the repair benchmark, Trace.phase marks recovery attempts explicitly.
    # Recognize them without changing the virtual simulator's request protocol.
    # Report/summary retries have different phases and must not be mistaken for actions.
    phase = bc.http_json.phase if isinstance(bc.http_json, Trace) else ""
    no_action_retry = no_action_retry or (phase.endswith("_retry") and
        (phase.startswith("branch_recursive_") or phase in
         ("branch_investigate_retry", "parent_investigate_retry")))
    # Preserve investigative reasoning. ACK, final report and compaction are
    # lightweight reporting tasks, so they may request no hidden reasoning.
    # This is best-effort: older inference servers may ignore the hint.
    request_cfg = cfg
    if final and cfg.get("_final_output_tokens") is not None:
        request_cfg = dict(cfg)
        request_cfg["max_output_tokens"] = cfg["_final_output_tokens"]
    elif summary and cfg.get("_summary_output_tokens") is not None:
        request_cfg = dict(cfg)
        request_cfg["max_output_tokens"] = cfg["_summary_output_tokens"]
    elif no_action_retry and cfg.get("_no_action_retry_output_tokens") is not None:
        request_cfg = dict(cfg)
        request_cfg["max_output_tokens"] = cfg["_no_action_retry_output_tokens"]
    # The first investigation request keeps its normal reasoning allowance.
    # Only a *failed* JSON-action generation switches to lightweight recovery.
    effort = (cfg.get("_no_action_retry_reasoning_effort") if no_action_retry
              else cfg.get("_report_reasoning_effort") if reporting or summary else None)
    result = bc.chat(request_cfg, model, messages, reasoning_effort=effort)
    content = result.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return str(content)


def model_final_report(cfg, model, messages, trace, expected, mission, control):
    """Score the model's actual report; retry only unfinished output.

    Retries use the *same* accumulated conversation, never oracle answers,
    and their inference work is included in token/time measurements.
    """
    trace.phase = "parent_final"
    final = call_parent(cfg, model, messages, reporting=True, final=True)
    first = score_final_report(final, expected, mission, control, trace.calls[-1])
    attempts = 1
    scored = first
    max_retries = cfg.get("_final_retries", 0)
    while (not scored["exact_final"] and attempts <= max_retries and
           (scored["final_output_truncation"] != "none" or not final.strip())):
        trace.phase = "parent_final_retry"
        retry_messages = messages + [{"role": "user", "content": (
            "The final report was not completed. Output ONLY the requested compact "
            "JSON object immediately. No analysis or commentary. Retain the original "
            "mission_id and control_code and all verified seals in their original batches.")}]
        final = call_parent(cfg, model, retry_messages, reporting=True, final=True)
        attempts += 1
        scored = score_final_report(final, expected, mission, control, trace.calls[-1])
    scored["final_attempts"] = attempts
    scored["final_first_status"] = first["final_status"]
    scored["final_recovered_on_retry"] = attempts > 1 and scored["exact_final"]
    scored["raw_final_excerpt"] = final[:1500]
    return scored


SUMMARY_REQUEST = (
    "PERIODIC CONTEXT COMPACTION. Summarize completed archive lookups so the\n"
    "conversation can discard bulky old archive rows. The original mission and\n"
    "control instructions are preserved separately. Keep every completed batch\n"
    "name and its exact seal (or clearly mark unknown); preserve uncertainties\n"
    "and relevant corrections. Do NOT invent data or use unstated answers.\n"
    "Return ONLY a concise memory note, ideally one line per batch, no archive\n"
    "rows and no commentary. Stay under {max_chars} characters."
)


def compact_history(cfg, model, messages, preserved, trace, max_chars, summary_request=SUMMARY_REQUEST):
    """Compact only model-visible history, preserving the original instructions.

    Summarization uses its own output allowance and reporting reasoning policy.
    Empty or truncated answers can be retried once (configurable). Never use
    benchmark oracle data, rewrite the original task or accept cut-off memory.
    """
    request = summary_request.format(max_chars=max_chars)
    request_messages = messages + [{"role": "user", "content": request}]
    max_retries = cfg.get("_summary_retries", 1)
    attempts = 0
    first_status = None
    status = "empty"
    summary = ""
    try:
        for attempt in range(max_retries + 1):
            trace.phase = "parent_summary" if attempt == 0 else "parent_summary_retry"
            current_messages = request_messages if attempt == 0 else messages + [
                {"role": "user", "content": request + "\n"
                 "Your previous summary attempt was empty or cut off. Return ONLY "
                 "completed batch names and exact verified seals, one per line. "
                 "No analysis; stay within the stated character limit."}]
            summary = call_parent(cfg, model, current_messages, reporting=True, summary=True).strip()
            attempts += 1
            last = trace.calls[-1] if trace.calls else {}
            cut_off = output_was_cut_off(last.get("finish_reason"), last.get("completion_tokens"),
                                         last.get("requested_max_tokens"), summary, None)
            if not summary:
                status = "empty"
            elif len(summary) > max_chars:
                status = "oversized"
            elif cut_off != "none":
                status = "truncated_" + cut_off
            else:
                status = "success"
            if first_status is None:
                first_status = status
            if status == "success":
                break
            # Retry only empty or cut-off output, never a fully produced
            # oversized note: a size violation is not safe memory compaction.
            if status not in ("empty", "truncated_confirmed", "truncated_suspected"):
                break
    finally:
        trace.summary_events.append({
            "attempts": attempts,
            "first_status": first_status or "error",
            "status": status if attempts else "error",
            "recovered_on_retry": attempts > 1 and status == "success",
            "chars": len(summary),
        })
    if status != "success":
        raise ValueError(f"Context summarization failed after {attempts} attempt(s): {status}"
                         + (f" ({len(summary)} > {max_chars} chars)" if status == "oversized" else ""))
    # The original task survives unchanged. Neither the full investigative
    # history nor either summarization request enters future parent calls.
    compacted = [dict(preserved[0]), dict(preserved[1]),
                 {"role": "assistant", "content": "COMPACT MEMORY OF COMPLETED LOOKUPS:\n" + summary}]
    return compacted, summary



def run_trial(mode, seed, tasks, records, cfg, model, context_limit, save_dir,
              summary_every=2, summary_max_chars=1800):
    if mode not in ("linear", "summarized", "branch"):
        raise ValueError(f"Unknown benchmark mode: {mode}")
    if summary_every < 1 or summary_max_chars < 100:
        raise ValueError("Invalid summarization settings")
    puzzle = create_puzzle(seed, tasks, records)
    messages = [
        {"role": "system", "content": "Follow user instructions exactly. Solve archive lookups precisely. Respond with JSON when required. No external actions or tools."},
        {"role": "user", "content": puzzle["brief"]},
    ]
    protected_intro = [dict(m) for m in messages]
    expected = {name: answer for name, _, answer in puzzle["tasks"]}
    result = {
        "mode": mode, "seed": seed, "tasks": tasks, "records_per_task": records,
        "model": model, "configured_context_limit": context_limit,
        "max_output_tokens": cfg["max_output_tokens"],
        "solved_subtasks": 0, "subtask_accuracy": 0, "mission_ok": False,
        "control_ok": False, "final_seals_correct": 0, "final_seals_total": tasks,
        "exact_final": False, "final_status": "not_attempted", "final_json_valid": False,
        "final_output_truncation": "unknown", "final_stop_reason": None,
        "visible_seals_correct": 0, "visible_seals_count": 0,
        "visible_mission_ok": False, "visible_control_ok": False,
        "any_stage_seals_correct": 0, "final_recovered_subtasks": 0,
        "error": "", "error_type": "none", "failed_phase": "",
        "reported_answers": {}, "expected_answers": expected, "lookup_breakdown": [],
        "summary_every": summary_every if mode == "summarized" else None,
        "summary_max_chars": summary_max_chars if mode == "summarized" else None,
        "summaries_generated": 0, "summary_chars_peak": 0, "summaries": [],
    }
    original_http = bc.http_json
    original_data = bc.DATA
    trace = Trace(mode, original_http)
    bc.http_json = trace
    bc.DATA = save_dir / "branches"
    started = time.perf_counter()
    stage = "setup"
    try:
        for name, side_prompt, answer_expected in puzzle["tasks"]:
            stage = f"lookup:{name}"
            if mode in ("linear", "summarized"):
                trace.phase = "parent_lookup"
                messages.append({"role": "user", "content": side_prompt})
                response = call_parent(cfg, model, messages)
                messages.append({"role": "assistant", "content": response})
                answer = extract_lookup_answer(response)
                reason = trace.calls[-1]["finish_reason"] if trace.calls else None
                lookup_status = "correct" if answer == answer_expected else ("wrong" if answer else "missing_or_unreadable")
            else:
                trace.phase = "branch_lookup"
                isolated = bc.run_branch(cfg, side_prompt, f"Task: {name}; look up only its requested seal.")
                answer = str(isolated.get("result") or "")
                reason = trace.calls[-1]["finish_reason"] if trace.calls else None
                lookup_status = "correct" if answer == answer_expected else ("wrong" if answer else "missing_or_unreadable")
                trace.phase = "parent_ack"
                compressed = {"batch": name, "status": isolated.get("status"),
                              "result": isolated.get("result"), "caveats": isolated.get("caveats")}
                messages.append({"role": "user", "content": "Isolated lookup finished. Keep this compact outcome for the final report: "
                                 + json.dumps(compressed) + "\nReply ONLY READY; do not output a report yet."})
                ack = call_parent(cfg, model, messages, reporting=True)
                messages.append({"role": "assistant", "content": ack})
            result["reported_answers"][name] = answer
            result["lookup_breakdown"].append({
                "batch": name, "answer": answer, "correct": answer == answer_expected,
                "status": lookup_status, "finish_reason": reason
            })
            result["solved_subtasks"] += 1
            if answer == answer_expected:
                result["subtask_accuracy"] += 1
            if (mode == "summarized" and result["solved_subtasks"] % summary_every == 0
                    and result["solved_subtasks"] < tasks):
                stage = f"summary:after:{name}"
                messages, summary = compact_history(cfg, model, messages, protected_intro, trace, summary_max_chars)
                result["summaries_generated"] += 1
                result["summary_chars_peak"] = max(result["summary_chars_peak"], len(summary))
                result["summaries"].append({"after_batch": name, "chars": len(summary), "text": summary})
        stage = "final_report"
        messages.append({"role": "user", "content": (
            "FINAL REPORT. Assemble the audit report from the seals obtained in earlier steps. "
            "Give ONLY valid JSON with the exact keys mission_id, control_code, seals. "
            "Use the original instruction's mission and control values. Don't explain."
        )})
        result.update(model_final_report(cfg, model, messages, trace, expected,
                                         puzzle["mission"], puzzle["control"]))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        result["error_type"] = ("summary_failed" if stage.startswith("summary:") and isinstance(exc, ValueError)
                                else classify_exception(exc))
        result["failed_phase"] = stage
        result["final_status"] = result["error_type"] if stage == "final_report" else "not_attempted"
    finally:
        # Count correct information found at ANY point separately from a complete
        # final report; don't turn incomplete JSON into a successful final.
        final_visible = result.get("visible_seals", {})
        direct = result["reported_answers"]
        result["final_recovered_subtasks"] = sum(
            direct.get(name) != answer and final_visible.get(name) == answer
            for name, answer in expected.items()
        )
        result["any_stage_seals_correct"] = sum(
            direct.get(name) == answer or final_visible.get(name) == answer
            for name, answer in expected.items()
        )
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result.update(trace.totals())
        result["calls"] = trace.calls
        # Flag risk without misreporting an over-limit prompt as confirmed failure.
        result["input_over_limit_risk"] = any(
            r["prompt_tokens"] is not None and r["prompt_tokens"] > context_limit
            for r in trace.calls
        )
        bc.http_json = original_http
        bc.DATA = original_data
    return result

def arguments(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark linear, summarized and branching context on Jan or a dedicated llama.cpp server")
    parser.add_argument("--engine", choices=("jan", "llama"), default="jan", help="Use Jan (legacy) or an independently managed llama.cpp server. For fresh server each trial use --engine llama")
    parser.add_argument("--llama-server", default="llama-server", help="Executable for standalone llama-server (system PATH or absolute path)")
    parser.add_argument("--llama-model", default=None, help="Override local_model.json MODEL_PATH or CONTEXT_BRANCH_MODEL with a GGUF file path")
    parser.add_argument("--llama-port", type=int, default=18089, help="Dedicated loopback port, default 18089; refuses to overwrite an existing listener")
    parser.add_argument("--llama-gpu-layers", type=int, default=99, help="GPU layers to offload (default 99; use 0 for CPU-only)")
    parser.add_argument("--llama-startup-timeout", type=float, default=180, help="Seconds to wait for local server readiness")
    parser.add_argument("--llama-arg", action="append", default=[], metavar="ARG", help="Additional single llama-server argument (repeat; use --llama-arg=--flag for options)")
    parser.add_argument("--check-llama-server", action="store_true", help="Validate binary and model without starting the server or running trials")
    parser.add_argument("--reuse-llama", action="store_true", help="Opt out of fresh llama-server between trials (by default each trial uses a NEW process)")
    parser.add_argument("--scenario", choices=("archive", "repair"), default="archive", help="Archive lookups (legacy) or stateful repair traps (new)")
    parser.add_argument("--modes", choices=("linear", "summarized", "branch", "recursive"),
                        nargs="+", default=["linear", "summarized", "branch"],
                        help="Competitors to run; add recursive for controlled nested branching (repair only)")
    parser.add_argument("--no-action-retries", type=int, default=1,
                        help="Retry invalid/empty investigation JSON action before spending a simulator action (0-2; default 1)")
    parser.add_argument("--no-action-retry-output-tokens", type=int, default=256,
                        help="Generation cap on each no-action recovery request (64-4096; default 256)")
    parser.add_argument("--no-action-retry-reasoning", choices=("auto", "none"), default="none",
                        help="Request no thinking on recovery retries only with llama.cpp (default none; auto skips hint)")
    parser.add_argument("--recursive-max-depth", type=int, default=3,
                        help="Maximum child levels in controlled recursive mode (0-3, default 3)")
    parser.add_argument("--tasks", type=int, nargs="+", default=None, help="Archive: 3 6 8; repair: 1 2 4 by default")
    parser.add_argument("--records", type=int, default=65, help="Rows in each archive (default 65)")
    parser.add_argument("--repeats", type=int, default=2, help="Different seeds per task level (default 2)")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--context-limit", type=int, default=8192, help="Context in tokens: passed to llama-server; for Jan used for reporting only")
    parser.add_argument("--max-output-tokens", type=int, default=1024, help="Investigation output cap (default 1024; equal for all modes)")
    parser.add_argument("--final-output-tokens", type=int, default=2048, help="Output cap for final reporting ONLY (default 2048)")
    parser.add_argument("--final-retries", type=int, default=1, help="Retries for truncated or empty final report (default 1; retries counted in time/tokens)")
    parser.add_argument("--report-reasoning", choices=("auto", "none"), default="none",
                        help="Request no thinking on ACK, summary and final-report calls with llama.cpp (default none); investigation reasoning stays enabled")
    parser.add_argument("--summary-output-tokens", type=int, default=2048,
                        help="Output cap only for periodic summaries (default 2048)")
    parser.add_argument("--summary-retries", type=int, default=1,
                        help="Retries for empty or truncated summaries (default 1)")
    parser.add_argument("--summary-every", type=int, default=2, help="Summarize history after every N completed lookups, except after the last (default: 2)")
    parser.add_argument("--summary-max-chars", type=int, default=1800, help="Maximum length of each compact memory note (default: 1800 chars)")
    parser.add_argument("--trap-log-lines", type=int, default=32, help="Repair: non-authoritative log lines per trap (default 32; 0-150)")
    parser.add_argument("--trap-density", type=int, choices=[1, 2, 3], default=2, help="Repair trap strength (1-3; default 2)")
    parser.add_argument("--recovery-log-lines", type=int, default=36, help="Repair: noisy lines per MANDATORY recovery shard (default 36; 0-120)")
    parser.add_argument("--nested-depth", type=int, choices=(0, 1, 2), default=2,
                        help="Repair: extra mandatory dependency layers before recovery (default 2; 0 reproduces v0.7)")
    parser.add_argument("--nested-log-lines", type=int, default=14,
                        help="Repair: noisy lines in EACH nested index/escrow shard (default 14; 0-120)")
    parser.add_argument("--pressure-records", type=int, default=4,
                        help="Repair: additional plausible near-match rows PER required evidence document (default 4; 0-160; 0 reproduces v0.8)")
    parser.add_argument("--pressure-detail-chars", type=int, default=48,
                        help="Repair: evidence metadata characters per injected row (default 48; 0-256)")
    parser.add_argument("--max-actions", type=int, default=32, help="Repair: maximum JSON decisions per incident (default 32; ideal run takes 12 + 3 per nested layer)")
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument("--reuse-jan", action="store_true",
                             help="Do not reset Jan between trials (default; real-world cache reuse)")
    reset_group.add_argument("--restart-jan-between-trials", action="store_true",
                             help="Verified Jan model unload/reload BEFORE EVERY trial, including the first (clean trials)")
    reset_group.add_argument("--restart-jan-between-modes", action="store_true",
                             help="Unload/reload only before the first trial and when the mode changes")
    parser.add_argument("--jan-router-url", type=str, default=None,
                        help="Optional localhost llama.cpp router URL (auto-detected from Jan worker if omitted)")
    parser.add_argument("--cooldown-seconds", type=float, default=8.0,
                        help="Seconds to pause AFTER unload, BEFORE reloading (default: 8)")
    parser.add_argument("--reset-timeout-seconds", type=float, default=180.0,
                        help="Maximum wait for each model state transition (default: 180)")
    parser.add_argument("--check-jan-reset", action="store_true",
                        help="Only verify model router discovery, without unloading anything or running trials")
    parser.add_argument("--output", type=Path, default=None, help="Result folder; defaults to project/data/benchmarks/<UTC time>")
    parser.add_argument("--dry-run", action="store_true", help="Generate scenarios & validate constraints; no Jan connection")
    parser.add_argument("--smoke", action="store_true", help="One repeat, 3 archive lookups or 1 repair per mode. Repair still uses selected trap/pressure settings.")
    args = parser.parse_args(argv)
    if args.tasks is None:
        args.tasks = [3, 6, 8] if args.scenario == "archive" else [1, 2, 4]
    if args.smoke:
        args.tasks, args.records, args.repeats = ([3], 12, 1) if args.scenario == "archive" else ([1], 12, 1)
    if not 64 <= args.max_output_tokens <= 4096:
        parser.error("--max-output-tokens must be between 64 and 4096")
    if not 64 <= args.final_output_tokens <= 4096 or not 0 <= args.final_retries <= 2:
        parser.error("--final-output-tokens must be 64-4096 and --final-retries 0-2")
    if not 64 <= args.summary_output_tokens <= 4096 or not 0 <= args.summary_retries <= 2:
        parser.error("--summary-output-tokens must be 64-4096 and --summary-retries 0-2")
    if args.summary_every < 1 or not 100 <= args.summary_max_chars <= 10000:
        parser.error("--summary-every must be >= 1 and --summary-max-chars between 100 and 10000")
    if (not 0 <= args.trap_log_lines <= 150 or not 0 <= args.recovery_log_lines <= 120
            or not 0 <= args.nested_log_lines <= 120 or not 12 <= args.max_actions <= 80
            or not 0 <= args.pressure_records <= 160 or not 0 <= args.pressure_detail_chars <= 256):
        parser.error("--trap-log-lines must be 0-150; recovery/nested lines 0-120; pressure-records 0-160; pressure-detail-chars 0-256; max-actions 12-80")
    if not 0 <= args.cooldown_seconds <= 600 or not 10 <= args.reset_timeout_seconds <= 1200:
        parser.error("--cooldown-seconds must be 0-600 and --reset-timeout-seconds 10-1200")
    if args.engine == "llama" and (args.restart_jan_between_trials or args.restart_jan_between_modes or args.check_jan_reset):
        parser.error("Jan router reset flags are not applicable to --engine llama (each llama trial is fresh by default)")
    if args.engine == "jan" and (args.reuse_llama or args.check_llama_server or args.llama_model or args.llama_arg):
        parser.error("llama options require --engine llama")
    if args.engine == "llama":
        try:
            args.llama_model = resolve_llama_model_path(args.llama_model)
        except ValueError as exc:
            parser.error(f"Invalid local model configuration: {exc}")
    if args.jan_router_url:
        try:
            args.jan_router_url = jan_reset.validate_router_url(args.jan_router_url)
        except jan_reset.ResetError as ex:
            parser.error(str(ex))
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    if "recursive" in args.modes and args.scenario != "repair":
        parser.error("--modes recursive is supported only for --scenario repair")
    if not 0 <= args.no_action_retries <= 2:
        parser.error("--no-action-retries must be 0-2")
    if not 64 <= args.no_action_retry_output_tokens <= 4096:
        parser.error("--no-action-retry-output-tokens must be 64-4096")
    if not 0 <= args.recursive_max_depth <= 3:
        parser.error("--recursive-max-depth must be 0-3")
    if args.records < 2 or args.repeats < 1 or not args.tasks or any(n < 1 for n in args.tasks):
        parser.error("--tasks, --records, and --repeats must be positive (records >= 2)")
    return args


def write_files(directory, data, cfg, args, reset_events=None):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": {"model": data[0]["model"] if data else "unknown", "tasks": args.tasks,
                  "records": args.records, "repeats": args.repeats, "context_limit": args.context_limit,
                  "token_count_note": "Exact only when token_counts_exact_from_api=true; otherwise heuristic chars/4",
                  "max_output_tokens": args.max_output_tokens,
                  "final_output_tokens": args.final_output_tokens,
                  "final_retries": args.final_retries,
                  "report_reasoning": args.report_reasoning if args.engine == "llama" else "auto",
                  "seed": args.seed,
                  "scoring_version": 3,
                  "benchmark_version": "1.2.0", "scenario": args.scenario,
                  "engine": args.engine,
                  "llama_reset_policy": ("reuse" if args.reuse_llama else "fresh_each_trial") if args.engine == "llama" else None,
                  "jan_reset_policy": (("each_trial" if args.restart_jan_between_trials else
                                       "mode_change" if args.restart_jan_between_modes else "reuse") if args.engine == "jan" else None),
                  "llama_server_executable": args.llama_server if args.engine == "llama" else None,
                  "llama_model_path": str(args.llama_model) if args.engine == "llama" else None,
                  "llama_gpu_layers": args.llama_gpu_layers if args.engine == "llama" else None,
                  "jan_reset_cooldown_seconds": args.cooldown_seconds,
                  "jan_reset_timeout_seconds": args.reset_timeout_seconds,
                  "trap_density": args.trap_density if args.scenario == "repair" else None,
                  "trap_log_lines": args.trap_log_lines if args.scenario == "repair" else None,
                  "recovery_log_lines": args.recovery_log_lines if args.scenario == "repair" else None,
                  "nested_depth": args.nested_depth if args.scenario == "repair" else None,
                  "nested_log_lines": args.nested_log_lines if args.scenario == "repair" else None,
                  "pressure_records": args.pressure_records if args.scenario == "repair" else None,
                  "pressure_detail_chars": args.pressure_detail_chars if args.scenario == "repair" else None,
                  "mandatory_recovery": args.scenario == "repair",
                  "max_actions": args.max_actions if args.scenario == "repair" else None,
                  "modes": args.modes, "recursive_max_depth": args.recursive_max_depth,
                  "no_action_retries": args.no_action_retries,
                  "no_action_retry_output_tokens": args.no_action_retry_output_tokens,
                  "no_action_retry_reasoning": args.no_action_retry_reasoning if args.engine == "llama" else "auto",
                  "summary_every": args.summary_every, "summary_max_chars": args.summary_max_chars,
                  "summary_output_tokens": args.summary_output_tokens,
                  "summary_retries": args.summary_retries,
                  "strict_pass_rule": "Complete valid JSON with every field and seal correct. Partial visible fields are diagnostics only."},
                   "trials": data, "resets": reset_events or []}, fh, indent=2, ensure_ascii=False)
    columns = [
        "mode", "seed", "tasks", "records_per_task", "model", "configured_context_limit", "max_output_tokens",
        "exact_final", "final_status", "final_json_valid", "final_output_truncation", "final_stop_reason",
        "final_attempts", "final_first_status", "final_recovered_on_retry",
        "compiled_verified_cases", "compiled_report_complete",
        "mission_ok", "control_ok", "visible_mission_ok", "visible_control_ok",
        "subtask_accuracy", "solved_subtasks", "final_recovered_subtasks", "any_stage_seals_correct",
        "final_seals_correct", "visible_seals_correct", "visible_seals_count", "final_seals_total",
        "peak_main_prompt_tokens", "peak_branch_prompt_tokens", "input_over_limit_risk",
        "total_prompt_tokens", "total_completion_tokens", "total_processed_tokens", "token_counts_exact_from_api",
        "elapsed_seconds", "api_calls", "api_error_calls", "length_stops",
        "empty_visible_calls", "reasoning_only_calls", "reported_reasoning_tokens",
        "reset_before_trial", "reset_status", "reset_seconds", "reset_method", "reset_memory_before_mb",
        "reset_memory_after_unload_mb", "reset_memory_after_reload_mb",
        "summary_every", "summary_max_chars", "summaries_generated", "summary_chars_peak",
        "summary_attempts", "summary_retries", "summary_retry_recoveries", "summary_failures",
        "error_type", "failed_phase", "error",
        "scenario", "trap_density", "trap_log_lines", "max_actions",
        "total_actions", "verified_cases", "invalid_actions", "wrong_repairs",
        "no_action_investigation_calls", "empty_investigation_replies",
        "no_action_retry_attempts", "no_action_retry_recoveries", "no_action_retry_failures", "out_of_scope_actions",
        "no_action_retry_inference_seconds", "no_action_retry_prompt_tokens",
        "no_action_retry_completion_tokens", "no_action_retry_total_tokens",
        "no_action_retry_token_counts_exact_from_api", "no_action_retry_empty_visible_calls",
        "recursive_branch_count", "recursive_context_count", "recursive_max_depth", "recursive_stage_successes", "recursive_stage_failures",
        "decoy_documents_opened", "trap_waste_chars",
        "recovery_log_lines", "recovery_completed", "recovery_attempts",
        "recovery_failures", "recovery_blocked", "recovery_shard_views", "forced_evidence_chars",
        "nested_depth", "nested_log_lines", "nested_index_completed", "nested_escrow_completed",
        "nested_index_attempts", "nested_escrow_attempts", "nested_index_failures", "nested_escrow_failures",
        "nested_blocked", "nested_shard_views", "nested_evidence_chars",
        "pressure_records", "pressure_detail_chars", "pressure_generated_chars", "pressure_evidence_chars",
        "authority_docs_viewed", "authority_evidence_chars", "authority_blocked"
    ]
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

    lines = [
        "# Context Branch benchmark — stage-scoped recursion and no-action recovery (v1.4.1 / scoring v3))", "",
        f"Scenario: **{args.scenario}** | Repair traps: density={args.trap_density}, "
        f"optional log lines={args.trap_log_lines}, required shard lines={args.recovery_log_lines}, "
        f"nested depth={args.nested_depth}, nested lines={args.nested_log_lines}, "
        f"pressure rows/document={args.pressure_records}, metadata chars/row={args.pressure_detail_chars}, "
        f"max actions={args.max_actions}" if args.scenario == "repair" else "Scenario: archive retrieval", "",
        f"Model: `{data[0]['model'] if data else 'unknown'}` | Configured context: {args.context_limit} | "
        f"Investigation output: {args.max_output_tokens} | Final output: {args.final_output_tokens} "
        f"| Trials completed/started: {len(data)}", "",
        "## Strict correctness and recoverable information", "",
        ("| Tasks | Mode | Valid complete reports | Verified incidents / attempted | " if args.scenario == "repair"
         else "| Tasks | Mode | Valid complete reports | Direct lookups correct / attempted | ")
        + "Information correct anywhere | Summaries | Peak MAIN tokens | Total tokens | Seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in args.tasks:
        for mode in args.modes:
            rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
            if not rows:
                continue
            avg = lambda key: round(sum(float(r[key]) for r in rows) / len(rows))
            successes = f"{sum(int(r['exact_final']) for r in rows)}/{len(rows)}"
            attempted = sum(r["solved_subtasks"] for r in rows)
            correct_lookups = f"{sum(r['subtask_accuracy'] for r in rows)}/{attempted}"
            anywhere = f"{sum(r['any_stage_seals_correct'] for r in rows)}/{n*len(rows)}"
            lines.append(f"| {n} | {mode} | {successes} | {correct_lookups} | {anywhere} | "
                         f"{sum(r['summaries_generated'] for r in rows)} | "
                         f"{avg('peak_main_prompt_tokens')} | {avg('total_processed_tokens')} | "
                         f"{round(sum(r['elapsed_seconds'] for r in rows)/len(rows),1)} |")
    lines += ["", "## Summarization reliability (generated by the same model)", "",
              "| Repairs/tasks | Mode | Summary attempts | Retries | Recovered after retry | Summary failures |",
              "|---:|---|---:|---:|---:|---:|"]
    for n in args.tasks:
        rows = [row for row in data if row["tasks"] == n and row["mode"] == "summarized"]
        if rows:
            lines.append(f"| {n} | summarized | {sum(r.get('summary_attempts', 0) for r in rows)} "
                         f"| {sum(r.get('summary_retries', 0) for r in rows)} "
                         f"| {sum(r.get('summary_retry_recoveries', 0) for r in rows)} "
                         f"| {sum(r.get('summary_failures', 0) for r in rows)} |")
    lines += ["", "## Final-report diagnostics (model-generated reports only)", "",
              "| Tasks | Mode | First report incomplete | Retried reports | Recovered via retry | Empty-content calls | Reasoning-only calls |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for n in args.tasks:
        for mode in args.modes:
            rows = [row for row in data if row["tasks"] == n and row["mode"] == mode]
            if rows:
                lines.append(f"| {n} | {mode} | {sum(r.get('final_first_status', 'success') != 'success' for r in rows)} "
                             f"| {sum(r.get('final_attempts', 1) > 1 for r in rows)} "
                             f"| {sum(bool(r.get('final_recovered_on_retry')) for r in rows)} "
                             f"| {sum(r.get('empty_visible_calls', 0) for r in rows)} "
                             f"| {sum(r.get('reasoning_only_calls', 0) for r in rows)} |")
    lines += ["", "Final-output retries run against the SAME model history and count toward total inference work. "
              "Compiled/verified outputs (if present) are orchestrator-generated and are NOT counted as model final-report success."]
    if args.scenario == "repair":
        lines += ["", "## Stage scope and no-action recovery", "",
                  "| Tasks | Mode | Rejected wrong-stage actions | No-action responses | Retries | Recovered on retry | Failed retries |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for n in args.tasks:
            for mode in args.modes:
                rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
                if rows:
                    lines.append(f"| {n} | {mode} | {sum(r.get('out_of_scope_actions', 0) for r in rows)} "
                                 f"| {sum(r.get('no_action_investigation_calls', 0) for r in rows)} "
                                 f"| {sum(r.get('no_action_retry_attempts', 0) for r in rows)} "
                                 f"| {sum(r.get('no_action_retry_recoveries', 0) for r in rows)} "
                                 f"| {sum(r.get('no_action_retry_failures', 0) for r in rows)} |")
    policy = (("fresh dedicated llama.cpp process per trial" if not args.reuse_llama else "reuse dedicated llama.cpp process")
              if args.engine == "llama" else
              "every trial" if args.restart_jan_between_trials else
              "on mode change" if args.restart_jan_between_modes else "reuse Jan model")
    lines += ["", "## Model reset and memory isolation", "",
              f"Engine: **{args.engine}**. Reset policy: **{policy}**.",
              f"Verified launches/resets: {sum(ev.get('status') == 'verified' for ev in (reset_events or []))}; "
              f"failed: {sum(ev.get('status') == 'failed' for ev in (reset_events or []))}.",
              ("For dedicated llama.cpp mode, only benchmark-owned server processes are stopped. "
               "Model loading is timed separately from trial runtime; server logs are in server_logs/. "
               "Jan and its router are never contacted by this engine."
               if args.engine == "llama" else
               "Legacy Jan model router resets are optional and may be unavailable on some Jan builds."), ""]
    if args.scenario == "repair":
        lines += ["", "## Repair traps: observable investigation behavior", "",
                  "| Tasks | Mode | Verified cases | Model actions | Wrong repairs | Decoy docs read | Trap output chars |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for n in args.tasks:
            for mode in args.modes:
                rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
                if rows:
                    lines.append(f"| {n} | {mode} | {sum(r['verified_cases'] for r in rows)}/{n * len(rows)} | "
                                 f"{sum(r['total_actions'] for r in rows)} | "
                                 f"{sum(r['wrong_repairs'] for r in rows)} | "
                                 f"{sum(r['decoy_documents_opened'] for r in rows)} | "
                                 f"{sum(r['trap_waste_chars'] for r in rows)} |")
        lines += ["", "## Investigation no-action diagnostics", "",
                  "Counts cover only investigation calls, not final reporting or summaries.",
                  "| Tasks | Mode | No valid JSON action | Empty model responses |",
                  "|---:|---|---:|---:|"]
        for n in args.tasks:
            for mode in args.modes:
                rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
                if rows:
                    lines.append(f"| {n} | {mode} | {sum(r.get('no_action_investigation_calls', 0) for r in rows)} "
                                 f"| {sum(r.get('empty_investigation_replies', 0) for r in rows)} |")
        lines += ["", "## Mandatory recovery investigation", "",
                  "Every incident requires two noisy recovery shards and a correct `recover` action before "
                  "the approvals ledger opens or a repair can be accepted. These reads are necessary work "
                  "and are counted separately from optional decoys.", "",
                  "| Tasks | Mode | Unlocked cases | Shards read | Recovery attempts | Bad recovery | Blocked actions | Required evidence chars |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for n in args.tasks:
            for mode in args.modes:
                rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
                if rows:
                    lines.append(f"| {n} | {mode} | {sum(r.get('recovery_completed', 0) for r in rows)}/{n * len(rows)} | "
                                 f"{sum(r.get('recovery_shard_views', 0) for r in rows)} | "
                                 f"{sum(r.get('recovery_attempts', 0) for r in rows)} | "
                                 f"{sum(r.get('recovery_failures', 0) for r in rows)} | "
                                 f"{sum(r.get('recovery_blocked', 0) for r in rows)} | "
                                 f"{sum(r.get('forced_evidence_chars', 0) for r in rows)} |")
        if args.nested_depth:
            lines += ["", "## Mandatory nested dependencies", "",
                      "Index (depth 2) must be rebuilt before escrow (depth 1) can be inspected; "
                      "escrow must be unlocked before the original recovery shards can be inspected. "
                      "Every stage requires both noisy shards and an exact CURRENT-run token. "
                      "These are enforced simulator gates, not optional decoys.", "",
                      "| Tasks | Mode | Index rebuilt | Escrow unlocked | Nested shards read | Nested evidence chars | Nested blocked | Failed index / escrow attempts |",
                      "|---:|---|---:|---:|---:|---:|---:|---:|"]
            for n in args.tasks:
                for mode in args.modes:
                    rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
                    if rows:
                        lines.append(f"| {n} | {mode} | {sum(r.get('nested_index_completed', 0) for r in rows)} | "
                                     f"{sum(r.get('nested_escrow_completed', 0) for r in rows)} | "
                                     f"{sum(r.get('nested_shard_views', 0) for r in rows)} | "
                                     f"{sum(r.get('nested_evidence_chars', 0) for r in rows)} | "
                                     f"{sum(r.get('nested_blocked', 0) for r in rows)} | "
                                     f"{sum(r.get('nested_index_failures', 0) for r in rows)} / "
                                     f"{sum(r.get('nested_escrow_failures', 0) for r in rows)} |")
        lines += ["", "## Required-evidence context pressure", "",
                  "Each required shard and each of the five authoritative evidence documents receives "
                  "additional *near-matching structured records*. The correct answer stays unchanged; "
                  "the model must cross-check CURRENT run/state and ACTIVE+SIGNED authority. "
                  "Injected records contain no operative instructions. With pressure enabled, "
                  "all five authoritative documents must be read before repair. "
                  "Extra generated chars are not automatically viewed: observed pressure "
                  "is counted only when the AI actually inspects those documents.", "",
                  "| Tasks | Mode | Extra chars generated | Extra chars actually viewed | Authority documents read | Authority evidence chars | Attempts blocked (unread evidence) |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for n in args.tasks:
            for mode in args.modes:
                rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
                if rows:
                    lines.append(f"| {n} | {mode} | {sum(r.get('pressure_generated_chars', 0) for r in rows)} | "
                                 f"{sum(r.get('pressure_evidence_chars', 0) for r in rows)} | "
                                 f"{sum(r.get('authority_docs_viewed', 0) for r in rows)}/{5*n*len(rows)} | "
                                 f"{sum(r.get('authority_evidence_chars', 0) for r in rows)} | "
                                 f"{sum(r.get('authority_blocked', 0) for r in rows)} |")
        lines += ["", "**Context-pressure caveat:** The benchmark does not rewind inside a repair; "
                  "nested stages share a context in ordinary branch mode, but recursive mode isolates each prerequisite. Very high pressure may exhaust "
                  "all modes. API token usage and actual truncation behavior depend on Jan.", ""]
        lines += ["", "A repair is counted as **verified** only after the AI successfully submits a policy-valid "
                  "release/route/certificate triple AND requests verification. A plausible written answer is not enough. "
                  "Wrong repairs return noisy simulator logs, costing context; decoy documents also add context. "
                  "A strict final pass still requires correct mission IDs and all verified seals.", ""]
    lines += [
        "", "A **strict successful report** must contain complete valid JSON, both original IDs, and every correct seal. "
        "Answers visible in cut-off JSON are *not* scored as successful reports.",
        "**Direct lookups** are graded from the immediate per-task reply, regardless of the final report. "
        "**Information correct anywhere** credits each seal if it is confirmed in an immediate reply "
        "or fully visible in the final response; this measures recoverable data, NOT completed work.",
        "", "## Outcome diagnostics", "",
        "| Tasks | Mode | Outcome | Trials |", "|---:|---|---|---:|",
    ]
    for n in args.tasks:
        for mode in args.modes:
            rows = [r for r in data if r["tasks"] == n and r["mode"] == mode]
            if not rows:
                continue
            statuses = sorted({r["final_status"] for r in rows})
            for status in statuses:
                lines.append(f"| {n} | {mode} | `{status}` | {sum(r['final_status'] == status for r in rows)} |")
    lines += [
        "", f"**Final-output truncation:** {sum(r.get('final_output_truncation') == 'confirmed' for r in data)} confirmed, "
        f"{sum(r.get('final_output_truncation') == 'suspected' for r in data)} suspected. "
        "Suspected means incomplete JSON near the output cap without an explicit API stop reason.",
        f"**API-reported length stops across all calls:** {sum(r.get('length_stops', 0) for r in data)}.",
        f"**Prompt size above configured limit (risk only):** {sum(r.get('input_over_limit_risk', False) for r in data)} trials. "
        "Jan may truncate inputs or interpret `usage` differently; this is NOT proof that a context overflow occurred.",
        f"**Recovered direct lookup misses in fully visible final fields:** {sum(r.get('final_recovered_subtasks', 0) for r in data)}.",
        "", "## Interpreting the scores", "",
        "- `success`: complete valid final report with all correct answers.",
        "- `incorrect_answer`: valid complete report with at least one wrong value.",
        "- `missing_fields`: valid JSON but a required field is absent or has the wrong type.",
        "- `malformed_json`: no complete JSON; not enough evidence to attribute cause to token limit.",
        "- `output_truncated_confirmed`: API explicitly reported `finish_reason=length` for an incomplete report.",
        "- `output_truncated_suspected`: incomplete report plus usage at the generation cap; API did not confirm truncation.",
        "- `context_overflow_confirmed`: exception explicitly indicates context overflow.",
        "- `timeout`: a definitive timeout; `connection_or_timeout_unknown` means the error was ambiguous.",
        "- `not_attempted`: a lookup or summarization failed before the final report request.",
        "- `summary_failed`: the summary stayed empty, truncated, or oversized after its retry allowance.",
        "", "**Token counts:** Actual Jan API `usage` when provided; otherwise rough chars/4 estimates, "
        "not tokenizer counts. Avoid comparing estimated and measured values as equivalent.",
        "**Fairness:** Same model, seed, synthetic cases, final-answer criteria, and output cap across three modes. "
        "Summarization uses real extra Jan inference calls, counted in time and token totals. The same original "
        "instructions are protected in all three modes; summarized mode must retain the seals in its model-written notes. "
        "Model sampling can still vary; compare multiple seeds/repeats. Branch selection is controlled, not autonomous.",
        "**Modes:** linear retains entire history; summarized periodically replaces old steps with a note "
        "written by the same model; branch isolates each case and returns an outcome. "
        f"Summarized cadence: every {args.summary_every} lookups, memory budget {args.summary_max_chars} characters. "
        f"Summary cap: {args.summary_output_tokens} tokens; retry allowance: {args.summary_retries}. "
        "Summaries, retry events and per-call traces are in results.json.\n",
        "**Limitations:** Repair scenario uses a VIRTUAL simulator, not real files, browser actions, shell, or credentials. "
        "Branching is selected by benchmark orchestration, not autonomously. Nested dependencies happen INSIDE each case's active context (the harness does not spawn nested branches for them). No tool inheritance or real Jan UI/MCP tool-selection evaluation. "
        "A server may silently truncate oversized inputs.",
        "", "The CSV contains all per-trial outcome and score fields. The JSON additionally records per-call finish reasons and per-batch judgments.",
    ]
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main(argv=None):
    args = arguments(argv)
    combos = [(n, args.seed + i * 10007 + n * 313) for n in args.tasks for i in range(args.repeats)]
    from trap_benchmark import create_trap_puzzle, run_trap_trial
    for n, seed in ([] if args.check_jan_reset or args.check_llama_server else combos):
        if args.scenario == "archive":
            puzzle = create_puzzle(seed, n, args.records)
            print(f"Scenario OK: {n} lookups x {args.records} rows; max side prompt {max(len(s) for _, s, _ in puzzle['tasks'])} chars", flush=True)
        else:
            puzzle = create_trap_puzzle(seed, n, args.trap_log_lines, args.trap_density, args.recovery_log_lines,
                                        args.nested_depth, args.nested_log_lines,
                                        args.pressure_records, args.pressure_detail_chars)
            mandatory_paths = {"registry/releases", "registry/approvals", "network/aliases",
                               "network/routes", "security/certificates"} | {
                               path for path in puzzle["cases"][0]["docs"]
                               if "/shard-" in path}
            required_chars = sum(len(puzzle["cases"][0]["docs"][path]) for path in mandatory_paths)
            largest_doc_chars = max(len(doc) for doc in puzzle["cases"][0]["docs"].values())
            print(f"  Required evidence per incident: ~{required_chars//4:,} estimated tokens of text "
                  f"({required_chars:,} chars), largest document ~{largest_doc_chars//4:,} tokens; "
                  f"injected extra chars={sum(puzzle['cases'][0]['pressure_deltas'].values()):,}.", flush=True)
            estimated_evidence = required_chars // 4
            # This is intentionally conservative: 4 chars/token is only a rough estimate
            # and excludes system instructions, actions, chat framing and prior responses.
            if estimated_evidence + args.max_output_tokens >= args.context_limit * 0.8:
                print("  WARNING: Mandatory evidence plus reserved output consumes much of the context budget "
                      f"(~{estimated_evidence:,} + {args.max_output_tokens:,} estimated tokens, "
                      f"before instructions/history; server limit {args.context_limit:,}). "
                      "All three modes may hit HTTP 400 inside a single repair. "
                      "Use smaller --recovery-log-lines / --pressure-records / --nested-depth, "
                      "or increase --context-limit.", flush=True)
            print(f"Scenario OK: {n} simulated repairs; traps={args.trap_density}, optional lines={args.trap_log_lines}, mandatory shard lines={args.recovery_log_lines}; "
                  f"nested depth={args.nested_depth}, nested lines={args.nested_log_lines}, "
                  f"pressure records={args.pressure_records}, detail chars={args.pressure_detail_chars}, "
                  f"evidence docs={len(puzzle['cases'][0]['docs'])}", flush=True)
    if args.dry_run:
        print("DRY RUN: no inference server contacted; reset feature is not executed.")
        return 0
    managed_server = None
    if args.engine == "llama":
        if not args.llama_model:
            print("Missing GGUF model path: set MODEL_PATH in local_model.json (copy local_model.example.json), "
                  "export CONTEXT_BRANCH_MODEL, or specify --llama-model /path/to/model.gguf.", file=sys.stderr)
            return 2
        managed_server = llama_runner.LocalLlamaServer(
            args.llama_server, args.llama_model, args.llama_port, args.context_limit,
            args.llama_gpu_layers, args.llama_startup_timeout, args.llama_arg)
        try:
            spec = managed_server.preflight()
            managed_server._ensure_port_free()
        except llama_runner.ServerError as exc:
            print(f"LLAMA PREFLIGHT FAILED: {exc}", file=sys.stderr)
            return 2
        print(f"Dedicated llama.cpp: {spec['binary']} | model: {spec['model']} | port: {args.llama_port}")
        if args.check_llama_server:
            print("LLAMA PREFLIGHT OK. No server started and no processes stopped.")
            return 0
        # Deliberately do not read Jan configuration or send requests to Jan.
        cfg = dict(bc.DEFAULTS)
        cfg.update({"api_base_url": managed_server.base_url + "/v1", "api_key": "", "max_branch_depth": 0,
                    "max_agent_turns": 1, "max_output_tokens": args.max_output_tokens,
                    "save_transcripts": False, "read_only_project_root": ""})
        model = "branch-benchmark"  # discovered from the dedicated API after startup
    else:
        try:
            cfg = bc.config()
            cfg["max_branch_depth"] = 0
            cfg["max_agent_turns"] = 1
            cfg["max_output_tokens"] = args.max_output_tokens
            cfg["save_transcripts"] = False
            cfg["read_only_project_root"] = ""
            model = bc.chosen_model(cfg)
        except Exception as exc:
            print(f"Jan connection error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    cfg["_recursive_max_depth"] = args.recursive_max_depth
    cfg["_no_action_retries"] = args.no_action_retries
    cfg["_no_action_retry_output_tokens"] = args.no_action_retry_output_tokens
    cfg["_no_action_retry_reasoning_effort"] = (
        "none" if args.engine == "llama" and args.no_action_retry_reasoning == "none" else None)
    cfg["_final_output_tokens"] = args.final_output_tokens
    cfg["_final_retries"] = args.final_retries
    cfg["_summary_output_tokens"] = args.summary_output_tokens
    cfg["_summary_retries"] = args.summary_retries
    # Per-request control is best-effort: older llama.cpp builds may ignore it.
    cfg["_report_reasoning_effort"] = "none" if args.engine == "llama" and args.report_reasoning == "none" else None
    resetter = None
    if args.engine == "jan" and (args.restart_jan_between_trials or args.restart_jan_between_modes or args.check_jan_reset):
        try:
            resetter = jan_reset.JanModelReset(
                model, cfg, router_url=args.jan_router_url,
                cooldown=args.cooldown_seconds, timeout=args.reset_timeout_seconds)
            info = resetter.check()
            print(f"Jan router found: {info['router_url']} | model: {info['router_model']} | status: {info['router_status']}", flush=True)
        except jan_reset.ResetError as ex:
            print(f"RESET PREFLIGHT FAILED: {ex}", file=sys.stderr)
            return 2
    if args.check_jan_reset:
        print("RESET CHECK OK: no model was unloaded and no trials were started.")
        return 0
    output = args.output or RESULTS / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.scenario)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        print(f"Output folder isn't empty: {output}. Choose --output another/folder", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    original_data = bc.DATA
    rows = []
    reset_events = []
    previous_mode = None
    try:
        if managed_server is not None:
            managed_server.log_dir = output / "server_logs"
        for n, seed in combos:
            # Rotate the three competitors to limit warm-up and ordering bias.
            ordered = tuple(args.modes)
            offset = ((seed - args.seed - n * 313) // 10007) % len(ordered)
            modes = ordered[offset:] + ordered[:offset]
            for mode in modes:
                # Reset only at trial boundaries, NEVER midway through a repair.
                should_reset = (resetter is not None and
                                (args.restart_jan_between_trials or previous_mode != mode))
                event = None
                if managed_server is not None:
                    # A new, owned inference process for every trial. Do not touch Jan.
                    if not args.reuse_llama or managed_server.process is None:
                        managed_server.stop()
                        print(f"Starting fresh llama-server before {mode} trial...", flush=True)
                        try:
                            event = managed_server.start()
                            event.update({'next_mode': mode, 'next_seed': seed, 'next_tasks': n})
                            reset_events.append(event)
                            print(f"  model ready in {event['elapsed_seconds']}s; PID {event['pid']} (excluded from trial runtime)", flush=True)
                        except Exception as ex:
                            event = {'status': 'failed', 'method': 'fresh_llama_process',
                                     'next_mode': mode, 'next_seed': seed, 'next_tasks': n, 'error': str(ex)[:300]}
                            reset_events.append(event)
                            write_files(output, rows, cfg, args, reset_events)
                            print(f"Dedicated llama.cpp start FAILED: {ex}. Partial results saved.", file=sys.stderr)
                            return 3
                    model = managed_server.model_id
                if should_reset:
                    print(f"Resetting Jan model before {mode} trial...", flush=True)
                    try:
                        event = resetter.reset()
                        event.update({'next_mode': mode, 'next_seed': seed, 'next_tasks': n})
                        reset_events.append(event)
                        print(f"  reset verified in {event['elapsed_seconds']}s (excluded from trial runtime)", flush=True)
                    except jan_reset.ResetError as ex:
                        event = getattr(ex, 'reset_record', {'status': 'failed', 'elapsed_seconds': 0})
                        event.update({'next_mode': mode, 'next_seed': seed, 'next_tasks': n,
                                      'error': str(ex)[:300]})
                        reset_events.append(event)
                        write_files(output, rows, cfg, args, reset_events)
                        print(f"Jan reset FAILED: {ex}. Stopping; partial results saved.", file=sys.stderr)
                        return 3
                previous_mode = mode
                print(f"Running {mode:10s}: {n} {args.scenario} cases, seed {seed}", flush=True)
                if args.scenario == "archive":
                    result = run_trial(mode, seed, n, args.records, cfg, model, args.context_limit,
                                       output / f"{n}_{seed}_{mode}", args.summary_every, args.summary_max_chars)
                else:
                    result = run_trap_trial(mode, seed, n, cfg, model, args.context_limit,
                                            output / f"{n}_{seed}_{mode}",
                                            sys.modules[__name__], args.summary_every, args.summary_max_chars,
                                            args.trap_log_lines, args.trap_density, args.max_actions, args.recovery_log_lines,
                                            args.nested_depth, args.nested_log_lines,
                                            args.pressure_records, args.pressure_detail_chars)
                result.update({
                    'reset_before_trial': bool(event),
                    'reset_status': event['status'] if event else 'not_requested',
                    'reset_seconds': event['elapsed_seconds'] if event else 0,
                    'reset_method': event['method'] if event else 'none',
                    'reset_memory_before_mb': event.get('memory_available_before_mb') if event else None,
                    'reset_memory_after_unload_mb': event.get('memory_available_after_unload_mb') if event else None,
                    'reset_memory_after_reload_mb': event.get('memory_available_after_reload_mb') if event else None,
                })
                if managed_server is not None:
                    # Detect unexpected model crashes rather than treating them as reasoning failures.
                    crashed = managed_server.process is None or managed_server.process.poll() is not None
                    if not args.reuse_llama:
                        managed_server.stop()  # Frees allocated VRAM and KV cache between trials.
                    if crashed:
                        result['inference_process_crashed'] = True
                rows.append(result)
                write_files(output, rows, cfg, args, reset_events)
                if managed_server is not None and result.get('inference_process_crashed'):
                    print("Dedicated inference process crashed; stopping. See server_logs. Partial results saved.", file=sys.stderr)
                    return 3
                gc.collect()  # Release Python trial objects; never clear a live model context.
                print(f"  final={result['final_status']}, lookups={result['subtask_accuracy']}/{result['solved_subtasks']}, visible={result['visible_seals_correct']}/{n}, peak-main={result['peak_main_prompt_tokens']}, sec={result['elapsed_seconds']}, error={result['error'] or '-'}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted: partial results saved.", file=sys.stderr)
        return 130
    finally:
        if managed_server is not None:
            managed_server.stop()  # Always clean up ONLY the subprocess spawned here (also on Ctrl+C).
        bc.DATA = original_data
    print(f"\nSaved: {output / 'SUMMARY.md'}\n       {output / 'results.csv'}\n       {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
