"""Persistent run budgets and lossless conversation maintenance for Jan branches.

No model answers or external state are invented here. Token estimates include
tool schemas and are calibrated against observed API usage; they are not a
replacement for the inference server's tokenizer or context limit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time


LIMIT_KEYS = ("context_limit_tokens", "context_safety_tokens", "max_total_model_calls",
              "max_total_branches", "max_total_tokens", "max_total_seconds")
CHECKPOINT_NOTE = ("CONTEXT CHECKPOINT: identical read-only observations were deduplicated. "
                   "Unique observations, delegated outcomes and task instructions are retained.")
READ_TOOLS = {"read_project_file", "search_project_files"}


class RunLimitExceeded(RuntimeError):
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


class DelegationRefused(ValueError):
    pass


def task_fingerprint(task, context=""):
    # The same task is not a smaller independent problem merely because its
    # capitalization/spacing differs. Distinct supplied evidence can matter.
    normalized = [" ".join(str(value).casefold().split()) for value in (task, context)]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode()).hexdigest()


def task_identity(task):
    return " ".join(str(task).casefold().split())


def estimate_prompt_tokens(messages, tools=None):
    body = {"messages": messages}
    if tools:
        body["tools"] = tools
    size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    return math.ceil(size / 3) + 8 * len(messages) + 32


class RunBudget:
    """One ledger shared by descendants and later resumes of the same root.

    Reserve the entire requested generation allowance before sending a request.
    Reconcile with API usage afterwards; keep the reservation on an API failure.
    Idle time between MCP calls is excluded from the cumulative active deadline.
    """

    def __init__(self, root_id, cfg, saved=None, persist=None):
        self.root_id = root_id
        self.limits = {key: cfg[key] for key in LIMIT_KEYS}
        self.state = {"model_calls": 0, "branches": 0, "total_tokens": 0,
                      "api_usage_calls": 0, "estimated_usage_calls": 0,
                      "elapsed_seconds": 0, "prompt_ratio": 1.0,
                      "task_fingerprints": [], "checkpoints": 0,
                      "deduplicated_reads": 0, "no_action_retries": 0,
                      "duplicate_delegations_blocked": 0}
        if saved:
            if saved.get("root_id") != root_id:
                raise ValueError("Run budget belongs to a different branch root")
            self.limits.update(saved["limits"])
            self.state.update(saved["state"])
        self._prior_elapsed = self.state["elapsed_seconds"]
        self._started = time.monotonic()
        self.persist = persist

    def elapsed(self):
        return self._prior_elapsed + max(0, time.monotonic() - self._started)

    def snapshot(self):
        state = copy.deepcopy(self.state)
        state["elapsed_seconds"] = self.elapsed()
        return {"version": 1, "root_id": self.root_id,
                "limits": dict(self.limits), "state": state}

    def flush(self):
        if self.persist:
            self.persist(self.snapshot())

    def public(self):
        data = self.snapshot()
        state = data.pop("state")
        state.pop("task_fingerprints", None)
        data.update(state)
        data["elapsed_seconds"] = round(data["elapsed_seconds"], 3)
        data["token_counts_exact_from_api"] = (self.state["model_calls"] > 0
                                               and self.state["estimated_usage_calls"] == 0
                                               and not self.state.get("pre_upgrade_usage_unknown"))
        return data

    def check(self):
        if self.elapsed() >= self.limits["max_total_seconds"]:
            raise RunLimitExceeded("run_time_limit", "The branch tree exhausted its active-time budget")
        if self.state["total_tokens"] > self.limits["max_total_tokens"]:
            raise RunLimitExceeded("run_token_limit", "The branch tree exhausted its total token budget")

    def remaining_seconds(self):
        self.check()
        return max(0.001, self.limits["max_total_seconds"] - self.elapsed())

    def add_branch(self, fingerprint):
        self.check()
        if fingerprint in self.state["task_fingerprints"]:
            self.state["duplicate_delegations_blocked"] += 1
            self.flush()
            raise DelegationRefused("This identical task and context were already delegated in this run")
        if self.state["branches"] >= self.limits["max_total_branches"]:
            raise RunLimitExceeded("run_branch_limit", "The branch tree exhausted its total branch budget")
        self.state["branches"] += 1
        self.state["task_fingerprints"].append(fingerprint)
        self.flush()

    def predicted_prompt(self, messages, tools=None):
        return math.ceil(estimate_prompt_tokens(messages, tools) * self.state["prompt_ratio"])

    def before_call(self, prompt_tokens, output_tokens):
        self.check()
        if self.state["model_calls"] >= self.limits["max_total_model_calls"]:
            raise RunLimitExceeded("run_call_limit", "The branch tree exhausted its total model-call budget")
        if prompt_tokens + output_tokens + self.limits["context_safety_tokens"] > self.limits["context_limit_tokens"]:
            raise RunLimitExceeded("context_budget", "Task instructions and unique observed evidence exceed the prompt budget; read smaller ranges or split the task")
        reserved = prompt_tokens + output_tokens
        if self.state["total_tokens"] + reserved > self.limits["max_total_tokens"]:
            raise RunLimitExceeded("run_token_limit", "The next request would exceed the branch tree's total token budget")
        self.state["model_calls"] += 1
        self.state["estimated_usage_calls"] += 1
        self.state["total_tokens"] += reserved
        self.flush()
        return {"reserved": reserved, "prompt": prompt_tokens, "output": output_tokens,
                "ratio": self.state["prompt_ratio"]}

    def after_call(self, reservation, usage):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        valid_prompt = type(prompt) is int and prompt >= 0
        valid_completion = type(completion) is int and completion >= 0
        actual = (prompt if valid_prompt else reservation["prompt"]) + (
            completion if valid_completion else reservation["output"])
        self.state["total_tokens"] += actual - reservation["reserved"]
        if valid_prompt and valid_completion:
            self.state["api_usage_calls"] += 1
            self.state["estimated_usage_calls"] -= 1
        if valid_prompt:
            raw_estimate = reservation["prompt"] / reservation["ratio"]
            self.state["prompt_ratio"] = max(self.state["prompt_ratio"], prompt / max(1, raw_estimate))
        self.flush()
        self.check()


def normalize_conversation(messages, system):
    """Keep complete assistant/tool exchanges; remove orphan legacy fragments.

    Save-time slicing in older releases could start a transcript with a tool
    result. Do not submit that invalid fragment to OpenAI-compatible servers.
    """
    result = [{"role": "system", "content": system}]
    dropped = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            dropped += 1
            index += 1
            continue
        role = message.get("role")
        if role == "system":
            index += 1
            continue
        if role == "tool":
            dropped += 1
            index += 1
            continue
        calls = message.get("tool_calls") if role == "assistant" else None
        if calls:
            ids = [call.get("id") for call in calls if isinstance(call, dict)]
            following = messages[index + 1:index + 1 + len(calls)]
            valid = (len(ids) == len(calls) and all(isinstance(i, str) and i for i in ids)
                     and len(set(ids)) == len(ids) and len(following) == len(calls)
                     and all(isinstance(m, dict) and m.get("role") == "tool"
                             and isinstance(m.get("tool_call_id"), str) for m in following)
                     and {m.get("tool_call_id") for m in following} == set(ids))
            if valid:
                result.extend(copy.deepcopy([message, *following]))
                index += len(calls) + 1
                continue
            dropped += 1
        elif role in ("user", "assistant"):
            result.append(copy.deepcopy(message))
        else:
            dropped += 1
        index += 1
    return result, dropped


def observation_key(call, reply):
    spec = call.get("function") or {}
    if spec.get("name") in READ_TOOLS:
        try:
            arguments = json.loads(spec.get("arguments") or "{}")
            return json.dumps([spec["name"], arguments, reply.get("content")],
                              sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            pass
    return None


def deduplicate_reads(messages):
    """Remove only exactly repeated read-only tool observations.

    Distinct versions of a source and every nested-branch result remain intact.
    Tool calls and their replies are removed together, including multi-call turns.
    """
    # Keep the LAST occurrence: A -> B -> A must still end in observation A.
    latest = {}
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if calls:
            replies = {m["tool_call_id"]: m for m in messages[index + 1:index + 1 + len(calls)]}
            for position, call in enumerate(calls):
                key = observation_key(call, replies[call["id"]])
                if key is not None:
                    latest[key] = (index, position)
    result = []
    removed = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            if message.get("content") != CHECKPOINT_NOTE:
                result.append(copy.deepcopy(message))
            index += 1
            continue
        replies = messages[index + 1:index + 1 + len(calls)]
        by_id = {reply["tool_call_id"]: reply for reply in replies}
        kept = []
        kept_replies = []
        for position, call in enumerate(calls):
            reply = by_id[call["id"]]
            key = observation_key(call, reply)
            if key is not None and latest[key] != (index, position):
                removed += 1
                continue
            kept.append(copy.deepcopy(call))
            kept_replies.append(copy.deepcopy(reply))
        if kept:
            assistant = copy.deepcopy(message)
            assistant["tool_calls"] = kept
            result.extend([assistant, *kept_replies])
        elif message.get("content"):
            result.append({"role": "assistant", "content": message["content"]})
        index += len(calls) + 1
    if removed:
        result.append({"role": "user", "content": CHECKPOINT_NOTE})
    return result, removed


def finish_interrupted_exchange(messages, reason):
    """Close pending tool calls with an explicit non-execution result on stop."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        done = {m.get("tool_call_id") for m in messages[index + 1:] if m.get("role") == "tool"}
        for call in calls:
            if call["id"] not in done:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": "Not executed: branch stopped (" + reason + ")"})
        break
