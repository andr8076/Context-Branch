# Jan runtime: v1.5.1

The MCP runner now checks the complete conversation before each model request,
keeps valid tool exchanges across resumes, and uses one persistent budget for a
root branch and all its descendants. Existing `config.json` files continue to
work; missing settings use the defaults below.

Version 1.5.1 fixes recovery of empty or unfinished final responses. A short
tool-action retry and a complete five-field answer need different allowances.
Final-only requests and forced final reports now use `final_output_tokens`.
Empty, truncated or malformed final replies retry with at least that allowance
and never receive less than the preceding request's output cap. Malformed tool
calls without cutoff evidence still use the short action-retry allowance.

## Update and check your installation

```bash
cd ~/Documents/Github/Context-Branch
git switch main
git pull --ff-only
python3 branch_context.py doctor
python3 branch_context.py demo
```

Enable Jan's Local API Server and load your model before the last two commands.
Keep your existing API URL/key settings. `doctor` contacts `/v1/models` and checks
that the selected model is advertised, including when `model` is explicitly set.
It does not generate text; `demo` tests an actual generation request. A successful
demo returns a branch with `status: success` and a short glaze recipe.

After pulling, restart the Branch Context MCP server in Jan (restarting Jan also
works), so it imports the updated Python files. The MCP command stays the same.
The standalone llama.cpp benchmark engine and `local_model.json` remain separate
from these Jan API settings.

If Jan is configured for a context size other than 8192 tokens, set
`context_limit_tokens` in `config.json` to that size or a smaller guard value.
This setting does not resize Jan's actual context window. Edit individual keys;
do not replace a working configuration with the example and lose your API key.

## Limits

| Setting | Default | Meaning |
| --- | ---: | --- |
| `context_limit_tokens` | 8192 | Per-request prompt + requested generation + safety reserve |
| `context_safety_tokens` | 256 | Reserved for tokenizer and chat-format differences |
| `max_total_model_calls` | 24 | All requests across the root, children, retries and resumes |
| `max_total_branches` | 8 | Total branch contexts, including the root |
| `max_total_tokens` | 50000 | Total prompt and completion tokens processed across the tree |
| `max_total_seconds` | 600 | Cumulative active runtime, excluding idle time between MCP calls |
| `final_output_tokens` | 2048 | Generation allowance for final-only requests and final-response recovery |
| `no_action_retries` | 1 | Additional attempts after an empty or unusable response |
| `retry_output_tokens` | 256 | Short action-recovery cap; does not apply to empty/truncated final replies |

The existing `max_agent_turns` and `max_branch_depth` still apply. An identical
task/context cannot create another child in the same run; a child cannot repeat
an ancestor's task with different context. All nested work uses the original root
limits. `branch_resume` and explicit `parent_id` calls reuse that root's ledger.
Changing config does not replenish an existing run. Start a new root branch for a
new allowance.

Before an API request, the runner reserves its estimated prompt and full requested
generation allowance. It reconciles that reservation with API-reported usage
after the response. If usage is missing or a request fails, the unknown portion
stays conservatively reserved. Failed calls and recovery attempts cost budget.
The larger final allowance must also fit the remaining context and shared run
budget; recovery cannot bypass any limit.

Prompt estimation includes tool schemas, UTF-8 content and message overhead. It
is calibrated upward using actual prompt-token usage when available. This is an
estimate, not the exact model tokenizer; the server still enforces its real limit.
Time limits are cooperative checks between operations, with each HTTP timeout
clamped to the remaining allowance. They are not process termination timers.

## What the guardrails do and do not establish

The configured context limit is an operational target supplied by you; the
runtime does not query Jan for the selected model's exact context window or
tokenizer. Before a request, it compares an estimated prompt size plus the full
requested output and safety reserve against that target. The estimate includes
message and tool-schema overhead, but it can over- or under-estimate the server's
actual tokenization.

| Signal | What it means | What it does not mean |
| --- | --- | --- |
| `stop_reason: context_budget` | The runtime's estimate exceeded the configured guard, so it stopped before sending that request. | Jan confirmed an overflow; the actual request would necessarily have failed. |
| `context_overflow_confirmed` | The server error text matched the runtime's context-capacity classifier. | Independent proof of the server's internal token count; classification depends on the error the server returned. |
| `usage.total_tokens` / per-call usage | API-reported usage when supplied; otherwise the reservation/estimate is retained. | An independently verified exact count. Actual usage can exceed the reservation; an over-budget request is detected after its response, not prevented in advance. |
| `output_truncated_confirmed` | The API reported a length stop for an incomplete response. | That the prompt exceeded the context window; output limits and context limits are distinct. |
| `status: success` | The model returned a success status in its report. | That the result is factually correct or externally verified. |

Treat guard stops as conservative control decisions, not measurements. Treat
server-classified errors as diagnostic evidence, not a substitute for inspecting
the server log. Leave headroom in `context_limit_tokens`; if the model's actual
context size is unknown, use a smaller configured guard and verify behavior with
representative requests. Review the evidence behind any important model-reported
conclusion.

Ledgers are private JSON files under `data/branches/budgets/`; this directory is
ignored by Git. Old branches begin a ledger on their first post-upgrade use.
Earlier usage cannot be reconstructed and is marked `pre_upgrade_usage_unknown`.

## Context checkpoints and file ranges

When approaching the request budget, the runner can remove identical repeated
results from its read-only project tools. It removes a tool call together with
its matching result, keeps the latest occurrence, and preserves distinct versions
of a source, original instructions and every child outcome. Checkpoints make no
model requests and invent no summaries or observations.

If unique evidence still cannot fit, the branch returns `blocked` with
`stop_reason: context_budget` before sending the oversized estimated request.
Unique evidence is not silently dropped. Read smaller ranges or start a smaller
side task. The model can now use:

```json
{"path":"src/example.py","start_line":90,"max_lines":30}
```

The read result includes line numbers and the next starting line when more text
is available. An individual oversized line is explicitly marked as truncated.
Project tools remain disabled unless `read_only_project_root` is configured.

## Resumes, results and diagnostics

With `save_transcripts: true`, saved history preserves complete assistant/tool
exchanges instead of keeping an arbitrary final 22 messages. Incomplete fragments
from older versions are removed before resuming, with an explicit instruction to
re-inspect necessary sources. With transcript saving off, resumes still use only
the saved result and new follow-up context.

Every result includes the current shared `usage` and a `stop_reason` when stopped
by a limit. `branch_get` reads current ledger usage even if another child has run
since that branch completed. Other stop reasons include `run_call_limit`,
`run_branch_limit`, `run_token_limit`, `run_time_limit`, and `no_usable_response`.
An exhausted cutoff is identified as `output_truncated_confirmed` when the API
reports a length stop, or `output_truncated_suspected` when usage reaches the cap
without an explicit stop reason. Incomplete JSON alone does not establish why it
stopped. Fully emitted top-level result text can be retained as partial evidence;
missing fields and unfinished strings are never fabricated into a success.

Returned `response_diagnostics` shows the last response status, first failure,
output cap/usage, finish reason, reported reasoning usage and retry recovery.
This is available even with transcript saving disabled. If the demo still fails,
send its output so these fields can distinguish cutoff, malformed output and
empty visible output. No configuration change is needed to enable these defaults.

The stored branch record's `runtime.calls` contains requested generation caps,
available API usage, finish reasons, empty-visible-output flags, retry flags,
latency and error types. It does not save reasoning text. Full transcripts are
still opt-in. A resource stop retains completed tool observations in saved
transcripts and marks pending calls as not executed.

`status: success` is the model's report. The handoff explicitly identifies it as
`model_reported`; project reads do not independently prove its conclusion.
Because all branch tools are read-only, `what_changed` is always `Nothing`.
The parent must verify conclusions and perform any required external actions.

The experimental benchmark and its simulator are separate from the MCP runtime.
Benchmark trials that exercise a real branch use an explicit request transport
and an isolated trial store; they do not change the runtime's API or state
settings. See [BENCHMARK.md](BENCHMARK.md) for experiment-specific behavior and
validation.

## Validation and next experiment

```bash
python3 -m unittest discover -s tests -q
```

Regression tests exercise the actual MCP runner with controlled model doubles:
repeated reads/resumes, shared nested budgets, deadline/token accounting,
duplicate/cycle rejection, partial stops, health checks, range reads and recovery.
These validate code behavior, not live Qwen accuracy or speed.
