# Evidence-gated repairs and reproducible context-pressure experiment

**Simulator rule update:** every repair now requires the model to inspect all five authoritative documents: `registry/releases`, `registry/approvals`, `network/aliases`, `network/routes`, and `security/certificates`, even when `--pressure-records 0`. Correct guesses without evidence are rejected and counted as `authority_blocked`. This changes the task protocol relative to older runs; do not compare pass rates across simulator-rule versions without noting the difference. `results.json` now includes `simulator_rules_version=2` and `mandatory_authority_inspection=true`.

**Live local model pressure comparison:** `checkpoint_stress.py` runs two Qwen/llama.cpp experiments with the *same model, case seed, evidence and action limits*, in independently launched model-server processes. Only the recursive checkpoint threshold changes. The script does not inject arbitrary padding or fake evidence; it reports when no checkpoint fired rather than claiming success.

```bash
cd ~/Documents/Github/Context-Branch
python3 checkpoint_stress.py
```

This defaults to 4,096 context tokens, 1 plausible extra record per evidence source, 8 recovery and nested log lines, 40 actions, and checkpoint thresholds **0 (off)** vs **0.50 (on)**. Override with e.g. `--context-limit 6144 --pressure-records 3 --max-actions 48 --checkpoint-threshold 0.55`. The GGUF path comes from ignored `local_model.json`; `--llama-model` can override it.

The script creates `data/benchmarks/pressure-study-<UTC timestamp>/off`, `.../on`, and `comparison.json`. A successful proof requires a **confirmed context overflow with checkpointing off**, a **checkpoint actually created with it on**, and a **verified final repair** in the checkpoint-on run. If both succeed or no checkpoint is created, the result is **inconclusive about necessity**, not a failure of the code. A checkpoint may also legitimately report that the unique required evidence cannot fit.

Offline regression tests also simulate a hard 4,096-token API limit and have the same *model double* repeatedly inspect authentic simulator documents in each arm. This tests checkpoint reconstruction and the actual error path but **is not a Qwen performance measurement**. Run with `python3 -m unittest discover -s tests -q`.

---

# v1.4.4 — pressure-aware recursive checkpoints and failure snapshots

Recursive repair stages can now **restart their private conversation before exhausting its context window**. The controller estimates upcoming prompt size, using actual API prompt-token counts when available. At the configured pressure threshold it rebuilds that stage's message list from:

- The original stage-only system instructions and *verified* child handoff.
- Exact simulator observations for documents already inspected by **that stage** (one copy per path).
- The last non-inspection action outcome, if present.

No expected repair, hidden oracle, fabricated summary, or child's transcript is inserted. Environment state is not reset, and checkpointing performs **zero simulator actions**. All model calls and tokens remain counted. If all previously observed evidence itself cannot fit, the stage fails explicitly instead of silently dropping it.

The default is `--recursive-checkpoint-threshold 0.78` (fraction of the context limit), `--recursive-checkpoint-max 4` per stage. Set `--recursive-checkpoint-threshold 0` to reproduce legacy behavior. Checkpoint details are in `recursive_events[].stages[].checkpoint_details`; totals appear as `recursive_checkpoints` and `recursive_checkpoint_blocked`. Pressure estimates are conservative heuristics and can still miss tokenizer differences, so these settings do **not** guarantee no context overflow.

When a simulator trial fails midway through an incident, its **partial simulator actions, completed prerequisites, document reads and nested stage events now survive in `results.json`**, marked `partial_failure` / `interrupted_batch`. An unverified incident remains unverified; the run does not fabricate a successful final report.

Compare controlled recursion with and without checkpointing using the **same seed**, changing only the threshold:

```bash
python3 benchmark.py --engine llama --scenario repair \
  --modes recursive --tasks 1 --repeats 1 --seed 20261026 \
  --context-limit 8192 --nested-depth 2 --nested-log-lines 8 \
  --recovery-log-lines 8 --pressure-records 0 --max-actions 32 \
  --stage-start-inspects 2 --recursive-checkpoint-threshold 0.78
```

No live Qwen improvement is claimed until this configuration is run. Autonomous branching is still a separate future milestone.

---

# v1.4.3 — low-overhead recursive stage starts

New recursive contexts now begin with a **limited, non-thinking inspection phase**: the model itself selects the first two *in-scope* document inspections, each with a 256-token output cap and a best-effort `reasoning_effort=none` hint on llama.cpp. This avoids spending the full 1,024-token investigation allowance on trivial guide/initial-evidence reads.

After two successful inspections, or immediately when the model chooses a non-inspection/invalid/out-of-scope action, the stage returns to **normal full-reasoning inference**. It never skips simulator actions, invents document contents, shares child histories, or changes the shared action budget. Unusable stage-start responses still use v1.4.2's bounded no-action retry path.

CLI settings: `--stage-start-inspects 2` (default; `0` disables), `--stage-start-output-tokens 256`, and `--stage-start-reasoning none` (use `auto` to omit the hint). This optimization only applies to `recursive` repair mode. Results include `stage_start_calls`, `stage_start_inference_seconds`, `stage_start_total_tokens`, `stage_start_empty_visible_calls` and per-stage `stage_start_inspections`.

To run the same scenario with and without the optimization, use the following command twice, changing just `--stage-start-inspects 2` to `0` for the control:

```bash
python3 benchmark.py --engine llama --scenario repair \
  --modes recursive --tasks 1 --repeats 1 --seed 20261026 \
  --context-limit 8192 --nested-depth 2 --nested-log-lines 8 \
  --recovery-log-lines 8 --pressure-records 0 --max-actions 32 \
  --no-action-retries 1 --stage-start-inspects 2
```

Results from scripted agents or GitHub Actions validate plumbing, **not** live Qwen performance. Compare verified repairs, context, tokens, *and wall-clock time* across several seeds before evaluating effectiveness.

---

## v1.4.1 controlled-recursion scope and no-action rules

- The controlled recursive solver gives each stage a stage-specific protocol and a filtered view of the *existing visible* `guide/index`; index/escrow/recovery/repair cannot read another stage's shards or execute its completion action. The simulator rejects wrong-stage actions before executing them and records `stage_scope_rejected`. Rejecting an out-of-scope action still consumes one shared action budget slot.
- After each verified child, its parent receives the stage name, a simulator-verified completion flag, and an explicit environment change (e.g. `registry/approvals unlocked`). Parent contexts never receive child investigation messages, tokens, or benchmark oracle answers.
- All four repair modes use the same configurable `--no-action-retries` (0–2, default 1). When a model produces empty/malformed/unusable JSON, the harness makes a bounded additional inference request with a brief reminder. This additional request counts in API calls, tokens, and elapsed time, but **doesn't call the simulator or spend a simulator action until a valid action is produced or all attempts fail**.
- The run records `no_action_investigation_calls` (unusable generations), `no_action_retry_attempts` (actual retry requests), `no_action_retry_recoveries`, `no_action_retry_failures`, `out_of_scope_actions`, and per-stage `rejected_out_of_scope`. `empty_investigation_replies` includes retry calls.
- These changes are benchmark orchestration, **not** autonomous branch selection and do not claim a successful live Qwen benchmark. Inspect totals across all branches and inference retries when comparing modes.

---

## v1.4 — controlled recursion and investigation diagnostics

The `recursive` strategy creates independent model histories for dependency
stages inside one repair: root repair → recovery → escrow → index. A child
updates the same simulator as its parent. Ancestors receive only a short,
verified completion note; shard transcripts remain isolated. All model actions
use the **same incident budget**, including child stages. The benchmark controls
delegation based on public dependency depth; the model does not autonomously
choose branches. No oracle repair solutions are supplied in prompts.

Use `--scenario repair --modes linear summarized branch recursive` to enable
four competitors. The default remains the original three. Supported recursion
depth limit: `--recursive-max-depth 0..3`. At `--nested-depth 2`, the
full chain needs depth 3; lower limits are scored as blocked rather than
implicitly skipping dependencies. In `archive` mode, recursive is unsupported.

New per-trial fields: `recursive_branch_count` (spawned children, excluding
root), `recursive_context_count`, `recursive_max_depth`,
`recursive_stage_successes`, `recursive_stage_failures`, and detailed
`recursive_events` in results.json. Existing total API tokens, peak branch
context and elapsed time include nested calls. New no-action metrics:
`no_action_investigation_calls` counts simulator responses where the model
produced no parseable action; `empty_investigation_replies` counts empty visible
text from investigation requests. No-action recovery is a **separate future
feature**, not silently activated here.

Testing: mocked-model benchmarks exercise dependency gating, branch isolation,
shared state, action cap and a synthetic per-context overflow. A passing
mock-model test says nothing definitive about your local Qwen model. See
[ROADMAP.md](ROADMAP.md) for recursive, autonomous and no-action goals.

---

## v1.3 summarization reliability

Summarization is a model-generated inference task, not free lossless compression. It now uses a separate `--summary-output-tokens` budget (default 2048), requests `reasoning_effort=none` when the reporting policy is enabled on standalone llama.cpp, and retries once (configurable `--summary-retries`) if the generated note is empty or cut off. The original mission instructions are kept unchanged across compaction, and the model never receives correct answers from the benchmark oracle.

A valid, complete summary must also fit the separate retained-memory size limit (`--summary-max-chars`, default 1800). Oversized complete notes are rejected rather than cropped; repeated empty/truncated notes produce `summary_failed`. Additional model requests are included in all cost and timing metrics.

New detailed fields in `results.json`: `summary_events` (each compaction's `first_status`, `status`, `attempts`, `recovered_on_retry`, `chars`), plus totals `summary_attempts`, `summary_retries`, `summary_retry_recoveries` and `summary_failures`. Aggregate counters also appear in `results.csv` and the markdown summary.

To exercise compaction, use at least three repair incidents per trial with the default `--summary-every 2`. A one-incident `--smoke` run does not exercise the summarizer.

---

## v1.2 reporting and diagnostics

When benchmarking local reasoning models, output tokens used for hidden reasoning can exhaust the final-report cap without producing text. The standalone llama.cpp engine requests `reasoning_effort=none` on ACK and final-report requests (not during investigations), allows `--final-output-tokens` (default 2048), and retries an empty or truncated final once (`--final-retries 1`). All costs count toward benchmark token totals and runtime. Do not interpret a successful retry as a first-attempt success; read `final_first_status`, `final_attempts`, and `final_recovered_on_retry`.

We report `empty_visible_calls`, `reasoning_only_calls` (only when the API exposes evidence), `reported_reasoning_tokens`, and per-call `requested_reasoning_effort`. Some server versions ignore per-request reasoning controls. For compatible engines you can specify `--report-reasoning auto` to omit that hint.

Only the **model-generated report** can pass `exact_final`. `compiled_report` / `compiled_report_complete` in the repair scenario are machine-assembled from simulator-verified outcomes and do not measure model recall. These are diagnostic, not a fourth competitor.

To test summarization you must run **three or more repairs** (default summary cadence is every two *completed* incidents, except the last). `--smoke` only runs one repair per mode.

---

# New in v1.1: dedicated llama.cpp benchmark engine

**Recommended for clean benchmarking:** use `--engine llama` to run a separate
`llama-server` process with your local GGUF model. It restarts between every
trial without attempting to authenticate to Jan's internal router. It does not
stop or modify Jan. See [DEDICATED_LLAMA.md](DEDICATED_LLAMA.md) for installation
and commands for Arch Linux / AMD ROCm.

The Jan MCP extension and old `--engine jan` path remain available, but the
Jan-specific router reset is not needed for this new mode.

---

## NEW in v1.0.4: read-only router credential fallback

When Jan's worker is detected but its private `/models` endpoint returns HTTP 401,
preflight now also tries the API key already present in this project's `config.json`
(or `JAN_BRANCH_API_KEY`, when supplied). It only sends that key to a localhost port
verified as being owned by Jan's llama worker. This is a GET-only preflight and
never prints keys. **The public key may not be the router key**; if Jan uses
separate credentials, the reset remains blocked. Jan has an open router-auth
bug report (#9092) with similar symptoms: do not bypass auth or kill its worker.

Use `python3 benchmark.py --check-jan-reset` after replacing `jan_reset.py`.
If it still fails, run benchmarks without the automatic-reset flag and let Jan
maintain its normal cache; don't treat those runs as cold-cache baselines.

---

**v1.0.3 authentication update:** On Linux Flatpak, Jan normally stores its logs in `~/.var/app/ai.jan.Jan/data/Jan/data/logs/app.log`. The router detector now checks that location in addition to the native/custom data location. It prioritizes an explicit `JAN_ROUTER_API_KEY`, accepts more worker launch-argument formats, and reports *no key found* versus *candidate keys rejected*, without showing secrets. Only an exact Jan-owned port is probed and `--check-jan-reset` remains read-only. **Live validation on Build-Arch is still required.**

## Reset preflight fix (v1.0.2)

Use `python3 benchmark.py --check-jan-reset` first. The detector now distinguishes
HTTP 401/403 authentication failures from 404/not-a-router responses, and tries Jan's
private router key only when the socket belongs to the Jan worker. Keys from Jan's worker
process or a matching port-specific app log entry are used in memory only and never appear
in benchmark results. `JAN_ROUTER_API_KEY` is an optional environment fallback.

No destructive action is taken during preflight. Automatic unload/reload remains opt-in.
The 8,192-token context budget, three competitor modes, and repair tasks are unchanged.

---

# Benchmark: Does branching context actually help?

## NEW — Verified model reset at trial boundaries (v1.0)

This release adds a **model unload/reload**, not a full Jan Desktop restart, with a fail-safe stop when any step cannot be verified. It uses llama.cpp's local router `POST /models/unload` and `POST /models/load`, then verifies the router reports the model as loaded and Jan's documented `GET /v1/models` API is available.

```bash
python3 benchmark.py --scenario repair --check-jan-reset
python3 benchmark.py --scenario repair --smoke --restart-jan-between-trials
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --pressure-records 4 --max-actions 32 \
  --restart-jan-between-trials --cooldown-seconds 8
```

Reset modes: default `--reuse-jan` (no resets); `--restart-jan-between-trials` (before **every** trial, including the first); `--restart-jan-between-modes` (before first trial and whenever mode changes). `--check-jan-reset` is non-destructive. Discovery normally identifies the Jan llama.cpp router through `ss -H -lntp` and the owning worker process. If it cannot identify exactly one matching router, stop and verify its port; you may pass `--jan-router-url http://127.0.0.1:PORT` (not normally the Jan API port).

Reset events are saved in `results.json`, and each trial's reset status, elapsed seconds and system available memory diagnostics are included in CSV. Reset time is excluded from model execution time. Jan Desktop is **never killed by this feature**. After a reset failure, no further trials are started and existing results are saved; in a failure scenario you may need to reload your model in Jan manually. No Jan 0.8.7 live verification has been performed here.

---

## Previous update — Required-evidence context pressure (v0.9)

The Repair Gauntlet now adds configurable, plausibly authoritative *near-matches* inside the mandatory investigation itself, instead of simply extending optional decoy logs. The extra records are added to the **five authoritative documents** (release inventory, signed approvals, alias graph, routes, certificates), the **two recovery shards**, and both shards of each enabled **index/escrow dependency**. At default nested depth 2, this is **11 separate evidence documents per incident**. Baseline authoritative entries are randomly interleaved with extra records. Some records are locally convincing but fail a cross-document check (e.g., high-ranked ACTIVE revisions are PENDING, SIGNED revisions are RETIRED, apparently LIVE routes refer to unrelated nodes or zones, and shard rows match only one of CURRENT/run).

A repair under positive pressure **cannot succeed without inspecting all five authoritative documents**, in addition to the already mandatory recovery and nested shards. This rule is described in the simulated guide/policy; it is applied to all three competitors equally. The oracle answers, stage tokens, and verification seals remain identical for a given incident seed as you vary only pressure settings. The extra noise is generated using a separate RNG stream, so later incidents in the same mission also retain identical ground truth.

### Settings

| Flag | Default | Range | Effect |
|---|---:|---:|---|
| `--pressure-records` | 4 | 0–160 | Extra near-matching rows **per required evidence document**. Set to 0 for v0.8-style evidence. |
| `--pressure-detail-chars` | 48 | 0–256 | Length of audit/trace metadata on **each** added row. |
| `--recovery-log-lines` | 36 | 0–120 | Original required replay rows in each recovery shard. |
| `--nested-log-lines` | 14 | 0–120 | Original required replay rows in each index/escrow shard. |
| `--trap-log-lines` | 32 | 0–150 | Optional decoy/failed-repair diagnostics (not required evidence). |

```bash
# Preview size and warnings WITHOUT accessing Jan:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 --dry-run \
  --pressure-records 4 --pressure-detail-chars 48 --context-limit 8192

# Compare three methods on seeded, difficult repairs:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --pressure-records 4 --pressure-detail-chars 48 \
  --max-actions 32 --context-limit 8192

# Strong pressure: may exceed one branch's context; preflight first:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 --dry-run \
  --nested-depth 2 --pressure-records 16 --pressure-detail-chars 96
```

**New metrics:** `pressure_generated_chars` is all extra evidence created, including unread documents. `pressure_evidence_chars` is the extra evidence actually viewed (counted once per document per incident), so an AI that skips records is not credited with processing them. `authority_docs_viewed` (out of five per attempted incident) and `authority_evidence_chars` measure authoritative source inspection. `authority_blocked` counts repairs rejected because the agent skipped a required authority source. Existing `forced_evidence_chars` and `nested_evidence_chars` measure required recovery and dependency shards, independently of optional `trap_waste_chars`.

The preflight prints approximate required evidence characters and chars/4 token estimates, plus a warning if mandatory text alone is above the configured context. This is deliberately a **rough pressure indicator**, not an actual tokenizer or proof of overflow. Main/child API usage and overflow diagnostics remain in the results. Expect an individual branch to overflow at extreme settings: **v0.9 still does not create subbranches inside an incident**. This exposes exactly where that next architectural change is needed.

Offline tests: **47 passed**, including seed invariance, the authority gate, multiple modes under added pressure, edge validation, and the earlier nested-dependency tests. No live Jan performance conclusion is claimed.

---

## NEW — Mandatory nested prerequisite chain (v0.8)

The Repair Gauntlet now supports **nested depth 0, 1, or 2** using `--nested-depth` (default **2**):

- **Depth 0:** v0.7 authorization recovery only.
- **Depth 1:** escrow authorization must be reconstructed from two noisy escrow shards before the recovery shards can be read.
- **Depth 2:** a damaged evidence index must first be rebuilt from two noisy index shards. Only then can the escrow shards be read and unlocked, followed by the original recovery shards and the repair.

For each step, `guide/index` supplies a **stage-specific run ID and AB/BA order**. Each shard includes the single authoritative `CURRENT` record for that run among stale and mismatched-current decoys. Model actions are `inspect`, `rebuild_index`, `unlock_escrow`, `recover`, `repair`, and `verify`. Every stage requires **both shards inspected** plus the exact joined token. State-machine gates block later stages even if the correct answer is guessed early. Everything is synthetic and runs in a closed simulator.

The required extra noise is controlled by `--nested-log-lines` (default **14 per shard**, range 0–120). The original required recovery noise still uses `--recovery-log-lines` (default 36 per shard). Unlike optional decoy documents, these shards **must** be read, allowing us to measure compulsory context pollution fairly.

**New CSV and JSON fields:** `nested_depth`, `nested_log_lines`, `nested_index_completed`, `nested_escrow_completed`, `nested_index_attempts`, `nested_escrow_attempts`, `nested_index_failures`, `nested_escrow_failures`, `nested_blocked`, `nested_shard_views`, and `nested_evidence_chars`. `SUMMARY.md` has a dedicated *Mandatory nested dependencies* table. Required nested evidence and v0.7 recovery evidence are reported separately from optional `trap_waste_chars`.

```bash
# Short 3-way comparison, 1 incident per mode:
python3 benchmark.py --scenario repair --smoke --nested-depth 2 --nested-log-lines 8

# Repeated comparison at 1, 2, and 4 incidents:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --nested-depth 2 --nested-log-lines 14 --recovery-log-lines 36 \
  --trap-density 3 --trap-log-lines 64 --max-actions 32 --context-limit 8192

# Reproduce v0.7 dependency difficulty (but with v0.8 reporting):
python3 benchmark.py --scenario repair --tasks 1 2 4 --nested-depth 0
```

At depth 2, a direct solution requires about **18 simulated model decisions**, plus any false starts. The default maximum is now **32** actions per incident; supported range is 12–80. `--nested-log-lines` is bounded 0–120. The old `--recovery-log-lines` remains 0–120.

**Research limitation:** The benchmark still isolates contexts **at incident boundaries only**. These nested prerequisites run *inside each incident's active context*. It does **not** yet test autonomous mid-incident creation of nested subbranches, and the real local model may fail if that child context becomes full. The smoke/automated tests use simulated agents; real performance needs a live Jan benchmark run.

---

## Previous update — Mandatory recovery traps (`--scenario repair`, v0.7)

The original archive lookup test is still available and still the default. The new repair scenario is designed to test the user's original **side-quest → solve a problem → return with the outcome** concept.

A generated mission contains multiple repair incidents (`batch_1`, `batch_2`, ...). For each incident, the model has a limited number of turns to operate a **text-only, in-memory virtual repair simulator**, using JSON actions:

```json
{"action":"inspect","path":"registry/releases"}
```

```json
{"action":"recover","token":"<joined fragments>"}
```

```json
{"action":"repair","release":"REL-...","route":"ROUTE-...","certificate":"CERT-..."}
```

```json
{"action":"verify"}
```

The model must find and reconcile evidence across **recovery shards, release rankings, signatures, approval status, alias chains, routes, and certificate validity**. Specifically:

**Mandatory complication:** Every case begins with `registry/approvals` locked. The model must read `recovery/shard-a` and `recovery/shard-b`. Each contains many unrelated historical records, including deceptive records with the right run ID but wrong state, or the right state but wrong run ID. `guide/index` specifies the unique current run and a randomized `AB` or `BA` joining order. The model must select the correct `state=CURRENT` fragment for the right run from each shard and submit `action=recover` with the combined token. The simulator refuses any repair, even a correct one, until this succeeds. Shard inspection cannot be bypassed by guessing.

This compulsory work is intentionally distinct from **optional** decoy logs and wrong repairs. It provides consistent context pressure for every method. The recovery burden is adjustable with `--recovery-log-lines` (default 36 lines **per shard**). Expect more tool decisions per case than v0.6.

Additionally:

1. A more recent release can be unsigned or retired. The correct release is not necessarily the highest revision or the same rank in every incident.
2. The primary route may have an expired certificate. The agent must work through the fallback dependency, or sometimes recognize that the primary is actually valid despite an outdated alert.
3. A similarly named staged route has a valid certificate but **is not permitted**. Valid certificates alone do not establish a valid repair.
4. An old quick-fix note offers a plausible but wrong shortcut. Following it can cause a rejected repair.
5. `logs/diagnostics` contains substantial plausible noise. Opening it pollutes context; a rejected repair automatically produces verbose diagnostic logs, so an incorrect decision genuinely costs context.
6. The agent **cannot receive a verification seal** until it applies exactly the valid repair and then explicitly verifies it. The final report must retain all verified seals and the original mission ID and control code.

### Fairness and scoring

- Same model, case seeds, evidence, action rules, per-call output cap, and per-incident action budget across **linear**, **summarized**, and **branch** modes. Each run starts with an independent simulator state.
- Each mode makes its own decisions from the observations it has been shown; neither prompts nor simulated tools reveal the known correct repair. The benchmark applies/validates chosen actions deterministically.
- Linear mode retains all case-specific actions and bulky observations. Summarized mode asks the same model to compact completed cases every `--summary-every` cases. Branch mode starts a fresh separate model conversation per case; only one outcome enters the parent's history.
- The benchmark orchestrates branch selection **by case boundary**; it does not test automatic or recursive branching inside a case. Mandatory investigations remain inside each isolated branch (they are not spun off as separate nested branches). A branch can still fill its *own* context, which is an important failure to report.
- `verified_cases` / `tasks` is the primary problem-solving metric. A strictly correct final JSON with mission/control IDs and all verified seals is the additional end-to-end success condition.
- `recovery_completed`, `recovery_attempts`, `recovery_failures`, `recovery_blocked`, `recovery_shard_views`, `forced_evidence_chars` measure mandatory investigative steps, separately from optional traps.
- `wrong_repairs`, `decoy_documents_opened`, `trap_waste_chars`, `invalid_actions`, `total_actions` expose optional detour and recovery cost. `trap_waste_chars` is simulator-output characters, **not** actual token counts. Overall model token usage and parent/branch context pressure remain separate measurements.
- The simulator is not an actual repair tool or Browser MCP, and the branch benchmark isolates Jan API message histories; it is not invoking the MCP server's `branch_solve` tool for each incident.

### Commands

```bash
# Generate/validate scenarios without Jan:
python3 benchmark.py --scenario repair --dry-run

# One case per mode; validates your local Jan connection and JSON action handling:
python3 benchmark.py --scenario repair --smoke

# Moderate multi-case test (one, two, and four incidents per trial):
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 2 --context-limit 8192

# More difficult (and slower): required noisy shards + optional traps:
python3 benchmark.py --scenario repair --tasks 1 2 4 --repeats 3 \
  --trap-density 3 --trap-log-lines 64 --recovery-log-lines 48 \
  --nested-depth 2 --max-actions 32 --context-limit 8192
```

Ranges: `--trap-density` 1–3, `--trap-log-lines` 0–150, `--recovery-log-lines` 0–120, `--max-actions` 12–80. Larger logs create longer detours and may exhaust even the isolated branch context. The maximum output per Jan request stays 1,024 tokens unless overridden with `--max-output-tokens`.

**Caution:** These simulations make many sequential Jan requests (the depth-2 case requires at least 18 decisions). The full comparison can be significantly slower than the archive benchmark. Start with `--smoke`. Offline automated tests include a fake-agent run that intentionally makes a wrong repair and must recover; these tests do **not** measure the real model's ability to solve the cases.

---

## Archive retrieval benchmark (`--scenario archive`, original test)

This is a controlled **three-way test** against **the same Jan Local API model**. It compares:

- **Linear:** Main AI processes each archive lookup directly, retaining every full archive and reply in its conversation.
- **Summarized:** The main AI handles lookups directly but periodically asks the **same model** to summarize its completed work. It replaces old archive history with this model-generated note while keeping the original mission instructions unchanged. This is an actual Jan inference call, not a programmatically generated perfect answer.
- **Branch:** Each archive lookup is solved by the existing `branch_context.py` isolated-branch runner. Main AI only receives the concise result, acknowledges it, and continues. Main AI gets an API turn after every completed branch to avoid overstating inference savings.

It is an independent Python test program; **you do not have to enable Branch Context as an MCP server in Jan** to run it. You only have to start Jan's Local API Server with a locally loaded model. For archive lookup mode, the actual `branch_context.py` runner is exercised. The new repair scenario deliberately uses separate benchmark-controlled Jan API conversations to simulate case isolation; it does not call the MCP `branch_solve` command.

## 1. Prepare Jan

1. Load the model you want to evaluate in Jan (e.g., Qwen3.5-9B).
2. Start **Settings > Local API Server**. Default endpoint is `http://127.0.0.1:1337/v1`.
3. Keep its context size and other model/quantization/GPU settings unchanged throughout the experiment. **Set 8192 context** for the first benchmark if that is the size you're testing.
4. Configure the API endpoint/key/model in project-local `config.json` if needed. `model: "auto"` selects the first exposed model. You can check it with `python3 branch_context.py doctor`.
5. Keep Jan's API server running; the script sends the test prompts **only** to the configured loopback endpoint.

Jan reference: https://docs.jan.ai/desktop/api-server

## 2. Run tests

Open a terminal inside this directory.

```bash
# Verify that test scenarios fit the branch's per-task input limit (no Jan connection)
python3 benchmark.py --dry-run

# Small live check (3 side tasks, 12 records, 1 repeat; all three modes, including compaction)
python3 benchmark.py --smoke

# Main benchmark (3, 6, 8 side tasks; 65 records per task; 2 different seeds per level)
python3 benchmark.py --tasks 3 6 8 --records 65 --repeats 2 --context-limit 8192

# More reliable comparison (longer run, greater variance coverage)
python3 benchmark.py --tasks 3 6 8 --records 65 --repeats 5 --context-limit 8192
```

The benchmark now requests **1,024 generated tokens per API call by default**, instead of the previous hard cap of 350. To use another limit, add `--max-output-tokens 1536` (range 64–4096). The same cap applies to all three modes, including summarization calls. If Jan uses an 8,192-token context window, remember that the prompt and generated output must fit within the model/backend limit; increasing the output budget does not increase context capacity.

The default summarizes after **every 2 completed lookups**, except after the final lookup, and requires each note to fit in **1,800 characters**. Tune this policy with `--summary-every 2 --summary-max-chars 1800`. An empty or oversized note is recorded as a summarization failure; it is not silently corrected with ground truth. All model-generated notes are recorded in `results.json`. Note that model summarization may itself fail if the existing context has already grown beyond Jan's limit.

For the first real benchmark, use **2 repeats**. If results look meaningful and the server is stable, use 5. The harness runs trials sequentially so all three approaches share the same GPU/VRAM allocation, not multiple models loaded at once. It rotates which mode runs first across repeats to reduce warm-up effects.

**Runtime caution:** the full default benchmark sends many local model requests; avoid competing applications or model reloads during the run. If the API server crashes, the error is recorded in the output rather than being counted as a correct result. Local API requests may have long latency on some models.

## 3. What each test actually does

A random but reproducible assignment includes a **mission ID** and a **control code** that must be retained to the end. It contains `N` independent tasks, each providing a table of 65 fake audit records and a target reference ID. The AI must retrieve the corresponding seal from the correct row and preserve it for the final report.

At the end, the AI must output JSON with the original mission ID, original control code and **all** correct seals. We know the ground-truth answers before the AI starts. The same seed is used in all three modes.

This exercises:

- Retaining original instructions from the beginning of the main task.
- Finding relevant details among many similar distractors.
- Carrying accurate results forward.
- Main conversation growth versus isolated branch growth.

It does **not** exercise browsing, autonomous branch selection, multi-step tool use, recursive nesting, or restoring Jan's actual chat transcript. Those need separate tests.

## 4. Outputs

Results are written **inside the project folder**:

```text
data/benchmarks/<UTC timestamp>/
  SUMMARY.md       # human-readable averages and success fractions
  results.csv      # per-trial metrics; easy to open in LibreOffice Calc
  results.json     # full per-request traces and reference answers
  3_<seed>_branch/branches/...  # test branch data, not personal content
  # summarized-mode notes and diagnostics live in results.json
```

`results.csv` columns now distinguish strict success from recoverable evidence. Key columns include:

| Metric | Meaning |
|---|---|
| `exact_final` | **Strict pass:** valid complete JSON with both IDs and every seal correct. |
| `final_status` | Outcome: success, incorrect_answer, missing_fields, malformed_json, output_truncated_confirmed/suspected, etc. |
| `final_json_valid` | Whether the *entire* final response was parseable as JSON (optionally inside a fenced code block). |
| `visible_seals_correct` | Correct, fully emitted seal values visible even inside incomplete/truncated JSON; diagnostic only. |
| `visible_mission_ok`, `visible_control_ok` | Whether fully emitted original IDs were visible, even in invalid JSON. |
| `any_stage_seals_correct` | Correct seals found in either a direct lookup reply or the visible final response. Not strict completion. |
| `final_recovered_subtasks` | Seals missed at intermediate-answer scoring but correctly present in final response. |
| `final_seals_correct` | Correct seals in final answer, 0 to N. |
| `subtask_accuracy` | Correct immediate lookup results, independently scored from the final response. Denominator is `solved_subtasks`, not necessarily N when interrupted. |
| `mission_ok`, `control_ok` | Did the model retain beginning-of-task instructions? |
| `peak_main_prompt_tokens` | Maximum active parent/main input, per request. Most important measure of context pressure. |
| `peak_branch_prompt_tokens` | Maximum active isolated branch input, per request. |
| `total_processed_tokens` | Sum of prompt + response tokens across **all** parent AND branch calls. This is work performed, not live VRAM usage. |
| `elapsed_seconds` | Total wall-clock duration of each test. |
| `api_calls` | Every completion request, including parent acknowledgements after branches. |
| `max_output_tokens` | Configured output cap applied equally to all three modes. |
| `length_stops` | Number of API calls that ended with `finish_reason=length`, if reported by Jan. |
| `input_over_limit_risk` | True when Jan reports more prompt tokens than the configured context size. This is only a risk indicator, not proof of an actual overflow. |
| `error_type`, `failed_phase` | Distinguishes explicit context-overflow messages, definite timeouts, ambiguous connection/timeout errors, and other errors, and identifies where they occurred. |
| `error` | API, model, capacity or other errors. |
| `token_counts_exact_from_api` | Whether Jan provided actual token usage for every request. |
| `summaries_generated` | Number of complete successful model-written context summaries for summarized mode. |
| `summary_chars_peak` | Largest compact-memory note in characters. |
| `summary_every`, `summary_max_chars` | Summarization cadence and memory-note size budget. |

**Completion diagnostics:** `results.json` includes each call’s requested `max_tokens` and API `finish_reason`, when available. For invalid final JSON, `finish_reason=length` means confirmed output truncation. When there is no explicit stop reason but `completion_tokens` is within 3 tokens of the requested cap, truncation is only **suspected**. Otherwise an incomplete output is simply `malformed_json`. Unknown causes are not guessed.

**Crucial scoring rule:** An incomplete report can contain correct mission IDs and some correct seals, but it will *not* pass `exact_final`. Those fully emitted fields count only toward `visible_*` and `any_stage_*` diagnostics. Conversely, a report that is entirely correct but was preceded by unparseable intermediate lookup replies is still a strict final success, and `final_recovered_subtasks` records that recovery.

**Token precision:** If Jan returns standard `usage.prompt_tokens` and `usage.completion_tokens`, the benchmark uses those. Otherwise the script falls back to a **rough characters ÷ 4** estimate, which is NOT an exact model tokenizer count. Do not interpret an estimated `8192` as a definite context-limit crossing. Check Jan's configured context and server logs if the API truncates inputs silently.

## 5. How to evaluate it

Branching is actually useful compared with **both** alternatives when it:

1. Completes more assignments accurately under context pressure.
2. Holds **peak main context** substantially lower than linear mode.
3. Does not drop vital answers from branch summaries.
4. Remains within tolerable latency and **total token** overhead.

Expected trade-off: branching may be **slower and use more total tokens on short tasks** because the parent and child make separate inference calls. Its value should emerge for *larger, longer* tasks.

A model can fail the `exact_final` condition for reasons unrelated to context (poor JSON formatting, lookup mistakes). Scoring v3 retains distinct failure causes, correct intermediate answers, and fully visible but incomplete final answers. Read these together.

**Three-way fairness:** All modes use the same model, seeds, test archives, original instructions, final scoring, and output token cap. The summarizer is run by the same model and receives only its visible conversation history, never benchmark ground truth. Its extra model calls count toward token and time totals, and a missing seal in its summary can cause a failed final report. The order of modes rotates between repeats. Branching and summarization are orchestrated by the benchmark, **not** autonomously selected by the model.

## 6. Offline harness tests

```bash
python3 -m unittest discover -s tests -v
```

These verify deterministic puzzle creation, complete and truncated JSON scoring, confirmed versus suspected output limits, context overflow versus ambiguous network failures, direct-versus-final answer reconciliation, all three competitors, summary compaction, **virtual repair verification**, **misleading evidence**, and **recovery from a failed repair**. **The fake server tests cannot establish any performance improvement by real AI models.**

## Security

All example audits are fake and generated locally. No passwords, cookies or other credentials are needed. Do not substitute real password-manager data into benchmark prompts or result logs. All generated files stay within this project's `data/` tree by default.

## Router autodetection troubleshooting (v1.0.1)

`python3 benchmark.py --check-jan-reset` is read-only. It checks for the selected benchmark model in Jan's internal router. The detector looks for TCP listening sockets owned by `jan-llama-worker` or its children using `ss`, then falls back to matching socket inodes under `/proc`. If discovery fails, the error includes worker PIDs, candidate ports, and any model IDs returned by a Jan-owned router. The public Jan API on port 1337 is **not necessarily** the internal router. Never use port 1337 as the router URL unless its root `/models` endpoint is independently verified to be a compatible llama.cpp router.
