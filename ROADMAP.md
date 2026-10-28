# v1.7 — adaptive surprise branches

The original autonomous-branching path is now represented by the opt-in
`adaptive` repair arm. It injects one explicit unexpected blocker at the first
repair attempt, records the parent's message list as a marker, forks that marker
into a private child context, and resumes the parent with only a compact verified
receipt. Results expose surprise triggers, marker creation, child work, parent
prompt size before/after resume, and transcript-leak checks.

Offline tests cover the complete marker → child → resume path. A live Qwen run
is still required to measure real model behavior and context savings.

Example:

```bash
python3 benchmark.py --engine llama --scenario repair \
  --modes adaptive --tasks 1 --repeats 3 --seed 20261026 \
  --context-limit 8192 --nested-depth 2 --nested-log-lines 8 \
  --recovery-log-lines 8 --pressure-records 0 --max-actions 48 \
  --autonomous-max-tokens 100000
```

The earlier `autonomous` arm remains available as a separate model-selected
delegation experiment for known prerequisite stages. It should not be treated
as the original surprise-and-resume milestone.

---

# v1.5 — Jan runtime reliability

Completed in the MCP runner: whole-request budget checks, persistent shared tree
limits, safe transcript/resume handling, read-result deduplication, paged project
reads, bounded no-action recovery, real model-list health checks and explicit
model-reported handoffs. See [RUNTIME.md](RUNTIME.md) for settings and validation.

Next: validate the runtime against the local Jan model; measure context isolation
against matched single-conversation and adaptive surprise runs across repeated
live seeds. The repair benchmark contains controlled recursion, legacy
model-selected delegation, and the adaptive marker/resume arm; none is evidence
of live Qwen behavior until run against the local model.

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

No live Qwen improvement is claimed until this configuration is run. Autonomous
branching has a separate benchmark arm and still needs live validation.

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

# Context Branch — development goals

Keep these three goals in scope while testing the architecture. They are separate
experiments so improvements in one cannot disguise regressions in another.

## 1. Recursive context branching — v1.4.1 (controlled benchmark with strict stage handoffs)

- **Current mechanism:** `--scenario repair --modes ... recursive` runs a
  benchmark-selected dependency tree: root repair → recovery → escrow → index.
  Only stages enabled by `--nested-depth` are opened.
- Each stage has its own API message history. No child transcript is copied
  into an ancestor's context; only a short simulator-verified completion fact.
- All stages operate on **one** `RepairSimulator` state. Child changes survive
  when the parent resumes. Every stage's model actions use the same per-repair
  `--max-actions` budget, preventing unlimited delegated work.
- Report spawned child branches, total contexts, max depth, stage successes,
  all model calls, peak branch/parent prompts, total tokens and actual repair
  verification. A fake-agent context-overflow test is **not** evidence of real
  Qwen performance.
- **Next validation:** run the same seeded nested repair with `branch` and
  `recursive` at 8192 tokens. Repeat across several seeds. A branch that fails
  must not be labeled successful merely because its parent stayed small.

**v1.4.1 completed:** stage-specific read/action allowlists, verified child-to-parent environment-state receipts, no child history sharing, and bounded no-action inference retry. A real Qwen rerun remains necessary.

## 2. Legacy model-selected delegation — benchmark slice implemented

- The opt-in `autonomous` repair mode lets the model choose **whether**, **when**,
  and **what** to delegate with an explicit `delegate` action. The root may also
  continue in place.
- Delegated stages use private contexts and return only simulator-verified state
  receipts. Depth, child count, shared action budget, lower-level ordering,
  shared prompt/completion token budget, duplicate targets and cycles are
  enforced by the harness.
- Results record accepted/rejected/unnecessary delegations, missed opportunities,
  branch count, context count, decisions and maximum depth. No benchmark oracle
  or child transcript is injected into a parent.
- This arm is useful for studying whether a model chooses to delegate known
  prerequisites, but it is not the original surprise-marker-resume design.
  Offline scripted agents validate plumbing only; repeated live Qwen runs are
  required before judging model behavior.

Example:

```bash
python3 benchmark.py --engine llama --scenario repair \
  --modes branch recursive autonomous --tasks 1 --repeats 3 \
  --seed 20261026 --context-limit 8192 --nested-depth 2 \
  --nested-log-lines 8 --recovery-log-lines 8 \
  --pressure-records 0 --max-actions 32 \
  --autonomous-max-children 8
```

## 3. Investigation responses with no usable action — bounded recovery implemented

- Log `no_action_investigation_calls` (no parseable simulator JSON action)
  and `empty_investigation_replies` (empty visible content) independently
  from summary and final-report failures. Both are in the results JSON/CSV.
- Determine whether the generation budget was consumed by hidden reasoning,
  output truncation, malformed JSON or genuine indecision before selecting a
  fallback.
- Both the repair benchmark and Jan MCP runner now use bounded same-context
  recovery. Every retry counts toward inference tokens/time; the MCP runner also
  charges the shared model-call budget. Neither controller invents an action.
- Keep the primary benchmark unbiased: do not silently enable this recovery
  for only one strategy or infer model success from simulated action text.

## Research rules

Use the same local GGUF, context limit, max generation limits and scenarios
across modes. Track tokens over **all** parents and children. Use a fresh
llama-server for each trial when possible; never kill unrelated servers. A
deterministic simulator can verify a repair, but only actual model responses
establish final-report correctness. The three goals above are not interchangeable.

