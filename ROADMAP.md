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

## 2. Autonomous branch decisions — next (not implemented)

- Let the local model choose **whether**, **when**, and **what** to delegate.
  It should be free to continue in place and should explain only the minimal
  input needed by a child (not its entire conversation).
- Use an explicit delegate action with bounded depth, a maximum number of
  children, a shared token/action budget, and a cycle/duplicate-subtask guard.
- Keep simulator validation and environment state outside chat context.
  Do not leak answers from the benchmark oracle or inject a correct branch plan.
- Benchmark decisions against both always-linear and controlled recursion.
  Measure unnecessary branches and missed opportunities, not just successes.

## 3. Investigation responses with no usable action — next (diagnostics added)

- Log `no_action_investigation_calls` (no parseable simulator JSON action)
  and `empty_investigation_replies` (empty visible content) independently
  from summary and final-report failures. Both are in the results JSON/CSV.
- Determine whether the generation budget was consumed by hidden reasoning,
  output truncation, malformed JSON or genuine indecision before selecting a
  fallback.
- Consider one bounded *same-context* retry with a concise JSON-action reminder,
  charging the retry for inference tokens/time/actions; never invent an action.
- Keep the primary benchmark unbiased: do not silently enable this recovery
  for only one strategy or infer model success from simulated action text.

## Research rules

Use the same local GGUF, context limit, max generation limits and scenarios
across modes. Track tokens over **all** parents and children. Use a fresh
llama-server for each trial when possible; never kill unrelated servers. A
deterministic simulator can verify a repair, but only actual model responses
establish final-report correctness. The three goals above are not interchangeable.
