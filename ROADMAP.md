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
