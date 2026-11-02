# Generality audit and evaluation gates

## Current evidence

The successful 1.12.1 live case proves one verified branch/resume execution.
It does not prove universal decomposition or net compute savings. Prerequisite
branching uses named simulator stages and simulator-certified completion.
The general runtime has model-reported completion instead. These are different
trust models and must be evaluated separately.

The 4096-token CSV establishes an estimated controller stop in adaptive mode
and a confirmed server refusal in linear mode. It does not establish child
transcript leakage or identify the exact rejected adaptive prompt. Old reclaimed
context metrics cover surprise branches only. Cumulative child inference tokens
are not the amount of context released.

## Foundation changes in 1.13.0

- Adaptive parent cleanup and request admission share calibrated accounting.
- Calibration is scoped to each conversation; child token density no longer
  sets the parent's multiplier. Estimates remain estimates, especially for new
  evidence whose token density differs from previous observations.
- `context_budget_checks` records prompt estimate, raw estimate, reserve, limit,
  scope, calibration samples, and admission result, including rejected requests.
- CLI repair experiments default to `--surprise-workload shared`. Every mode
  gets the same disturbance and blocker documents, and action instructions
  explain blocker resolution. `none` disables it for all modes; `legacy`
  explicitly restores adaptive-only disturbance. Direct Python callers retain
  legacy behavior unless `_surprise_workload` is specified.
- `python3 audit_replay.py path/to/results.json` replays recorded environment
  actions without inference. It checks simulator outcomes, not model reasoning
  or context savings. Unsupported controller-only events fail explicitly.

## Gates before further live testing

1. Extend replay into a controller-level transcript fixture. Assert exact
   unfinished-evidence retention, immutable goals, child isolation, retrieval,
   and repeatable budget diagnostics. Existing deterministic fake-agent tests
   cover some of these properties but are not substitutes for live transcripts.
2. Unify the accounting interface across the general runtime, recursive stages,
   and baseline modes; audit output allowances and retry allowances against the
   actual request payload. Current conversation calibration fixes are adaptive.
3. Replace full-parent surprise forks with explicit task inputs and retrievable
   evidence references. Define return contracts and verifier hooks independent
   of simulator stage names. Do not discard evidence solely because a model
   claims completion.
4. Establish component competence with sufficient context, then compare
   identical workloads with equivalent total work budgets and model settings.
   Shared disturbances alone do not eliminate differences in controller hints,
   stage-start reasoning settings, or request limits.
5. Evaluate a held-out task family, such as repository investigation, without
   adding domain-specific stage names to the controller. Publish failures as
   well as successes before claiming generality.

Do not increase live test size until the relevant gate passes. Report verified
completion, peak active context, total prompt/completion tokens, retrieval work,
elapsed time, and failure category separately. Token counts do not measure energy.
