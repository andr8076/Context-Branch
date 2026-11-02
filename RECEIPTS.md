# Receipt and evidence lifecycle

`receipt_archive.ReceiptArchive` is wired into `branch_context.run_branch`.
Nested and automatically isolated child work returns a durable receipt before
the parent resumes. Simulator controllers are separate and do not yet use it.
The runtime derives receipts from the existing structured final response;
there is no additional receipt-generation model call.

Receipts contain outcome, findings, sources, changes, and unresolved issues.
All five fields are text. They are capped as a whole and marked model-reported
unless an application verifier explicitly accepts the outcome. A verifier is
trusted application code; it must check the actual completion condition.
Archiving and schema validation alone do not establish factual correctness.

`handoff(parent, child, receipt, fits=..., verifier=...)` checks the proposed
parent context, saves the receipt and complete observed child history to an
atomic JSON archive, syncs it to disk, appends the receipt, then clears the
child list. Failures before the handoff preserve both active histories. Do not
mutate these lists concurrently. External references to child data are not
erased; this component releases the managed active list, not every memory copy.

Receipts expose an archive identifier and message count. Their source field
should identify relevant message indexes. `retrieve` returns bounded exact
character ranges from an archived message. `recall` appends a range only when
the parent's budget admits it. Historical observations are explicitly labeled
and must not be treated as current state without rechecking. Retrieval does
not automatically reload the rest of the child history. Archive files persist
across manager/process restarts and may contain sensitive observed data.

Run the model-free lifecycle checks:

```bash
python3 -m unittest discover -s tests -p test_receipt_archive.py -v
python3 -m unittest discover -s tests -p test_runtime_receipts.py -v
```

The continuation test intentionally omits an exact value from the receipt,
reopens storage with a fresh manager, retrieves the omitted evidence, and uses
it in a subsequent calculation. Other checks exercise verification failure,
storage failure, budget refusal, and invalid archive references. They prove
controller behavior, not a model's ability to write or interpret good receipts.

Runtime receipts remain model-reported and preserve success/partial/blocked
status, findings, a historical message-index directory, and unresolved work.
Evidence includes prior working-memory snapshots. The parent can call
`read_child_evidence(archive, message_index, start=0, limit=1200)` for exact
omitted details, including on a later `branch_resume`. Only archives returned
to that branch are accessible through this tool. Retrieval occupies a normal
tool turn; subsequent inference uses the shared root call/token budget.
Both receipts and retrieval are checked against calibrated prompt estimates,
tool schemas, output reserve, and context safety allowance. Oversized retrieval
is refused rather than cutting serialized JSON. Token estimates remain estimates.

Archives persist under the branch data directory's `receipts/`, even with
`save_transcripts=false`. If a receipt cannot be stored or admitted, the runtime
reports a blocked handoff and retains a recovery transcript. Optional saved
transcripts are not erased. Disk retention/garbage collection is not implemented.

The scripted runtime tests cover continuation after release, later retrieval,
shared accounting, partial outcomes, storage failures, budget refusal, and
oversized receipts. They establish controller behavior, not universal model
reliability or net inference savings. The next live test should target this
runtime path; simulator benchmarks do not test this integration.
