# Receipt and evidence lifecycle

`receipt_archive.ReceiptArchive` is a standalone, domain-independent component.
It is not yet wired into the existing model runtime or simulator controllers.
The caller generates the receipt and supplies its context-budget predicate.

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
```

The continuation test intentionally omits an exact value from the receipt,
reopens storage with a fresh manager, retrieves the omitted evidence, and uses
it in a subsequent calculation. Other checks exercise verification failure,
storage failure, budget refusal, and invalid archive references. They prove
controller behavior, not a model's ability to write or interpret good receipts.

Next integration gate: expose this lifecycle through the general runtime's
child-return and evidence-read tools, charge retrieval to its shared budgets,
and test receipt generation and later retrieval with scripted model replies.
Keep the existing runtime behavior until that integration is tested.
