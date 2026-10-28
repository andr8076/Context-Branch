#!/usr/bin/env python3
"""Run a controlled live llama.cpp checkpoint-on/off repair comparison.

A real inference run is needed to establish Qwen behavior. Offline tests only
validate simulator rules and the checkpoint mechanism. No fake padding is added.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import benchmark


def parser():
    p = argparse.ArgumentParser(description="Live checkpoint pressure experiment (fresh llama-server per arm)")
    p.add_argument("--context-limit", type=int, default=4096)
    p.add_argument("--checkpoint-threshold", type=float, default=0.50)
    p.add_argument("--checkpoint-max", type=int, default=8)
    p.add_argument("--pressure-records", type=int, default=1)
    p.add_argument("--recovery-log-lines", type=int, default=8)
    p.add_argument("--nested-log-lines", type=int, default=8)
    p.add_argument("--max-actions", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--stage-start-inspects", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--output", type=Path, default=None,
                   help="New, empty directory for both independent runs")
    p.add_argument("--llama-model", type=str, default=None,
                   help="Optional override for local_model.json")
    p.add_argument("--llama-server", type=str, default="llama-server")
    return p


def run(argv=None):
    args = parser().parse_args(argv)
    if not 0.50 <= args.checkpoint_threshold <= 0.90:
        raise SystemExit("--checkpoint-threshold must be between 0.50 and 0.90")
    output = args.output or (Path("data/benchmarks") /
             ("pressure-study-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for label, threshold in (("off", 0.0), ("on", args.checkpoint_threshold)):
        print(f"\n--- Context checkpoints {label.upper()}, threshold={threshold} ---", flush=True)
        trial_args = [
            "--engine", "llama", "--llama-server", args.llama_server,
            "--scenario", "repair", "--modes", "recursive",
            "--tasks", "1", "--repeats", "1", "--seed", str(args.seed),
            "--context-limit", str(args.context_limit), "--nested-depth", "2",
            "--nested-log-lines", str(args.nested_log_lines),
            "--recovery-log-lines", str(args.recovery_log_lines),
            "--pressure-records", str(args.pressure_records),
            "--max-actions", str(args.max_actions),
            "--stage-start-inspects", str(args.stage_start_inspects),
            "--recursive-checkpoint-threshold", str(threshold),
            "--recursive-checkpoint-max", str(args.checkpoint_max),
            "--output", str(output / label),
        ]
        if args.llama_model:
            trial_args += ["--llama-model", args.llama_model]
        status = benchmark.main(trial_args)
        result_path = output / label / "results.json"
        if not result_path.exists():
            print(f"Arm {label} could not be evaluated; benchmark returned {status}.")
            return 3
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if not data.get("trials"):
            print(f"Arm {label} returned no trials.")
            return 3
        rows[label] = data["trials"][0]

    control, treatment = rows["off"], rows["on"]
    summary = {
        "experiment": "real_local_inference_checkpoint_comparison",
        "seed": args.seed, "context_limit": args.context_limit,
        "settings_same_except_checkpoint_threshold": True,
        "off": {field: control.get(field) for field in
                ("verified_cases", "exact_final", "error_type", "recursive_checkpoints",
                 "recursive_checkpoint_blocked", "recursive_checkpoint_compactions",
                 "recursive_checkpoint_block_reasons", "final_status", "failed_phase",
                 "authority_docs_viewed", "total_actions", "elapsed_seconds",
                 "total_processed_tokens", "peak_branch_prompt_tokens")},
        "on": {field: treatment.get(field) for field in
               ("verified_cases", "exact_final", "error_type", "recursive_checkpoints",
                "recursive_checkpoint_blocked", "recursive_checkpoint_compactions",
                "recursive_checkpoint_block_reasons", "final_status", "failed_phase",
                "authority_docs_viewed", "total_actions", "elapsed_seconds",
                "total_processed_tokens", "peak_branch_prompt_tokens")},
    }
    control_success = control.get("exact_final") and control.get("verified_cases") == 1
    treatment_success = treatment.get("exact_final") and treatment.get("verified_cases") == 1
    if not treatment.get("recursive_checkpoints"):
        verdict = "INCONCLUSIVE: checkpoint did not activate; increase legitimate workload or adjust context limit."
    elif control.get("error_type") == "context_overflow_confirmed" and treatment_success:
        verdict = "SUCCESSFUL DEMONSTRATION: control overflowed; checkpoint run verified the repair."
    elif treatment_success and not control_success:
        verdict = ("CHECKPOINT ROBUSTNESS WIN: checkpoint run completed while control did not; "
                   f"control outcome={control.get('final_status') or control.get('error_type')}. "
                   "The cause may be action exhaustion or model variance, not necessarily overflow.")
    elif (control.get("error_type") == "context_overflow_confirmed" and
          treatment.get("error_type") == "context_checkpoint_blocked"):
        verdict = ("SAFE CHECKPOINT BLOCK: control overflowed and checkpointing refused to discard "
                   "evidence, but the treatment did not complete. Reduce evidence, lower the output "
                   "reserve, or use a larger context.")
    elif treatment_success:
        verdict = "CHECKPOINT VERIFIED: it activated and the run completed; control did not prove necessity."
    else:
        verdict = "CHECKPOINT ACTIVATED BUT DID NOT COMPLETE: inspect both traces and stage events."
    summary["verdict"] = verdict
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + verdict)
    print(json.dumps(summary, indent=2))
    print("Comparison saved:", output / "comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

