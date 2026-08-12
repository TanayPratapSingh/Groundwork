#!/usr/bin/env python3
"""Run the full evaluation and the release gate.

    python3 scripts/run_eval.py                  # 1200-question stratified sample
    python3 scripts/run_eval.py -n 400           # faster inner loop
    python3 scripts/run_eval.py --full           # all 11,873 questions
    python3 scripts/run_eval.py --set-baseline   # record this run as the baseline
    python3 scripts/run_eval.py --no-guardrails  # RAG metrics only

Exit code is 0 when the gate passes and 1 when it fails, so this drops
straight into CI without a wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundwork.config import ARTIFACT_DIR, SETTINGS  # noqa: E402
from groundwork.evaluation.report import (  # noqa: E402
    render_markdown,
    write_baseline,
    write_json,
    write_markdown,
)
from groundwork.evaluation.runner import full_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--questions", type=int, default=1200,
                        help="stratified sample size (default 1200)")
    parser.add_argument("--full", action="store_true", help="evaluate every question")
    parser.add_argument("--no-guardrails", action="store_true",
                        help="skip guardrail training and scoring")
    parser.add_argument("--set-baseline", action="store_true",
                        help="write this run's metrics as the regression baseline")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    n = 100_000 if args.full else args.questions

    print("=" * 72)
    print("Groundwork evaluation")
    print(f"  provider: {SETTINGS.provider}"
          f"{'  (no API key set, using the deterministic offline provider)' if SETTINGS.provider == 'offline' else ''}")
    print("=" * 72 + "\n")

    run = full_evaluation(
        n_questions=n,
        include_guardrails=not args.no_guardrails,
        verbose=not args.quiet,
    )

    json_path = write_json(run)
    md_path = write_markdown(run)
    print(f"\nWrote {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path}")
    print(f"Wrote {md_path.relative_to(Path.cwd()) if md_path.is_relative_to(Path.cwd()) else md_path}")

    if args.set_baseline:
        baseline_path = write_baseline(run)
        print(f"Wrote {baseline_path.name} (future runs are compared against it)")

    if not args.quiet:
        after = run.calibration_after
        if after.get("bins"):
            from groundwork.evaluation.calibration import CalibrationReport, Bin
            report = CalibrationReport(
                ece=after["ece"], mce=after["mce"], brier=after["brier"],
                auc=after["auc"], accuracy=after["accuracy"],
                base_rate=after["base_rate"], n=after["n"],
                bins=[Bin(b["range"][0], b["range"][1], b["count"],
                          b["mean_confidence"], b["accuracy"]) for b in after["bins"]],
                temperature=after["temperature"],
            )
            print("\nReliability diagram (post-calibration)")
            print(report.reliability_plot())

    return 0 if run.gate.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
