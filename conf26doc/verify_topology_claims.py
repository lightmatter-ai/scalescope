#!/usr/bin/env python3
"""Extract winning configurations per topology from the precursor sweeps.

Reads the per-topology sweep summaries, filters to the best configuration
per (batch, context, imbalance) point, collapses context (verifying
context-invariance), and writes compact CSVs to paper_artifacts/.

The context-invariance check is structural: if the winning configuration for
a (batch, imbalance) cell differs across contexts, all violations are
reported at the end and the process exits non-zero.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BEST_TAG = "BEST_PLACEMENT_BY_BATCH_CONTEXT_IMBALANCE"

KEEP_COLS = [
    "batch",
    "imbalance",
    "colocated_experts",
    "tp_moe_shape",
    "router_mode",
    "g",
    "num_complete_sets",
]

MAX_OFFENDERS = 8


@dataclass
class CheckResults:
    """Accumulator for verification outcomes across all topologies."""

    passed: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def _record_pass(self, label: str) -> None:
        print(f"  PASS: {label}")
        self.passed += 1

    def _record_fail(self, section: str, label: str) -> None:
        self.failed += 1
        self.failures.append((section, label))

    def summary_line(self) -> str:
        total = self.passed + self.failed
        if self.failed == 0:
            return f"RESULT: ALL CHECKS PASSED ({self.passed}/{total})"
        return f"RESULT: {self.failed}/{total} CHECKS FAILED"


def load_best(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["Best"] == BEST_TAG].copy()


def verify_context_invariant(best: pd.DataFrame, name: str, results: CheckResults) -> pd.DataFrame:
    """Record whether configs are identical across contexts for each (batch, imbalance).

    Always returns a collapsed frame. If context-invariance fails we
    still proceed with first-context-per-(batch, imbalance) so the
    extraction can complete and surface additional information.
    """
    config_cols = [c for c in KEEP_COLS if c not in ("batch", "imbalance")]
    grouped = best.groupby(["batch", "imbalance"])[config_cols].nunique()
    violations = grouped[(grouped > 1).any(axis=1)]
    if violations.empty:
        results._record_pass(f"[{name}] context-invariant")
    else:
        print(f"  FAIL: [{name}] configuration varies with context")
        print(f"    expected: nunique == 1 for {config_cols}")
        print(f"    actual  : {len(violations)} (batch, imbalance) cells differ across contexts")
        print(f"    offenders (first {min(MAX_OFFENDERS, len(violations))} of {len(violations)}):")
        head = violations.head(MAX_OFFENDERS)
        for idx, row in head.iterrows():
            print(f"      (batch={idx[0]}, imbalance={idx[1]}): {row.to_dict()}")
        results._record_fail(name, "context-invariant")
    collapsed = best[KEEP_COLS].drop_duplicates().sort_values(["batch", "imbalance"])
    return collapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precursor-dir",
        default="out/conf26/paper_artifacts/precursor_sweeps",
        help="Directory containing per-topology summary CSVs",
    )
    parser.add_argument(
        "--out-dir",
        default="out/conf26/paper_artifacts",
        help="Output directory for extracted CSVs",
    )
    args = parser.parse_args()

    precursor = Path(args.precursor_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = CheckResults()
    for name, filename in [
        ("fc128x32", "summary_fc128x32.csv"),
        ("torus3d", "summary_torus3d.csv"),
        ("hyperx", "summary_hyperx.csv"),
    ]:
        path = precursor / filename
        print(f"\n{'='*50}")
        print(f"Processing {name}: {path}")
        print(f"{'='*50}")

        best = load_best(path)
        print(f"  {len(best)} best rows from {len(pd.read_csv(path))} total")

        collapsed = verify_context_invariant(best, name, results)

        out_path = out / f"{name}_best_configs.csv"
        collapsed.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}")

    print(f"\n{'='*50}")
    print(results.summary_line())
    if results.failed:
        print("Failed checks:")
        for section, label in results.failures:
            print(f"  - [{section}] {label}")
    print(f"{'='*50}")

    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
