#!/usr/bin/env python3
"""Extract winning configurations per topology and verify key paper claims.

Reads the per-topology sweep summaries, filters to the best configuration
per (batch, context, imbalance) point, collapses context (verifying
context-invariance), and writes compact CSVs to paper_artifacts/.

Also runs checks against claims made in Section III-D of the paper.

Failures are recorded and reported at the end rather than aborting on the
first mismatch, so a single run surfaces every regression at once. The
process exits non-zero iff any check failed.
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
    """Accumulator for verification outcomes across all topology claim checks."""

    passed: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def _record_pass(self, label: str) -> None:
        print(f"  PASS: {label}")
        self.passed += 1

    def _record_fail(self, section: str, label: str) -> None:
        self.failed += 1
        self.failures.append((section, label))

    @staticmethod
    def _print_offenders(bad: pd.DataFrame, column: str) -> None:
        """Print up to ``MAX_OFFENDERS`` rows that violate the claim."""
        cols = [c for c in ("batch", "imbalance", column) if c in bad.columns]
        head = bad[cols].head(MAX_OFFENDERS).to_dict("records")
        print(f"    offenders (first {min(MAX_OFFENDERS, len(bad))} of {len(bad)}):")
        for rec in head:
            print(f"      {rec}")

    def check_all_equal(
        self, section: str, label: str, df_subset: pd.DataFrame, column: str, expected
    ) -> None:
        """Assert ``df_subset[column] == expected`` for every row in the subset.

        On failure, prints the expected value, the set of distinct actual
        values, and a sample of the offending rows; does not raise.
        """
        if len(df_subset) == 0:
            print(f"  SKIP: {label} (empty subset)")
            return
        actual_series = df_subset[column]
        if (actual_series == expected).all():
            self._record_pass(label)
            return
        bad = df_subset[actual_series != expected]
        unique_actual = sorted(actual_series.unique().tolist())
        print(f"  FAIL: {label}")
        print(f"    expected: {column} == {expected} (over {len(df_subset)} rows)")
        print(f"    actual  : {unique_actual}")
        self._print_offenders(bad, column)
        self._record_fail(section, label)

    def check_condition(
        self,
        section: str,
        label: str,
        condition: bool,
        *,
        expected,
        actual,
    ) -> None:
        """Generic boolean check with explicit expected/actual reporting."""
        if condition:
            self._record_pass(label)
            return
        print(f"  FAIL: {label}")
        print(f"    expected: {expected}")
        print(f"    actual  : {actual}")
        self._record_fail(section, label)

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
    still proceed with first-context-per-(batch, imbalance) so downstream
    claim checks can run and surface additional information.
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


def verify_fc(df: pd.DataFrame, results: CheckResults) -> None:
    """Verify FC128x32 claims from Section III-D."""
    print("\nFC128x32 claim checks:")
    section = "fc128x32"

    results.check_all_equal(section, "g=1 everywhere", df, "g", 1)
    results.check_all_equal(section, "s=32 everywhere", df, "num_complete_sets", 32)

    small = df[df["batch"] <= 128]
    results.check_all_equal(section, "B<=128 -> c=1024", small, "colocated_experts", 1024)
    results.check_all_equal(section, "B<=128 -> shape=1x128", small, "tp_moe_shape", "1x128")

    large_rho8 = df[(df["batch"] >= 1024) & (df["imbalance"] >= 4)]
    results.check_all_equal(section, "B>=1024, rho>=4 -> c=32", large_rho8, "colocated_experts", 32)
    results.check_all_equal(
        section, "B>=1024, rho>=4 -> shape=1x4", large_rho8, "tp_moe_shape", "1x4"
    )


def verify_torus(df: pd.DataFrame, results: CheckResults) -> None:
    """Verify 3D torus claims from Section III-D."""
    print("\nTorus claim checks:")
    section = "torus3d"

    results.check_all_equal(section, "c=32 everywhere", df, "colocated_experts", 32)
    results.check_all_equal(section, "s=1 everywhere", df, "num_complete_sets", 1)

    small = df[df["batch"] <= 128]
    results.check_all_equal(section, "B<=128 -> g=32", small, "g", 32)
    results.check_all_equal(section, "B<=128 -> shape=1x128", small, "tp_moe_shape", "1x128")
    results.check_all_equal(
        section, "B<=128 -> replicated router", small, "router_mode", "replicated"
    )

    b1024 = df[df["batch"] == 1024]
    results.check_all_equal(section, "B=1024 -> shape=16x8", b1024, "tp_moe_shape", "16x8")
    results.check_all_equal(section, "B=1024 -> sharded router", b1024, "router_mode", "sharded")
    results.check_all_equal(section, "B=1024 -> g=32", b1024, "g", 32)

    b4096 = df[df["batch"] == 4096]
    results.check_all_equal(section, "B=4096 -> g=8", b4096, "g", 8)
    results.check_all_equal(section, "B=4096 -> shape=4x32", b4096, "tp_moe_shape", "4x32")

    config_cols = [c for c in KEEP_COLS if c not in ("batch", "imbalance")]
    per_batch = df.groupby("batch")[config_cols].nunique()
    invariant_mask = (per_batch == 1).all(axis=1)
    invariant = bool(invariant_mask.all())
    if invariant:
        results._record_pass("insensitive to imbalance")
    else:
        bad_batches = per_batch[~invariant_mask]
        print("  FAIL: insensitive to imbalance")
        print(f"    expected: per-batch nunique == 1 across {config_cols}")
        print(f"    actual  : {len(bad_batches)} batch(es) show variation with rho")
        head = bad_batches.head(MAX_OFFENDERS)
        for batch, row in head.iterrows():
            print(f"      batch={batch}: {row.to_dict()}")
        results._record_fail(section, "insensitive to imbalance")


def verify_hyperx(df: pd.DataFrame, results: CheckResults) -> None:
    """Verify HyperX claims from Section III-D."""
    print("\nHyperX claim checks:")
    section = "hyperx"

    results.check_all_equal(section, "sharded router uniformly", df, "router_mode", "sharded")

    low = df[(df["batch"] <= 32) & (df["imbalance"] <= 16)]
    results.check_all_equal(section, "small B / low rho -> g=1", low, "g", 1)
    results.check_all_equal(section, "small B / low rho -> c=32", low, "colocated_experts", 32)
    results.check_all_equal(
        section, "small B / low rho -> shape=1x128", low, "tp_moe_shape", "1x128"
    )

    severe = df[(df["imbalance"] >= 32) & (df["batch"] >= 128)]
    results.check_all_equal(
        section,
        "severe imbalance -> c=1024 (full colocation)",
        severe,
        "colocated_experts",
        1024,
    )

    extreme = df[(df["batch"] == 4096) & (df["imbalance"] >= 16)]
    results.check_all_equal(section, "B=4096, rho>=16 -> s=2", extreme, "num_complete_sets", 2)


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
    extracted = {}
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
        extracted[name] = collapsed

        out_path = out / f"{name}_best_configs.csv"
        collapsed.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}")

    verify_fc(extracted["fc128x32"], results)
    verify_torus(extracted["torus3d"], results)
    verify_hyperx(extracted["hyperx"], results)

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
