#!/usr/bin/env python3
"""Extract winning configurations per topology and verify key paper claims.

Reads the per-topology sweep summaries, filters to the best configuration
per (batch, context, imbalance) point, collapses context (verifying
context-invariance), and writes compact CSVs to paper_artifacts/.

Also runs assertions against claims made in Section III-D of the paper.
"""

import argparse
import sys
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


def load_best(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["Best"] == BEST_TAG].copy()


def verify_context_invariant(best: pd.DataFrame, name: str) -> pd.DataFrame:
    """Assert config is identical across contexts for each (batch, imbalance).

    Returns the context-collapsed frame.
    """
    config_cols = [c for c in KEEP_COLS if c not in ("batch", "imbalance")]
    grouped = best.groupby(["batch", "imbalance"])[config_cols].nunique()
    violations = grouped[(grouped > 1).any(axis=1)]
    if not violations.empty:
        print(f"FAIL [{name}]: configuration varies with context at:", file=sys.stderr)
        print(violations, file=sys.stderr)
        sys.exit(1)
    collapsed = best[KEEP_COLS].drop_duplicates().sort_values(["batch", "imbalance"])
    print(f"  PASS: context-invariant ({len(collapsed)} unique points)")
    return collapsed


def verify_fc(df: pd.DataFrame) -> None:
    """Verify FC128x32 claims from Section III-D."""
    print("\nFC128x32 claim checks:")

    assert (df["g"] == 1).all(), "FAIL: g != 1 found"
    print("  PASS: g=1 everywhere")

    assert (df["num_complete_sets"] == 32).all(), "FAIL: s != 32 found"
    print("  PASS: s=32 everywhere")

    small = df[df["batch"] <= 128]
    assert (small["colocated_experts"] == 1024).all(), "FAIL: c != 1024 at B<=128"
    assert (small["tp_moe_shape"] == "1x128").all(), "FAIL: shape != 1x128 at B<=128"
    print("  PASS: B<=128 -> c=1024, 1x128")

    large_rho8 = df[(df["batch"] >= 1024) & (df["imbalance"] >= 4)]
    assert (large_rho8["colocated_experts"] == 32).all(), "FAIL: c != 32 at B>=1024, rho>=4"
    assert (large_rho8["tp_moe_shape"] == "1x4").all(), "FAIL: shape != 1x4 at B>=1024, rho>=4"
    print("  PASS: B>=1024, rho>=4 -> c=32, 1x4")


def verify_torus(df: pd.DataFrame) -> None:
    """Verify 3D torus claims from Section III-D."""
    print("\nTorus claim checks:")

    assert (df["colocated_experts"] == 32).all(), "FAIL: c != 32 found"
    print("  PASS: c=32 everywhere")

    assert (df["num_complete_sets"] == 1).all(), "FAIL: s != 1 found"
    print("  PASS: s=1 everywhere")

    small = df[df["batch"] <= 128]
    assert (small["g"] == 32).all(), "FAIL: g != 32 at B<=128"
    assert (small["tp_moe_shape"] == "1x128").all(), "FAIL: shape != 1x128 at B<=128"
    assert (small["router_mode"] == "replicated").all(), "FAIL: not replicated at B<=128"
    print("  PASS: B<=128 -> g=32, 1x128, replicated")

    b1024 = df[df["batch"] == 1024]
    assert (b1024["tp_moe_shape"] == "16x8").all(), "FAIL: shape != 16x8 at B=1024"
    assert (b1024["router_mode"] == "sharded").all(), "FAIL: not sharded at B=1024"
    assert (b1024["g"] == 32).all(), "FAIL: g != 32 at B=1024"
    print("  PASS: B=1024 -> 16x8, sharded, g=32")

    b4096 = df[df["batch"] == 4096]
    assert (b4096["g"] == 8).all(), "FAIL: g != 8 at B=4096"
    assert (b4096["tp_moe_shape"] == "4x32").all(), "FAIL: shape != 4x32 at B=4096"
    print("  PASS: B=4096 -> g=8, 4x32")

    config_cols = [c for c in KEEP_COLS if c not in ("batch", "imbalance")]
    per_batch = df.groupby("batch")[config_cols].nunique()
    invariant = (per_batch == 1).all(axis=1).all()
    assert invariant, "FAIL: config varies with imbalance"
    print("  PASS: insensitive to imbalance")


def verify_hyperx(df: pd.DataFrame) -> None:
    """Verify HyperX claims from Section III-D."""
    print("\nHyperX claim checks:")

    assert (df["router_mode"] == "sharded").all(), "FAIL: non-sharded router found"
    print("  PASS: sharded router uniformly")

    low = df[(df["batch"] <= 32) & (df["imbalance"] <= 16)]
    assert (low["g"] == 1).all(), "FAIL: g != 1 at small B / low rho"
    assert (low["colocated_experts"] == 32).all(), "FAIL: c != 32 at small B / low rho"
    assert (low["tp_moe_shape"] == "1x128").all(), "FAIL: shape != 1x128 at small B / low rho"
    print("  PASS: small B / low rho -> g=1, c=32, 1x128")

    severe = df[(df["imbalance"] >= 32) & (df["batch"] >= 128)]
    if not severe.empty:
        assert (severe["colocated_experts"] == 1024).all(), "FAIL: c != 1024 at severe imbalance"
        print("  PASS: severe imbalance -> c=1024 (full colocation)")

    extreme = df[(df["batch"] == 4096) & (df["imbalance"] >= 16)]
    if not extreme.empty:
        assert (extreme["num_complete_sets"] == 2).all(), "FAIL: s != 2 at B=4096, rho>=16"
        print("  PASS: B=4096, rho>=16 -> s=2")


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

    results = {}
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

        collapsed = verify_context_invariant(best, name)
        results[name] = collapsed

        out_path = out / f"{name}_best_configs.csv"
        collapsed.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}")

    verify_fc(results["fc128x32"])
    verify_torus(results["torus3d"])
    verify_hyperx(results["hyperx"])

    print(f"\n{'='*50}")
    print("RESULT: ALL CHECKS PASSED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
