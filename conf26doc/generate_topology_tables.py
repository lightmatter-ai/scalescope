#!/usr/bin/env python3
"""Generate cross-topology speedup tables from per-topology sweep summaries.

Reads the HyperX, 3D torus, and FC128x32 summary CSVs produced by the
publication sweeps, selects the best configuration per (batch, context,
imbalance) point, and computes pairwise TPOT speedup ratios across
topologies.
"""
import argparse
from pathlib import Path

import pandas as pd


def _read_summary(path: Path, topology: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary for {topology}: {path}")
    df = pd.read_csv(path)
    required = {
        "batch",
        "context",
        "imbalance",
        "colocated_experts",
        "router_mode",
        "g",
        "tp_moe_shape",
        "expert_tp",
        "overall_s",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    df["topology"] = topology
    return df


def _best_by_workload(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    idx = df.groupby(group_cols)["overall_s"].idxmin()
    selected_cols = [
        *group_cols,
        "colocated_experts",
        "router_mode",
        "g",
        "tp_moe_shape",
        "expert_tp",
        "overall_s",
    ]
    if "topology" not in group_cols:
        selected_cols.insert(len(group_cols), "topology")
    return (
        df.loc[idx, selected_cols]
        .copy()
        .sort_values(group_cols if "topology" in group_cols else group_cols + ["topology"])
        .reset_index(drop=True)
    )


def _cross_topology_speedups(
    best_per_topology: pd.DataFrame,
    output_path: Path,
) -> None:
    key_cols = ["batch", "context", "imbalance"]
    piv = (
        best_per_topology.pivot_table(
            index=key_cols,
            columns="topology",
            values="overall_s",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    required_topologies = ["HyperX", "Torus", "FC128x32"]
    missing = [c for c in required_topologies if c not in piv.columns]
    if missing:
        raise ValueError(f"Cannot compute speedups; missing topologies: {missing}")

    # Keep only rows where all three topology values are present.
    piv = piv.dropna(subset=required_topologies).copy()

    piv["HyperX_vs_Torus"] = piv["Torus"] / piv["HyperX"]
    piv["HyperX_vs_FC128x32"] = piv["FC128x32"] / piv["HyperX"]
    piv["Torus_vs_FC128x32"] = piv["FC128x32"] / piv["Torus"]
    piv = piv.sort_values(key_cols).reset_index(drop=True)
    piv.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate cross-topology speedup tables from conf26 summary CSVs."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/conf26/build"),
        help="Directory containing and receiving conf26 CSV files.",
    )
    parser.add_argument(
        "--hyperx-summary",
        type=Path,
        default=Path("out/conf26/build/sweeps/generalized_ep_hyperx/summary.csv"),
    )
    parser.add_argument(
        "--torus-summary",
        type=Path,
        default=Path("out/conf26/build/sweeps/generalized_ep_torus3d/summary.csv"),
    )
    parser.add_argument(
        "--fc-summary",
        type=Path,
        default=Path("out/conf26/build/sweeps/fc128x32_disagg_ep_baseline/summary.csv"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    hyperx = _read_summary(args.hyperx_summary, "HyperX")
    torus = _read_summary(args.torus_summary, "Torus")
    fc = _read_summary(args.fc_summary, "FC128x32")
    all_df = pd.concat([hyperx, torus, fc], ignore_index=True)

    key_cols = ["batch", "context", "imbalance", "topology"]

    best_overall = _best_by_workload(all_df, key_cols)
    _cross_topology_speedups(
        best_overall,
        args.out_dir / "cross_topology_speedups.csv",
    )

    print("Wrote:", args.out_dir / "cross_topology_speedups.csv")


if __name__ == "__main__":
    main()
