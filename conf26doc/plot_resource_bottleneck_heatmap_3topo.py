#!/usr/bin/env python3
"""Plot a three-topology bottleneck heatmap (Figure 5).

For each topology (HyperX, 3D torus, FC128x32) at a fixed imbalance rho,
classifies every (batch, context) operating point by its dominant resource
cost — communication or memory — and renders the result as a side-by-side
heatmap grid with per-cell percentage annotations.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.patches import FancyBboxPatch

COLOR = {"comm": "#F28E2B", "mem": "#72B7B2", "compute": "#4C78A8"}
LABEL = {"comm": "C", "mem": "M", "compute": "P"}


def _prepare(summary_csv: Path, rho: float) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    need = [
        "batch",
        "context",
        "imbalance",
        "overall_s",
        "projection_compute_s",
        "projection_mem_s",
        "projection_comm_s",
        "core_attn_compute_s",
        "core_attn_mem_s",
        "core_attn_comm_s",
        "moe_compute_s",
        "moe_mem_s",
        "moe_comm_s",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {summary_csv}: {missing}")
    d = df[df["imbalance"] == rho].copy()
    if d.empty:
        raise ValueError(f"No rows with imbalance={rho} in {summary_csv}")
    idx = d.groupby(["batch", "context"])["overall_s"].idxmin()
    w = d.loc[idx].copy()
    w["compute_s"] = w["projection_compute_s"] + w["core_attn_compute_s"] + w["moe_compute_s"]
    w["mem_s"] = w["projection_mem_s"] + w["core_attn_mem_s"] + w["moe_mem_s"]
    w["comm_s"] = w["projection_comm_s"] + w["core_attn_comm_s"] + w["moe_comm_s"]

    def _dom(r: pd.Series) -> str:
        vals = {
            "compute": float(r["compute_s"]),
            "mem": float(r["mem_s"]),
            "comm": float(r["comm_s"]),
        }
        return max(vals, key=vals.get)

    w["dominant"] = w.apply(_dom, axis=1)
    w["dominant_pct"] = w.apply(
        lambda r: 100.0
        * float(r[r["dominant"] + "_s"])
        / (float(r["compute_s"]) + float(r["mem_s"]) + float(r["comm_s"])),
        axis=1,
    )
    return w


def _plot_panel(
    ax,
    df: pd.DataFrame,
    title: str,
    batches: list[int],
    contexts: list[int],
    *,
    context_increases_up: bool,
    show_axis_titles: bool,
) -> None:
    cell_w = 1.10
    cell_h = 0.80
    pad = 0.08
    for i, c in enumerate(contexts):
        for j, b in enumerate(batches):
            row = df[(df["batch"] == b) & (df["context"] == c)]
            if row.empty:
                continue
            r = row.iloc[0]
            dom = str(r["dominant"])
            pct = float(r["dominant_pct"])
            x = j * (cell_w + pad)
            y = (
                i * (cell_h + pad)
                if context_increases_up
                else (len(contexts) - 1 - i) * (cell_h + pad)
            )
            rect = FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.02",
                facecolor=COLOR[dom],
                edgecolor="#F5F5F5",
                linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(
                x + cell_w / 2,
                y + cell_h * 0.62,
                LABEL[dom],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
            ax.text(
                x + cell_w / 2,
                y + cell_h * 0.28,
                f"{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
            )

    for j, b in enumerate(batches):
        ax.text(
            j * (cell_w + pad) + cell_w / 2,
            -0.20,
            str(b),
            ha="center",
            va="top",
            fontsize=8.8,
            color="#444",
        )
    for i, c in enumerate(contexts):
        y = (
            (i * (cell_h + pad) + cell_h / 2)
            if context_increases_up
            else ((len(contexts) - 1 - i) * (cell_h + pad) + cell_h / 2)
        )
        c_label = f"{int(c/1024)}K" if c < 1_000_000 else "2M"
        ax.text(-0.18, y, c_label, ha="right", va="center", fontsize=8.8, color="#444")

    mid_x = (len(batches) - 1) / 2 * (cell_w + pad) + cell_w / 2
    mid_y = (len(contexts) - 1) / 2 * (cell_h + pad) + cell_h / 2
    if show_axis_titles:
        ax.text(mid_x, -0.48, "Batch", ha="center", fontsize=9.2, color="#333")
        ax.text(
            -0.58,
            mid_y,
            "Context",
            ha="center",
            va="center",
            fontsize=9.2,
            color="#333",
            rotation=90,
        )
    ax.text(
        mid_x,
        len(contexts) * (cell_h + pad) + 0.24,
        title,
        ha="center",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color="#222",
    )

    ax.set_xlim(-0.10, len(batches) * (cell_w + pad) + 0.02)
    ax.set_ylim(-0.62, len(contexts) * (cell_h + pad) + 0.42)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=float, default=8.0)
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Exclude rows with batch < min-batch (default: 1).",
    )
    parser.add_argument(
        "--context-increases-up",
        action="store_true",
        help="Plot context so values increase upward on Y axis.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not render the figure-level title.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not render the figure-level legend text.",
    )
    parser.add_argument(
        "--single-axis-labels",
        action="store_true",
        help="Render shared figure-level Batch/Context labels only once.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("out/conf26/build/resource_bottleneck_heatmap_3topo_rho32.png"),
    )
    parser.add_argument(
        "--hyperx",
        type=Path,
        default=Path("out/conf26/build/sweeps/generalized_ep_hyperx/summary.csv"),
    )
    parser.add_argument(
        "--torus",
        type=Path,
        default=Path("out/conf26/build/sweeps/generalized_ep_torus3d/summary.csv"),
    )
    parser.add_argument(
        "--fc",
        type=Path,
        default=Path("out/conf26/build/sweeps/fc128x32_disagg_ep_baseline/summary.csv"),
    )
    args = parser.parse_args()

    hx = _prepare(args.hyperx, args.rho)
    tr = _prepare(args.torus, args.rho)
    fc = _prepare(args.fc, args.rho)
    hx = hx[hx["batch"] >= args.min_batch].copy()
    tr = tr[tr["batch"] >= args.min_batch].copy()
    fc = fc[fc["batch"] >= args.min_batch].copy()

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    _plot_panel(
        axes[0],
        hx,
        "HyperX",
        batches=sorted(hx["batch"].astype(int).unique().tolist()),
        contexts=sorted(hx["context"].astype(int).unique().tolist()),
        context_increases_up=args.context_increases_up,
        show_axis_titles=not args.single_axis_labels,
    )
    _plot_panel(
        axes[1],
        tr,
        "Torus 3D",
        batches=sorted(tr["batch"].astype(int).unique().tolist()),
        contexts=sorted(tr["context"].astype(int).unique().tolist()),
        context_increases_up=args.context_increases_up,
        show_axis_titles=not args.single_axis_labels,
    )
    _plot_panel(
        axes[2],
        fc,
        "FC128x32",
        batches=sorted(fc["batch"].astype(int).unique().tolist()),
        contexts=sorted(fc["context"].astype(int).unique().tolist()),
        context_increases_up=args.context_increases_up,
        show_axis_titles=not args.single_axis_labels,
    )
    if args.single_axis_labels:
        fig.text(0.5, 0.03, "Batch", ha="center", fontsize=9.4, color="#333333")
        fig.text(0.01, 0.5, "Context", va="center", rotation=90, fontsize=9.4, color="#333333")
    if not args.no_title:
        fig.suptitle(
            f"Dominant resource bottleneck by regime (rho={int(args.rho) if float(args.rho).is_integer() else args.rho})",
            y=1.02,
            fontsize=11,
        )
    if not args.no_legend:
        fig.text(0.5, 0.01, "Legend: C=Communication, M=Memory, P=Compute", ha="center", fontsize=9)
    fig.tight_layout(w_pad=1.1)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=250, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Wrote:", args.out_png)


if __name__ == "__main__":
    main()
