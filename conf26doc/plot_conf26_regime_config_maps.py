#!/usr/bin/env python3
"""Plot winner configuration grids (batch × imbalance) from conf26 summary CSVs.

Reads per-row sweep results, picks minimum overall_s per (batch, imbalance) at a
fixed context length, and renders colored cells (TP_MoE shape) with c, g, s labels.

By default writes three PNGs (one per topology). Optional ``--combined`` also writes
a single figure with all three panels side by side.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# Locked paper palette (Figure 4 and companion regime maps).
# Keep this mapping stable for artifact reproducibility.
PAPER_SHAPE_COLORS: dict[str, str] = {
    "1x128": "#2C6A8F",
    "1x64": "#4A8DB8",
    "1x32": "#7CB8DA",
    "4x32": "#A3BECC",
    "8x32": "#9BC4D8",
    "16x8": "#A3BECC",
    "16x128": "#9B7EB6",
    "32x128": "#7A5C98",
    "1x4": "#C9A66B",
    "1x1": "#8B7355",
}
PAPER_TEXT_COLORS: dict[str, str] = {
    "1x128": "#FFFFFF",
    "1x64": "#FFFFFF",
    "1x32": "#1A3A4A",
    "4x32": "#1A3A4A",
    "8x32": "#1A3A4A",
    "16x8": "#1A3A4A",
    "16x128": "#FFFFFF",
    "32x128": "#FFFFFF",
    "1x4": "#1A3A4A",
    "1x1": "#FFFFFF",
}

DEFAULT_RHOS = (1.0, 4.0, 8.0, 16.0, 32.0)
BATCHES_HYPERX_TORUS = (1, 16, 32, 128, 1024, 4096)
BATCHES_FC = (32, 128, 1024, 4096)


def _palette_colors(tp_moe_shape: str) -> tuple[str, str]:
    if tp_moe_shape not in PAPER_SHAPE_COLORS:
        raise ValueError(
            f"Unknown tp_moe_shape '{tp_moe_shape}' for locked paper palette. "
            f"Add it explicitly to PAPER_SHAPE_COLORS and PAPER_TEXT_COLORS."
        )
    if tp_moe_shape not in PAPER_TEXT_COLORS:
        raise ValueError(
            f"Missing text color for tp_moe_shape '{tp_moe_shape}' in locked paper palette."
        )
    return PAPER_SHAPE_COLORS[tp_moe_shape], PAPER_TEXT_COLORS[tp_moe_shape]


def get_regime_map(summary_csv: Path, context: int) -> dict[tuple[int, float], dict]:
    df = pd.read_csv(summary_csv)
    if "context" not in df.columns:
        raise ValueError(f"Missing 'context' column in {summary_csv}")
    sub = df[df["context"] == context].copy()
    if sub.empty:
        raise ValueError(f"No rows with context={context} in {summary_csv}")

    best: dict[tuple[int, float], dict] = {}
    for row in sub.itertuples(index=False):
        batch = int(row.batch)
        rho = float(row.imbalance)
        overall = float(row.overall_s)
        key = (batch, rho)
        rec = {
            "overall": overall,
            "g": int(row.g),
            "coloc": int(row.colocated_experts),
            "tp_moe_shape": str(row.tp_moe_shape),
            "router": str(row.router_mode),
            "sets": int(row.num_complete_sets),
        }
        if key not in best or overall < best[key]["overall"]:
            best[key] = rec
    return best


def plot_regime_panel(
    ax,
    best: dict[tuple[int, float], dict],
    batches: tuple[int, ...],
    rhos: tuple[float, ...],
    title: str,
    *,
    cell_w: float,
    cell_h: float,
    pad: float,
) -> None:
    for i, b in enumerate(batches):
        for j, r in enumerate(rhos):
            key = (b, r)
            if key not in best:
                continue
            v = best[key]
            shape = v["tp_moe_shape"]
            color, tc = _palette_colors(shape)

            x = j * (cell_w + pad)
            y = (len(batches) - 1 - i) * (cell_h + pad)

            rect = FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="#F5F5F5",
                linewidth=1.2,
            )
            ax.add_patch(rect)

            ax.text(
                x + cell_w / 2,
                y + cell_h * 0.72,
                shape,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=tc,
            )
            ax.text(
                x + cell_w / 2,
                y + cell_h * 0.44,
                f"c={v['coloc']}  g={v['g']}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=tc,
                alpha=0.85,
            )
            ax.text(
                x + cell_w / 2,
                y + cell_h * 0.2,
                f"s={v['sets']}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=tc,
                alpha=0.85,
            )

    for j, r in enumerate(rhos):
        ax.text(
            j * (cell_w + pad) + cell_w / 2,
            -0.2,
            str(int(r)),
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
        )

    for i, b in enumerate(batches):
        y = (len(batches) - 1 - i) * (cell_h + pad) + cell_h / 2
        ax.text(-0.2, y, str(b), ha="right", va="center", fontsize=9, color="#444444")

    mid_x = (len(rhos) - 1) / 2 * (cell_w + pad) + cell_w / 2
    mid_y = (len(batches) - 1) / 2 * (cell_h + pad) + cell_h / 2
    ax.text(mid_x, -0.68, "Imbalance factor (ρ)", ha="center", fontsize=9.5, color="#333333")
    ax.text(
        -0.85,
        mid_y,
        "Batch size",
        ha="center",
        va="center",
        fontsize=9.5,
        rotation=90,
        color="#333333",
    )
    ax.text(
        mid_x,
        len(batches) * (cell_h + pad) + 0.35,
        title,
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
    )

    ax.set_xlim(-0.12, len(rhos) * (cell_w + pad) + 0.05)
    ax.set_ylim(-0.7, len(batches) * (cell_h + pad) + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")


def _prefixed_name(out_dir: Path, prefix: str, stem: str) -> Path:
    return out_dir / f"{prefix}{stem}.png"


def main() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/conf26/build/sweeps"),
        help="Directory containing summary CSV subfolders and receiving output PNGs.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=1024,
        help="Context length to fix for the regime map (default: 1024).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Prepended to each output filename stem (e.g. prefix 'paper_' yields "
            "paper_hyperx_regime_map.png). Use empty string for default stems."
        ),
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write a single three-panel figure (all topologies side by side).",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    paths = {
        "hyperx": out_dir / "generalized_ep_hyperx" / "summary.csv",
        "torus": out_dir / "generalized_ep_torus3d" / "summary.csv",
        "fc": out_dir / "fc128x32_disagg_ep_baseline" / "summary.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} summary: {p}")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})

    cell_w = 1.12
    cell_h = 0.82
    pad = 0.06

    hyperx_best = get_regime_map(paths["hyperx"], args.context)
    torus_best = get_regime_map(paths["torus"], args.context)
    fc_best = get_regime_map(paths["fc"], args.context)

    written: list[Path] = []

    for best, batches, title, stem in (
        (hyperx_best, BATCHES_HYPERX_TORUS, "HyperX", "hyperx_regime_map"),
        (torus_best, BATCHES_HYPERX_TORUS, "Torus 3D", "torus3d_regime_map"),
        (fc_best, BATCHES_FC, "FC128x32", "fc128x32_regime_map"),
    ):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
        plot_regime_panel(
            ax,
            best,
            batches,
            DEFAULT_RHOS,
            title,
            cell_w=cell_w,
            cell_h=cell_h,
            pad=pad,
        )
        out_path = _prefixed_name(out_dir, prefix, stem)
        fig.savefig(
            out_path,
            dpi=250,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        written.append(out_path)

    if args.combined:
        fig3, axes = plt.subplots(1, 3, figsize=(18.5, 5.2))
        plot_regime_panel(
            axes[0],
            hyperx_best,
            BATCHES_HYPERX_TORUS,
            DEFAULT_RHOS,
            "HyperX",
            cell_w=cell_w,
            cell_h=cell_h,
            pad=pad,
        )
        plot_regime_panel(
            axes[1],
            torus_best,
            BATCHES_HYPERX_TORUS,
            DEFAULT_RHOS,
            "Torus 3D",
            cell_w=cell_w,
            cell_h=cell_h,
            pad=pad,
        )
        plot_regime_panel(
            axes[2],
            fc_best,
            BATCHES_FC,
            DEFAULT_RHOS,
            "FC128x32",
            cell_w=cell_w,
            cell_h=cell_h,
            pad=pad,
        )
        plt.tight_layout(w_pad=1.2)
        three_png = _prefixed_name(out_dir, prefix, "regime_config_maps_3topo")
        fig3.savefig(
            three_png,
            dpi=250,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig3)
        written.append(three_png)

    for p in written:
        print("Wrote:", p)


if __name__ == "__main__":
    main()
