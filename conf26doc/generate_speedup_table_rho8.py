#!/usr/bin/env python3
"""Extract a fixed-rho slice from the cross-topology speedup CSV and emit
Table II artifacts (CSV + LaTeX) comparing HyperX against 3D torus and
FC128x32 baselines.
"""

import argparse
from pathlib import Path

import pandas as pd

COL_VS_TORUS = "HyperX_vs_Torus"
COL_VS_FC = "HyperX_vs_FC128x32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("out/conf26/build/cross_topology_speedups.csv"),
        help="Cross-topology speedups CSV.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=8.0,
        help="Fixed imbalance value (default: 8).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("out/conf26/build/topo_speedups_rho8_table.csv"),
        help="Output CSV path for table rows.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=Path("out/conf26/build/topo_speedups_rho8_table.tex"),
        help="Output LaTeX table path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    need = ["batch", "context", "imbalance", COL_VS_TORUS, COL_VS_FC]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {missing}")

    contexts = [1024, 32768, 131072, 2097152]
    batches = [32, 128, 1024, 4096]
    rows: list[dict[str, float | int | str]] = []

    for t in contexts:
        for b in batches:
            sel = df[(df["context"] == t) & (df["batch"] == b) & (df["imbalance"] == args.rho)]
            if sel.empty:
                raise ValueError(
                    f"Missing row for context={t}, batch={b}, rho={args.rho} in {args.input_csv}"
                )
            r = sel.iloc[0]
            rows.append(
                {
                    "context": int(t),
                    "batch": int(b),
                    "rho": float(args.rho),
                    "hyperx_vs_torus": float(r[COL_VS_TORUS]),
                    "hyperx_vs_fc128x32": float(r[COL_VS_FC]),
                }
            )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    def _ctx_label(c: int) -> str:
        if c == 1024:
            return "1K"
        if c == 32768:
            return "32K"
        if c == 131072:
            return "131K"
        if c == 2097152:
            return "2M"
        return str(c)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{HyperX speedup over torus and FC128x32 after per-topology optimization",
        f"at representative operating points with fixed $\\rho={int(args.rho) if float(args.rho).is_integer() else args.rho}$. Speedup $>1$ means HyperX is faster.}}",
        "\\label{tab:topo-speedups-rho8}",
        "\\small",
        "\\begin{tabular}{rrcc}",
        "\\toprule",
        "$t$ & $B$ & HyperX vs.\\ Torus & HyperX vs.\\ FC128x32 \\\\",
        "\\midrule",
    ]
    for i, c in enumerate(contexts):
        block = out[out["context"] == c]
        for _, r in block.iterrows():
            lines.append(
                f"{_ctx_label(int(r['context']))} & {int(r['batch'])} & "
                f"${r['hyperx_vs_torus']:.2f}\\times$ & "
                f"${r['hyperx_vs_fc128x32']:.2f}\\times$ \\\\"
            )
        if i < len(contexts) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    args.out_tex.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", args.out_csv)
    print("Wrote:", args.out_tex)


if __name__ == "__main__":
    main()
