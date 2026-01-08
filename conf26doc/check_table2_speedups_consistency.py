#!/usr/bin/env python3
"""Verify paper Table 2 values against canonical rho=8 speedup CSV."""

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEX_PATH = REPO_ROOT / "out/conf26/paper_artifacts/table2_topo_speedups_rho8.tex"
CSV_PATH = REPO_ROOT / "out/conf26/paper_artifacts/precursor_sweeps/cross_topology_speedups.csv"

EXPECTED_CONTEXTS = ("1K", "32K", "131K", "2M")
EXPECTED_BATCHES = (32, 128, 1024, 4096)
CONTEXT_LABEL_TO_TOKENS = {
    "1K": "1024",
    "32K": "32768",
    "131K": "131072",
    "2M": "2097152",
}


def _norm_imbalance(value: str) -> str:
    return str(int(float(value.strip())))


def _parse_tex_rows(tex_text: str) -> list[tuple[str, int, float, float]]:
    row_pattern = re.compile(
        r"\s*([0-9A-Za-z]+)\s*&\s*(\d+)\s*&\s*"
        r"\$([0-9]+(?:\.[0-9]+)?)\\times\$\s*&\s*"
        r"\$([0-9]+(?:\.[0-9]+)?)\\times\$\s*\\\\"
    )
    rows: list[tuple[str, int, float, float]] = []
    for match in row_pattern.finditer(tex_text):
        context_label, batch, vs_torus, vs_fc = match.groups()
        rows.append((context_label, int(batch), float(vs_torus), float(vs_fc)))
    return rows


def _load_csv_rows(csv_path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    out: dict[tuple[str, int], tuple[float, float]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "batch",
            "context",
            "imbalance",
            "HyperX_vs_Torus",
            "HyperX_vs_FC128x32",
        }
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            if _norm_imbalance(row["imbalance"]) != "8":
                continue
            key = (row["context"].strip(), int(float(row["batch"].strip())))
            out[key] = (
                float(row["HyperX_vs_Torus"].strip()),
                float(row["HyperX_vs_FC128x32"].strip()),
            )
    return out


def main() -> int:
    if not TEX_PATH.is_file():
        print(f"ERROR: missing TeX table: {TEX_PATH}")
        return 1
    if not CSV_PATH.is_file():
        print(f"ERROR: missing canonical CSV: {CSV_PATH}")
        return 1

    tex_rows = _parse_tex_rows(TEX_PATH.read_text())
    if not tex_rows:
        print(f"ERROR: no table rows parsed from {TEX_PATH}")
        return 1

    csv_rows = _load_csv_rows(CSV_PATH)

    missing_in_csv: list[tuple[str, int]] = []
    value_mismatches: list[str] = []

    for context_label, batch, tex_vs_torus, tex_vs_fc in tex_rows:
        context_tokens = CONTEXT_LABEL_TO_TOKENS.get(context_label)
        if context_tokens is None:
            value_mismatches.append(f"Unknown context label in TeX row: {context_label!r}")
            continue

        key = (context_tokens, batch)
        if key not in csv_rows:
            missing_in_csv.append((context_label, batch))
            continue

        csv_vs_torus, csv_vs_fc = csv_rows[key]
        # Table stores values rounded to 2 decimals.
        if round(csv_vs_torus, 2) != round(tex_vs_torus, 2):
            value_mismatches.append(
                f"({context_label}, {batch}) HyperX vs Torus mismatch: "
                f"tex={tex_vs_torus:.2f} csv={csv_vs_torus:.6f}"
            )
        if round(csv_vs_fc, 2) != round(tex_vs_fc, 2):
            value_mismatches.append(
                f"({context_label}, {batch}) HyperX vs FC128x32 mismatch: "
                f"tex={tex_vs_fc:.2f} csv={csv_vs_fc:.6f}"
            )

    expected_rows = {(c, b) for c in EXPECTED_CONTEXTS for b in EXPECTED_BATCHES}
    tex_row_keys = {(c, b) for c, b, _, _ in tex_rows}
    missing_in_tex = sorted(expected_rows - tex_row_keys)
    extra_in_tex = sorted(tex_row_keys - expected_rows)

    print("Table2 consistency report")
    print(f"- TeX rows parsed: {len(tex_rows)}")
    print(f"- CSV rows at rho=8: {len(csv_rows)}")
    print(f"- Missing TeX rows in CSV: {len(missing_in_csv)}")
    print(f"- Value mismatches: {len(value_mismatches)}")
    print(f"- Missing expected rows in TeX: {len(missing_in_tex)}")
    print(f"- Extra rows in TeX: {len(extra_in_tex)}")

    for context_label, batch in missing_in_csv:
        print(f"  MISSING CSV ROW: ({context_label}, {batch})")
    for msg in value_mismatches:
        print(f"  MISMATCH: {msg}")
    for row in missing_in_tex:
        print(f"  MISSING TEX ROW: {row}")
    for row in extra_in_tex:
        print(f"  EXTRA TEX ROW: {row}")

    ok = not missing_in_csv and not value_mismatches and not missing_in_tex and not extra_in_tex
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
