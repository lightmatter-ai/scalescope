import csv
from pathlib import Path

import pytest

from conf26doc.plot_conf26_regime_config_maps import (
    BATCHES_FC,
    BATCHES_HYPERX_TORUS,
    DEFAULT_RHOS,
    get_regime_map,
)
from conf26doc.plot_resource_bottleneck_heatmap_3topo import _prepare

pytestmark = pytest.mark.conf26_paper_regression


REPO_ROOT = Path(__file__).resolve().parents[1]
PRECURSOR_DIR = REPO_ROOT / "out/conf26/paper_artifacts/precursor_sweeps"

SUMMARY_PATHS = {
    "hyperx": PRECURSOR_DIR / "summary_hyperx.csv",
    "torus3d": PRECURSOR_DIR / "summary_torus3d.csv",
    "fc128x32": PRECURSOR_DIR / "summary_fc128x32.csv",
}

CANONICAL_SPEEDUPS_CSV = PRECURSOR_DIR / "cross_topology_speedups.csv"

# ---------------------------------------------------------------------------
# Table II regression (rho=8)
# ---------------------------------------------------------------------------

EXPECTED_TABLE2_RHO8 = {
    ("1K", 32): (9.68, 11.66),
    ("1K", 128): (8.64, 10.07),
    ("1K", 1024): (5.35, 4.98),
    ("1K", 4096): (2.75, 1.86),
    ("32K", 32): (9.44, 11.36),
    ("32K", 128): (7.94, 9.23),
    ("32K", 1024): (4.13, 3.86),
    ("32K", 4096): (2.12, 1.55),
    ("131K", 32): (8.75, 10.52),
    ("131K", 128): (6.40, 7.41),
    ("131K", 1024): (2.67, 2.52),
    ("131K", 4096): (1.53, 1.26),
    ("2M", 32): (3.95, 4.62),
    ("2M", 128): (1.99, 2.18),
    ("2M", 1024): (1.16, 1.15),
    ("2M", 4096): (1.05, 1.02),
}

CONTEXT_LABEL_BY_TOKENS = {
    1024: "1K",
    32768: "32K",
    131072: "131K",
    2097152: "2M",
}


def _load_canonical_rows() -> list[dict[str, str]]:
    if not CANONICAL_SPEEDUPS_CSV.is_file():
        raise FileNotFoundError(
            f"Missing canonical regression input: {CANONICAL_SPEEDUPS_CSV}. "
            "Regenerate publication outputs before running this test."
        )
    with CANONICAL_SPEEDUPS_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _norm_imbalance(value: str) -> int:
    return int(float(value.strip()))


def test_table2_rho8_values_match_paper_regression():
    rows = _load_canonical_rows()
    observed: dict[tuple[str, int], tuple[float, float]] = {}

    for row in rows:
        if _norm_imbalance(row["imbalance"]) != 8:
            continue
        context_tokens = int(float(row["context"]))
        batch = int(float(row["batch"]))
        context_label = CONTEXT_LABEL_BY_TOKENS[context_tokens]
        observed[(context_label, batch)] = (
            round(float(row["HyperX_vs_Torus"]), 2),
            round(float(row["HyperX_vs_FC128x32"]), 2),
        )

    assert set(observed) == set(EXPECTED_TABLE2_RHO8), (
        "Row set mismatch for paper Table II regression.\n"
        f"Observed keys: {sorted(observed)}\n"
        f"Expected keys: {sorted(EXPECTED_TABLE2_RHO8)}"
    )

    for key, expected in EXPECTED_TABLE2_RHO8.items():
        assert observed[key] == expected, (
            f"Paper Table II regression changed at {key}: "
            f"observed={observed[key]} expected={expected}"
        )


def test_headline_max_speedups_regression():
    """Verify the 'up to X x' claims in the Introduction and Conclusion.

    The paper reports 'up to 11.7x over the rack baseline and up to 9.7x
    over the torus'.  These are the rho=8 maximums from Table II (11.66
    and 9.68, rounding to 11.7 and 9.7 at one decimal).
    """
    rows = _load_canonical_rows()
    rho8_rows = [r for r in rows if _norm_imbalance(r["imbalance"]) == 8]
    max_vs_torus = max(float(r["HyperX_vs_Torus"]) for r in rho8_rows)
    max_vs_fc = max(float(r["HyperX_vs_FC128x32"]) for r in rho8_rows)

    assert round(max_vs_torus, 2) == 9.68
    assert round(max_vs_fc, 2) == 11.66
    assert round(max_vs_torus, 1) == 9.7
    assert round(max_vs_fc, 1) == 11.7


# ---------------------------------------------------------------------------
# Figure 4 regime-map regression
# ---------------------------------------------------------------------------

# Tuple schema:
# (batch, rho, tp_moe_shape, colocated_experts, g, router_mode, num_complete_sets)
EXPECTED_FIG4 = {
    "hyperx": [
        (1, 1, "1x128", 32, 1, "sharded", 1),
        (1, 4, "1x128", 32, 1, "sharded", 1),
        (1, 8, "1x128", 32, 1, "sharded", 1),
        (1, 16, "1x128", 32, 1, "sharded", 1),
        (1, 32, "1x128", 32, 1, "sharded", 1),
        (16, 1, "1x128", 32, 1, "sharded", 1),
        (16, 4, "1x128", 32, 1, "sharded", 1),
        (16, 8, "1x128", 32, 1, "sharded", 1),
        (16, 16, "1x128", 32, 1, "sharded", 1),
        (16, 32, "1x128", 32, 1, "sharded", 1),
        (32, 1, "1x128", 32, 1, "sharded", 1),
        (32, 4, "1x128", 32, 1, "sharded", 1),
        (32, 8, "1x128", 32, 1, "sharded", 1),
        (32, 16, "1x128", 32, 1, "sharded", 1),
        (32, 32, "1x32", 8, 4, "sharded", 1),
        (128, 1, "1x128", 32, 1, "sharded", 1),
        (128, 4, "1x128", 32, 1, "sharded", 1),
        (128, 8, "1x32", 8, 4, "sharded", 1),
        (128, 16, "1x32", 8, 4, "sharded", 1),
        (128, 32, "32x128", 1024, 1, "sharded", 1),
        (1024, 1, "1x64", 16, 2, "sharded", 1),
        (1024, 4, "1x32", 8, 4, "sharded", 1),
        (1024, 8, "1x32", 8, 4, "sharded", 1),
        (1024, 16, "8x32", 64, 4, "sharded", 1),
        (1024, 32, "32x128", 1024, 1, "sharded", 1),
        (4096, 1, "1x64", 16, 2, "sharded", 1),
        (4096, 4, "1x32", 8, 4, "sharded", 1),
        (4096, 8, "1x32", 8, 4, "sharded", 1),
        (4096, 16, "16x128", 1024, 1, "sharded", 2),
        (4096, 32, "16x128", 1024, 1, "sharded", 2),
    ],
    "torus3d": [
        (1, 1, "1x128", 32, 32, "replicated", 1),
        (1, 4, "1x128", 32, 32, "replicated", 1),
        (1, 8, "1x128", 32, 32, "replicated", 1),
        (1, 16, "1x128", 32, 32, "replicated", 1),
        (1, 32, "1x128", 32, 32, "replicated", 1),
        (16, 1, "1x128", 32, 32, "replicated", 1),
        (16, 4, "1x128", 32, 32, "replicated", 1),
        (16, 8, "1x128", 32, 32, "replicated", 1),
        (16, 16, "1x128", 32, 32, "replicated", 1),
        (16, 32, "1x128", 32, 32, "replicated", 1),
        (32, 1, "1x128", 32, 32, "replicated", 1),
        (32, 4, "1x128", 32, 32, "replicated", 1),
        (32, 8, "1x128", 32, 32, "replicated", 1),
        (32, 16, "1x128", 32, 32, "replicated", 1),
        (32, 32, "1x128", 32, 32, "replicated", 1),
        (128, 1, "1x128", 32, 32, "replicated", 1),
        (128, 4, "1x128", 32, 32, "replicated", 1),
        (128, 8, "1x128", 32, 32, "replicated", 1),
        (128, 16, "1x128", 32, 32, "replicated", 1),
        (128, 32, "1x128", 32, 32, "replicated", 1),
        (1024, 1, "16x8", 32, 32, "sharded", 1),
        (1024, 4, "16x8", 32, 32, "sharded", 1),
        (1024, 8, "16x8", 32, 32, "sharded", 1),
        (1024, 16, "16x8", 32, 32, "sharded", 1),
        (1024, 32, "16x8", 32, 32, "sharded", 1),
        (4096, 1, "4x32", 32, 8, "sharded", 1),
        (4096, 4, "4x32", 32, 8, "sharded", 1),
        (4096, 8, "4x32", 32, 8, "sharded", 1),
        (4096, 16, "4x32", 32, 8, "sharded", 1),
        (4096, 32, "4x32", 32, 8, "sharded", 1),
    ],
    "fc128x32": [
        (32, 1, "1x128", 1024, 1, "sharded", 32),
        (32, 4, "1x128", 1024, 1, "sharded", 32),
        (32, 8, "1x128", 1024, 1, "sharded", 32),
        (32, 16, "1x128", 1024, 1, "sharded", 32),
        (32, 32, "1x128", 1024, 1, "sharded", 32),
        (128, 1, "1x128", 1024, 1, "sharded", 32),
        (128, 4, "1x128", 1024, 1, "sharded", 32),
        (128, 8, "1x128", 1024, 1, "sharded", 32),
        (128, 16, "1x128", 1024, 1, "sharded", 32),
        (128, 32, "1x128", 1024, 1, "sharded", 32),
        (1024, 1, "1x1", 8, 1, "replicated", 32),
        (1024, 4, "1x4", 32, 1, "sharded", 32),
        (1024, 8, "1x4", 32, 1, "sharded", 32),
        (1024, 16, "1x4", 32, 1, "sharded", 32),
        (1024, 32, "1x4", 32, 1, "sharded", 32),
        (4096, 1, "1x4", 32, 1, "sharded", 32),
        (4096, 4, "1x4", 32, 1, "sharded", 32),
        (4096, 8, "1x4", 32, 1, "sharded", 32),
        (4096, 16, "1x4", 32, 1, "sharded", 32),
        (4096, 32, "1x4", 32, 1, "sharded", 32),
    ],
}

# ---------------------------------------------------------------------------
# Figure 5 bottleneck-grid regression (rho=8, min_batch=32)
# ---------------------------------------------------------------------------

# Tuple schema: (context_tokens, batch, dominant_resource, dominant_pct_rounded_1dp)
EXPECTED_FIG5 = {
    "hyperx": [
        (1024, 32, "comm", 64.1),
        (1024, 128, "comm", 66.3),
        (1024, 1024, "comm", 74.5),
        (1024, 4096, "comm", 79.4),
        (32768, 32, "comm", 62.3),
        (32768, 128, "comm", 60.2),
        (32768, 1024, "comm", 53.5),
        (32768, 4096, "comm", 50.6),
        (131072, 32, "comm", 57.2),
        (131072, 128, "mem", 52.4),
        (131072, 1024, "mem", 69.8),
        (131072, 4096, "mem", 74.3),
        (2097152, 32, "mem", 77.6),
        (2097152, 128, "mem", 90.6),
        (2097152, 1024, "mem", 96.3),
        (2097152, 4096, "mem", 97.0),
    ],
    "torus3d": [
        (1024, 32, "comm", 93.2),
        (1024, 128, "comm", 92.5),
        (1024, 1024, "comm", 92.5),
        (1024, 4096, "comm", 92.4),
        (32768, 32, "comm", 92.9),
        (32768, 128, "comm", 91.5),
        (32768, 1024, "comm", 86.2),
        (32768, 4096, "comm", 76.6),
        (131072, 32, "comm", 92.0),
        (131072, 128, "comm", 88.3),
        (131072, 1024, "comm", 71.1),
        (131072, 4096, "comm", 50.1),
        (2097152, 32, "comm", 77.6),
        (2097152, 128, "comm", 52.2),
        (2097152, 1024, "mem", 83.2),
        (2097152, 4096, "mem", 92.8),
    ],
    "fc128x32": [
        (1024, 32, "mem", 95.7),
        (1024, 128, "mem", 95.4),
        (1024, 1024, "mem", 94.4),
        (1024, 4096, "mem", 92.8),
        (32768, 32, "mem", 95.7),
        (32768, 128, "mem", 95.5),
        (32768, 1024, "mem", 94.8),
        (32768, 4096, "mem", 94.3),
        (131072, 32, "mem", 95.8),
        (131072, 128, "mem", 95.6),
        (131072, 1024, "mem", 95.6),
        (131072, 4096, "mem", 96.4),
        (2097152, 32, "mem", 96.2),
        (2097152, 128, "mem", 96.9),
        (2097152, 1024, "mem", 98.4),
        (2097152, 4096, "mem", 98.9),
    ],
}


def _assert_precursor_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing paper precursor summary: {path}. "
            "Run paper artifact generation before regression tests."
        )


def test_figure4_regime_grid_regression():
    for topology, summary_path in SUMMARY_PATHS.items():
        _assert_precursor_exists(summary_path)
        best = get_regime_map(summary_path, context=1024)
        batches = BATCHES_FC if topology == "fc128x32" else BATCHES_HYPERX_TORUS

        observed: list[tuple[int, int, str, int, int, str, int]] = []
        for batch in batches:
            for rho in DEFAULT_RHOS:
                row = best[(batch, float(rho))]
                observed.append(
                    (
                        batch,
                        int(rho),
                        str(row["tp_moe_shape"]),
                        int(row["coloc"]),
                        int(row["g"]),
                        str(row["router"]),
                        int(row["sets"]),
                    )
                )
        assert observed == EXPECTED_FIG4[topology]


def test_figure5_bottleneck_grid_regression():
    for topology, summary_path in SUMMARY_PATHS.items():
        _assert_precursor_exists(summary_path)
        prepared = _prepare(summary_path, rho=8.0)
        prepared = prepared[prepared["batch"] >= 32].copy()
        batches = sorted(prepared["batch"].astype(int).unique().tolist())
        contexts = sorted(prepared["context"].astype(int).unique().tolist())

        observed: list[tuple[int, int, str, float]] = []
        for context in contexts:
            for batch in batches:
                row = prepared[
                    (prepared["batch"] == batch) & (prepared["context"] == context)
                ].iloc[0]
                observed.append(
                    (
                        int(context),
                        int(batch),
                        str(row["dominant"]),
                        round(float(row["dominant_pct"]), 1),
                    )
                )
        assert observed == EXPECTED_FIG5[topology]
