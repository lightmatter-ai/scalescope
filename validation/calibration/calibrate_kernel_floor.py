"""Fit and audit the per-model kernel-launch floor from the measured panel.

The kernel-launch floor (`KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` in
`run_validation_panel.py`) is a per-layer-per-decode-step host overhead the
roofline does not model. It is hand-set in the driver; this script computes
what those values *should* be from the panel, and audits how well the current
ones are centered. It is read-only: it loads an existing
`validation_panel_summary.yaml` and never runs ScaleScope or edits any config.

It covers every model with a measured per-token latency, in BOTH the disagg
and non-disagg regimes (the floor is injected into every decode prediction
regardless). Outputs, in order:

  0. FITTED FLOOR (primary). For every model, the suggested floor in us/layer.
     The floor is additive to decode TPOT and the operating point (batch=conc)
     is fixed, so the value that lands prediction on measurement is recovered
     exactly from any panel, regardless of the floor that generated it:
         floor_fit = floor_current + (measured_tpot - predicted_tpot) / n_layers
     Reported as the median of per-row fits over clean small-batch rows (where
     the floor dominates and the prefill α-correction is ~1), with a per-GPU
     median so a per-(model, gpu_class) split is an easy call.
  1. Per-GPU small-batch centering: geomean TPOT ratio (predicted / measured)
     per GPU class. ~1.00x means the current floor is centered there.
  2. Matched-cell head-to-head (models on >= 2 GPU classes): compares identical
     (TP, conc, ISL, OSL) cells between GPU pairs, isolating the per-GPU
     component the single shared floor cannot capture.
  3. Coverage: cross-checks every floor-table model against the panel, so
     throughput-only models (no measured TPOT) are reported as not-fittable
     rather than silently omitted.

A |geomean - 1| above --flag-threshold (or a matched-pair gap above it) is
flagged as a candidate for promoting the floor key to (model, gpu_class).
The fitted values are suggestions; applying them is a manual edit to the
driver followed by re-running the panel.

Run:
    python validation/calibration/calibrate_kernel_floor.py
    python validation/calibration/calibrate_kernel_floor.py --model llama3_70b
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

VALIDATION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PANEL = VALIDATION_DIR / "validation_results" / "validation_panel_summary.yaml"
DRIVER = VALIDATION_DIR / "run_validation_panel.py"
MODEL_ZOO = VALIDATION_DIR / "model_zoo"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _n_layers(model_id: str) -> int:
    """Transformer layer count for a model, read from its model_zoo spec.

    The kernel-launch floor is a per-layer-per-decode-step overhead, so the
    fit divides the measured per-token residual by this count.
    """
    path = MODEL_ZOO / f"{model_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No model_zoo spec for {model_id!r} at {path}; cannot fit a "
            f"per-layer floor without its layer count."
        )
    with path.open() as f:
        spec = yaml.safe_load(f)
    return int(spec["n_layers"])


def _implied_floor_sec(row: dict, floor_current: float, n_layers: int) -> float:
    """Exact per-row floor that would make this prediction match measurement.

    The floor enters decode TPOT additively (`predicted = roofline +
    floor * n_layers`) and does not move the operating point (batch = conc is
    fixed), so the roofline term is identical whatever floor was used. Hence
    the floor that lands prediction on the measured value is recovered in one
    step from any panel, regardless of the floor that generated it:

        floor_new = floor_current + (measured_tpot - predicted_tpot) / n_layers

    (TPOT in ms -> seconds via /1000; result in seconds/layer.)
    """
    gap_sec_per_layer = (row["measured_tpot_ms"] - row["predicted_tpot_ms"]) / 1000.0 / n_layers
    return floor_current + gap_sec_per_layer


def _load_floor_table() -> dict[str, dict[str, float]]:
    """Import KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER from the panel driver."""
    spec = importlib.util.spec_from_file_location("_rmp_floor", DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_rmp_floor"] = module
    spec.loader.exec_module(module)
    table = module.KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER
    out: dict[str, dict[str, float]] = {}
    for model_id, by_gpu in table.items():
        if not isinstance(by_gpu, dict) or not by_gpu:
            raise TypeError(
                f"KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER[{model_id!r}] must be "
                f"a non-empty {{gpu_class: seconds}} map, got {by_gpu!r}"
            )
        out[model_id] = dict(by_gpu)
    return out


def _floor_for_row(floor_table: dict[str, dict[str, float]], row: dict) -> float:
    model_id = row["model_id"]
    gpu = row["gpu"]
    if model_id not in floor_table:
        raise KeyError(
            f"No kernel-launch floor for model {model_id!r} " f"(known: {sorted(floor_table)})"
        )
    by_gpu = floor_table[model_id]
    if gpu not in by_gpu:
        raise KeyError(
            f"No kernel-launch floor for ({model_id}, {gpu}). " f"Known GPUs: {sorted(by_gpu)}"
        )
    return by_gpu[gpu]


def _decode_rows(panel: dict) -> list[dict]:
    """Decode records (disagg AND non-disagg) with a usable TPOT ratio.

    Selection is by data *property*, not by source name or regime, so every
    model with a measured per-token latency is audited and adding a new
    benchmark source cannot silently bypass this check:

      - `tpot_ratio is None` -> skip. The kernel-launch floor only acts in
        the latency-bound decode regime; throughput-only sources/rows (e.g.
        TRT-LLM perf-overview) expose no TPOT ratio and are correctly
        irrelevant here.

    Both regimes are kept: the per-model floor is injected into every decode
    prediction regardless of disagg, so its centering is checked against the
    naive `tpot_ratio` of all decode rows. (The non-disagg-only α-correction
    is a separate adjustment and is intentionally NOT used here — in the
    small-batch floor regime α≈1, so naive and corrected coincide, and using
    the naive ratio keeps disagg and non-disagg on the same footing.)

    A row that qualifies (has tpot_ratio) but is missing any structured shape
    field the driver is expected to emit (`gpu_class`, `tp`, `conc`, `isl`,
    `osl`, `disagg`) is a hard error, not a silent skip: a new decode source
    must be made to emit them deliberately.
    """
    shape_fields = (
        "gpu_class",
        "tp",
        "conc",
        "isl",
        "osl",
        "disagg",
        "predicted_tpot_ms",
        "measured_tpot_ms",
    )
    rows = []
    for r in panel["results"]:
        if r.get("tpot_ratio") is None:
            continue
        missing = [k for k in shape_fields if r.get(k) is None]
        if missing:
            raise ValueError(
                f"Record {r['record_id']!r} qualifies for the kernel-floor "
                f"audit (has tpot_ratio) but the panel summary is missing "
                f"structured field(s) {missing} (source={r.get('source')!r}). "
                f"The driver must emit these for every decode source."
            )
        rows.append(
            {
                "model_id": r["model_id"],
                "gpu": r["gpu_class"],
                "tp": int(r["tp"]),
                "conc": int(r["conc"]),
                "isl": int(r["isl"]),
                "osl": int(r["osl"]),
                "disagg": bool(r["disagg"]),
                "tpot_ratio": float(r["tpot_ratio"]),
                "predicted_tpot_ms": float(r["predicted_tpot_ms"]),
                "measured_tpot_ms": float(r["measured_tpot_ms"]),
                # Non-disagg rows carry a contamination flag (capacity /
                # queue saturation); disagg rows have no such probe -> clean.
                "contaminated": bool(r.get("contaminated")),
            }
        )
    return rows


def _audit_model_floor(
    model_id: str,
    rows: list[dict],
    floor_by_gpu: dict[str, float] | None,
    small_batch_conc: int,
    flag_threshold: float,
) -> None:
    """Print the kernel-floor diagnostic for one model and flag floor errors.

    Context: `KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` is keyed on (model, gpu).
    This function audits whether each listed GPU's current floor is
    centered, using small-batch decode rows where the floor dominates.
    It does NOT judge which GPU is "better" hardware — `tpot_ratio` is
    predicted/measured, so the question is purely whether ScaleScope's floor is
    correctly centered, per GPU.

    Two complementary views (printed):

      1. Per-GPU small-batch centering: geomean of `tpot_ratio` over the
         model's small-batch rows, separately for each GPU class. ~1.00x =
         the shared floor is centered for that (model, GPU); a consistent
         offset (|geomean - 1| > flag_threshold, marked "off-center") means
         it is not. This view can be confounded because each GPU may run a
         different workload mix.

      2. Matched-cell head-to-head (only when the model spans >= 2 GPU
         classes): for each GPU pair, compare predictions on the IDENTICAL
         (tp, conc, isl, osl) cells both ran, removing the workload-mix
         confound. A persistent offset here ("per-GPU floor warranted") is
         clean evidence the floor should be split into a per-(model,
         gpu_class) value rather than one shared number.

    So the "comparing GPUs" step is a calibration verdict on the floor, not a
    hardware comparison: it answers "is one model-level floor good enough, or
    does GPU class matter?". Output is printed to stdout only (no return, no
    file writes); `flag` suffixes are the actionable signal.
    """
    by_gpu: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu"]].append(r)

    if floor_by_gpu is None:
        floor_str = "n/a"
    else:
        floor_str = ", ".join(
            f"{gpu} {val * 1e6:.0f} us/layer" for gpu, val in sorted(floor_by_gpu.items())
        )
    regimes = sorted({"disagg" if r["disagg"] else "non-disagg" for r in rows})
    print(f"\n=== {model_id}  ({'+'.join(regimes)}, current floor: {floor_str}) ===")
    print(f"  GPU classes in panel: {', '.join(sorted(by_gpu))}")

    # 1) Per-GPU small-batch centering.
    print(f"  Small-batch (conc <= {small_batch_conc}) geomean TPOT ratio (pred/meas):")
    for gpu in sorted(by_gpu):
        sb = [r["tpot_ratio"] for r in by_gpu[gpu] if r["conc"] <= small_batch_conc]
        if not sb:
            print(f"    {gpu:6}: (no conc <= {small_batch_conc} rows)")
            continue
        g = _geomean(sb)
        flag = "  <-- off-center" if abs(g - 1.0) > flag_threshold else ""
        print(f"    {gpu:6}: {g:.3f}x  (N={len(sb)}){flag}")

    # 2) Matched-cell head-to-head for GPU pairs (isolates the per-GPU effect).
    gpus = sorted(by_gpu)
    if len(gpus) < 2:
        return
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["tp"], r["conc"], r["isl"], r["osl"])][r["gpu"]].append(r["tpot_ratio"])
    # Compare every unordered GPU pair (a, b). The per-GPU small-batch
    # centering above can be confounded: each GPU class may run a different
    # mix of (tp, conc, isl, osl) shapes, so a raw geomean difference could
    # just reflect a different workload mix rather than a true per-GPU floor
    # error. The head-to-head below removes that confound by comparing only
    # cells where BOTH GPUs ran the identical (tp, conc, isl, osl) shape.
    for i in range(len(gpus)):
        for j in range(i + 1, len(gpus)):
            a, b = gpus[i], gpus[j]
            # all_ratios: one rb/ra per shared cell (full conc range).
            # sb_ratios: the same, restricted to small-batch cells, where
            # the kernel floor dominates and a per-GPU bias is clearest.
            all_ratios, sb_ratios = [], []
            for key, per_gpu in cells.items():
                # Only matched cells contribute: both GPUs must have a row
                # at this exact (tp, conc, isl, osl). Cells run on just one
                # GPU are skipped so workload mix can't bias the result.
                if a in per_gpu and b in per_gpu:
                    # Average duplicate rows within the cell (e.g. multiple
                    # frameworks for the same shape) before taking the ratio.
                    ra = sum(per_gpu[a]) / len(per_gpu[a])
                    rb = sum(per_gpu[b]) / len(per_gpu[b])
                    # rb/ra > 1 means b's prediction runs hotter than a's at
                    # the same workload — the residual the shared single-value
                    # floor leaves on b relative to a.
                    all_ratios.append(rb / ra)
                    if key[1] <= small_batch_conc:  # key = (tp, conc, isl, osl)
                        sb_ratios.append(rb / ra)
            if not all_ratios:
                continue  # the two GPUs never ran a common shape; nothing to compare.
            # Geomean (not arithmetic mean) because these are multiplicative
            # ratios: it is symmetric, so a 1.1x bias one way and 1/1.1x the
            # other cancel to 1.0 instead of a spurious >1 average.
            gall = _geomean(all_ratios)
            gsb = _geomean(sb_ratios) if sb_ratios else float("nan")
            # A persistent matched-cell offset can't be a workload-mix
            # artifact, so it is direct evidence the single shared floor
            # should be split into a per-(model, gpu_class) value.
            flag = "  <-- per-GPU floor warranted" if abs(gall - 1.0) > flag_threshold else ""
            print(
                f"  Matched cells {b}/{a}: geomean(all)={gall:.3f}x "
                f"(N={len(all_ratios)}), small-batch={gsb:.3f}x{flag}"
            )


def _report_fitted_floors(
    models: list[str],
    by_model: dict[str, list[dict]],
    floor_table: dict[str, dict[str, float]],
    small_batch_conc: int,
) -> None:
    """Print the fitted kernel-launch floor (us/layer) per (model, gpu).

    Inverts the audit. For each listed (model, gpu) it takes the clean
    small-batch decode rows and, per row, computes the exact additive
    floor that lands ScaleScope's prediction on the measured TPOT (see
    `_implied_floor_sec`). The reported value is the MEDIAN of those
    per-row floors.

    These are suggestions for `KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER`; this script
    never edits the driver. Throughput-only models (no measured TPOT) cannot be
    fit and are reported separately by `_report_coverage`.
    """
    print("\n=== Fitted kernel-launch floor (clean small-batch, exact additive) ===")
    print(
        "  floor_fit = floor_current + median[(measured_tpot - predicted_tpot) / n_layers], "
        f"over clean rows with conc <= {small_batch_conc}. Units: us/layer."
    )
    for model_id in models:
        if model_id not in floor_table:
            print(
                f"  {model_id:14}: no current floor in KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER; skipping."
            )
            continue
        n_layers = _n_layers(model_id)
        clean_sb = [
            r for r in by_model[model_id] if r["conc"] <= small_batch_conc and not r["contaminated"]
        ]
        if not clean_sb:
            print(f"  {model_id:14}: no clean small-batch rows; cannot fit.")
            continue
        by_gpu: dict[str, list[dict]] = defaultdict(list)
        for r in clean_sb:
            by_gpu[r["gpu"]].append(r)
        for gpu, gpu_rows in sorted(by_gpu.items()):
            current = _floor_for_row(floor_table, gpu_rows[0])
            implied_us = [_implied_floor_sec(r, current, n_layers) * 1e6 for r in gpu_rows]
            print(
                f"  {model_id:14} {gpu:6} (n_layers={n_layers}): "
                f"current {current * 1e6:.0f} -> fit {_median(implied_us):.0f} us/layer  "
                f"[range {min(implied_us):.0f}-{max(implied_us):.0f}, N={len(gpu_rows)}]"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--panel", default=str(DEFAULT_PANEL), help="validation_panel_summary.yaml path."
    )
    parser.add_argument(
        "--model", action="append", default=None, help="Restrict to model_id(s). Repeatable."
    )
    parser.add_argument(
        "--small-batch-conc", type=int, default=8, help="conc <= this is 'small batch'."
    )
    parser.add_argument(
        "--flag-threshold", type=float, default=0.03, help="|geomean-1| above this is flagged."
    )
    args = parser.parse_args(argv)

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel summary not found: {panel_path}. Run run_validation_panel.py first."
        )
    with panel_path.open() as f:
        panel = yaml.safe_load(f)

    floor_table = _load_floor_table()
    rows = _decode_rows(panel)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_id"]].append(r)

    models = sorted(by_model)
    if args.model is not None:
        models = [m for m in models if m in set(args.model)]

    print(f"Loaded {len(rows)} decode rows across {len(by_model)} model(s).")

    # Primary output: the fitted floor for every model in the panel.
    _report_fitted_floors(models, by_model, floor_table, args.small_batch_conc)

    # Detailed per-model diagnostics (centering + per-GPU head-to-head).
    for model_id in models:
        _audit_model_floor(
            model_id,
            by_model[model_id],
            floor_table.get(model_id),
            args.small_batch_conc,
            args.flag_threshold,
        )

    _report_coverage(panel, floor_table, audited=set(by_model))
    return 0


def _report_coverage(
    panel: dict, floor_table: dict[str, dict[str, float]], *, audited: set[str]
) -> None:
    """Cross-check every floor-table model against what the audit covered.

    A model carries a kernel-launch floor but is auditable only if the panel
    has at least one decode row (a measured TPOT) for it. This makes the
    coverage explicit instead of leaving throughput-only models (no TPOT)
    silently absent from the report above.
    """
    panel_models = {r["model_id"] for r in panel["results"]}
    print("\n=== Floor-table coverage ===")
    for model_id in sorted(floor_table):
        if model_id in audited:
            print(f"  {model_id:14}: audited")
        elif model_id not in panel_models:
            print(f"  {model_id:14}: not in this panel (filtered out?)")
        else:
            # Present in the panel but no row exposes a TPOT ratio.
            print(f"  {model_id:14}: throughput-only (no measured TPOT) — not auditable here")


if __name__ == "__main__":
    raise SystemExit(main())
