"""Measured-panel validation driver — TRT-LLM (H100) + InferenceX (GB200 NVL72 disagg, B200 non-disagg).

Loops over every record in `trtllm_perf_overview.yaml` and
`inferencex_raw/*.json` whose hardware × model × parallelism shape is
supported by ScaleScope today, invokes `scalescope.run_scalescope` once
per record at the operating-point batch implied by the measurement, and
emits a comparison markdown table.

Run from the repo root:
    # Full panel (canonical output to validation_results/validation_panel_summary.{md,yaml}):
    python validation/run_validation_panel.py

    # Smoke-test the first 3 records of each loaded source:
    python validation/run_validation_panel.py --limit 3

    # One corpus at a time:
    python validation/run_validation_panel.py --source trtllm
    python validation/run_validation_panel.py --source inferencex

    # Fine-grained subset (all filters are AND-combined; values within
    # a single flag are OR-combined):
    python validation/run_validation_panel.py \
        --source inferencex --model minimax_m2_5
    python validation/run_validation_panel.py \
        --gpu-class B200 --gpu-class GB200
    python validation/run_validation_panel.py \
        --record-contains isl8192_osl1024 --record-contains conc512

    # Dry-run — list what would run, no ScaleScope invocations, no files
    # written:
    python validation/run_validation_panel.py \
        --source inferencex --model minimax_m2_5 --list-only

Any filter (--source != 'all', --limit, --model, --gpu-class,
--record-contains) auto-suffixes outputs with '__partial' so smoke
tests don't clobber the canonical summary. Override with
`--output-suffix ''` to force overwrite, or `--output-suffix __foo`
for a named subset.

Operating-point batch selection (so ScaleScope and the measurement
compare at the same point on the per-token throughput curve):

  - TRT-LLM perf-overview: TPOT is not exposed publicly, so we sweep
    ScaleScope across BATCH_SWEEP_GRID and pick the batch whose
    predicted throughput is closest (in log space) to the measured
    throughput. The reported `predicted_tpot_ms` and `predicted_sys_tps`
    are taken at that batch — i.e. ScaleScope's "what batch is the
    measurement secretly running at?" answer, with the corresponding
    TPOT alongside.
  - **InferenceX (PD-disaggregated only)**: the row carries the
    concurrency `conc` directly (= the number of users the benchmark
    drove against the decode pool). ScaleScope is invoked once at
    `global_batch = conc`. The prefill engine is ignored entirely — we
    model only the decode pool. This is the cleanest comparison
    available: no Little's-Law back-derivation, no chunked-prefill
    interleaving, no sweep guesswork.

Outputs:
    validation_results/validation_panel_summary.md           (markdown table)
    validation_results/validation_panel_summary.yaml         (machine-readable summary)
    validation_results/runs/<record_id>.yaml                 (per-record ScaleScope output)
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE
RESULTS_DIR = VALIDATION_DIR / "validation_results"
RUNS_DIR = RESULTS_DIR / "runs"

# Point matplotlib at a repo-local writable cache so each ScaleScope subprocess
# doesn't pay a ~10s font-cache rebuild when the default matplotlib cache dir
# isn't writable (e.g. `~/.matplotlib` root-owned from a prior sudo run, or a
# read-only / sandboxed $HOME). Without this the panel runtime is ~8x slower
# because every subprocess re-imports matplotlib and rebuilds the font cache.
_MPL_CACHE_DIR = REPO_ROOT / ".mplcache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

HARDWARE_LIB = REPO_ROOT / "hardware_specs" / "hardware_library.yaml"
WORKLOAD_TEMPLATES = {
    "fp8": VALIDATION_DIR / "inference_decode_specs" / "fp8_decode_template.yaml",
    "fp4": VALIDATION_DIR / "inference_decode_specs" / "fp4_decode_template.yaml",
}


def _require_2d_workload_templates() -> None:
    """Decode combine is 2D-only."""
    required_keys = ("tp_attn_spec", "tp_proj_spec", "tp_moe_spec")
    for name, path in WORKLOAD_TEMPLATES.items():
        spec = yaml.safe_load(path.read_text())
        for key in required_keys:
            block = spec[key]
            variant = block["variant"]
            if variant != "SHARD_WEIGHTS_2D":
                raise ValueError(
                    f"{path}: {key}.variant must be SHARD_WEIGHTS_2D; "
                    f"got {variant!r} in the {name} template."
                )
            if block.get("split_2d") is None:
                raise ValueError(f"{path}: {key}.split_2d is required for SHARD_WEIGHTS_2D.")


MODEL_ZOO = VALIDATION_DIR / "model_zoo"
TOPO_PHYS = VALIDATION_DIR / "topology_zoo" / "physical"
TOPO_LOG = VALIDATION_DIR / "topology_zoo" / "logical"

# Per-model host kernel-launch/dispatch floor (seconds per layer per decode
# step), injected into each generated workload spec. This is an empirical
# runtime overhead the roofline does not model; it is fit to the clean
# small-batch (TP=1, conc <= 8) latency-bound residual of each model in the
# measured panel. The lookup is intentionally fail-fast (no silent default):
# every model with a Case must be listed here. llama3_1_405b and llama3_1_8b
# carry representative values for the TRT-LLM throughput regime.
KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER: dict[str, float] = {
    "llama3_70b": 42e-6,
    "llama3_1_405b": 45e-6,
    "llama3_1_8b": 24e-6,
    "minimax_m2_5": 90e-6,
    "gpt_oss_120b": 32e-6,
}

# Per-model MoE software overhead (seconds per *remote* routed token-expert
# pair per layer per pass). Volume is N_rows × (D−1)/D on decode. Dense
# shapes are structurally zero. gpt_oss_120b is an EP2 single-DP ISL=1024
# NNLS of the zero-pair residuals (186 ns; all-EP NNLS was 163 ns, 1.14×,
# accepted). See validation/fit_ep_dispatch_overhead.py.
MOE_TOKEN_PAIR_OVERHEAD_SEC: dict[str, float] = {
    "llama3_70b": 0.0,
    "llama3_1_405b": 0.0,
    "llama3_1_8b": 0.0,
    "minimax_m2_5": 0.0,
    "gpt_oss_120b": 186e-9,
}

# FP4 model_ids explicitly promoted into the canonical panel despite the
# default FP4 exclusion. ScaleScope undercounts *generic* FP4 cost (no dequant
# / block-scaling machinery — see §4 of modeling_improvements_todo.md), so FP4
# is dropped by default. gpt_oss_120b is the one FP4 model we keep: it has no
# FP8 variant in the corpora and is individually calibrated (kernel-launch
# floor + disagg α) and validated, so excluding it would drop the only
# gpt_oss coverage. Other FP4 records stay excluded unless --include-fp4 is
# passed; this allowlist is an always-include override, not a blanket opt-in.
FP4_CANONICAL_MODEL_IDS: frozenset[str] = frozenset({"gpt_oss_120b"})

TRTLLM_YAML = VALIDATION_DIR / "benchmark_data" / "trtllm_perf_overview.yaml"
INFERENCEX_DIR = VALIDATION_DIR / "benchmark_data" / "inferencex_raw"

# InferenceX display-name -> ScaleScope model_zoo filename (no extension).
# Only models with a model_zoo YAML and a Case in INFERENCEX_CASE_TABLE will
# actually get exercised; the rest are silently filtered.
_INFERENCEX_DISPLAY_TO_SS = {
    "gpt-oss-120b": "gpt_oss_120b",
    "MiniMax-M2.5": "minimax_m2_5",
    "Llama-3.3-70B-Instruct-FP8": "llama3_70b",
}

# InferenceX `hardware` (lowercase) -> ScaleScope gpu_class label used in
# the case-table keys.
_INFERENCEX_HW_TO_GPU_CLASS = {
    "h100": "H100",
    "b200": "B200",
    "gb200": "GB200",
}

# Per-chip HBM capacity (bytes), copied verbatim from
# `hardware_specs/hardware_library.yaml` (`num_stack × size_per_stack`).
# Used to flag InferenceX rows where ScaleScope's predicted
# `total_peak_memory_per_xpu_bytes` exceeds the chip's HBM — those rows
# cannot physically run the labelled `conc` and the framework must be
# internally serving an effective batch much smaller than `conc`,
# making `median_tpot` a per-step measurement on a small effective
# batch rather than the steady-state cost the row's labelling implies.
HBM_CAPACITY_BYTES: dict[str, int] = {
    "H100": 5 * 16_000_000_000,  # 80 GB SXM5 (5 × 16 GB HBM3 stacks)
    "B200": 8 * 22_500_000_000,  # 180 GB SXM5 (8 × 22.5 GB HBM3e stacks)
    "GB200": 8 * 24_000_000_000,  # 192 GB GB200 NVL72 (8 × 24 GB HBM3e stacks)
}

# `mean_e2el / (OSL × median_tpot)` > this threshold flags a row as
# queue-saturated — the framework is admitting requests faster than the
# decode pool can clear them, so reported `median_tpot` is per-step
# time on a smaller effective decode batch than the labelled `conc`,
# not the steady-state cost of decoding all `conc` users in parallel.
# 1.5× allows for ramp-up + drain at the head/tail of the run while
# still catching real saturation (typically ≥2× when the cap actually
# bites).
QUEUEING_FACTOR_SATURATED_THRESHOLD: float = 1.5


# ---------------------------------------------------------------------------
# Case table — one entry per (model, GPU-class, GPU-count) combination
# ---------------------------------------------------------------------------
# Each case fixes the parallelism hypothesis. Records that don't match a
# case (e.g. H200 / B200 today) are skipped with status "deferred".
@dataclass(frozen=True)
class Case:
    case_id: str
    model_yaml: str  # name under model_zoo/ (no extension)
    phys_topo: str  # name under topology_zoo/physical/ (no extension)
    logical_topo: str  # name under topology_zoo/logical/ (no extension)
    parallelism_label: str  # e.g. "TP=8", "TP=1+EP=8"
    is_moe: bool
    num_colocated_experts: int  # 1 for dense; full expert count for colocated MoE
    batch_quantum: int  # global_batch must be a multiple of this
    precision: str  # "fp8" | "fp4" — picks workload template + float_size_bytes


TRTLLM_CASE_TABLE: dict[tuple[str, str, int], Case] = {
    # (model_id, gpu_class, num_gpus) -> Case
    # H100x{1,2,8} TRT-LLM perf-overview FP8 cases.
    ("llama3_1_405b", "H100", 8): Case(
        "llama3_1_405b__h100_8x_tp8",
        "llama3_1_405b",
        "h100_8x",
        "h100_8x_tp8",
        "TP=8",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_1_8b", "H100", 1): Case(
        "llama3_1_8b__h100_1x_tp1",
        "llama3_1_8b",
        "h100_1x",
        "h100_1x_tp1",
        "TP=1",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "H100", 2): Case(
        "llama3_70b__h100_2x_tp2",
        "llama3_70b",
        "h100_2x",
        "h100_2x_tp2",
        "TP=2",
        False,
        1,
        1,
        "fp8",
    ),
}


# InferenceX cases — keyed on the full parallelism + precision shape
# because multiple Cases can share `(model, gpu_class, num_gpus)`
# (e.g. 4 GPUs with three different `(decode_tp, decode_ep, dp_attn)`
# shapes; same MiniMax shape served at FP4 and FP8). Includes both
# Phase-3 disagg-decode rows (gpt-oss-120b on GB200) and Phase-4
# non-disagg continuous-batch rows (MiniMax-M2.5 on B200), the latter
# of which get a downstream prefill-overhead α-correction (see
# `_alpha_for_inferencex_record`).
#
# Key tuple: (model_id, gpu_class, num_gpus,
#             decode_tp, decode_ep, decode_dp_attention,
#             precision)
INFERENCEX_CASE_TABLE: dict[tuple[str, str, int, int, int, bool, str], Case] = {
    # ──────────────────────────────────────────────────────────────────
    # Phase 3 — gpt-oss-120b on GB200 NVL72, PD-disaggregated decode pool.
    # MXFP4 native quantisation; ScaleScope runs at `fp4` template.
    # ──────────────────────────────────────────────────────────────────
    # Clean-TP shapes (Phase 3A). All 128 experts colocated, TP-sharded along d_ff.
    ("gpt_oss_120b", "GB200", 2, 2, 1, False, "fp4"): Case(
        "gpt_oss_120b__gb200_2x_tp2",
        "gpt_oss_120b",
        "gb200_2x",
        "gb200_2x_tp2",
        "TP=2",
        True,
        128,
        1,
        "fp4",
    ),
    ("gpt_oss_120b", "GB200", 4, 4, 1, False, "fp4"): Case(
        "gpt_oss_120b__gb200_4x_tp4",
        "gpt_oss_120b",
        "gb200_4x",
        "gb200_4x_tp4",
        "TP=4",
        True,
        128,
        1,
        "fp4",
    ),
    ("gpt_oss_120b", "GB200", 8, 8, 1, False, "fp4"): Case(
        "gpt_oss_120b__gb200_8x_tp8",
        "gpt_oss_120b",
        "gb200_8x",
        "gb200_8x_tp8",
        "TP=8",
        True,
        128,
        1,
        "fp4",
    ),
    # Hybrid shapes (Phase 3B). `num_colocated_experts = 128 / EP_SU`.
    ("gpt_oss_120b", "GB200", 2, 2, 2, True, "fp4"): Case(
        "gpt_oss_120b__gb200_2x_tp2_dpattn_ep2",
        "gpt_oss_120b",
        "gb200_2x",
        "gb200_2x_tp2_dpattn_ep2",
        "TP_grp=2+DPattn+EP=2",
        True,
        64,
        1,
        "fp4",
    ),
    ("gpt_oss_120b", "GB200", 4, 4, 4, True, "fp4"): Case(
        "gpt_oss_120b__gb200_4x_tp4_dpattn_ep4",
        "gpt_oss_120b",
        "gb200_4x",
        "gb200_4x_tp4_dpattn_ep4",
        "TP_grp=4+DPattn+EP=4",
        True,
        32,
        1,
        "fp4",
    ),
    ("gpt_oss_120b", "GB200", 4, 2, 2, True, "fp4"): Case(
        "gpt_oss_120b__gb200_4x_tp2_dpattn_ep2_dp2",
        "gpt_oss_120b",
        "gb200_4x",
        "gb200_4x_tp2_dpattn_ep2_dp2",
        "TP_grp=2+DPattn+EP=2+DP=2",
        True,
        64,
        1,
        "fp4",
    ),
    ("gpt_oss_120b", "GB200", 12, 4, 4, True, "fp4"): Case(
        "gpt_oss_120b__gb200_12x_tp4_dpattn_ep4_dp3",
        "gpt_oss_120b",
        "gb200_12x",
        "gb200_12x_tp4_dpattn_ep4_dp3",
        "TP_grp=4+DPattn+EP=4+DP=3",
        True,
        32,
        1,
        "fp4",
    ),
    ("gpt_oss_120b", "GB200", 8, 2, 1, False, "fp4"): Case(
        "gpt_oss_120b__gb200_8x_tp2_dp4",
        "gpt_oss_120b",
        "gb200_8x",
        "gb200_8x_tp2_dp4",
        "TP=2+DP=4",
        True,
        128,
        1,
        "fp4",
    ),
    # ──────────────────────────────────────────────────────────────────
    # Phase 4 — MiniMax-M2.5 on B200, non-disagg continuous batching.
    # All EP=1: 256 experts colocated on each chip, TP-sharded along
    # d_ff axis. Both native FP8 (E4M3) and operator-quantised FP4
    # variants appear in the InferenceX corpus. α-correction applied
    # downstream because these rows pay chunked-prefill overhead.
    # ──────────────────────────────────────────────────────────────────
    ("minimax_m2_5", "B200", 1, 1, 1, False, "fp4"): Case(
        "minimax_m2_5__b200_1x_tp1__fp4",
        "minimax_m2_5",
        "b200_1x",
        "b200_1x_tp1",
        "TP=1",
        True,
        256,
        1,
        "fp4",
    ),
    ("minimax_m2_5", "B200", 2, 2, 1, False, "fp4"): Case(
        "minimax_m2_5__b200_2x_tp2__fp4",
        "minimax_m2_5",
        "b200_2x",
        "b200_2x_tp2",
        "TP=2",
        True,
        256,
        1,
        "fp4",
    ),
    ("minimax_m2_5", "B200", 2, 2, 1, False, "fp8"): Case(
        "minimax_m2_5__b200_2x_tp2__fp8",
        "minimax_m2_5",
        "b200_2x",
        "b200_2x_tp2",
        "TP=2",
        True,
        256,
        1,
        "fp8",
    ),
    ("minimax_m2_5", "B200", 4, 4, 1, False, "fp4"): Case(
        "minimax_m2_5__b200_4x_tp4__fp4",
        "minimax_m2_5",
        "b200_4x",
        "b200_4x_tp4",
        "TP=4",
        True,
        256,
        1,
        "fp4",
    ),
    ("minimax_m2_5", "B200", 4, 4, 1, False, "fp8"): Case(
        "minimax_m2_5__b200_4x_tp4__fp8",
        "minimax_m2_5",
        "b200_4x",
        "b200_4x_tp4",
        "TP=4",
        True,
        256,
        1,
        "fp8",
    ),
    ("minimax_m2_5", "B200", 8, 8, 1, False, "fp4"): Case(
        "minimax_m2_5__b200_8x_tp8__fp4",
        "minimax_m2_5",
        "b200_8x",
        "b200_8x_tp8",
        "TP=8",
        True,
        256,
        1,
        "fp4",
    ),
    # ──────────────────────────────────────────────────────────────────
    # Llama-3.3-70B-Instruct-FP8 (uses llama3_70b.yaml)
    # ──────────────────────────────────────────────────────────────────
    ("llama3_70b", "H100", 2, 2, 1, False, "fp8"): Case(
        "llama3_70b__h100_2x_tp2__fp8",
        "llama3_70b",
        "h100_2x",
        "h100_2x_tp2",
        "TP=2",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "H100", 4, 4, 1, False, "fp8"): Case(
        "llama3_70b__h100_4x_tp4__fp8",
        "llama3_70b",
        "h100_4x",
        "h100_4x_tp4",
        "TP=4",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "H100", 8, 8, 1, False, "fp8"): Case(
        "llama3_70b__h100_8x_tp8__fp8",
        "llama3_70b",
        "h100_8x",
        "h100_8x_tp8",
        "TP=8",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "B200", 1, 1, 1, False, "fp4"): Case(
        "llama3_70b__b200_1x_tp1__fp4",
        "llama3_70b",
        "b200_1x",
        "b200_1x_tp1",
        "TP=1",
        False,
        1,
        1,
        "fp4",
    ),
    ("llama3_70b", "B200", 1, 1, 1, False, "fp8"): Case(
        "llama3_70b__b200_1x_tp1__fp8",
        "llama3_70b",
        "b200_1x",
        "b200_1x_tp1",
        "TP=1",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "B200", 2, 2, 1, False, "fp4"): Case(
        "llama3_70b__b200_2x_tp2__fp4",
        "llama3_70b",
        "b200_2x",
        "b200_2x_tp2",
        "TP=2",
        False,
        1,
        1,
        "fp4",
    ),
    ("llama3_70b", "B200", 2, 2, 1, False, "fp8"): Case(
        "llama3_70b__b200_2x_tp2__fp8",
        "llama3_70b",
        "b200_2x",
        "b200_2x_tp2",
        "TP=2",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "B200", 4, 4, 1, False, "fp4"): Case(
        "llama3_70b__b200_4x_tp4__fp4",
        "llama3_70b",
        "b200_4x",
        "b200_4x_tp4",
        "TP=4",
        False,
        1,
        1,
        "fp4",
    ),
    ("llama3_70b", "B200", 4, 4, 1, False, "fp8"): Case(
        "llama3_70b__b200_4x_tp4__fp8",
        "llama3_70b",
        "b200_4x",
        "b200_4x_tp4",
        "TP=4",
        False,
        1,
        1,
        "fp8",
    ),
    ("llama3_70b", "B200", 8, 8, 1, False, "fp4"): Case(
        "llama3_70b__b200_8x_tp8__fp4",
        "llama3_70b",
        "b200_8x",
        "b200_8x_tp8",
        "TP=8",
        False,
        1,
        1,
        "fp4",
    ),
    ("llama3_70b", "B200", 8, 8, 1, False, "fp8"): Case(
        "llama3_70b__b200_8x_tp8__fp8",
        "llama3_70b",
        "b200_8x",
        "b200_8x_tp8",
        "TP=8",
        False,
        1,
        1,
        "fp8",
    ),
}


def _case_for(record: Record) -> Case | None:
    """Source-aware Case lookup.

    InferenceX records carry the full decode-stage parallelism shape; we
    use it (not just `num_gpus`) because multiple Cases can share a
    `(model_id, gpu_class, num_gpus)` triple. TRT-LLM has one Case per
    triple. Each source has its own case table; an unrecognized source
    fails fast rather than silently routing into a table that does not
    model it.
    """
    match record.source:
        case "inferencex":
            if (
                record.decode_tp is None
                or record.decode_ep is None
                or record.decode_dp_attention is None
                or record.precision is None
            ):
                raise ValueError(
                    f"InferenceX record {record.record_id!r} is missing decode shape / precision fields."
                )
            return INFERENCEX_CASE_TABLE.get(
                (
                    record.model_id,
                    record.gpu_class,
                    record.num_gpus,
                    record.decode_tp,
                    record.decode_ep,
                    record.decode_dp_attention,
                    record.precision,
                )
            )
        case "trtllm_perf_overview":
            return TRTLLM_CASE_TABLE.get((record.model_id, record.gpu_class, record.num_gpus))
        case _:
            raise ValueError(
                f"No case table for record source {record.source!r} "
                f"(record {record.record_id!r})."
            )


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    """Normalised view of one measured datapoint."""

    record_id: str  # stable, filesystem-safe
    source: str  # "trtllm_perf_overview" | "inferencex"
    submitter: str | None  # InferenceX framework, None for TRT-LLM
    system: str  # human-readable hardware string
    gpu_class: str  # "H100" / "H200" / "B200" / "GH200" / "GB200"
    num_gpus: int
    model_id: str
    benchmark_label: str  # e.g. "llama2-70b-99 Server" or "ISL=1024,OSL=2048"
    isl: int
    osl: int
    measured_sys_tps: float  # system aggregate output tokens/sec
    measured_tpot_ms: float | None  # P50 if available, else None (TRT-LLM)
    # InferenceX-only parallelism shape — None for TRT-LLM rows because
    # that source doesn't expose decode-stage parallelism. Used by the
    # InferenceX case lookup to disambiguate hybrid DP-attention / EP
    # configurations that share the same `(model, gpu_class, num_gpus)`.
    decode_tp: int | None = None
    decode_ep: int | None = None
    decode_dp_attention: bool | None = None
    # InferenceX-only deployment + measurement fields. `disagg` decides
    # whether the prefill-overhead α-correction is applied; `precision`
    # is part of the InferenceX case key (one model on the same GPU
    # class and parallelism shape can be served at multiple precisions);
    # `mean_e2el_s` is the per-user end-to-end latency in seconds and is
    # needed by the α-correction to derive samples-per-second from
    # `conc / mean_e2el`.
    disagg: bool | None = None
    precision: str | None = None
    mean_e2el_s: float | None = None
    conc: int | None = None


# Maps a TRT-LLM table row "model_id" to the ScaleScope model-zoo name.
_TRT_MODEL_TO_SS = {
    "llama3_1_8b": "llama3_1_8b",
    "llama3_3_70b_fp8": "llama3_70b",  # structurally identical to Llama-3.1-70B
    "llama3_1_405b_fp8": "llama3_1_405b",
    # FP4 variants are deferred (no FP4 in ScaleScope yet).
}


def _load_trtllm_records() -> list[Record]:
    if not TRTLLM_YAML.exists():
        raise FileNotFoundError(f"Missing TRT-LLM YAML: {TRTLLM_YAML}")
    with TRTLLM_YAML.open() as f:
        data = yaml.safe_load(f)
    out: list[Record] = []
    for trt_model_id, info in data["models"].items():
        ss_model_id = _TRT_MODEL_TO_SS.get(trt_model_id)
        if ss_model_id is None:
            continue  # deferred (FP4)
        precision = info["precision"]
        if precision != "fp8":
            continue
        for run in info["runs"]:
            gpu = run["gpu"]
            tp = int(run["tp"])
            isl = int(run["isl"])
            osl = int(run["osl"])
            tps = float(run["total_output_tps"])
            record_id = f"trtllm__{trt_model_id}__{gpu}__tp{tp}__isl{isl}_osl{osl}"
            out.append(
                Record(
                    record_id=record_id,
                    source="trtllm_perf_overview",
                    submitter=None,
                    system=f"{gpu}_x{tp}_TRT-LLM",
                    gpu_class=gpu,
                    num_gpus=tp,
                    model_id=ss_model_id,
                    benchmark_label=f"ISL={isl}, OSL={osl}, TP={tp}",
                    isl=isl,
                    osl=osl,
                    measured_sys_tps=tps,
                    measured_tpot_ms=None,
                )
            )
    return out


def _load_inferencex_records() -> list[Record]:
    """Load InferenceX rows (both disagg and non-disagg) from
    `inferencex_raw/*.json`.

    Both deployment modes are passed through to the Record; whether a
    given (model, gpu_class, parallelism, precision) is actually
    runnable is decided downstream by the `INFERENCEX_CASE_TABLE`
    lookup. Disagg rows feed ScaleScope's steady-state decode model
    cleanly; non-disagg rows pay chunked-prefill overhead which is
    quantified separately via the α-correction (see
    `_alpha_for_inferencex_record`).

    Note: the prefill-side fields (``prefill_*``, ``num_prefill_gpu``)
    are not loaded — α-correction is parameterised by the *decode*
    pool's parallelism and the row's measured samples-per-second
    (derived from `conc / mean_e2el`).
    """
    if not INFERENCEX_DIR.exists():
        return []
    out: list[Record] = []
    for json_path in sorted(INFERENCEX_DIR.glob("*.json")):
        display_name = json_path.stem
        ss_model_id = _INFERENCEX_DISPLAY_TO_SS.get(display_name)
        if ss_model_id is None:
            continue  # model not modeled by ScaleScope today; skipped silently
        with json_path.open() as f:
            rows = json.load(f)
        for row in rows:
            gpu_class = _INFERENCEX_HW_TO_GPU_CLASS.get(row["hardware"])
            if gpu_class is None:
                continue
            disagg = bool(row["disagg"])
            num_decode_gpu = int(row["num_decode_gpu"])
            decode_tp = int(row["decode_tp"])
            decode_ep = int(row["decode_ep"])
            decode_dp_attention = bool(row["decode_dp_attention"])
            metrics = row.get("metrics") or {}
            median_tpot_s = metrics.get("median_tpot")
            if median_tpot_s is None:
                continue
            mean_e2el_s = metrics.get("mean_e2el")  # may be None for some rows
            isl = int(row["isl"])
            osl = int(row["osl"])
            conc = int(row["conc"])
            precision = row["precision"]
            framework = row["framework"]
            date = row.get("date", "")
            # Shape suffix uniquely identifies the parallelism flavour
            # (and deployment mode + precision) so multiple Cases that share
            # `(model, gpu, num_gpus)` get distinct record IDs and don't
            # overwrite each other's output YAML.
            shape_suffix = (
                f"tp{decode_tp}_ep{decode_ep}"
                f"{'_dpattn' if decode_dp_attention else ''}"
                f"_{'disagg' if disagg else 'nondisagg'}"
            )
            record_id = (
                f"inferencex__{ss_model_id}__{gpu_class.lower()}_{num_decode_gpu}x_{shape_suffix}"
                f"__{precision}__isl{isl}_osl{osl}__conc{conc}__{framework}__{date}"
            )
            # ScaleScope's `tokens_per_second` at global_batch=N is the
            # per-decode-step aggregate output rate `N / TPOT_user`. The
            # equivalent on the measurement side is `conc / median_tpot`
            # (algebraically equal to `conc × median_intvty`, since
            # intvty == 1/TPOT in steady state).
            #
            # We deliberately do NOT use InferenceX's `tput_per_gpu` here:
            # that metric is `(input + output tokens) / runtime / total
            # deployment GPUs` (i.e. averaged over prefill + decode pools),
            # and would give numbers ~6-9× higher than the decode pool's
            # actual output rate at long ISL.
            measured_sys_tps = float(conc) / float(median_tpot_s)
            out.append(
                Record(
                    record_id=record_id,
                    source="inferencex",
                    submitter=framework,
                    system=f"{gpu_class}_x{num_decode_gpu}_{framework}",
                    gpu_class=gpu_class,
                    num_gpus=num_decode_gpu,
                    model_id=ss_model_id,
                    benchmark_label=f"ISL={isl}, OSL={osl}, conc={conc}, {framework}",
                    isl=isl,
                    osl=osl,
                    measured_sys_tps=measured_sys_tps,
                    measured_tpot_ms=float(median_tpot_s) * 1000.0,
                    decode_tp=decode_tp,
                    decode_ep=decode_ep,
                    decode_dp_attention=decode_dp_attention,
                    disagg=disagg,
                    precision=precision,
                    mean_e2el_s=float(mean_e2el_s) if mean_e2el_s is not None else None,
                    conc=conc,
                )
            )
    return out


def _runnable_records(records: Iterable[Record]) -> list[Record]:
    return [r for r in records if _case_for(r) is not None]


# ---------------------------------------------------------------------------
# Operating-point batch selection
# ---------------------------------------------------------------------------
# Batch sweep used for TRT-LLM records (no measured TPOT). For each
# record we run ScaleScope at every batch in this grid and pick the
# batch whose predicted throughput is closest (log-space) to measured.
BATCH_SWEEP_GRID = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]


def _trtllm_sweep_batches(case: Case) -> list[int]:
    """Return the BATCH_SWEEP_GRID rounded UP to the case quantum, deduped."""
    seen: set[int] = set()
    out: list[int] = []
    for b in BATCH_SWEEP_GRID:
        # Round b UP to the next multiple of batch_quantum (ceil division
        # `(b + q - 1) // q` times q), then floor at one quantum so the
        # batch is never 0. With batch_quantum == 1 this is just `b`; it
        # only bites for cases that require batch divisibility (e.g. DP>1
        # decode, where global_batch must split evenly across replicas).
        b_q = max(
            case.batch_quantum,
            ((b + case.batch_quantum - 1) // case.batch_quantum) * case.batch_quantum,
        )
        if b_q not in seen:
            seen.add(b_q)
            out.append(b_q)
    return out


# ---------------------------------------------------------------------------
# ScaleScope invocation
# ---------------------------------------------------------------------------
def _patch_workload(
    template: Path,
    output: Path,
    *,
    global_batch: int,
    prefill_seq_len: int,
    num_colocated_experts: int,
    is_dense: bool,
    kernel_launch_latency_per_layer_sec: float,
    moe_token_pair_overhead_sec: float,
) -> None:
    with template.open() as f:
        spec = yaml.safe_load(f)
    spec["global_batch_size"] = int(global_batch)
    spec["prefill_seq_len"] = int(prefill_seq_len)
    spec["kernel_launch_latency_per_layer_sec"] = float(kernel_launch_latency_per_layer_sec)
    spec["moe_token_pair_overhead_sec"] = float(moe_token_pair_overhead_sec)
    spec["moe_spec_v2"]["num_colocated_experts"] = int(num_colocated_experts)
    if is_dense:
        # A dense (single-expert) model has no routing skew: all tokens go to
        # the one expert, so rho is definitionally 1.0. The shared template's
        # MoE skew (rho=4.0) is infeasible at num_regular_experts == 1, where
        # the realizable bound is rho <= E/top_k = 1.
        spec["moe_spec_v2"]["expert_imbalance"] = 1.0
    with output.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def _run_scalescope(record: Record, case: Case, *, global_batch: int) -> dict:
    workload_path = RUNS_DIR / f"_workload__{record.record_id}.yaml"
    output_yaml = RUNS_DIR / f"{record.record_id}.yaml"
    # Pre-emptively delete any stale per-record output left behind by a
    # previous panel run. Without this, a record that succeeded on an
    # earlier run but fails now (e.g. tripped by a fail-fast guard) would
    # leave the old success yaml on disk, and downstream analyses keyed
    # off `runs/<record_id>.yaml` would silently read stale predictions.
    # The workload yaml is left in place so a post-mortem can see exactly
    # what input produced the failure.
    output_yaml.unlink(missing_ok=True)
    template = WORKLOAD_TEMPLATES[case.precision]
    # Dense models are modeled as a single expert (num_regular_experts == 1);
    # detect them here so the workload uses rho = 1.0 instead of the template's
    # MoE skew. (is_moe is True at 1/1, so test the count directly.)
    with (MODEL_ZOO / f"{case.model_yaml}.yaml").open() as f:
        model_cfg = yaml.safe_load(f)
    is_dense = int(model_cfg["num_regular_experts"]) <= 1
    if case.model_yaml not in KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER:
        raise KeyError(
            f"No kernel-launch floor calibrated for model '{case.model_yaml}'. "
            f"Add it to KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER in run_validation_panel.py "
            f"(known: {sorted(KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER)})."
        )
    if case.model_yaml not in MOE_TOKEN_PAIR_OVERHEAD_SEC:
        raise KeyError(
            f"No EP dispatch overhead calibrated for model '{case.model_yaml}'. "
            f"Add it to MOE_TOKEN_PAIR_OVERHEAD_SEC in run_validation_panel.py "
            f"(known: {sorted(MOE_TOKEN_PAIR_OVERHEAD_SEC)})."
        )
    # Decode TPOT must be averaged over the generation: the per-step context
    # grows ISL -> ISL+OSL as tokens are emitted, so the KV-cache read cost
    # rises across the run. We evaluate a single representative step at the
    # MEAN decode context (ISL + OSL/2). Because the per-step time is affine in
    # context (KV read and attn-core compute are both linear in sequence
    # length), the single-step value at the mean context equals the true
    # average over the generation. Using the first-token context (= ISL) alone
    # systematically under-predicts TPOT for long-OSL workloads (validated:
    # median ratio 0.81x -> 0.90x on the isl1024/osl8192 llama3_70b records).
    decode_mean_context = record.isl + record.osl // 2
    _patch_workload(
        template,
        workload_path,
        global_batch=global_batch,
        prefill_seq_len=decode_mean_context,
        num_colocated_experts=case.num_colocated_experts,
        is_dense=is_dense,
        kernel_launch_latency_per_layer_sec=KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER[case.model_yaml],
        moe_token_pair_overhead_sec=MOE_TOKEN_PAIR_OVERHEAD_SEC[case.model_yaml],
    )
    cmd = [
        sys.executable,
        "-m",
        "scalescope.run_scalescope",
        "--hardware-lib",
        str(HARDWARE_LIB),
        "--model-spec",
        str(MODEL_ZOO / f"{case.model_yaml}.yaml"),
        "--workload-spec",
        str(workload_path),
        "--physical-topology",
        str(TOPO_PHYS / f"{case.phys_topo}.yaml"),
        "--logical-topology",
        str(TOPO_LOG / f"{case.logical_topo}.yaml"),
        "--pass-mode",
        "inference_decode",
        "--output-file",
        str(output_yaml),
        "--raw-output",
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        # Defensive belt-and-suspenders: if ScaleScope wrote a partial
        # yaml before crashing, ensure it's gone so nobody reads it.
        output_yaml.unlink(missing_ok=True)
        return {
            "ok": False,
            "error": (res.stderr or res.stdout or "<no output>").strip()[-500:],
            "global_batch": global_batch,
        }
    with output_yaml.open() as f:
        rows = yaml.safe_load(f)
    if not isinstance(rows, list) or len(rows) != 1:
        return {
            "ok": False,
            "error": f"Expected 1 row, got {len(rows) if isinstance(rows, list) else 'non-list'}",
            "global_batch": global_batch,
        }
    t = rows[0]["time"]
    tpot_user = t.get("tpot_user_seconds")
    tps = t.get("tokens_per_second")
    if tpot_user is None or tps is None:
        return {
            "ok": False,
            "error": f"ScaleScope returned None for tpot/tps (tpot={tpot_user}, tps={tps})",
            "global_batch": global_batch,
        }
    # Peak resident memory per chip (weights + KV cache + activations +
    # runtime). Used by the InferenceX HBM-feasibility check; required
    # because rows whose predicted peak exceeds chip HBM cannot
    # physically run the labelled batch and contaminate any
    # measured-vs-predicted aggregate. `--raw-output` makes ScaleScope
    # emit this as a raw byte count rather than a human-readable string.
    mem_fp = rows[0].get("memory_footprint") or {}
    peak_bytes = mem_fp.get("total_peak_memory_per_xpu_bytes")
    if peak_bytes is None:
        raise RuntimeError(
            f"ScaleScope output {output_yaml} missing "
            f"memory_footprint.total_peak_memory_per_xpu_bytes — cannot "
            f"compute HBM-feasibility flag."
        )
    return {
        "ok": True,
        "global_batch": global_batch,
        "tpot_user_ms": tpot_user * 1000.0,
        "tokens_per_second": tps,
        "ttft_forward_ms": (t.get("ttft_forward_seconds") or 0.0) * 1000.0,
        "peak_memory_per_xpu_bytes": peak_bytes,
    }


# ---------------------------------------------------------------------------
# Prefill-overhead α-correction for non-disagg InferenceX records
# ---------------------------------------------------------------------------
# Why this exists: non-disaggregated serving runs prefill and decode on the
# SAME chips under continuous batching. While one request decodes, chunks of
# other in-flight requests' prefill share the same compute units, so the
# measured per-user TPOT carries an interleaved-prefill tax. ScaleScope
# models a single pure-decode step at a fixed batch and ignores prefill
# entirely, so its predicted TPOT is systematically optimistic for these
# rows — and the size of that gap scales with how much of each chip's
# compute prefill is consuming, which varies row-to-row with concurrency,
# ISL and parallelism. Comparing raw against measured would therefore
# conflate a real modelling limitation (no continuous batching) with
# ScaleScope's decode accuracy.
#
# The correction makes the comparison apples-to-apples by estimating the
# steady-state fraction of per-chip compute that prefill consumes
# (`prefill_frac`) and inflating the pure-decode prediction by `1/α`. It is
# a first-principles compute-budget (FLOPS) argument, NOT a ScaleScope
# internal: prefill FLOPS are derived from the model_zoo spec (see
# `_prefill_facts_from_spec`) and the per-chip ceiling from
# `hardware_library.yaml`.
# Limiting behaviour: α→1 when prefill load is negligible (correction is a
# no-op), so it cannot manufacture accuracy where the decode model is
# already right — its near-no-op median on the clean non-disagg subset is
# the evidence that chunked prefill is not the dominant gap there. Disagg
# rows place prefill on a separate GPU pool and so need no correction.
#
# Parameterised on the InferenceX Record + Case.
#
# Steady-state per-chip FLOPS demand from prefill chunks of OTHER
# in-flight requests:
#
#     samples_per_sec_per_chip = (conc / mean_e2el) / dp_replicas
#     per_chip_flops_per_token = (2 × attn_params_total
#                                + 2 × ffn_total_active_params × ep_chip_share)
#                              ÷ tp_chip_count
#     prefill_frac = samples_per_sec_per_chip × ISL × per_chip_flops_per_token
#                  ÷ flops_budget
#     α = 1 − prefill_frac
#     corrected_tpot = pure_decode_tpot ÷ α
#
# `flops_budget` is the per-chip FLOPS ceiling at the row's actual
# precision, read from `hardware_library.yaml` (`nominal × efficiency`).


def _load_per_chip_flops_budget() -> dict[tuple[str, str], float]:
    """Build the (gpu_class, precision) -> effective FLOPS/sec/chip table from
    `hardware_library.yaml`, so the α-correction's compute ceiling stays in
    sync with the single source of truth ScaleScope itself uses. For each
    `<GPU>_COMPUTE` component, every `<precision>_nominal` entry becomes
    `(GPU, precision) -> nominal × efficiency`.
    """
    with HARDWARE_LIB.open() as f:
        compute = yaml.safe_load(f)["components"]["compute"]
    budget: dict[tuple[str, str], float] = {}
    suffix = "_COMPUTE"
    for component_name, spec in compute.items():
        if not component_name.endswith(suffix):
            raise ValueError(f"Unexpected compute component name: {component_name!r}")
        gpu_class = component_name[: -len(suffix)]
        # PyYAML parses unquoted scientific notation without a decimal point
        # (e.g. `3958e12`) as a str, so coerce nominal FLOPS to float.
        efficiency = float(spec["efficiency"])
        for key, value in spec.items():
            if not key.endswith("_nominal") or value is None:
                continue
            precision = key[: -len("_nominal")]
            budget[(gpu_class, precision)] = float(value) * efficiency
    return budget


# (gpu_class, precision) -> effective FLOPS/sec per chip.
PER_CHIP_FLOPS_BUDGET: dict[tuple[str, str], float] = _load_per_chip_flops_budget()


@dataclass(frozen=True)
class PrefillFacts:
    """Per-layer attention + active-FFN parameter counts used by the
    α-correction, derived from the model_zoo spec ScaleScope consumes (see
    `_prefill_facts_from_spec`).
    """

    attn_params_per_layer: float  # Q+K+V+O weight count for one layer
    n_layers: int  # transformer layers (assumed full-attention)
    ffn_active_params_per_layer: float  # active experts × matrices × d_model × intermediate


# Per-expert weight-matrix count by MLP kind (gated SwiGLU = gate+up+down).
_MLP_MATRIX_COUNT: dict[str, int] = {"gated_3matrix": 3}


@lru_cache(maxsize=None)
def _prefill_facts_from_spec(model_yaml: str) -> PrefillFacts:
    """Derive the α-correction's per-layer FLOPS facts from a model_zoo spec.

    Reads the same YAML ScaleScope runs against, so the correction's prefill
    compute cannot drift from the modelled architecture. Per layer:
      - attention (Q+K+V+O) = 2 × d_model × head_dim × (num_heads + num_kv_heads)
      - active FFN = (selected + shared) experts × matrices × d_model
                     × (mlp_ratio × d_model)
    """
    with (MODEL_ZOO / f"{model_yaml}.yaml").open() as f:
        spec = yaml.safe_load(f)
    d_model = spec["d_model"]
    head_dim = spec["head_dim"]
    num_heads = spec["num_heads"]
    num_kv_heads = spec["num_kv_heads"]
    mlp_kind = spec["mlp_kind"]
    if mlp_kind not in _MLP_MATRIX_COUNT:
        raise ValueError(
            f"{model_yaml}: unsupported mlp_kind {mlp_kind!r} for prefill-FLOPS "
            f"derivation (known: {sorted(_MLP_MATRIX_COUNT)})."
        )
    intermediate = spec["mlp_ratio"] * d_model
    active_experts = spec["num_selected_experts"] + spec["num_shared_experts"]
    return PrefillFacts(
        attn_params_per_layer=2.0 * d_model * head_dim * (num_heads + num_kv_heads),
        n_layers=spec["n_layers"],
        ffn_active_params_per_layer=(
            active_experts * _MLP_MATRIX_COUNT[mlp_kind] * d_model * intermediate
        ),
    )


def _queueing_factor(record: Record) -> float | None:
    """Return `mean_e2el / (OSL × median_tpot)` for an InferenceX row.

    Acts as a saturation probe: a value near 1× means each user spends
    its entire wall time decoding (steady-state batched behaviour, the
    regime ScaleScope models). Values significantly > 1× mean a chunk
    of each user's e2e latency is spent in the framework's wait queue,
    so reported `median_tpot` is per-step time on a smaller effective
    decode batch than `conc` — the row's labelling lies, and
    measured-vs-predicted ratios at the labelled batch are meaningless.

    Returns None if the record is missing `mean_e2el_s` (some
    InferenceX rows have it nulled out for very short runs).
    """
    if record.mean_e2el_s is None or record.measured_tpot_ms is None or record.osl <= 0:
        return None
    naive_e2el_s = record.osl * (record.measured_tpot_ms / 1000.0)
    if naive_e2el_s <= 0:
        return None
    return record.mean_e2el_s / naive_e2el_s


def _alpha_for_inferencex_record(record: Record, case: Case) -> tuple[float | None, dict]:
    """Return (alpha, debug) for a non-disagg InferenceX record.

    `alpha = 1 − prefill_frac`. Returns (None, debug) if any
    prerequisite is missing or if the precision/GPU pair has no FLOPS
    budget. Caller is responsible for skipping the correction when
    `record.disagg is True`.
    """
    debug: dict = {}
    facts = _prefill_facts_from_spec(case.model_yaml)
    if record.precision is None or record.gpu_class is None:
        return None, {"reason": "missing precision/gpu_class"}
    budget = PER_CHIP_FLOPS_BUDGET.get((record.gpu_class, record.precision))
    if budget is None:
        return None, {"reason": f"no FLOPS budget for ({record.gpu_class},{record.precision})"}
    if record.mean_e2el_s is None or record.mean_e2el_s <= 0:
        return None, {"reason": "missing mean_e2el_s"}
    if record.decode_tp is None or record.decode_ep is None:
        return None, {"reason": "missing decode shape"}

    # Derive parallelism counts from the record's decode shape. For all
    # currently registered non-disagg Cases this is "clean TP" (one TP
    # group per replica, EP=1, no DP-attention), but the formula is
    # general.
    tp_chip_count = record.decode_tp
    ep_chip_count = record.decode_ep
    dp_replicas = record.num_gpus // record.decode_tp
    if dp_replicas < 1:
        return None, {
            "reason": f"bad dp_replicas={dp_replicas} (num_gpus={record.num_gpus}, decode_tp={record.decode_tp})"
        }

    if record.conc is None:
        return None, {"reason": "missing conc"}
    samples_per_sec_total = float(record.conc) / record.mean_e2el_s
    samples_per_sec_per_chip = samples_per_sec_total / dp_replicas

    attn_params_total = facts.attn_params_per_layer * facts.n_layers
    ffn_active_total = facts.ffn_active_params_per_layer * facts.n_layers
    ep_chip_share = 1.0 / ep_chip_count
    per_chip_flops_per_token = (
        2.0 * attn_params_total + 2.0 * ffn_active_total * ep_chip_share
    ) / tp_chip_count
    prefill_compute_per_sec = (
        samples_per_sec_per_chip * float(record.isl) * per_chip_flops_per_token
    )
    prefill_frac = prefill_compute_per_sec / budget
    alpha = 1.0 - prefill_frac
    debug.update(
        samples_per_sec_total=samples_per_sec_total,
        samples_per_sec_per_chip=samples_per_sec_per_chip,
        per_chip_flops_per_token=per_chip_flops_per_token,
        prefill_compute_per_sec=prefill_compute_per_sec,
        prefill_frac=prefill_frac,
        alpha=alpha,
        flops_budget=budget,
        dp_replicas=dp_replicas,
        tp_chip_count=tp_chip_count,
        ep_chip_count=ep_chip_count,
    )
    return alpha, debug


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--source",
        choices=["trtllm", "inferencex", "all"],
        default="all",
        help="Which corpus to use. 'all' = trtllm + inferencex.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N runnable records (smoke-test mode).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Keep only records with this `model_id` (e.g. 'minimax_m2_5', "
            "'gpt_oss_120b'). Repeatable; multiple values are OR-combined. "
            "Combined with other filters using AND."
        ),
    )
    parser.add_argument(
        "--gpu-class",
        action="append",
        default=None,
        metavar="CLASS",
        help=(
            "Keep only records on this GPU class (e.g. 'B200', 'H100', "
            "'GB200'). Repeatable; multiple values are OR-combined."
        ),
    )
    parser.add_argument(
        "--record-contains",
        action="append",
        default=None,
        metavar="SUBSTR",
        help=(
            "Keep only records whose record_id contains this substring "
            "(case-sensitive). Repeatable; multiple values are OR-combined "
            "(any-match)."
        ),
    )
    parser.add_argument(
        "--include-fp4",
        action="store_true",
        help=(
            "Include FP4 records in the panel. FP4 is EXCLUDED by default "
            "because ScaleScope undercounts FP4 cost — it does not model the "
            "extra dequant / block-scaling machinery FP4 requires — which "
            "biases the predicted/measured comparison. Pass this flag to run "
            "the FP4 cases anyway."
        ),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help=(
            "Print the post-filter runnable record IDs (one per line) and "
            "exit without running ScaleScope or writing any output files. "
            "Useful for verifying a filter set before committing to a run."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to summary filenames "
            "(e.g. '__smoke' -> 'validation_panel_summary__smoke.{md,yaml}'). "
            "Auto-set to '__partial' when any filter is active "
            "(--source != 'all', --limit, --model, --gpu-class, "
            "--record-contains), to avoid clobbering canonical full-run "
            "summaries. Pass empty string '' to force overwrite."
        ),
    )
    args = parser.parse_args(argv)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _require_2d_workload_templates()

    # Auto-suffix partial runs so smoke tests don't clobber the canonical
    # summary. The full panel is "--source=all with no filters"; anything
    # else writes to *__partial.{md,yaml} unless --output-suffix='' is
    # explicitly set.
    any_filter_active = (
        args.source != "all"
        or args.limit is not None
        or bool(args.model)
        or bool(args.gpu_class)
        or bool(args.record_contains)
    )
    if args.output_suffix is None:
        if any_filter_active:
            suffix = "__partial"
        elif args.include_fp4:
            # FP4-excluded is the canonical full run; including FP4 produces a
            # superset, so route it to its own file rather than clobber.
            suffix = "__with_fp4"
        else:
            suffix = ""
    else:
        suffix = args.output_suffix

    records: list[Record] = []
    if args.source in ("trtllm", "all"):
        records += _load_trtllm_records()
    if args.source in ("inferencex", "all"):
        records += _load_inferencex_records()

    runnable = _runnable_records(records)
    deferred = [r for r in records if r not in runnable]

    # Apply subset filters AFTER `runnable` / `deferred` are computed so
    # the printed counts continue to reflect "how much could we run if
    # we lifted the filter". Each filter (model, gpu-class, record
    # substring) is intersected with the others (AND across flags);
    # multiple values within a single flag are OR-combined.
    filter_model_ids: set[str] | None = set(args.model) if args.model else None
    filter_gpu_classes: set[str] | None = set(args.gpu_class) if args.gpu_class else None
    filter_record_contains: list[str] | None = (
        list(args.record_contains) if args.record_contains else None
    )

    def _record_passes_filters(rec: Record) -> bool:
        if filter_model_ids is not None and rec.model_id not in filter_model_ids:
            return False
        if filter_gpu_classes is not None and rec.gpu_class not in filter_gpu_classes:
            return False
        if filter_record_contains is not None and not any(
            s in rec.record_id for s in filter_record_contains
        ):
            return False
        return True

    filtered_out = 0
    if filter_model_ids or filter_gpu_classes or filter_record_contains:
        before = len(runnable)
        runnable = [r for r in runnable if _record_passes_filters(r)]
        filtered_out = before - len(runnable)

    # FP4 is excluded by default: ScaleScope undercounts FP4 cost (no dequant /
    # block-scaling machinery), so those rows bias the comparison. The Case
    # precision is authoritative for what ScaleScope actually runs at.
    fp4_excluded = 0
    if not args.include_fp4:
        before = len(runnable)
        runnable = [
            r
            for r in runnable
            if _case_for(r).precision != "fp4" or r.model_id in FP4_CANONICAL_MODEL_IDS
        ]
        fp4_excluded = before - len(runnable)

    if args.limit is not None:
        runnable = runnable[: args.limit]

    msg = (
        f"Total records: {len(records)}; runnable today: {len(runnable)}; "
        f"deferred: {len(deferred)}."
    )
    if filtered_out:
        msg += f" Filtered out by --model/--gpu-class/--record-contains: {filtered_out}."
    if fp4_excluded:
        msg += f" Excluded {fp4_excluded} FP4 record(s) (pass --include-fp4 to run them)."
    if args.limit is not None:
        msg += f" Limited to first {args.limit}."
    print(msg)

    if args.list_only:
        # Print one record_id per line in the same order the run loop
        # would visit them, then exit cleanly without writing any files
        # or invoking ScaleScope.
        for rec in runnable:
            case = _case_for(rec)
            case_id = case.case_id if case is not None else "<no-case>"
            print(f"{rec.record_id}\tcase={case_id}")
        print(f"# {len(runnable)} record(s) would run.")
        return 0

    import math

    # ------------------------------------------------------------------
    # Pre-flight summary + progress bar.
    #
    # We report TWO counts at the top so the operator can sanity-check
    # scope before committing to a long run:
    #
    #   * `cases`: distinct measured records that will be compared
    #     against ScaleScope. One row in the output table per case.
    #   * `scalescope invocations`: actual subprocess calls. Each
    #     inferencex case = 1 invocation; each trtllm_perf_overview case
    #     = `len(BATCH_SWEEP_GRID)` invocations because we sweep batch
    #     size and pick the best-fit point.
    #
    # The progress bar advances per ScaleScope invocation so it stays
    # smooth across the long trtllm tail; each per-record `print(...)`
    # below is routed through `tqdm.write` so it interleaves cleanly
    # above the bar.
    invocations_per_record: list[int] = []
    for record in runnable:
        case = _case_for(record)
        if case is None:  # pragma: no cover — runnable filter guarantees a hit
            raise RuntimeError(f"Internal: no Case for runnable record {record.record_id!r}")
        if record.source == "trtllm_perf_overview":
            invocations_per_record.append(len(_trtllm_sweep_batches(case)))
        else:
            invocations_per_record.append(1)
    total_invocations = sum(invocations_per_record)
    print(
        f"Run plan: {len(runnable)} case(s) -> {total_invocations} ScaleScope "
        f"invocation(s) (1 per inferencex case, "
        f"{len(BATCH_SWEEP_GRID)}-batch sweep per trtllm case).",
        flush=True,
    )

    # `mininterval=2.0` keeps the bar repainting at least every 2 s even when
    # a single invocation takes longer (typical: 3-15 s). `miniters=1` makes
    # every completed invocation trigger a repaint regardless of timing, so
    # both `tail -f` users and TTY users see fresh state without waiting for
    # tqdm's default rate-limit. `set_postfix(...)` below shows the active
    # record id so a single line of the captured log reveals what is running.
    progress = tqdm(
        total=total_invocations,
        desc="ScaleScope",
        unit="invoc",
        dynamic_ncols=True,
        leave=True,
        mininterval=2.0,
        miniters=1,
    )

    # Heartbeat cadence: a flushed `tqdm.write(...)` summary every N
    # completed invocations. tqdm's own bar uses carriage returns which can
    # be lost in some captured logs; the heartbeat is a plain newline-
    # terminated line, so `grep '^Heartbeat'` over the captured log always
    # reveals real progress.
    HEARTBEAT_EVERY = 25
    completed_invocations = 0

    def _heartbeat(force: bool = False) -> None:
        if force or completed_invocations % HEARTBEAT_EVERY == 0:
            tqdm.write(
                f"Heartbeat: {completed_invocations}/{total_invocations} invocations "
                f"complete ({100.0 * completed_invocations / max(1, total_invocations):.1f}%).",
            )

    def _bump_progress(n: int, record_id: str) -> None:
        nonlocal completed_invocations
        completed_invocations += n
        progress.update(n)
        progress.set_postfix_str(record_id, refresh=False)
        _heartbeat()

    rows: list[dict] = []
    for record, record_invocations in zip(runnable, invocations_per_record):
        case = _case_for(record)
        if case is None:  # pragma: no cover — runnable filter guarantees a hit
            raise RuntimeError(f"Internal: no Case for runnable record {record.record_id!r}")
        row: dict = {
            "record_id": record.record_id,
            "source": record.source,
            "submitter": record.submitter,
            "system": record.system,
            "model_id": record.model_id,
            "gpu_class": record.gpu_class,
            "benchmark": record.benchmark_label,
            "isl": record.isl,
            "osl": record.osl,
            "case": case.case_id,
            "parallelism": case.parallelism_label,
            "measured_sys_tps": record.measured_sys_tps,
            "measured_tpot_ms": record.measured_tpot_ms,
        }
        if record.source == "inferencex":
            # ScaleScope is invoked once at global_batch = `conc` (the
            # row's reported concurrency = number of simultaneous users
            # = exactly the steady-state decode-pool batch). Disagg rows
            # need no further correction; non-disagg rows pay
            # chunked-prefill overhead which we quantify with the
            # α-correction below.
            if record.conc is None:
                row.update(ok=False, error="inferencex: missing conc on Record")
                rows.append(row)
                _bump_progress(record_invocations, record.record_id)
                continue
            global_batch = max(
                case.batch_quantum, (record.conc // case.batch_quantum) * case.batch_quantum
            )
            tqdm.write(
                f"  [{record.record_id}] case={case.case_id} batch={global_batch} (conc) ...",
            )
            ss = _run_scalescope(record, case, global_batch=global_batch)
            _bump_progress(1, record.record_id)
            row["global_batch"] = global_batch
            row["batch_selection"] = "inferencex_conc"
            row["disagg"] = record.disagg
            # Structured decode shape consumed by the kernel-floor audit
            # (calibration/calibrate_kernel_floor.py) so it never has to
            # re-parse record_id.
            row["tp"] = record.decode_tp
            row["conc"] = record.conc
            if ss["ok"]:
                predicted_tpot = ss["tpot_user_ms"]
                predicted_tps = ss["tokens_per_second"]
                row.update(
                    ok=True,
                    predicted_sys_tps=predicted_tps,
                    predicted_tpot_ms=predicted_tpot,
                    tps_ratio=predicted_tps / record.measured_sys_tps,
                    tpot_ratio=predicted_tpot / record.measured_tpot_ms,
                )
                # HBM-capacity feasibility check. ScaleScope models the
                # peak resident state (weights + full KV cache for all
                # `conc` users + activations); if that exceeds the
                # chip's HBM the labelled batch cannot physically run
                # and the framework must be admitting <conc users at a
                # time. Caller flags these rows separately from the
                # legitimate model-vs-measurement aggregate.
                peak_bytes = ss["peak_memory_per_xpu_bytes"]
                hbm_capacity = HBM_CAPACITY_BYTES.get(record.gpu_class)
                if hbm_capacity is None:
                    raise RuntimeError(
                        f"No HBM_CAPACITY_BYTES entry for gpu_class "
                        f"{record.gpu_class!r}; add it before running."
                    )
                gib = 1024**3
                row["peak_memory_per_chip_gib"] = peak_bytes / gib
                row["hbm_capacity_gib"] = hbm_capacity / gib
                row["peak_to_hbm_ratio"] = peak_bytes / hbm_capacity
                row["infeasible_capacity"] = peak_bytes > hbm_capacity
                # Queue-saturation probe (non-disagg only — disagg rows
                # don't expose a coherent mean_e2el to compute against,
                # and they're not subject to the same scheduling-queue
                # contention because prefill lives on a separate pool).
                if record.disagg is False:
                    qf = _queueing_factor(record)
                    row["queueing_factor"] = qf
                    row["saturated"] = qf is not None and qf > QUEUEING_FACTOR_SATURATED_THRESHOLD
                    row["contaminated"] = bool(row["infeasible_capacity"] or row["saturated"])
                # Non-disagg rows get the prefill-overhead α-correction:
                # `T_real_pred = T_pure_decode / (1 − prefill_frac)`.
                # Disagg rows skip it (prefill runs on a separate pool).
                if record.disagg is False:
                    alpha, alpha_dbg = _alpha_for_inferencex_record(record, case)
                    row["alpha_debug"] = alpha_dbg
                    if alpha is not None:
                        row["alpha"] = alpha
                        row["prefill_frac"] = alpha_dbg.get("prefill_frac")
                        if alpha > 0:
                            corrected_tpot = predicted_tpot / alpha
                            row["corrected_tpot_ms"] = corrected_tpot
                            row["corrected_tpot_ratio"] = corrected_tpot / record.measured_tpot_ms
                            row["corrected_tps"] = predicted_tps * alpha
                            row["corrected_tps_ratio"] = (
                                predicted_tps * alpha
                            ) / record.measured_sys_tps
                        else:
                            # α ≤ 0 means the steady-state prefill load
                            # exceeds the chip's FLOPS budget (saturation
                            # regime). Report unbounded correction
                            # rather than silently substituting a value.
                            row["corrected_tpot_ms"] = None
                            row["corrected_tpot_ratio"] = None
                            row["corrected_tps"] = None
                            row["corrected_tps_ratio"] = None
            else:
                row.update(ok=False, error=ss["error"])
        elif record.source == "trtllm_perf_overview":
            # Sweep: pick the batch whose predicted throughput is closest
            # (log-space) to the measured throughput.
            sweep = _trtllm_sweep_batches(case)
            tqdm.write(
                f"  [{record.record_id}] case={case.case_id} sweep={sweep} ...",
            )
            sweep_results: list[dict] = []
            for b in sweep:
                ss = _run_scalescope(record, case, global_batch=b)
                if ss["ok"]:
                    sweep_results.append(ss)
                _bump_progress(1, f"{record.record_id}@b={b}")
            row["batch_selection"] = "trtllm_sweep_match_throughput"
            row["sweep_grid"] = sweep
            if not sweep_results:
                row.update(ok=False, error="all sweep points failed")
            else:
                # Pick the batch whose predicted throughput is closest to the
                # measurement. We minimise |log(pred) - log(meas)| =
                # |log(pred/meas)|, i.e. the multiplicative (ratio) error, not
                # the absolute tok/s gap. Three reasons:
                #   1. Dynamic range: throughput rises ~proportionally with
                #      batch across BATCH_SWEEP_GRID (1..4096), so absolute
                #      gaps at the high-batch end dwarf those at the low end;
                #      a linear metric would be biased toward whichever region
                #      has the larger scale. Log space normalises that.
                #   2. Symmetry: a 2x over- and a 2x under-prediction are
                #      equidistant in log space. Linear distance favours
                #      under-predictions (an undershoot is bounded by `meas`,
                #      an overshoot is unbounded), e.g. meas=1000 makes
                #      pred=500 look "closer" (Δ=500) than pred=2000 (Δ=1000)
                #      even though both are exactly 2x off.
                #   3. Consistency: the report scores quality by the ratio
                #      pred/meas against ±X% bands (and the geomean), so the
                #      selected batch is the one whose ratio is closest to
                #      1.00x in the same multiplicative sense.
                best = min(
                    sweep_results,
                    key=lambda s: abs(
                        math.log(s["tokens_per_second"]) - math.log(record.measured_sys_tps)
                    ),
                )
                row.update(
                    ok=True,
                    global_batch=best["global_batch"],
                    predicted_sys_tps=best["tokens_per_second"],
                    predicted_tpot_ms=best["tpot_user_ms"],
                    tps_ratio=best["tokens_per_second"] / record.measured_sys_tps,
                    tpot_ratio=None,
                    sweep_throughputs={
                        s["global_batch"]: round(s["tokens_per_second"], 1) for s in sweep_results
                    },
                )
        else:
            raise ValueError(f"Unknown source: {record.source}")
        rows.append(row)

    _heartbeat(force=True)
    progress.close()

    now = datetime.datetime.now()
    timestamp_iso = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_fn = now.strftime("%Y%m%d_%H%M%S")

    yaml_dict = {"timestamp": timestamp_iso, "results": rows}
    yaml_text = yaml.safe_dump(yaml_dict, sort_keys=False)
    md_text = _format_markdown(rows, deferred_count=len(deferred), timestamp=timestamp_iso)

    summary_yaml_ts = RESULTS_DIR / f"validation_panel_summary_{timestamp_fn}{suffix}.yaml"
    summary_md_ts = RESULTS_DIR / f"validation_panel_summary_{timestamp_fn}{suffix}.md"
    summary_yaml = RESULTS_DIR / f"validation_panel_summary{suffix}.yaml"
    summary_md = RESULTS_DIR / f"validation_panel_summary{suffix}.md"

    summary_yaml_ts.write_text(yaml_text)
    summary_md_ts.write_text(md_text)
    summary_yaml.write_text(yaml_text)
    summary_md.write_text(md_text)

    n_ok = sum(1 for r in rows if r.get("ok"))
    print()
    print(f"Wrote {summary_yaml_ts} and {summary_yaml}")
    print(f"Wrote {summary_md_ts} and {summary_md}")
    print(f"OK: {n_ok}/{len(rows)} records ran successfully.")
    return 0


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------
def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _within(ratios: list[float], lo: float, hi: float) -> float:
    """Percentage of `ratios` falling within the closed band [lo, hi]."""
    return 100.0 * sum(1 for r in ratios if lo <= r <= hi) / len(ratios)


# Accuracy bands reported for every predicted/measured ratio series. A ratio
# of 1.00× is a perfect match; the bands count how many records land within
# ±10% / ±25% / ±50% of the measurement.
_ACCURACY_BANDS: list[tuple[str, float, float]] = [
    ("±10%", 0.90, 1.10),
    ("±25%", 0.75, 1.25),
    ("±50%", 0.50, 1.50),
]


def _stat_row(bucket: str, metric: str, ratios: list[float]) -> str | None:
    """One markdown table row: N, geomean, median, min, max, ±band coverage.

    Returns None for an empty series so the caller can skip the metric.
    """
    if not ratios:
        return None
    srt = sorted(ratios)
    n = len(srt)
    bands = " | ".join(f"{_within(srt, lo, hi):.0f}%" for _, lo, hi in _ACCURACY_BANDS)
    return (
        f"| {bucket} | {metric} | {n} | {_geomean(srt):.2f}× | "
        f"{srt[n // 2]:.2f}× | {srt[0]:.2f}× | {srt[-1]:.2f}× | {bands} |"
    )


def _format_markdown(rows: list[dict], *, deferred_count: int, timestamp: str | None = None) -> str:
    out: list[str] = []
    out.append("# Validation-panel comparison — ScaleScope vs. published benchmarks\n")
    if timestamp:
        out.append(f"**Run timestamp**: `{timestamp}`\n")
    out.append(
        "Each row is one measured datapoint compared to a ScaleScope "
        "decode-step prediction at the operating-point batch.\n"
    )
    out.append(
        "- **TRT-LLM perf-overview**: batch chosen by sweeping ScaleScope "
        "across `BATCH_SWEEP_GRID` and picking the batch whose predicted "
        "throughput is closest (log-space) to the measured value.\n"
        "- **InferenceX (disagg decode pool)**: batch = the row's `conc` "
        "field directly. ScaleScope's steady-state decode model matches the "
        "decode pool of a PD-disaggregated deployment exactly; prefill is "
        "ignored.\n"
    )
    ok_rows = [r for r in rows if r.get("ok")]
    failed_rows = [r for r in rows if not r.get("ok")]
    out.append(
        f"- Records compared: **{len(rows)}** (OK: **{len(ok_rows)}**, failed at runtime: **{len(failed_rows)}**)"
    )
    out.append(
        f"- Deferred at pre-flight (other HW / unsupported case shapes / unsupported models): **{deferred_count}**"
    )
    out.append("")

    # Runtime-failure breakdown. We classify failed rows by the leading
    # error signature so the operator can tell at a glance whether the
    # losses are (a) the multi-attention-domain decode fail-fast surfacing
    # unsupported modeling, or (b) something else that warrants
    # investigation. See conf26_followup/modeling_improvements_todo.md §5.
    if failed_rows:
        phase_a_rows: list[dict] = []
        other_rows: list[dict] = []
        for r in failed_rows:
            err = r.get("error") or ""
            if "NotImplementedError" in err and "attn_domains_per_full_expert_set" in err:
                phase_a_rows.append(r)
            else:
                other_rows.append(r)

        out.append("## Failed at runtime — breakdown")
        out.append("")
        if phase_a_rows:
            out.append(f"### Multi-attention-domain decode fail-fast ({len(phase_a_rows)} records)")
            out.append("")
            out.append(
                "These records hit the multi-attention-domain decode guard in "
                "`scalescope/comm_model.py`: the decode comm model currently "
                "assumes one attention domain per expert set "
                "(`attn_domains_per_full_expert_set == 1`). Configs where the "
                "expert set spans multiple attention TP domains "
                "(centralized-pool / DP-attention + EP regime) require the "
                "scale-up / scale-out dispatch + combine modeling described "
                "in `conf26_followup/modeling_improvements_todo.md §5`, which "
                "is not yet implemented; these rows are excluded from the "
                "aggregates below."
            )
            out.append("")
            # Roll up by case label so the affected topology family is
            # visible without dumping the per-record list.
            case_counts: dict[str, int] = {}
            case_ratios: dict[str, tuple[int, int, int]] = {}
            for r in phase_a_rows:
                c = r.get("case", "<unknown>")
                case_counts[c] = case_counts.get(c, 0) + 1
                if c not in case_ratios:
                    err = r.get("error") or ""
                    import re as _re

                    m = _re.search(
                        r"attn_domains_per_full_expert_set=(\d+).*?"
                        r"full_expert_set_xpu_count=(\d+), attn_tp_size=(\d+)",
                        err,
                    )
                    if m:
                        case_ratios[c] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            out.append(
                "| Records | Case (topology family) | attn_domains_per_full_expert_set | full_expert_set_xpu_count | attn_tp_size |"
            )
            out.append("|---:|---|---:|---:|---:|")
            for c in sorted(case_counts, key=lambda k: -case_counts[k]):
                ratio_str = ""
                fes_str = ""
                atp_str = ""
                if c in case_ratios:
                    ratio, fes, atp = case_ratios[c]
                    ratio_str = str(ratio)
                    fes_str = str(fes)
                    atp_str = str(atp)
                out.append(f"| {case_counts[c]} | `{c}` | {ratio_str} | {fes_str} | {atp_str} |")
            out.append("")
        if other_rows:
            out.append(f"### Other runtime errors ({len(other_rows)} records)")
            out.append("")
            for r in other_rows:
                err_lines = (r.get("error") or "").strip().splitlines()
                err_head = err_lines[-1] if err_lines else "<no error>"
                out.append(f"- `{r['record_id']}` — case `{r.get('case', '?')}` — {err_head[:200]}")
            out.append("")
    if ok_rows:
        out.append("## Aggregate stats")
        out.append("")
        out.append(
            "Accuracy scoreboard over all OK records. Ratio = predicted / "
            "measured, so `geomean = 1.00×` is unbiased; the `±X%` columns give "
            "the fraction of records whose ratio lands within that band of the "
            "measurement. InferenceX rows are scored on per-user TPOT; TRT-LLM "
            "uses system TPS (per-user TPOT is not published). The α-correction "
            "(`pure_decode / (1 − prefill_frac)`) is only meaningful for the "
            "non-disagg rows, where chunked prefill interleaves with decode. "
            "The final row pools every clean record on its best available "
            "metric (α-corrected TPOT for non-disagg, naive TPOT for disagg, "
            "else TPS) and excludes the contaminated non-disagg rows "
            "(measurement artifacts, see Findings).\n"
        )
        out.append("| Bucket | Metric | N | Geomean | Median | Min | Max | ±10% | ±25% | ±50% |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        def _row(bucket: str, metric: str, ratios: list[float]) -> None:
            line = _stat_row(bucket, metric, ratios)
            if line is not None:
                out.append(line)

        def _vals(subset: list[dict], key: str) -> list[float]:
            return [r[key] for r in subset if r.get(key) is not None]

        trt_rows = [r for r in ok_rows if r["source"] == "trtllm_perf_overview"]
        disagg_rows = [
            r for r in ok_rows if r["source"] == "inferencex" and r.get("disagg") is True
        ]
        nondis_rows = [
            r for r in ok_rows if r["source"] == "inferencex" and r.get("disagg") is False
        ]
        nd_clean = [r for r in nondis_rows if not r.get("contaminated")]
        nd_contam = [r for r in nondis_rows if r.get("contaminated")]

        # For InferenceX, TPS ratio == 1 / (TPOT ratio), so only TPOT is
        # scored; TPS is scored only for TRT-LLM (no published TPOT).
        _row("TRT-LLM perf-overview", "TPS", _vals(trt_rows, "tps_ratio"))
        _row("InferenceX disagg", "TPOT", _vals(disagg_rows, "tpot_ratio"))
        _row("InferenceX non-disagg — clean", "TPOT naive", _vals(nd_clean, "tpot_ratio"))
        _row(
            "InferenceX non-disagg — clean", "TPOT α-corr", _vals(nd_clean, "corrected_tpot_ratio")
        )
        _row("InferenceX non-disagg — contaminated", "TPOT", _vals(nd_contam, "tpot_ratio"))
        _row(
            "InferenceX non-disagg — contaminated",
            "TPOT α-corr",
            _vals(nd_contam, "corrected_tpot_ratio"),
        )

        # Pooled accuracy: each clean record's best available ratio —
        # α-corrected TPOT where the correction applies (non-disagg), naive
        # TPOT for disagg (no chunked-prefill overlap to correct), else TPS
        # for TRT-LLM throughput rows.
        def _primary_ratio(r: dict) -> float | None:
            if r.get("corrected_tpot_ratio") is not None:
                return r["corrected_tpot_ratio"]
            if r.get("tpot_ratio") is not None:
                return r["tpot_ratio"]
            return r.get("tps_ratio")

        overall: list[float] = []
        for r in ok_rows:
            if r.get("contaminated"):
                continue
            primary = _primary_ratio(r)
            if primary is not None:
                overall.append(primary)
        _row("**All clean records**", "TPOT/TPS", overall)
        out.append("")

        # Per-(model, framework, GPU) breakdown of the same pooled
        # best-metric ratios, sorted worst-first (largest |log geomean|),
        # so the biggest modeling gaps are at the top. Same clean-record
        # filter and metric selection as the pooled row above.
        groups: dict[tuple[str, str, str], list[float]] = {}
        for r in ok_rows:
            if r.get("contaminated"):
                continue
            primary = _primary_ratio(r)
            if primary is None:
                continue
            key = (
                r["model_id"],
                r.get("submitter") or "trtllm",
                r.get("gpu_class") or "?",
            )
            groups.setdefault(key, []).append(primary)
        if groups:
            out.append("### Per model / framework")
            out.append("")
            out.append(
                "Clean records only, best available metric per record (same "
                "selection as the pooled row above). Sorted worst-first by "
                "mean |log error| — unlike the geomean, this does not let "
                "symmetric over/under scatter cancel out, so the top rows "
                "are where modeling attention pays off most.\n"
            )
            out.append(
                "| Model | Framework | GPU | N | Mean \\|err\\| | Geomean | Median | Min | Max | ±10% | ±25% | ±50% |"
            )
            out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

            def _mean_abs_log_err(ratios: list[float]) -> float:
                return sum(abs(math.log(r)) for r in ratios) / len(ratios)

            for (model_id, framework, gpu), ratios in sorted(
                groups.items(), key=lambda kv: -_mean_abs_log_err(kv[1])
            ):
                srt = sorted(ratios)
                n = len(srt)
                bands = " | ".join(f"{_within(srt, lo, hi):.0f}%" for _, lo, hi in _ACCURACY_BANDS)
                mean_err_pct = 100.0 * (math.exp(_mean_abs_log_err(srt)) - 1.0)
                out.append(
                    f"| {model_id} | {framework} | {gpu} | {n} | "
                    f"{mean_err_pct:.0f}% | {_geomean(srt):.2f}× | {srt[n // 2]:.2f}× | "
                    f"{srt[0]:.2f}× | {srt[-1]:.2f}× | {bands} |"
                )
            out.append("")

        if nd_contam:
            n_infeasible = sum(1 for r in nd_contam if r.get("infeasible_capacity"))
            n_saturated = sum(1 for r in nd_contam if r.get("saturated"))
            out.append(
                f"Contamination breakdown: `infeasible_capacity=True` on "
                f"{n_infeasible} row(s) (predicted peak > chip HBM); "
                f"`saturated=True` (queueing_factor > "
                f"{QUEUEING_FACTOR_SATURATED_THRESHOLD:g}×) on {n_saturated} "
                f"row(s). Contaminated rows are excluded from the pooled "
                f'"All clean records" line — see the per-record table below '
                f"for their `HBM%` and `Q-factor` flags."
            )
            out.append("")

    # ---- InferenceX disagg subset ----
    ix_rows_ok = [r for r in rows if r["source"] == "inferencex" and r.get("ok")]
    ix_disagg = [r for r in ix_rows_ok if r.get("disagg") is True]
    ix_nondis = [r for r in ix_rows_ok if r.get("disagg") is False]
    if ix_disagg:
        out.append("## InferenceX disagg-decode records — GB200 NVL72")
        out.append("")
        out.append(
            "Decode-only pool, `global_batch = conc`. The closest 1.00× ratio "
            "is the IDEAL match — both TPOT (per-user latency) and TPS "
            "(system throughput) ratios are reported so the residual is "
            "decomposable into per-step time error vs. throughput error.\n"
        )
        out.append(
            "| System | Model | Framework | Parallelism | ISL | OSL | Conc | "
            "TPOT meas (ms) | TPOT pred (ms) | TPOT ratio | TPS meas | TPS pred | TPS ratio |"
        )
        out.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in ix_disagg:
            out.append(
                f"| {r['system']} | {r['model_id']} | {r['submitter']} | {r['parallelism']} "
                f"| {r['isl']} | {r['osl']} | {r['global_batch']} "
                f"| {r['measured_tpot_ms']:.2f} | {r['predicted_tpot_ms']:.2f} "
                f"| {r['tpot_ratio']:.2f}× "
                f"| {r['measured_sys_tps']:.0f} | {r['predicted_sys_tps']:.0f} "
                f"| {r['tps_ratio']:.2f}× |"
            )
        out.append("")

    # ---- InferenceX non-disagg subset ----
    if ix_nondis:
        out.append("## InferenceX non-disagg records — B200 continuous batching")
        out.append("")
        out.append(
            "Non-disaggregated rows pay chunked-prefill overhead in steady "
            "state. The α-correction `T_real_pred = T_pure_decode / (1 − "
            "prefill_frac)` is applied with prefill_frac derived from real "
            "model architecture × measured `samples_per_sec = conc / "
            "mean_e2el`, ÷ the chip's per-precision FLOPS budget. `α = 1` "
            "(no prefill load) recovers the naive ratio; α < 1 lifts the "
            "predicted TPOT toward the measured value.\n"
        )
        out.append(
            "Two extra columns flag rows where the labelled `conc` is not "
            "what the framework actually served per decode step. **`HBM%`** "
            "= ScaleScope-predicted peak resident memory ÷ chip HBM "
            "capacity; values > 100% mean the labelled batch cannot fit "
            "and the framework must be admitting fewer than `conc` users "
            "concurrently. **`Q-factor`** = `mean_e2el / (OSL × "
            "median_tpot)`; values near 1× mean each user spends its "
            "entire wall-clock decoding (clean steady state), values "
            f"≥ {QUEUEING_FACTOR_SATURATED_THRESHOLD:g}× mean a chunk of "
            "each user's wall time is spent in the scheduler's wait "
            "queue, so reported `median_tpot` is per-step time on a "
            "smaller effective decode batch than labelled. Rows where "
            "either flag fires are marked `⚠` and excluded from the "
            "headline median in the aggregate stats above.\n"
        )
        out.append(
            "| Flag | System | Model | Framework | Parallelism | ISL | OSL | Conc | "
            "Precision | TPOT meas (ms) | TPOT pred (ms) | TPOT ratio | α | "
            "Corrected TPOT (ms) | HBM% | Q-factor | Corrected ratio |"
        )
        out.append(
            "|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in ix_nondis:
            alpha = r.get("alpha")
            corr_tpot = r.get("corrected_tpot_ms")
            corr_ratio = r.get("corrected_tpot_ratio")
            alpha_s = f"{alpha:.2f}" if alpha is not None else "n/a"
            corr_tpot_s = f"{corr_tpot:.2f}" if corr_tpot is not None else "n/a"
            corr_ratio_s = f"{corr_ratio:.2f}×" if corr_ratio is not None else "n/a"
            # Pull precision from the case_id suffix `__fp4` / `__fp8`
            prec = (
                "fp4"
                if r["case"].endswith("__fp4")
                else ("fp8" if r["case"].endswith("__fp8") else "?")
            )
            hbm_ratio = r.get("peak_to_hbm_ratio")
            qf = r.get("queueing_factor")
            hbm_s = f"{hbm_ratio*100:.0f}%" if hbm_ratio is not None else "n/a"
            qf_s = f"{qf:.2f}×" if qf is not None else "n/a"
            flags = []
            if r.get("infeasible_capacity"):
                flags.append("OOM")
            if r.get("saturated"):
                flags.append("QUEUE")
            flag_s = "⚠ " + ",".join(flags) if flags else ""
            out.append(
                f"| {flag_s} | {r['system']} | {r['model_id']} | {r['submitter']} | {r['parallelism']} "
                f"| {r['isl']} | {r['osl']} | {r['global_batch']} | {prec} "
                f"| {r['measured_tpot_ms']:.2f} | {r['predicted_tpot_ms']:.2f} "
                f"| {r['tpot_ratio']:.2f}× | {alpha_s} | {corr_tpot_s} "
                f"| {hbm_s} | {qf_s} | {corr_ratio_s} |"
            )
        out.append("")

    out.append("## TRT-LLM perf-overview records")
    out.append("")
    out.append(
        "Each row reports ScaleScope's prediction at the batch (from "
        "BATCH_SWEEP_GRID) whose predicted throughput was closest in log "
        "space to the measured throughput. `TPS ratio` of 1.0× by "
        "construction is the IDEAL match; deviation tells us TPS at the "
        "best batch is still off (gap is then in HBM-BW efficiency or "
        "in the FLOPS efficiency factor).\n"
    )
    out.append(
        "| GPU | Model | TP | ISL | OSL | Best batch | TPS meas | TPS pred @ batch | TPOT pred (ms) | TPS ratio |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["source"] != "trtllm_perf_overview" or not r.get("ok"):
            continue
        gpu = r["system"].split("_")[0]
        # case format: <model>__h100_Nx_tpK[_dpL|_epM]
        tp_segment = next(
            (s for s in r["case"].split("__")[-1].split("_") if s.startswith("tp")), "tp1"
        )
        tp = tp_segment.replace("tp", "")
        out.append(
            f"| {gpu} | {r['model_id']} | {tp} | {r['isl']} | {r['osl']} "
            f"| {r['global_batch']} | {r['measured_sys_tps']:.0f} | {r['predicted_sys_tps']:.0f} "
            f"| {r['predicted_tpot_ms']:.2f} | {r['tps_ratio']:.2f}× |"
        )
    out.append("")

    failed = [r for r in rows if not r.get("ok")]
    if failed:
        out.append("## Failed records")
        out.append("")
        for r in failed:
            out.append(f"- `{r['record_id']}`: {r.get('error', '<no error>')[:200]}")
        out.append("")

    # Findings — derived from the actual rows. The contaminated-subset
    # caveat is computed inside the findings loop (where its row stats are
    # in scope) but rendered later under Notes & caveats.
    contam_caveat: str | None = None
    if ok_rows:
        out.append("## Findings")
        out.append("")
        out.append(
            "1. **TRT-LLM throughput error correlates with OSL.** Across all "
            "H100 FP8 throughput records (Llama-3.1-8B TP=1, Llama-3.3-70B "
            "TP=2, Llama-3.1-405B TP=8), ScaleScope over-predicts system TPS "
            "at very long OSL and under-predicts at very short OSL (where the "
            "real system amortises prefill overhead across many decode steps). "
            "See the Aggregate stats table for the per-bucket ratio band and "
            "Notes & caveats below for the underlying KV-cache mechanism."
        )
        out.append("")
        ix_disagg_ok = [
            r for r in ok_rows if r["source"] == "inferencex" and r.get("disagg") is True
        ]
        ix_nondis_ok = [
            r for r in ok_rows if r["source"] == "inferencex" and r.get("disagg") is False
        ]
        if ix_disagg_ok:
            tpot_ratios = sorted(
                r["tpot_ratio"] for r in ix_disagg_ok if r.get("tpot_ratio") is not None
            )
            median_tpot_ratio = tpot_ratios[len(tpot_ratios) // 2] if tpot_ratios else float("nan")
            # Phase 3A = clean-TP-only cases (TP=N spanning N chips, no
            # DP-attention, no EP, no outer DP). Everything else is
            # Phase 3B (DP-attention + EP hybrids, including the
            # TP=2+DP=4 outer-DP shape).
            phase3a_cases = {
                "gpt_oss_120b__gb200_2x_tp2",
                "gpt_oss_120b__gb200_4x_tp4",
                "gpt_oss_120b__gb200_8x_tp8",
            }
            ix_clean = [r for r in ix_disagg_ok if r["case"] in phase3a_cases]
            ix_hybrid = [r for r in ix_disagg_ok if r["case"] not in phase3a_cases]
            clean_tpot = sorted(
                r["tpot_ratio"] for r in ix_clean if r.get("tpot_ratio") is not None
            )
            hybrid_tpot = sorted(
                r["tpot_ratio"] for r in ix_hybrid if r.get("tpot_ratio") is not None
            )
            clean_median = clean_tpot[len(clean_tpot) // 2] if clean_tpot else float("nan")
            hybrid_median = hybrid_tpot[len(hybrid_tpot) // 2] if hybrid_tpot else float("nan")
            ix_ok = ix_disagg_ok
            out.append(
                "2. **InferenceX disagg decode (gpt-oss-120b on GB200 NVL72): "
                f"median TPOT ratio {median_tpot_ratio:.2f}× across "
                f"{len(ix_ok)} rows.** This is the cleanest comparison in the "
                f"panel — `global_batch = conc` is reported directly, the "
                f"prefill engine sits on a separate GPU pool (no "
                f"chunked-prefill interleaving), and no overlap-efficiency "
                f"factor is applied. Coverage: TP=2/4/8, ISL=1024/8192, "
                f"OSL=1024, conc=1-3072. By shape: clean-TP "
                f"({len(ix_clean)} rows) at {clean_median:.2f}× median, "
                f"DP-attention + EP hybrids ({len(ix_hybrid)} rows) at "
                f"{hybrid_median:.2f}× median — the hybrid shapes carry the "
                f"fitted serial EP dispatch term "
                f"(moe_token_pair_overhead_sec, see "
                f"MOE_TOKEN_PAIR_OVERHEAD_SEC and modeling_"
                f"improvements_todo.md §14); the residual misses there are "
                f"the DP-replicated cells with saturation-flat measured TPOT "
                f"(over-predicted) and the EP4 shapes (modestly under). "
                f"See Notes & caveats below."
            )
            out.append("")

        if ix_nondis_ok:
            # Split into clean (HBM-feasible & non-saturated) and
            # contaminated (OOM-infeasible or queue-saturated). The
            # headline finding uses the clean subset only — the
            # contaminated rows are a separate story (Finding 4) and
            # putting them in the headline median produces aggregates
            # that are dominated by measurement artifacts rather than
            # ScaleScope's modelling error.
            nd_clean = [r for r in ix_nondis_ok if not r.get("contaminated")]
            nd_contam = [r for r in ix_nondis_ok if r.get("contaminated")]
            nd_infeas = [r for r in nd_contam if r.get("infeasible_capacity")]
            nd_sat = [r for r in nd_contam if r.get("saturated")]

            nd_tpot = sorted(r["tpot_ratio"] for r in nd_clean if r.get("tpot_ratio") is not None)
            nd_corr = sorted(
                r["corrected_tpot_ratio"]
                for r in nd_clean
                if r.get("corrected_tpot_ratio") is not None
            )
            nd_alpha = sorted(r["alpha"] for r in nd_clean if r.get("alpha") is not None)
            nd_pfrac = sorted(
                r["prefill_frac"] for r in nd_clean if r.get("prefill_frac") is not None
            )
            nd_tpot_med = nd_tpot[len(nd_tpot) // 2] if nd_tpot else float("nan")
            nd_corr_med = nd_corr[len(nd_corr) // 2] if nd_corr else float("nan")
            nd_alpha_min = nd_alpha[0] if nd_alpha else float("nan")
            nd_pfrac_max = nd_pfrac[-1] if nd_pfrac else float("nan")
            out.append(
                "3. **InferenceX non-disagg (MiniMax-M2.5 on B200), clean "
                "subset: the α-correction is correctly a no-op here**. "
                f"Across the {len(nd_clean)} clean non-disagg rows "
                f"(TP=1/2/4/8, FP4 + FP8, ISL=1024/8192, OSL=1024/8192, "
                f"conc=4-512, HBM-feasible peak memory AND queueing_factor ≤ "
                f"{QUEUEING_FACTOR_SATURATED_THRESHOLD:g}×) the naive TPOT "
                f"median is {nd_tpot_med:.2f}× and the α-corrected median is "
                f"{nd_corr_med:.2f}× — a <2% shift, because the worst-case "
                f"prefill_frac is only {nd_pfrac_max*100:.1f}% "
                f"(`α_min = {nd_alpha_min:.2f}`). This is the α-correction's "
                f"validity test: chunked prefill is NOT the dominant gap "
                f"mechanism in this serving regime. The residual "
                f"~{abs(1.0-nd_tpot_med)*100:.0f}% sits mostly at high "
                f"concurrency (conc=256-512), where measured TPOT rises "
                f"faster than predicted — continuous-batching/queueing plus "
                f"linearly-growing KV-cache traffic the single-step model "
                f"omits — and partly from the TP all-reduce's fixed latency "
                f"floor at small per-chip decode batches (small messages are "
                f"latency-bound), which ScaleScope's size-based comm model "
                f"under-counts. See Notes & caveats below."
            )
            out.append("")

            if nd_contam:
                # Headline numbers for the contamination story.
                contam_tpot = sorted(
                    r["tpot_ratio"] for r in nd_contam if r.get("tpot_ratio") is not None
                )
                contam_med = contam_tpot[len(contam_tpot) // 2] if contam_tpot else float("nan")
                contam_max = contam_tpot[-1] if contam_tpot else float("nan")
                qf_vals = sorted(
                    r["queueing_factor"] for r in nd_contam if r.get("queueing_factor") is not None
                )
                qf_max = qf_vals[-1] if qf_vals else float("nan")
                hbm_vals = sorted(
                    r["peak_to_hbm_ratio"]
                    for r in nd_contam
                    if r.get("peak_to_hbm_ratio") is not None
                )
                hbm_max = hbm_vals[-1] if hbm_vals else float("nan")
                contam_caveat = (
                    "- **InferenceX non-disagg, CONTAMINATED subset: the "
                    "labelled `conc` is not what the framework is actually "
                    "decoding per step**. "
                    f"{len(nd_contam)} of the {len(ix_nondis_ok)} non-disagg "
                    f"MiniMax rows fail one of two physical-feasibility "
                    f"checks: **{len(nd_infeas)}** have ScaleScope-predicted "
                    f"`total_peak_memory_per_xpu_bytes` > chip HBM "
                    f"(worst case {hbm_max*100:.0f}% — the labelled batch "
                    f"cannot fit), and **{len(nd_sat)}** have "
                    f"`mean_e2el / (OSL × median_tpot)` > "
                    f"{QUEUEING_FACTOR_SATURATED_THRESHOLD:g}× "
                    f"(worst case {qf_max:.1f}× — each user spends most "
                    f"of its wall-clock waiting in the scheduler queue). "
                    f"For these rows, vllm is internally admitting a much "
                    f"smaller effective decode batch than `conc` and "
                    f"reporting `median_tpot` against THAT batch — so "
                    f"ScaleScope's prediction at the *labelled* `conc` "
                    f"(which models the steady-state cost the framework "
                    f"would pay IF it could actually batch all `conc` "
                    f"users) is structurally larger than the measurement, "
                    f"producing the misleading {contam_med:.2f}× median "
                    f"and {contam_max:.2f}× worst-case in this subset. "
                    f"Worked example: `B200_x1 minimax_m2_5 TP=1 ISL=8192 "
                    f"OSL=1024 conc=512 fp4`. ScaleScope predicts "
                    f"peak=353.9 GiB on a B200 (211% of HBM) and "
                    f"queueing_factor = 13.0× (`mean_e2el` 344.9 s vs "
                    f"naive `OSL × tpot` 26.5 s). The implied effective "
                    f"decode batch is far below the labelled 512 "
                    f"(conc/13 ≈ 39 users); ScaleScope's prediction only "
                    f"falls to the measured 25.84 ms near conc≈64 "
                    f"(23.70 ms), versus 73.09 ms at the labelled conc=512 "
                    f"(2.83× off). The flag isolates this measurement-"
                    f"interpretation issue from genuine modelling errors "
                    f"elsewhere in the panel.\n"
                )

    out.append("## Notes & caveats")
    out.append("")
    out.append(
        "- **Continuous batching is not modelled.** ScaleScope predicts "
        "single-step decode at a fixed batch. Non-disaggregated serving "
        "interleaves chunked prefill with decode steps; that overhead is "
        "quantified separately via the α-correction on the InferenceX "
        "non-disagg rows.\n"
    )
    out.append(
        "- **TRT-LLM batch is selected by sweep**: ScaleScope is run at "
        "every batch in `BATCH_SWEEP_GRID = [1,4,16,64,128,256,512,1024,"
        "2048,4096]` and the row picks the batch whose predicted "
        "throughput is closest (in log space) to the measured value. "
        "The `TPS ratio` column therefore measures residual error AT "
        "THE BEST AVAILABLE BATCH; if no batch in the grid produced a "
        "matching throughput, the best-fit ratio will deviate from 1.0×. "
        "Re-run trtllm-bench locally to capture the actual operating "
        "batch and per-user TPOT for a tighter comparison.\n"
    )
    out.append(
        "- **Linearly-growing KV-cache traffic is not modelled.** Decode KV "
        "is read at a fixed per-step size rather than growing with the "
        "generated length, so ScaleScope over-predicts throughput at long "
        "OSL (TRT-LLM) and under-predicts the per-step TPOT slope at high "
        "concurrency (non-disagg).\n"
    )
    out.append(
        "- **Sliding-window / hybrid attention is modelled via the "
        "`attention_layers` blend** (gpt-oss caps KV history at 128 tokens "
        "on half its layers; the model spec carries the sliding/full layer "
        "mix, so KV traffic at long ISL reflects it).\n"
    )
    out.append(
        "- **The kernel-launch floor is keyed on model only, not "
        "(model, GPU).** `KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` carries one "
        "value per model, applied across every GPU class it runs on. "
        "Kernel-launch/dispatch latency is physically a host + GPU-runtime "
        "cost, so it has a per-GPU component the single value cannot "
        "capture. On Llama-3.3-70B (the only model spanning two GPU "
        "classes in the panel), matched (TP, conc, ISL, OSL) cells show "
        "ScaleScope predicts ~6% higher TPOT on B200 than H100 at small "
        "batch (~9% pooled): the shared 42 µs floor is centered on H100 "
        "(small-batch geomean 1.00×) and slightly over-predicts B200 "
        "(1.05×). The bias is within the ±10% band and small next to the "
        "OSL-driven KV-cache error above, which is common to both GPUs.\n"
    )
    if contam_caveat is not None:
        out.append(contam_caveat)

    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
