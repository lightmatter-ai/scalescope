# ScaleScope Measured-Panel Validation

This package validates ScaleScope's decode-phase predictions against **measured**
public inference benchmarks. For every supported benchmark record it invokes
ScaleScope once at the operating-point batch implied by the measurement and
compares predicted vs. measured TPOT / throughput, emitting a markdown table and
a machine-readable summary.

## What it covers

Measurement corpora (in `benchmark_data/`):

- **TRT-LLM perf-overview** (`trtllm_perf_overview.yaml`) — FP8 throughput rows
  (Llama-3.1-8B, Llama-3.3-70B, Llama-3.1-405B on H100), saturating-batch regime.
- **InferenceX** (`inferencex_raw/*.json`) — PD-disaggregated decode pools
  (gpt-oss-120b on GB200 NVL72) and non-disaggregated continuous batching
  (MiniMax-M2.5 on B200, Llama-3.3-70B on H100/B200).

Models exercised (the panel selection, in `model_zoo/`): `llama3_70b`,
`llama3_1_8b`, `llama3_1_405b`, `minimax_m2_5`, `gpt_oss_120b`. A record only
runs if its `(model, hardware, parallelism)` shape has a matching entry in the
`TRTLLM_CASE_TABLE` / `INFERENCEX_CASE_TABLE` inside the driver; everything else
is reported as `deferred` and skipped.

## Prerequisites

Install ScaleScope into the active environment and run from the **repo root**
(the driver shells out to `python -m scalescope.run_scalescope` with
`cwd=<repo root>` and reads `hardware_specs/hardware_library.yaml`):

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

## Running the panel

All commands are run from the repo root.

```bash
# Full canonical panel -> validation_results/validation_panel_summary.{md,yaml}
python validation/run_validation_panel.py

# Dry run: list the records that would execute, no ScaleScope invocations
python validation/run_validation_panel.py --list-only

# Smoke-test the first N records of each loaded source
python validation/run_validation_panel.py --limit 3

# One corpus at a time
python validation/run_validation_panel.py --source trtllm
python validation/run_validation_panel.py --source inferencex

# Fine-grained subset (filters AND across flags, OR within a flag)
python validation/run_validation_panel.py --source inferencex --model minimax_m2_5
python validation/run_validation_panel.py --gpu-class B200 --gpu-class GB200
```

Any filter (`--source != all`, `--limit`, `--model`, `--gpu-class`,
`--record-contains`) suffixes the outputs with `__partial` so a partial run never
clobbers the canonical summary. FP4 records are excluded by default (ScaleScope
does not model generic FP4 dequant/block-scaling); `gpt_oss_120b` is the one
calibrated FP4 model kept in the canonical panel. Pass `--include-fp4` to run the
rest.

## Refreshing the raw data (`extraction/`)

The two measurement corpora are snapshots of external sources; `extraction/`
holds the scripts that regenerate them. Both require network access.

```bash
# InferenceX: hit the public REST API per model, archive JSON to
# benchmark_data/inferencex_raw/<Model>.json (the panel loader reads these verbatim).
python validation/extraction/inferencex_fetch.py

# TRT-LLM perf-overview: scrape NVIDIA's HTML throughput tables into the panel
# YAML schema. Writes to stdout by default; pass --out to persist.
python validation/extraction/trtllm_perf_overview_fetch.py \
    --out validation/benchmark_data/trtllm_perf_overview.yaml

# Parse a previously-saved page instead of fetching (offline / reproducible):
python validation/extraction/trtllm_perf_overview_fetch.py --html page.html
```

The TRT-LLM scraper emits only the models listed in `KNOWN_MODELS` (those with a
ScaleScope `model_zoo` spec) and fails fast if an expected model table is missing
or the page layout changed. The checked-in `trtllm_perf_overview.yaml` may carry
extra hand-written `note:` annotations the scraper does not reproduce; diff the
generated output against it rather than blindly overwriting.

## Calibration / auditing (`calibration/`)

`calibration/` holds read-only analyses of the empirical knobs the panel
injects. They consume `validation_results/validation_panel_summary.yaml` (run the
panel first) and never modify configs.

Floor first, then pair. Procedure, commands, and holdouts:
[`calibration/README.md`](calibration/README.md).

```bash
# Floor: suggested us/layer + per-GPU centering audit
python validation/calibration/calibrate_kernel_floor.py
python validation/calibration/calibrate_kernel_floor.py --model llama3_70b

# Pair: LS on a zero-pair dump (EP2 ISL=1024 fit; EP4/DP/ISL=8192 holdout)
python validation/fit_ep_dispatch_overhead.py \
    validation/validation_results/validation_panel_summary__p1_zeropair.yaml

# Current driver table vs recovered fits
python validation/calibration/compare_floor_pair.py \
    --panel validation/validation_results/validation_panel_summary__tokensat.yaml \
    --zeropair-panel validation/validation_results/validation_panel_summary__p1_zeropair.yaml
```

`KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` and `MOE_TOKEN_PAIR_OVERHEAD_SEC` are
hand-set in the driver. The scripts only print suggestions. Applying a
value is a manual edit followed by a panel rerun.

## Inputs and outputs

```
validation/
  run_validation_panel.py         # the driver
  validation_results/             # GENERATED (gitignored)
    validation_panel_summary.md   #   comparison table
    validation_panel_summary.yaml #   machine-readable summary
    runs/<record_id>.yaml         #   per-record ScaleScope output + workload spec
  benchmark_data/                 # checked-in measurement corpora
    trtllm_perf_overview.yaml     #   TRT-LLM perf-overview records
    inferencex_raw/*.json         #   InferenceX records
  extraction/                     # pull raw data from the external sources
    inferencex_fetch.py           #   InferenceX REST API -> benchmark_data/inferencex_raw/*.json
    trtllm_perf_overview_fetch.py #   NVIDIA perf-overview HTML -> trtllm_perf_overview.yaml
  calibration/                    # read-only fit + audit of empirical model knobs
    README.md                     #   floor-then-pair procedure
    calibrate_kernel_floor.py     #   fits (model, gpu) kernel floor (us/layer)
    compare_floor_pair.py         #   driver table vs recovered floor + pair
    fit_emem.py                   #   per-GPU HBM efficiency from an ideal-e panel
  fit_ep_dispatch_overhead.py     # pair LS on a zero-pair gpt-oss EP dump
  inference_decode_specs/         # fp8 / fp4 decode workload templates
  model_zoo/                      # transformer configs for the selected models
  topology_zoo/{physical,logical}/# per-(hardware × parallelism) topology configs
```

Each generated workload spec injects a host kernel-launch/dispatch
floor keyed on `(model, gpu_class)` (`KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER`
in the driver), an empirical runtime overhead the roofline does not
model. `calibration/calibrate_kernel_floor.py` fits those values from
the panel and audits their centering (see `calibration/README.md`).

Everything under `validation_results/` is regenerated on each run and is not checked in.
