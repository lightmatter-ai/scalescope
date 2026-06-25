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

```bash
# Fit + audit the per-model kernel-launch floor. Prints, for every model in
# the panel: the suggested floor (us/layer), small-batch centering per GPU
# class, and a matched-cell head-to-head between GPU classes for multi-GPU
# models.
python validation/calibration/calibrate_kernel_floor.py
python validation/calibration/calibrate_kernel_floor.py --model llama3_70b
```

`KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` is hand-set in the driver and keyed on
model only. This script computes what each value *should* be — the floor is
additive to decode TPOT at the fixed `batch = conc` operating point, so the
fit is recovered exactly from the panel as
`floor_current + (measured_tpot − predicted_tpot) / n_layers`, taken as the
median over clean small-batch rows. It also audits how well the current value
is centered (e.g. on Llama-3.3-70B the shared floor sits on H100 at
small-batch geomean ~1.00× but slightly over-predicts B200 ~1.05×) and flags
models where promoting the key to `(model, gpu_class)` would help. Applying a
fitted value is a manual edit to the driver dict followed by re-running the
panel; the script never modifies configs.

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
    calibrate_kernel_floor.py     #   fits per-model kernel floor (us/layer) + per-GPU centering audit
  inference_decode_specs/         # fp8 / fp4 decode workload templates
  model_zoo/                      # transformer configs for the selected models
  topology_zoo/{physical,logical}/# per-(hardware × parallelism) topology configs
```

Each generated workload spec injects a per-model host kernel-launch/dispatch
floor (`KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` in the driver), an empirical
runtime overhead the roofline does not model, calibrated to the clean
small-batch residual of each model. `calibration/calibrate_kernel_floor.py`
fits those values from the panel and audits their centering (see above).

Everything under `validation_results/` is regenerated on each run and is not checked in.
