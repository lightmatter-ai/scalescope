# ScaleScope Measured-Panel Validation

This package validates ScaleScope's decode-phase predictions against **measured**
public inference benchmarks. For every supported benchmark record it invokes
ScaleScope once at the operating-point batch implied by the measurement and
compares predicted vs. measured TPOT / throughput, emitting a markdown table and
a machine-readable summary.

## What it covers

Measurement corpora (in `measurements/`):

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
# Full canonical panel -> results/measured_panel_summary.{md,yaml}
python validation/measurements/run_measured_panel.py

# Dry run: list the records that would execute, no ScaleScope invocations
python validation/measurements/run_measured_panel.py --list-only

# Smoke-test the first N records of each loaded source
python validation/measurements/run_measured_panel.py --limit 3

# One corpus at a time
python validation/measurements/run_measured_panel.py --source trtllm
python validation/measurements/run_measured_panel.py --source inferencex

# Fine-grained subset (filters AND across flags, OR within a flag)
python validation/measurements/run_measured_panel.py --source inferencex --model minimax_m2_5
python validation/measurements/run_measured_panel.py --gpu-class B200 --gpu-class GB200
```

Any filter (`--source != all`, `--limit`, `--model`, `--gpu-class`,
`--record-contains`) suffixes the outputs with `__partial` so a partial run never
clobbers the canonical summary. FP4 records are excluded by default (ScaleScope
does not model generic FP4 dequant/block-scaling); `gpt_oss_120b` is the one
calibrated FP4 model kept in the canonical panel. Pass `--include-fp4` to run the
rest.

## Inputs and outputs

```
validation/
  measurements/
    run_measured_panel.py        # the driver
    trtllm_perf_overview.yaml     # TRT-LLM perf-overview records
    inferencex_raw/*.json         # InferenceX records
    results/                      # GENERATED (gitignored)
      measured_panel_summary.md   #   comparison table
      measured_panel_summary.yaml #   machine-readable summary
      runs/<record_id>.yaml       #   per-record ScaleScope output + workload spec
  inference_decode_specs/         # fp8 / fp4 decode workload templates
  model_zoo/                      # transformer configs for the selected models
  topology_zoo/{physical,logical}/# per-(hardware × parallelism) topology configs
```

Each generated workload spec injects a per-model host kernel-launch/dispatch
floor (`KERNEL_LAUNCH_FLOOR_SEC_PER_LAYER` in the driver), an empirical
runtime overhead the roofline does not model, calibrated to the clean
small-batch residual of each model.

Everything under `results/` is regenerated on each run and is not checked in.
