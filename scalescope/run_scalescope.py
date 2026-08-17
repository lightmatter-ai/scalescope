"""CLI orchestrator and end-to-end analysis driver.

Parses arguments, loads hardware/model/workload/topology specs, constructs
logical topology views, invokes compute/comm/memory models, applies overlap and
pipeline bubble models, verifies memory footprints, builds a results DataFrame, and
produces structured YAML summaries (optionally human-formatted) and plots.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import scalescope.plotting as plotting
from scalescope.comm_model import get_transformer_comm_stats
from scalescope.compute_model import (
    backward_pass_flops_per_chip,
    calculate_embedding_compute,
    calculate_unembedding_compute,
    decode_attention_compute_breakdown_per_layer,
    forward_decode_flops_per_chip,
    forward_prefill_flops_per_chip,
    parameter_update_flops,
    prefill_attention_compute_breakdown_per_layer,
)
from scalescope.decode_layout_trace import (
    build_decode_2dtp_mesh,
    build_decode_2dtp_trace,
    LayoutTrace,
    NULL_LAYOUT_TRACE,
)
from scalescope.ep_strategy import (
    compute_expert_activation_cardinality,
    infer_expert_placement_per_xpu,
    validate_expert_imbalance_realizability,
)
from scalescope.hardware_spec import HardwareRegistry
from scalescope.memory_model import (
    _backward_attn_tp_1d_memory_traffic,
    _backward_attn_tp_2d_memory_traffic,
    _backward_moe_tp_1d_memory_traffic,
    _backward_moe_tp_2d_memory_traffic,
    _decode_attention_attn_tp_1d_memory_traffic,
    _decode_attention_attn_tp_2d_memory_traffic,
    _decode_attention_projection_tp_1d_memory_traffic,
    _decode_attention_projection_tp_2d_memory_traffic,
    _forward_pass_prefill_attention_attn_tp_1d_memory_traffic,
    _forward_pass_prefill_attention_attn_tp_2d_memory_traffic,
    _forward_pass_prefill_attention_projection_tp_1d_memory_traffic,
    _forward_pass_prefill_attention_projection_tp_2d_memory_traffic,
    _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic,
    _forward_pass_prefill_moe_ffn_tp_2d_memory_traffic,
    calculate_attn_backward_transient_peak_memory_footprint,
    calculate_attn_forward_transient_peak_memory_footprint_tp1d,
    calculate_attn_forward_transient_peak_memory_footprint_tp2d,
    calculate_attn_optimizer_states_memory_footprint,
    calculate_attn_saved_activations_memory_footprint,
    calculate_attn_weights_memory_footprint,
    calculate_embedding_memory_traffic_1d_tp_bwd,
    calculate_embedding_memory_traffic_1d_tp_fwd,
    calculate_ffn_moe_backward_transient_peak_memory_footprint,
    calculate_ffn_moe_forward_transient_peak_memory_footprint_tp1d,
    calculate_ffn_moe_forward_transient_peak_memory_footprint_tp2d,
    calculate_ffn_moe_optimizer_states_memory_footprint,
    calculate_ffn_moe_saved_activations_memory_footprint,
    calculate_ffn_moe_weights_memory_footprint,
    calculate_unembedding_memory_traffic_1d_tp_bwd,
    calculate_unembedding_memory_traffic_1d_tp_fwd,
    get_transformer_mem_stats_per_layer,
    LayerMemoryTraffic,
    MemoryFootprint,
)
from scalescope.model_spec import (
    attention_layer_classes,
    resolve_kv_heads_per_partition,
    TransformerConfig,
)
from scalescope.pp_strategy import get_pp_strategy
from scalescope.runtime_overhead_model import moe_token_pair_overhead_per_pass
from scalescope.stats import CommStats
from scalescope.sweep_dsl import (
    apply_case_to_specs,
    generate_cases,
    LeafSpec,
    NodeSpec,
    SweepConfig,
)
from scalescope.topology_spec import LogicalTopology, PhysicalTopology
from scalescope.tp_strategy import (
    resolve_tp2d_split,
    validate_1d_tp_invariants_attn,
    validate_2d_tp_invariants_attn,
    validate_2d_tp_invariants_moe,
    validate_2d_tp_invariants_proj,
)
from scalescope.utils import (
    calculate_overlapped_time,
    format_bytes,
    format_flops,
    format_time,
    PassMode,
)
from scalescope.workload_spec import (
    DecodeCrossDomainEPMode,
    InferenceDecodeSpec,
    InferencePrefillSpec,
    MoERouterSyncMode,
    TensorParallelVariant,
    TP2DProjAxisRole,
    TrainSpec,
    WorkloadSpec,
)


def pass_mode_from_string(value: str) -> PassMode:
    """
    Converts a string to a PassMode enum member by looking up its value.
    This is used as a custom type for argparse.
    """
    try:
        # Looks up the member by its string value.
        return PassMode(value)
    except ValueError:
        # If the value is not found, a ValueError is raised.
        # We re-raise it as an ArgumentTypeError for a clean argparse error message.
        valid_choices = [item.value for item in PassMode]
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid pass mode. Please choose from: {valid_choices}"
        )


def parse_arguments():
    """Parse command-line arguments for the scaling analysis."""

    parser = argparse.ArgumentParser(
        description="Run model performance analysis based on specified configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware configuration
    parser.add_argument(
        "--hardware-lib",
        type=str,
        required=True,  # No default, must be provided
        help="Path to the hardware library YAML file (defining components, nodes, topologies).",
    )

    # Model and Training Specs
    parser.add_argument(
        "--model-spec",
        type=str,
        required=True,  # No default, must be provided
        help="Path to the transformer model configuration YAML file.",
    )
    parser.add_argument(
        "--workload-spec",
        type=str,
        required=True,  # No default, must be provided
        help="Path to the workload specification YAML file (batch size, parallelism degrees, etc.).",
    )

    # Topology configuration
    parser.add_argument(
        "--physical-topology",  # Argument for the physical topology file path
        type=str,
        required=True,  # No default, must be provided
        metavar="PHYSICAL_TOPOLOGY_YAML_PATH",
        help="Path to the YAML file defining physical datacenter topology."
        " Example: --physical-topology physical_topology.yaml",
    )
    parser.add_argument(
        "--logical-topology",  # Argument for the logical topology file path
        type=str,
        required=True,  # No default, must be provided
        metavar="LOGICAL_TOPOLOGY_YAML_PATH",
        help="Path to the YAML file defining the logical network topology."
        " Example: --logical-topology logical_topology.yaml",
    )

    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help=(
            "Optional path to the sweep definition YAML. "
            "When omitted, runs a single default point (no sweep overrides)."
        ),
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,  # Default value is None if not provided
        metavar="OUTPUT_PATH",
        help="Optional path to save the results/output to a file."
        " If not specified, output might go to stdout or a default location.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Optional directory to write sweep artifacts: manifest.yaml and per-run outputs in runs/.",
    )
    parser.add_argument(
        "--save-run-artifacts",
        action="store_true",
        help=(
            "When --output-dir is set, also write one YAML per sweep point under runs/. "
            "Disabled by default to reduce I/O for large sweeps."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="scalescope_run",
        help="Name used for the output subdirectory when --output-dir is set.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="When --output-dir is set, do not append a timestamp to the run directory name (useful for tests/CI).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When --output-dir is set and the run directory already exists, overwrite files in-place.",
    )

    parser.add_argument(
        "--pass-mode",
        default=PassMode.TRAINING,  # Default to training mode
        type=pass_mode_from_string,
        choices=list(PassMode),
        help=(
            "The pass mode to execute.\n"
            "Choices are:\n"
            "  training:          Run a full training forward pass.\n"
            "  inference_prefill: Run the inference prefill/prompt phase.\n"
            "  inference_decode:  Run the inference decode/generation phase."
        ),
    )

    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Disable human-readable formatting in the YAML file and show raw numerical values instead.",
    )
    parser.add_argument(
        "--quiet-layout-trace",
        action="store_true",
        help=(
            "Suppress decode layout trace build/print (expected vs actual tables). "
            "Recommended for large sweeps."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress verbose console output: layout traces and per-row statistics. "
            "Recommended for large sweeps and automated pipelines."
        ),
    )

    args = parser.parse_args()

    # Perform basic validation on paths
    required_paths_to_check = {
        "Hardware library": args.hardware_lib,
        "Physical Topology": args.physical_topology,
        "Logical Topology": args.logical_topology,
        "Model spec": args.model_spec,
        "Workload spec": args.workload_spec,
    }
    for name, path in required_paths_to_check.items():
        if not os.path.exists(path):
            parser.error(f"{name} file not found: {path}")
    if args.sweep_config is not None and not os.path.exists(args.sweep_config):
        parser.error(f"Sweep config file not found: {args.sweep_config}")

    return args


def _git_head_sha(repo_root: Path) -> str | None:
    """Best-effort git SHA capture for provenance."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def _stable_run_id(*, sweep_case: dict, labels: dict) -> str:
    """Stable ID for a sweep point derived from (case, labels)."""
    payload = {"case": sweep_case, "labels": labels}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(blob).hexdigest()[:16]


def _default_single_point_sweep_config() -> SweepConfig:
    """Return a single identity sweep point when --sweep-config is omitted."""
    return SweepConfig(
        leaves={
            "__base": LeafSpec(name="__base", paths=[], cases=[[]], generator=None, labels=[{}])
        },
        sweep=NodeSpec(kind="ref", ref="__base"),
        report=None,
    )


def _extract_sweep_case_from_row(row: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {}

    def _set_nested(d: dict, dotted_key: str, value: object) -> None:
        parts = dotted_key.split(".")
        cur: dict = d
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value

    for k, v in row.items():
        if isinstance(k, str) and k.startswith("sweep."):
            dotted_key = k[len("sweep.") :]  # noqa E203
            _set_nested(out, dotted_key, v)
    return out


def _resolve_run_output_dir(args) -> Path:
    base = Path(args.output_dir).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir_name = args.run_name if args.no_timestamp else f"{args.run_name}__{stamp}"
    run_dir = base / run_dir_name
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output dir already exists: {run_dir}. Use --overwrite or --no-timestamp/--run-name to change it."
        )
    if run_dir.exists() and args.overwrite:
        # Avoid stale artifacts from previous runs (e.g., old run YAMLs/summary rows)
        # mixing with the new run when reusing the same output directory.
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.save_run_artifacts:
        (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_by_path(obj: object, path: str):
    """Extract a nested value from dict-like objects using dotted paths (e.g. 'time.total_time_seconds')."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def _where_predicate_ok(row: dict, pred: dict) -> bool:
    lhs = row.get(pred["col"])
    rhs = pred.get("value")
    op = pred["op"]
    try:
        if op == "==":
            return lhs == rhs
        if op == "!=":
            return lhs != rhs
        if op == "<":
            return lhs < rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        if op == ">=":
            return lhs >= rhs
    except Exception:
        return False
    return False


def _summary_row_from_record(record: dict, report_spec) -> dict:
    """Extract the flat report columns for one structured record.

    Lets callers build the small report row per record inside their iteration and
    discard the (heavyweight) full structured record immediately, instead of
    accumulating every record just to extract a handful of columns at the end.
    """
    return {col.name: _extract_by_path(record, col.path) for col in report_spec.columns}


def _write_summary_csv(*, run_dir: Path, manifest: dict, rows: list[dict], report_spec) -> None:
    """Write summary.csv from pre-extracted flat report rows (see
    `_summary_row_from_record`). Each entry of ``rows`` already maps report
    column name -> value; this only computes BEST markers and writes the CSV."""
    if report_spec is None:
        return

    # Compute BEST markers (optional).
    best_col = "Best"
    for r in rows:
        r[best_col] = ""
    if report_spec.best:
        for rule in report_spec.best:
            eligible_idx = list(range(len(rows)))
            if rule.where:
                eligible_idx = [
                    i
                    for i in eligible_idx
                    if all(
                        _where_predicate_ok(rows[i], {"col": p.col, "op": p.op, "value": p.value})
                        for p in rule.where
                    )
                ]

            # Group and select winner by metric.
            groups: dict[tuple, list[int]] = {}
            for i in eligible_idx:
                key = tuple(rows[i].get(k) for k in rule.group_by)
                groups.setdefault(key, []).append(i)

            for _, idxs in groups.items():

                def metric_val(ii: int):
                    return rows[ii].get(rule.metric)

                # Prefer numeric comparison; if missing, ignore those rows.
                valid = [ii for ii in idxs if metric_val(ii) is not None]
                if not valid:
                    continue
                if rule.mode == "min":
                    winner = min(valid, key=lambda ii: metric_val(ii))
                else:
                    winner = max(valid, key=lambda ii: metric_val(ii))
                cur = rows[winner].get(best_col) or ""
                rows[winner][best_col] = (cur + " " + rule.name).strip()

    # Write CSV.
    import csv

    out_path = run_dir / "summary.csv"
    fieldnames = [c.name for c in report_spec.columns] + [best_col]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Also record in manifest that summary exists.
    manifest.setdefault("artifacts", {})
    manifest["artifacts"]["summary_csv"] = "summary.csv"


def main():
    try:
        args = parse_arguments()
        quiet = bool(args.quiet) or bool(args.quiet_layout_trace)
        enable_decode_layout_trace = not quiet

        print("--- Parsed Arguments ---")
        print(f"Hardware Library: {args.hardware_lib}")
        print(f"Model Spec:       {args.model_spec}")
        print(f"Workload Spec:    {args.workload_spec}")
        print(f"Physical Topology File: {args.physical_topology}")
        print(f"Logical Topology File: {args.logical_topology}")
        print(f"Sweep Configuration: {args.sweep_config or '(none; single default point)'}")

    except SystemExit:
        # argparse raises SystemExit on error or --help, allow it to exit cleanly
        print("Error parsing command line arguments - exiting")
        raise
    except Exception as e:
        print(f"Error parsing command line arguments - exiting: {e}")
        raise

    base_specs = {}
    try:
        model_spec = TransformerConfig.load_from_yaml(args.model_spec)

        # Load the appropriate spec based on the pass mode
        if args.pass_mode == PassMode.TRAINING:
            workload_spec = TrainSpec.load_from_yaml(args.workload_spec)
        elif args.pass_mode == PassMode.INFERENCE_PREFILL:
            workload_spec = InferencePrefillSpec.load_from_yaml(args.workload_spec)
        elif args.pass_mode == PassMode.INFERENCE_DECODE:

            workload_spec = InferenceDecodeSpec.load_from_yaml(args.workload_spec)
        else:
            raise ValueError(f"Unsupported pass mode for spec loading: {args.pass_mode}")

        # Now, retrieve the actual node objects using the registry when needed
        # Create and load the hardware registry
        hw_registry = HardwareRegistry()  # Instantiate the registry
        hardware_lib_yaml = args.hardware_lib
        try:
            if not os.path.exists(hardware_lib_yaml):
                raise FileNotFoundError(f"Hardware library not found: {hardware_lib_yaml}")
            hw_registry.load_from_yaml(hardware_lib_yaml)  # Load the hardware library
            print(f"Successfully loaded hardware library from '{hardware_lib_yaml}")
        except Exception as e:
            print(f"FATAL: Could not load hardware library. Error: {e}")
            exit(1)

        # Load the physical topology, resolving the node reference via the registry
        phys_topo = PhysicalTopology.load_from_yaml(args.physical_topology, hw_registry)
        print(
            f"Loaded Physical Topology: {phys_topo.name} using Node: {phys_topo.node_definition.node_name}"
        )
        print(f"  Shape: {phys_topo.shape}, Total Nodes: {phys_topo.total_nodes}")

        # Load the logical topology, providing the physical topology it maps onto
        log_topo = LogicalTopology.load_from_yaml(args.logical_topology, phys_topo)
        print(
            f"Loaded Logical Topology: {log_topo.name} on Physical: {log_topo.physical_topology.name}"
        )
        print(f"  Logical Shape: {log_topo.shape}")

        # dp_topology = NetworkTopology.load_from_yaml(args.dp_topology_spec)
        # tp_topology = NetworkTopology.load_from_yaml(args.tp_topology_spec)

        base_specs["model_spec"] = model_spec
        base_specs["workload_spec"] = workload_spec
        base_specs["physical_topology"] = phys_topo
        base_specs["logical_topology"] = log_topo

        if args.sweep_config:
            sweep_cfg = SweepConfig.load_from_yaml(args.sweep_config)
            print("\nLoaded sweep DSL config.")
        else:
            sweep_cfg = _default_single_point_sweep_config()
            print("\nNo sweep config provided; using single default point.")

    except Exception as e:
        print(f"Error loading configuration files: {e}")
        raise

    # Do not mutate or replace model_spec here; compute per-stage layers later per sweep iteration
    # Build results dataframe (separate from plotting)
    df = build_results_df(
        base_specs,
        sweep_cfg,
        args.pass_mode,
        enable_decode_layout_trace=enable_decode_layout_trace,
    )

    # Save sweep results if requested
    if args.output_file:
        # Save a detailed YAML file
        if df.shape[0] > 0:
            # Convert each row to a structured dictionary
            structured_results = [
                structure_run_results(row, not args.raw_output) for _, row in df.iterrows()
            ]
            with open(args.output_file, "w") as f:
                # Add a document separator for clarity
                f.write("# Detailed run statistics for each point in the sweep\n")
                yaml.dump(
                    structured_results, f, indent=2, sort_keys=False, default_flow_style=False
                )
            print(f"Saved detailed results to {args.output_file}")

    # Save sweep artifacts directory if requested
    if args.output_dir:
        repo_root = Path(__file__).resolve().parents[1]
        run_dir = _resolve_run_output_dir(args)
        git_sha = _git_head_sha(repo_root)

        runs_index = []
        # Only the small flat report rows are retained across the whole sweep.
        # Full structured records are heavyweight nested dicts; for large sweeps
        # accumulating all of them (on top of the results DataFrame) exhausts RAM
        # and thrashes swap, so we extract the report row per iteration and let
        # each full record be garbage-collected immediately. Full records are
        # only retained when we must write per-run YAML artifacts.
        summary_rows: list[dict] = []
        report_spec = sweep_cfg.report
        num_runs = 0
        total_records = int(df.shape[0])
        if args.save_run_artifacts:
            print(f"Writing run artifacts: {total_records} files")
        else:
            print(
                "Run artifacts disabled (use --save-run-artifacts to enable per-point YAML files)."
            )
        # Building structured records (`structure_run_results` per row) is a
        # pure-Python per-row pass that is slow for large sweeps (hundreds of
        # thousands of rows). Always show a progress bar for this phase,
        # regardless of whether per-point YAML artifacts are written.
        artifact_iter = tqdm(
            df.iterrows(),
            total=total_records,
            desc="Building records" if not args.save_run_artifacts else "Artifact progress",
            unit="run",
            dynamic_ncols=True,
            # leave=True so close() paints the final, accurate frame
            # (total/total) instead of clearing the last throttled refresh,
            # which otherwise leaves a misleading sub-total count on screen.
            leave=True,
        )

        for step, (_, row) in enumerate(artifact_iter, start=1):
            record = structure_run_results(row, not args.raw_output)
            sweep_case = record.get("sweep_case") or {}
            labels = record.get("labels") or {}
            run_id = _stable_run_id(sweep_case=sweep_case, labels=labels)
            record["run_id"] = run_id
            num_runs += 1
            if report_spec is not None:
                summary_rows.append(_summary_row_from_record(record, report_spec))
            if args.save_run_artifacts:
                run_rel = Path("runs") / f"{run_id}.yaml"
                run_path = run_dir / run_rel
                with open(run_path, "w") as f:
                    yaml.dump(record, f, indent=2, sort_keys=False, default_flow_style=False)
                runs_index.append(
                    {
                        "run_id": run_id,
                        "path": str(run_rel),
                        "labels": labels,
                        "sweep_case": sweep_case,
                    }
                )

        artifact_iter.refresh()
        artifact_iter.close()

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"head_sha": git_sha},
            "inputs": {
                "hardware_lib": args.hardware_lib,
                "physical_topology": args.physical_topology,
                "logical_topology": args.logical_topology,
                "workload_spec": args.workload_spec,
                "model_spec": args.model_spec,
                "sweep_config": args.sweep_config,
                "pass_mode": (
                    args.pass_mode.value
                    if hasattr(args.pass_mode, "value")
                    else str(args.pass_mode)
                ),
                "raw_output": bool(args.raw_output),
            },
            "run": {
                "run_name": args.run_name,
                "timestamped": (not args.no_timestamp),
            },
            "num_runs": num_runs,
            "num_run_artifacts": len(runs_index),
            "save_run_artifacts": bool(args.save_run_artifacts),
            "runs": runs_index if args.save_run_artifacts else [],
        }
        # Optional summary/reporting (driven by sweep config's `report:` section).
        _write_summary_csv(
            run_dir=run_dir, manifest=manifest, rows=summary_rows, report_spec=report_spec
        )
        with open(run_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f, indent=2, sort_keys=False, default_flow_style=False)
        print(f"Saved sweep artifacts to {run_dir}")

    # Display per-row summary to console (suppressed by --quiet)
    if not quiet:
        try:
            print("\n--- Row-wise Statistics ---")
            for i, row in df.iterrows():
                try:
                    tp = row.get(plotting.tp_xpu_count_label, None)
                    dp = row.get(plotting.dp_ranks_label, None)
                    secs = row[plotting.time_to_train_sec_label]
                    days = row[plotting.time_to_train_day_label]
                    tps = row.get(plotting.tokens_per_second_label, None)
                    print(
                        f"row {i+1}: TP_XPU_Count={tp}, DP_Ranks={dp}, Time_Sec={secs}, Time_Day={days}, Tokens_Per_Sec={tps}"
                    )
                    print(f"workload_time_days: {days}")
                    print(f"workload_time_seconds: {secs}")
                except Exception as e:
                    print(f"row {i+1}: (unable to print all stats: {e})")
        except Exception as e:
            print(f"Warning: Failed to compute/print summary stats: {e}")

    # Optional plotting
    try:
        produce_plots(df)
    except Exception as e:
        if os.environ.get("SCALESCOPE_PLOT"):
            print(f"Warning: plotting failed: {e}")


_decode_trace_emitted: set[str] = set()


def _expert_activation_diagnostics(
    *,
    workload_spec: WorkloadSpec,
    model_spec: TransformerConfig,
) -> dict:
    """Build a dict of expert activation cardinality diagnostics for run output.

    Always returns the configured policy and ``rho``. When the workload has
    routed experts (``num_regular_experts > 0`` and ``num_selected_experts > 0``)
    also returns ``(E_support, E_active, active_expert_fraction)`` computed at
    ``tokens_per_set = batch_size_per_full_expert_set * seq_length`` so the
    diagnostics describe the same per-step routing scope the memory model uses.

    Diagnostics are computed regardless of which residency policy is in effect
    — under ``RESIDENT_FULL_SET`` they describe what active-subset accounting
    *would* produce for the same workload, which is useful for sweeping policy
    decisions.
    """
    moe_v2 = workload_spec.moe_spec_v2
    diagnostics = {
        plotting.expert_imbalance_rho_label: moe_v2.expert_imbalance,
        plotting.expert_weight_residency_policy_label: moe_v2.expert_weight_residency_policy.value,
    }
    if model_spec.num_regular_experts <= 0 or model_spec.num_selected_experts <= 0:
        return diagnostics
    tokens_per_set = (
        workload_spec.get_resolved_batches().batch_size_per_full_expert_set
        * workload_spec.seq_length
    )
    cardinality = compute_expert_activation_cardinality(
        tokens_per_set=tokens_per_set,
        num_selected_experts=model_spec.num_selected_experts,
        num_regular_experts=model_spec.num_regular_experts,
        num_colocated_experts=moe_v2.num_colocated_experts,
        expert_imbalance=moe_v2.expert_imbalance,
        routing_distribution=moe_v2.expert_routing_distribution,
    )
    diagnostics[plotting.expert_support_size_label] = cardinality.e_support
    diagnostics[plotting.expert_active_count_label] = cardinality.e_active
    diagnostics[plotting.active_expert_fraction_label] = cardinality.active_expert_fraction
    return diagnostics


def _resolve_decode_trace_params(
    *,
    model_spec: TransformerConfig,
    moe_config,
    proj_tp_topology_view,
    attn_tp_topology_view,
    moe_tp_topology_view,
    proj_x_dims,
    proj_yz_dims,
    attn_yz_dims,
    moe_x_dims,
    moe_yz_dims,
    g: int,
    router_sync_mode,
):
    """Extract resolved TP params needed for trace construction."""

    def _prod(shape, indices):
        if indices is None:
            return 1
        p = 1
        for idx in indices:
            p *= shape[idx]
        return p

    shape_proj = proj_tp_topology_view.shape
    shape_moe = moe_tp_topology_view.shape
    tp_x = _prod(shape_proj, proj_x_dims)
    tp_yz_proj = _prod(shape_proj, proj_yz_dims)
    tp_y = tp_yz_proj * g
    if moe_x_dims is None and moe_yz_dims is None:
        # 1D MoE TP: treat full MoE mesh size as effective YZ and X=1 for trace consistency.
        moe_tp_x = 1
        moe_tp_yz = _prod(shape_moe, range(len(shape_moe)))
    else:
        moe_tp_x = _prod(shape_moe, moe_x_dims)
        moe_tp_yz = _prod(shape_moe, moe_yz_dims)

    n_experts = model_spec.num_regular_experts
    top_k = model_spec.num_selected_experts
    c = moe_config.num_colocated_experts

    if hasattr(router_sync_mode, "value"):
        s = str(router_sync_mode.value).lower()
    elif isinstance(router_sync_mode, str):
        s = router_sync_mode.lower()
    else:
        raise ValueError(
            "router_sync_mode must have .value or be str, got " f"{type(router_sync_mode).__name__}"
        )
    if "sharded" in s:
        router_mode = "sharded"
    elif "replicated" in s:
        router_mode = "replicated"
    else:
        raise ValueError(
            f"router_sync_mode value must contain 'sharded' or 'replicated', got {s!r}"
        )

    return dict(
        tp_x=tp_x,
        tp_y=tp_y,
        moe_tp_x=moe_tp_x,
        moe_tp_yz=moe_tp_yz,
        g=g,
        c=c,
        n_experts=n_experts,
        top_k=top_k,
        router_mode=router_mode,
        attn_domains_per_full_expert_set=int(moe_config.attn_domains_per_full_expert_set),
        full_expert_sets_per_attn_domain=int(moe_config.full_expert_sets_per_attn_domain),
    )


def _emit_decode_layout_trace(
    *,
    model_spec: TransformerConfig,
    workload_spec,
    moe_config,
    attn_tp_topology_view,
    proj_tp_topology_view,
    moe_tp_topology_view,
    attn_x_dims,
    attn_yz_dims,
    proj_x_dims,
    proj_yz_dims,
    moe_x_dims,
    moe_yz_dims,
    g: int,
    router_sync_mode,
    actual_trace: "LayoutTrace | None" = None,
):
    """Build the expected decode pipeline layout trace and compare against actual.

    Prints the expected trace table and any diffs with the actual trace once per
    unique (tp_x, tp_y, g, c, router_mode) combination.
    """
    import logging

    log = logging.getLogger("scalescope.decode_layout_trace")

    params = _resolve_decode_trace_params(
        model_spec=model_spec,
        moe_config=moe_config,
        proj_tp_topology_view=proj_tp_topology_view,
        attn_tp_topology_view=attn_tp_topology_view,
        moe_tp_topology_view=moe_tp_topology_view,
        proj_x_dims=proj_x_dims,
        proj_yz_dims=proj_yz_dims,
        attn_yz_dims=attn_yz_dims,
        moe_x_dims=moe_x_dims,
        moe_yz_dims=moe_yz_dims,
        g=g,
        router_sync_mode=router_sync_mode,
    )

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    context_len = workload_spec.prefill_seq_len

    trace_key = (
        params["tp_x"],
        params["tp_y"],
        params["moe_tp_x"],
        params["moe_tp_yz"],
        params["g"],
        params["c"],
        params["router_mode"],
        params["attn_domains_per_full_expert_set"],
        params["full_expert_sets_per_attn_domain"],
        model_spec.d_model,
        model_spec.mlp_hidden_size,
        n_batch,
        context_len,
    )
    if trace_key in _decode_trace_emitted:
        return
    _decode_trace_emitted.add(trace_key)

    try:
        expected = build_decode_2dtp_trace(
            d_model=model_spec.d_model,
            d_ff=model_spec.mlp_hidden_size,
            n_heads=model_spec.num_heads,
            n_kv_heads=model_spec.num_kv_heads,
            head_dim=model_spec.head_dim,
            n_experts=params["n_experts"],
            top_k=params["top_k"],
            tp_x=params["tp_x"],
            tp_y=params["tp_y"],
            g=params["g"],
            c=params["c"],
            attn_domains_per_full_expert_set=(params["attn_domains_per_full_expert_set"]),
            full_expert_sets_per_attn_domain=(params["full_expert_sets_per_attn_domain"]),
            router_mode=params["router_mode"],
            batch_tokens=n_batch,
            dtype_bytes=workload_spec.float_size_bytes,
            context_len=context_len,
            attention_layers=attention_layer_classes(model_spec),
            moe_tp_x=params["moe_tp_x"],
            moe_tp_yz=params["moe_tp_yz"],
        )
        print(
            f"\n--- Decode 2D-TP Layout Trace [expected] (router_mode={params['router_mode']}, batch={n_batch}, ctx={context_len}) ---"
        )
        print(expected.format_table())
        print()

        if actual_trace is not None and actual_trace.records:
            actual_trace.assert_layer_boundary()
            print(
                f"--- Decode 2D-TP Layout Trace [actual from code] (batch={n_batch}, ctx={context_len}) ---"
            )
            print(actual_trace.format_table())
            print()

            diffs = actual_trace.compare_with(expected)
            if diffs:
                print("=== Expected vs Actual DIFFERENCES ===")
                for d in diffs:
                    print(f"  DIFF: {d}")
            else:
                print("=== Expected vs Actual: MATCH (all partitions consistent) ===")
            print()

    except Exception as exc:
        log.warning("Decode layout trace failed: %s", exc, exc_info=True)


def get_model_stats(
    cluster_physical_topology,
    cluster_logical_topology,
    model_spec: TransformerConfig,
    workload_spec: WorkloadSpec,
    pass_mode,
    *,
    enable_decode_layout_trace: bool = True,
):

    d_model = model_spec.d_model
    # Allow different TP strategies for attention vs MoE (explicit V2 specs).
    attn_tp_spec = workload_spec.get_attn_tp_spec()
    proj_tp_spec = workload_spec.get_proj_tp_spec()
    moe_tp_spec = workload_spec.get_moe_tp_spec()
    attn_tp_variant = attn_tp_spec.variant
    proj_tp_variant = proj_tp_spec.variant
    moe_tp_variant = moe_tp_spec.variant
    data_parallel_variant = workload_spec.dp_spec.variant
    # Derive PP stages strictly from the logical topology (PP tag size)
    pp_stages = cluster_logical_topology.get_mapping_for_tag("PP").size

    # Extract parameters from transformer config
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()

    # Embedding/Unembedding accumulators
    embedding_compute_time = 0.0
    embedding_flops = 0
    unembedding_compute_time = 0.0
    unembedding_flops = 0
    embedding_mem_time = 0.0  # forward
    embedding_mem_vol = 0  # forward
    embedding_mem_time_bwd = 0.0
    embedding_mem_vol_bwd = 0
    unembedding_mem_time = 0.0  # forward
    unembedding_mem_vol = 0  # forward
    unembedding_mem_time_bwd = 0.0
    unembedding_mem_vol_bwd = 0

    # create views for DP and TP and EP
    ep_topology_view = cluster_logical_topology.create_view_from_prefix(
        view_name="EP_View", prefix="EP", global_sharing_context="max_exclusive"
    )

    pp_topology_view = cluster_logical_topology.create_view(
        view_name="PP_View", relevant_tags=["PP"], global_sharing_context="max_exclusive"
    )

    tp_topology_view = cluster_logical_topology.create_view_from_prefix(
        view_name="TP_View", prefix="TP", global_sharing_context="max_exclusive"
    )

    attn_tp_topology_view = tp_topology_view.create_view_from_prefix(
        view_name="TP_Attn_View", prefix="TP_Attn", global_sharing_context="max_exclusive"
    )

    # Optional TP_Proj view: only created/used if tp_proj_spec is explicitly provided in the workload.
    # If omitted, we explicitly reuse TP_Attn_View (current behavior).
    if getattr(workload_spec, "tp_proj_spec", None) is not None:
        proj_tp_topology_view = tp_topology_view.create_view_from_prefix(
            view_name="TP_Proj_View", prefix="TP_Proj", global_sharing_context="max_exclusive"
        )
    else:
        proj_tp_topology_view = attn_tp_topology_view

    # For now, require proj TP variant to match attn TP variant (keeps integration simple and avoids silent changes).
    if proj_tp_variant != attn_tp_variant:
        raise ValueError(
            f"tp_proj_spec.variant ({proj_tp_variant}) must match tp_attn_spec.variant ({attn_tp_variant}) for now."
        )

    moe_tp_topology_view = tp_topology_view.create_view_from_prefix(
        view_name="TP_MoE_View", prefix="TP_MoE", global_sharing_context="max_exclusive"
    )

    dp_topology_view = cluster_logical_topology.create_view_from_prefix(
        view_name="DP_View", prefix="DP", global_sharing_context="max_exclusive"
    )

    # Build a combined DP_Attn view from split SU/SO dims when present:
    dp_attn_su_size = cluster_logical_topology.get_mapping_for_tag("DP_Attn_SU").size
    dp_attn_so_size = cluster_logical_topology.get_mapping_for_tag("DP_Attn_SO").size
    combined_dp_attn_size = dp_attn_su_size * dp_attn_so_size
    # Use the slowest link as the base
    attn_dp_topology_view = cluster_logical_topology.create_virtual_view_from_tag(
        view_name="DP_Attn_View",
        base_tag="DP_Attn_SO",
        virtual_size=combined_dp_attn_size,
        global_sharing_context="max_exclusive",
    )

    moe_dp_topology_view = dp_topology_view.create_view(
        view_name="DP_MoE", relevant_tags=["DP_MoE"], global_sharing_context="max_exclusive"
    )

    # Configure MoE from v2 spec and derive EP placement directly.
    #
    # Policy: require EP_SU to be present in the logical topology. We do not accept any alternate
    # physical-topology knobs for this (lab-style and production topologies should encode it in EP_SU).
    v2_spec = workload_spec.moe_spec_v2
    try:
        _ = cluster_logical_topology.get_mapping_for_tag("EP_SU").size
    except Exception as e:
        raise ValueError(
            "Logical topology must define `EP_SU` (size >= 1). ScaleScope uses EP_SU to define the "
            f"expert high-bandwidth group. Error: {e}"
        ) from e

    if pass_mode == PassMode.INFERENCE_DECODE:
        assert isinstance(workload_spec, InferenceDecodeSpec)
        router_sync_mode = workload_spec.decode_moe_router_sync_mode
    else:
        router_sync_mode = MoERouterSyncMode.REPLICATED_ROUTER

    # Resolve router_shard_factor: number of ranks across which the router
    # weight is sharded.  After AG(X) in 2D TP the router sees D/Y_eff, so
    # sharded_router uses Y_eff; replicated_router is always 1.
    # For 1D TP, sharded_router uses the full attention TP size.
    if router_sync_mode == MoERouterSyncMode.REPLICATED_ROUTER:
        _router_shard_factor = 1
    elif router_sync_mode == MoERouterSyncMode.SHARDED_ROUTER:
        if moe_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
            _resolved_moe_split = resolve_tp2d_split(
                tp_view=moe_tp_topology_view,
                split_2d=moe_tp_spec.split_2d,
                view_name="TP_MoE_router",
            )
            _router_shard_factor = _resolved_moe_split.tp_yz
        else:
            _router_shard_factor = attn_tp_topology_view.num_chips
    else:
        raise ValueError(f"Unsupported router_sync_mode: {router_sync_mode}")

    # Top-k realizability check: under top-k routing with distinct experts per
    # token, rho is bounded by `num_regular_experts / num_selected_experts`.
    # Fail fast here, before any compute/comm/memory work begins, with
    # `model_spec` and `v2_spec` both resolved.
    validate_expert_imbalance_realizability(
        expert_imbalance=v2_spec.expert_imbalance,
        num_regular_experts=model_spec.num_regular_experts,
        num_selected_experts=model_spec.num_selected_experts,
    )

    moe_config = infer_expert_placement_per_xpu(
        model_spec=model_spec,
        moe_v2=v2_spec,
        attn_tp_size=attn_tp_topology_view.num_chips,
        xpus_per_colocated_group=moe_tp_topology_view.num_chips,
        ep_chips_per_pp_stage=attn_dp_topology_view.num_chips * attn_tp_topology_view.num_chips,
        router_sync_mode=router_sync_mode,
        router_shard_factor=_router_shard_factor,
        tp_moe_a2a_replication_optimization_enabled=workload_spec.tp_moe_a2a_replication_optimization,
    )
    full_sets_per_attn_domain = moe_config.full_expert_sets_per_attn_domain
    if full_sets_per_attn_domain is None:
        raise ValueError("Resolved placement must define full_expert_sets_per_attn_domain")
    try:
        expert_set_mapping = cluster_logical_topology.get_mapping_for_tag("Expert_Set_Y")
    except Exception as e:
        if full_sets_per_attn_domain > 1:
            raise ValueError(
                "Multiple expert sets in one attention domain require explicit "
                "logical tag `Expert_Set_Y`."
            ) from e
        expert_set_topology_view = None
    else:
        if expert_set_mapping.size != full_sets_per_attn_domain:
            raise ValueError(
                f"Expert_Set_Y ({expert_set_mapping.size}) must equal "
                f"full_expert_sets_per_attn_domain ({full_sets_per_attn_domain})"
            )
        expert_set_topology_view = cluster_logical_topology.create_view(
            view_name="Expert_Set_Y_View",
            relevant_tags=["Expert_Set_Y"],
            global_sharing_context="concurrent_baseline",
        )
    # MoE placement fields in `moe_config` drive both EP A2A and TP-MoE comm paths.
    # We intentionally use the same placement-locality semantics for both (smooth remote fraction
    # over DP feeder groups and colocated expert groups) so TP1D/TP2D comparisons are consistent.
    # The only non-uniform attention-DP partition currently supported is the
    # scale-out-spanning wide-EP decode case with B < D_feed. It assigns one
    # request to each of B active feeders and leaves the other physical
    # attention domains idle. All other paths retain exact DP divisibility.
    allow_sparse_attention_domains = (
        pass_mode == PassMode.INFERENCE_DECODE
        and moe_config.attn_domains_per_full_expert_set > 1
        and moe_config.full_expert_sets_per_attn_domain == 1
        and moe_config.num_complete_expert_sets == 1
        and workload_spec.decode_cross_domain_ep_mode == DecodeCrossDomainEPMode.EXPLICIT_OWNER_A2AV
    )
    # Attach resolved batch sizes to the workload spec (global-first derivation)
    workload_spec = workload_spec.with_resolved_batches(
        attn_dp_topology_view.num_chips,
        moe_config.num_complete_expert_sets,
        allow_sparse_attention_domains=allow_sparse_attention_domains,
    )
    batch_sizes = workload_spec.get_resolved_batches()
    batch_size_per_dp = batch_sizes.batch_size_per_dp

    # ------------------------------------------------------------------------------------------
    # Decode-only policy: projection_groups (current implementation: TP-2D only)
    #
    # Semantics (global, not comm-only):
    # - We reduce the *effective* TP2D YZ degree for the projection path by a factor of g.
    # - This models splitting projection work across g independent YZ subgroups: comm collectives
    #   run on a smaller group, but per-chip projection compute and projection memory traffic
    #   grow by g because sharding shrinks.
    # - Attention-core sharding (and KV-cache footprint) is unaffected: only the projection view
    #   is derived. Physical XPU counts/topology are unchanged; we only create derived
    #   *model views* for costs.
    # - Concept generalizes to 1D TP; current implementation requires TP-2D (see field docstring
    #   on `projection_groups`).
    # ------------------------------------------------------------------------------------------
    def _resize_view_tag_size(*, view, tag: str, new_size: int, view_name: str):
        mapping = view.get_mapping_for_tag(tag)
        if mapping.size == new_size:
            return view
        new_dims = []
        for d in view.dimensions:
            d_tag = d.logical_tag.value if isinstance(d.logical_tag, Enum) else d.logical_tag
            if d_tag == tag:
                new_dims.append(d.model_copy(update={"size": new_size}))
            else:
                new_dims.append(d)
        return view.model_copy(update={"name": view_name, "dimensions": new_dims})

    if pass_mode == PassMode.INFERENCE_DECODE:
        assert isinstance(workload_spec, InferenceDecodeSpec)
        g = int(getattr(workload_spec, "projection_groups", 1) or 1)
        if g < 1:
            raise ValueError("projection_groups must be >= 1")
        if g > 1:
            if attn_tp_variant != TensorParallelVariant.SHARD_WEIGHTS_2D:
                raise ValueError(
                    "projection_groups > 1 currently requires TP-2D attention (variant: SHARD_WEIGHTS_2D)."
                )
            if attn_tp_spec.split_2d is None or proj_tp_spec.split_2d is None:
                raise ValueError(
                    "projection_groups > 1 requires explicit tp_attn_spec.split_2d and tp_proj_spec.split_2d."
                )
            if len(attn_tp_spec.split_2d.yz_tags) == 0 or len(proj_tp_spec.split_2d.yz_tags) == 0:
                raise ValueError(
                    "projection_groups requires non-empty yz_tags for both attn and proj split_2d specs."
                )
            proj_yz_tags = proj_tp_spec.split_2d.yz_tags
            proj_yz_size = 1
            for proj_yz_tag in proj_yz_tags:
                proj_yz_size *= proj_tp_topology_view.get_mapping_for_tag(proj_yz_tag).size
            if proj_yz_size % g != 0:
                raise ValueError(
                    f"projection_groups={g} must divide projection TP2D YZ size {proj_yz_size} "
                    f"(yz_tags={proj_yz_tags})."
                )

            # IMPORTANT: This policy is intended to affect *projection-domain* work (Q/K/V projections and
            # output projection) without changing attention-core sharding / KV-cache footprint.
            #
            # Therefore, we only shrink the *projection* effective YZ degree. Attention TP2D (and KV-cache
            # sequence sharding on YZ) remains determined by TP_Attn_View.
            #
            # If projection and attention share the same physical TP view (common case), we must create a
            # derived projection view rather than aliasing the attention view.
            if proj_tp_topology_view is attn_tp_topology_view:
                proj_tp_topology_view = proj_tp_topology_view.model_copy(
                    update={"name": f"{proj_tp_topology_view.name}__proj_domain"}
                )
            # Distribute the projection-domain shrink factor `g` across multiple YZ tags.
            # We walk tags in reverse order and, for each tag, take the largest divisor that
            # divides both (a) the remaining `g` factor and (b) that tag's current size. Each
            # accepted divisor reduces that tag size and updates the remaining factor. If any
            # factor remains at the end, this split is infeasible for the current yz_tags sizes.
            remaining_divisor = g
            for proj_yz_tag in reversed(proj_yz_tags):
                if remaining_divisor == 1:
                    break
                current_size = proj_tp_topology_view.get_mapping_for_tag(proj_yz_tag).size
                tag_divisor = remaining_divisor
                while tag_divisor > 1 and (
                    remaining_divisor % tag_divisor != 0 or current_size % tag_divisor != 0
                ):
                    tag_divisor -= 1
                if tag_divisor > 1:
                    proj_tp_topology_view = _resize_view_tag_size(
                        view=proj_tp_topology_view,
                        tag=proj_yz_tag,
                        new_size=current_size // tag_divisor,
                        view_name=f"{proj_tp_topology_view.name}__{proj_yz_tag}_div{tag_divisor}",
                    )
                    remaining_divisor //= tag_divisor

            if remaining_divisor != 1:
                raise ValueError(
                    f"projection_groups={g} could not be distributed across projection yz_tags={proj_yz_tags}."
                )

    # Per-layer host kernel-launch/dispatch floor, applied once per layer per pass.
    kernel_launch_floor_per_pass = (
        workload_spec.get_layers_per_pp_stage() * workload_spec.kernel_launch_latency_per_layer_sec
    )

    match pass_mode:
        case PassMode.TRAINING:
            assert isinstance(workload_spec, TrainSpec)
            total_train_tokens = workload_spec.total_train_tokens
            seq_length = workload_spec.seq_length
            train_steps, _ = workload_spec.calculate_number_of_train_steps()
            moe_runtime_overhead_per_pass = moe_token_pair_overhead_per_pass(
                model_spec=model_spec,
                moe_config=moe_config,
                workload_spec=workload_spec,
                n_seq=seq_length,
            )
            (
                batch_compute_time_forward_pass,
                batch_flops_forward_pass,
                attn_flops_forward_pass,
                moe_flops_forward_pass,
                compute_time_attn_fwd,
                compute_time_moe_fwd,
            ) = forward_prefill_flops_per_chip(
                proj_tp_topology_view,  # proj_tp_topology
                attn_tp_topology_view,
                moe_tp_topology_view,
                model_spec,
                moe_config,
                workload_spec,
            )
            batch_compute_time_forward_pass += (
                kernel_launch_floor_per_pass + moe_runtime_overhead_per_pass
            )
            compute_time_moe_fwd += moe_runtime_overhead_per_pass

            # Attention compute breakdown: projection vs core attention (per-batch, per-chip)
            attn_proj_compute_time_fwd_per_layer, _, attn_core_compute_time_fwd_per_layer, _ = (
                prefill_attention_compute_breakdown_per_layer(
                    proj_tp_topology=proj_tp_topology_view,
                    attn_tp_topology=attn_tp_topology_view,
                    transformer_config=model_spec,
                    workload_spec=workload_spec,
                )
            )
            attn_proj_compute_time_fwd = (
                attn_proj_compute_time_fwd_per_layer * workload_spec.get_layers_per_pp_stage()
            )
            attn_core_compute_time_fwd = (
                attn_core_compute_time_fwd_per_layer * workload_spec.get_layers_per_pp_stage()
            )

            # Embedding/Unembedding compute (prefill)
            embedding_time, embedding_flops_chip = calculate_embedding_compute(
                attn_tp_topology_view, model_spec, workload_spec
            )
            unembedding_time, unembedding_flops_chip = calculate_unembedding_compute(
                attn_tp_topology_view, model_spec, workload_spec
            )
            embedding_compute_time += embedding_time
            embedding_flops += embedding_flops_chip
            unembedding_compute_time += unembedding_time
            unembedding_flops += unembedding_flops_chip
            # Embedding/Unembedding compute accounted once per forward pass

            # backward pass
            (
                batch_compute_time_backward_pass,
                batch_flops_backward_pass,
                attn_flops_backward_pass,
                moe_flops_backward_pass,
                compute_time_attn_bwd,
                compute_time_moe_bwd,
            ) = backward_pass_flops_per_chip(
                proj_tp_topology_view,  # proj_tp_topology
                attn_tp_topology_view,
                moe_tp_topology_view,
                model_spec,
                moe_config,
                workload_spec,
            )
            batch_compute_time_backward_pass += (
                kernel_launch_floor_per_pass + moe_runtime_overhead_per_pass
            )
            compute_time_moe_bwd += moe_runtime_overhead_per_pass
            compute_time_param_update, flops_param_update = parameter_update_flops(
                attn_tp_topology_view, model_spec, moe_config, workload_spec
            )

        case PassMode.INFERENCE_PREFILL:
            assert isinstance(workload_spec, InferencePrefillSpec)
            seq_length = workload_spec.seq_length
            # inference does not have train steps
            train_steps = 1
            total_train_tokens = workload_spec.global_batch_size * seq_length * train_steps
            moe_runtime_overhead_per_pass = moe_token_pair_overhead_per_pass(
                model_spec=model_spec,
                moe_config=moe_config,
                workload_spec=workload_spec,
                n_seq=seq_length,
            )
            (
                batch_compute_time_forward_pass,
                batch_flops_forward_pass,
                attn_flops_forward_pass,
                moe_flops_forward_pass,
                compute_time_attn_fwd,
                compute_time_moe_fwd,
            ) = forward_prefill_flops_per_chip(
                proj_tp_topology_view,  # proj_tp_topology
                attn_tp_topology_view,
                moe_tp_topology_view,
                model_spec,
                moe_config,
                workload_spec,
            )
            batch_compute_time_forward_pass += (
                kernel_launch_floor_per_pass + moe_runtime_overhead_per_pass
            )
            compute_time_moe_fwd += moe_runtime_overhead_per_pass

            # Attention compute breakdown: projection vs core attention (per-batch, per-chip)
            attn_proj_compute_time_fwd_per_layer, _, attn_core_compute_time_fwd_per_layer, _ = (
                prefill_attention_compute_breakdown_per_layer(
                    proj_tp_topology=proj_tp_topology_view,
                    attn_tp_topology=attn_tp_topology_view,
                    transformer_config=model_spec,
                    workload_spec=workload_spec,
                )
            )
            attn_proj_compute_time_fwd = (
                attn_proj_compute_time_fwd_per_layer * workload_spec.get_layers_per_pp_stage()
            )
            attn_core_compute_time_fwd = (
                attn_core_compute_time_fwd_per_layer * workload_spec.get_layers_per_pp_stage()
            )

            # backward pass
            # For inference, we only compute the forward pass time
            batch_compute_time_backward_pass = 0
            batch_flops_backward_pass = 0
            attn_flops_backward_pass = 0
            moe_flops_backward_pass = 0
            compute_time_attn_bwd = 0
            compute_time_moe_bwd = 0
            compute_time_param_update = 0
            flops_param_update = 0

        case PassMode.INFERENCE_DECODE:
            assert isinstance(workload_spec, InferenceDecodeSpec)
            prefill_seq_len = workload_spec.prefill_seq_len
            num_decode_steps = workload_spec.num_generated_tokens
            seq_length = num_decode_steps
            # inference does not have train steps
            train_steps = 1
            total_train_tokens = workload_spec.global_batch_size * num_decode_steps * train_steps

            moe_runtime_overhead_per_step = moe_token_pair_overhead_per_pass(
                model_spec=model_spec,
                moe_config=moe_config,
                workload_spec=workload_spec,
                n_seq=1,
            )

            batch_compute_time_forward_pass = 0.0
            batch_flops_forward_pass = 0
            attn_flops_forward_pass = 0
            moe_flops_forward_pass = 0
            compute_time_attn_fwd = 0
            compute_time_moe_fwd = 0
            attn_proj_compute_time_fwd = 0.0
            attn_core_compute_time_fwd = 0.0

            for i in range(num_decode_steps):
                n_seq_so_far = prefill_seq_len + i
                (
                    step_attn_proj_time,
                    _,
                    step_attn_core_time,
                    _,
                ) = decode_attention_compute_breakdown_per_layer(
                    proj_tp_topology=proj_tp_topology_view,
                    attn_tp_topology=attn_tp_topology_view,
                    transformer_config=model_spec,
                    workload_spec=workload_spec,
                    n_seq_so_far=n_seq_so_far,
                )
                # Scale per-layer per-chip breakdown to per-PP-stage totals for this decode step
                step_attn_proj_time *= workload_spec.get_layers_per_pp_stage()
                step_attn_core_time *= workload_spec.get_layers_per_pp_stage()
                attn_proj_compute_time_fwd += step_attn_proj_time
                attn_core_compute_time_fwd += step_attn_core_time
                (
                    step_time,
                    step_flops,
                    step_attn_flops,
                    step_moe_flops,
                    step_attn_time,
                    step_moe_time,
                ) = forward_decode_flops_per_chip(
                    proj_tp_topology_view,  # proj_tp_topology
                    attn_tp_topology_view,
                    moe_tp_topology_view,
                    model_spec,
                    moe_config,
                    workload_spec,
                    n_seq_so_far,
                )
                batch_compute_time_forward_pass += (
                    step_time + kernel_launch_floor_per_pass + moe_runtime_overhead_per_step
                )
                batch_flops_forward_pass += step_flops
                attn_flops_forward_pass += step_attn_flops
                moe_flops_forward_pass += step_moe_flops
                compute_time_attn_fwd += step_attn_time
                compute_time_moe_fwd += step_moe_time + moe_runtime_overhead_per_step

                # Embedding/Unembedding compute per decode step with explicit seq_len=1
                embedding_time, embedding_flops_chip = calculate_embedding_compute(
                    attn_tp_topology_view, model_spec, workload_spec, seq_len=1
                )
                unembedding_time, unembedding_flops_chip = calculate_unembedding_compute(
                    attn_tp_topology_view, model_spec, workload_spec, seq_len=1
                )
                embedding_compute_time += embedding_time
                embedding_flops += embedding_flops_chip
                unembedding_compute_time += unembedding_time
                unembedding_flops += unembedding_flops_chip

            # backward pass
            # For inference, we only compute the forward pass time
            batch_compute_time_backward_pass = 0
            batch_flops_backward_pass = 0
            attn_flops_backward_pass = 0
            moe_flops_backward_pass = 0
            compute_time_attn_bwd = 0
            compute_time_moe_bwd = 0
            compute_time_param_update = 0
            flops_param_update = 0

        case _:
            raise ValueError(
                f"Unsupported pass mode: {pass_mode}. Supported modes are TRAINING, INFERENCE_PREFILL, and INFERENCE_DECODE."
            )

    # fold param update into backward pass
    batch_compute_time_backward_pass += compute_time_param_update
    batch_flops_backward_pass += flops_param_update

    # batch_mem_time_forward_pass, batch_mem_forward_pass = forward_pass_mem_per_chip(tp_node, tp_topology, transformer_config, batch_size_per_dp, n_seq, float_size)
    # batch_mem_time_backward_pass, batch_mem_backward_pass = backward_pass_mem_per_chip(tp_node, tp_topology, transformer_config, batch_size_per_dp, n_seq, optimizer=optimizer, float_size=float_size)

    def _dims_from_split_or_default(tp_view, tp_spec_v2):
        if tp_spec_v2.variant != TensorParallelVariant.SHARD_WEIGHTS_2D:
            return None, None
        if tp_spec_v2.split_2d is None:
            raise ValueError(
                "Explicit TP-2D split is required (split_2d). "
                "Provide `split_2d: {x_tags: [...], yz_tags: [...]}` for any *variant: SHARD_WEIGHTS_2D*."
            )
        resolved = resolve_tp2d_split(tp_view=tp_view, split_2d=tp_spec_v2.split_2d, view_name="TP")
        return resolved.x_dims, resolved.yz_dims

    # Derive explicit 2D dims for Attn / Proj / MoE when needed
    attn_x_dims, attn_yz_dims = _dims_from_split_or_default(attn_tp_topology_view, attn_tp_spec)
    proj_x_dims, proj_yz_dims = _dims_from_split_or_default(proj_tp_topology_view, proj_tp_spec)
    moe_x_dims, moe_yz_dims = _dims_from_split_or_default(moe_tp_topology_view, moe_tp_spec)

    if attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        validate_2d_tp_invariants_attn(model_spec, attn_tp_topology_view, attn_x_dims, attn_yz_dims)
        validate_2d_tp_invariants_proj(model_spec, proj_tp_topology_view, proj_x_dims, proj_yz_dims)
    elif attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_1D:
        validate_1d_tp_invariants_attn(model_spec, attn_tp_topology_view)
    if moe_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        validate_2d_tp_invariants_moe(model_spec, moe_tp_topology_view, moe_x_dims, moe_yz_dims)

    # --- Decode layout trace: actual trace object (populated during first decode step) ---
    _actual_layout_trace = NULL_LAYOUT_TRACE
    if (
        enable_decode_layout_trace
        and pass_mode == PassMode.INFERENCE_DECODE
        and attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D
    ):
        params = _resolve_decode_trace_params(
            model_spec=model_spec,
            moe_config=moe_config,
            proj_tp_topology_view=proj_tp_topology_view,
            attn_tp_topology_view=attn_tp_topology_view,
            moe_tp_topology_view=moe_tp_topology_view,
            proj_x_dims=proj_x_dims,
            proj_yz_dims=proj_yz_dims,
            attn_yz_dims=attn_yz_dims,
            moe_x_dims=moe_x_dims,
            moe_yz_dims=moe_yz_dims,
            g=g,
            router_sync_mode=router_sync_mode,
        )
        _actual_layout_trace = LayoutTrace(
            mesh=build_decode_2dtp_mesh(
                params["tp_x"],
                params["tp_y"],
                params["g"],
                params["n_experts"],
                params["c"],
                params["moe_tp_x"],
                params["moe_tp_yz"],
                params["full_expert_sets_per_attn_domain"],
            )
        )

    # TODO(MB) : move this to config
    activation_checkpointing = False  # (workload_spec.activation_recomputation != 0)

    device_for_map = workload_spec.memory_device_map_traffic
    # peak memory footprint calculation
    device_for_fp = workload_spec.memory_device_map_footprint
    match pass_mode:
        case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
            fwd_traffic_per_layer, bwd_traffic_per_layer = get_transformer_mem_stats_per_layer(
                attn_tp_topology_view,
                moe_tp_topology_view,
                model_spec,
                moe_config,
                batch_size_per_dp,
                seq_length,
                attn_tp_variant,
                moe_tp_variant,
                attn_x_dims,
                attn_yz_dims,
                moe_x_dims,
                moe_yz_dims,
                pass_mode,
                workload_spec.float_size_bytes,
                proj_tp_topology=proj_tp_topology_view,
                proj_x_dims=proj_x_dims,
                proj_yz_dims=proj_yz_dims,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for_map,
                workload_spec=workload_spec,
                attn_dp_size=attn_dp_topology_view.num_chips,
                moe_dp_size=moe_dp_topology_view.num_chips,
            )
            # Embedding/Unembedding memory traffic
            embed_tr = calculate_embedding_memory_traffic_1d_tp_fwd(
                name="embedding",
                model_spec=model_spec,
                workload_spec=workload_spec,
                attn_tp_topology=attn_tp_topology_view,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for_map,
                pass_mode=pass_mode,
            )
            unembed_tr = calculate_unembedding_memory_traffic_1d_tp_fwd(
                name="unembedding",
                model_spec=model_spec,
                workload_spec=workload_spec,
                attn_tp_topology=attn_tp_topology_view,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for_map,
                pass_mode=pass_mode,
            )
            embed_stats = embed_tr.get_stats(1)
            unembed_stats = unembed_tr.get_stats(1)
            # Backward components (only in training)
            if pass_mode == PassMode.TRAINING:
                embed_bwd = calculate_embedding_memory_traffic_1d_tp_bwd(
                    name="embedding_bwd",
                    model_spec=model_spec,
                    workload_spec=workload_spec,
                    attn_tp_topology=attn_tp_topology_view,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for_map,
                    pass_mode=pass_mode,
                )
                unembed_bwd = calculate_unembedding_memory_traffic_1d_tp_bwd(
                    name="unembedding_bwd",
                    model_spec=model_spec,
                    workload_spec=workload_spec,
                    attn_tp_topology=attn_tp_topology_view,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for_map,
                    pass_mode=pass_mode,
                )
                embed_bwd_stats = embed_bwd.get_stats(1)
                unembed_bwd_stats = unembed_bwd.get_stats(1)
                embedding_mem_time_bwd += embed_bwd_stats.get("embedding_bwd_time_s", 0.0)
                embedding_mem_vol_bwd += embed_bwd_stats.get("embedding_bwd_total_bytes", 0)
                unembedding_mem_time_bwd += unembed_bwd_stats.get("unembedding_bwd_time_s", 0.0)
                unembedding_mem_vol_bwd += unembed_bwd_stats.get("unembedding_bwd_total_bytes", 0)

        case PassMode.INFERENCE_DECODE:
            prefill_seq_len = workload_spec.prefill_seq_len
            num_decode_steps = workload_spec.num_generated_tokens
            fwd_traffic_per_layer = LayerMemoryTraffic(
                name="fwd_decode_total",
                device_for=device_for_map,
                node=attn_tp_topology_view.physical_topology.node_definition,
            )
            bwd_traffic_per_layer = LayerMemoryTraffic(
                name="bwd_decode_total",
                device_for=device_for_map,
                node=attn_tp_topology_view.physical_topology.node_definition,
            )
            for i in range(num_decode_steps):
                n_seq_so_far = prefill_seq_len + i
                fwd_traffic_per_layer_per_step, bwd_traffic_per_layer_per_step = (
                    get_transformer_mem_stats_per_layer(
                        attn_tp_topology_view,
                        moe_tp_topology_view,
                        model_spec,
                        moe_config,
                        batch_size_per_dp,
                        n_seq_so_far,
                        attn_tp_variant,
                        moe_tp_variant,
                        attn_x_dims,
                        attn_yz_dims,
                        moe_x_dims,
                        moe_yz_dims,
                        pass_mode,
                        workload_spec.float_size_bytes,
                        proj_tp_topology=proj_tp_topology_view,
                        proj_x_dims=proj_x_dims,
                        proj_yz_dims=proj_yz_dims,
                        activation_checkpointing=activation_checkpointing,
                        device_for=device_for_map,
                        workload_spec=workload_spec,
                        attn_dp_size=attn_dp_topology_view.num_chips,
                        moe_dp_size=moe_dp_topology_view.num_chips,
                    )
                )

                fwd_traffic_per_layer += fwd_traffic_per_layer_per_step
                bwd_traffic_per_layer += bwd_traffic_per_layer_per_step

                # Embedding/Unembedding memory traffic per decode step
                try:
                    ws_step = workload_spec.model_copy(update={"seq_length": 1})
                except Exception:
                    ws_step = workload_spec
                    try:
                        object.__setattr__(ws_step, "seq_length", 1)
                    except Exception:
                        pass
                embed_tr = calculate_embedding_memory_traffic_1d_tp_fwd(
                    name="embedding_step",
                    model_spec=model_spec,
                    workload_spec=ws_step,
                    attn_tp_topology=attn_tp_topology_view,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for_map,
                    pass_mode=pass_mode,
                    seq_len=1,
                )
                unembed_tr = calculate_unembedding_memory_traffic_1d_tp_fwd(
                    name="unembedding_step",
                    model_spec=model_spec,
                    workload_spec=ws_step,
                    attn_tp_topology=attn_tp_topology_view,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for_map,
                    pass_mode=pass_mode,
                    seq_len=1,
                )
                es = embed_tr.get_stats(1)
                us = unembed_tr.get_stats(1)
                embedding_mem_time += es.get("embedding_step_time_s", 0.0)
                embedding_mem_vol += es.get("embedding_step_total_bytes", 0)
                unembedding_mem_time += us.get("unembedding_step_time_s", 0.0)
                unembedding_mem_vol += us.get("unembedding_step_total_bytes", 0)

        case _:
            raise ValueError(f"Unsupported pass mode: {pass_mode}")

    node = attn_tp_topology_view.physical_topology.node_definition

    fwd_mem_stats = fwd_traffic_per_layer.get_stats(num_layers_per_pp_stage)
    bwd_mem_stats = bwd_traffic_per_layer.get_stats(num_layers_per_pp_stage)

    # Fold in embedding/unembedding memory to forward totals
    if pass_mode in (PassMode.TRAINING, PassMode.INFERENCE_PREFILL):
        embedding_mem_time += embed_stats.get("embedding_time_s", 0.0)
        embedding_mem_vol += embed_stats.get("embedding_total_bytes", 0)
        unembedding_mem_time += unembed_stats.get("unembedding_time_s", 0.0)
        unembedding_mem_vol += unembed_stats.get("unembedding_total_bytes", 0)

    mem_stats = {**fwd_mem_stats, **bwd_mem_stats}
    mem_stats["total_memory_B"] = (
        fwd_mem_stats[f"{fwd_traffic_per_layer.name}_total_bytes"]
        + bwd_mem_stats[f"{bwd_traffic_per_layer.name}_total_bytes"]
    )
    mem_stats["total_memory_time_s"] = (
        fwd_mem_stats[f"{fwd_traffic_per_layer.name}_time_s"]
        + bwd_mem_stats[f"{bwd_traffic_per_layer.name}_time_s"]
    )
    # memory_bandwidth = node.memory_bandwidth
    # mem_stats["memory_bandwidth_utilization"] = mem_stats["total_memory_B"] / mem_stats["total_memory_time_s"] / memory_bandwidth if mem_stats["total_memory_time_s"] > 0 else 0
    mem_stats["tp_attn_variant"] = attn_tp_variant
    mem_stats["tp_moe_variant"] = moe_tp_variant
    mem_stats["tp_attn_x_dims"] = attn_x_dims
    mem_stats["tp_attn_yz_dims"] = attn_yz_dims
    mem_stats["tp_moe_x_dims"] = moe_x_dims
    mem_stats["tp_moe_yz_dims"] = moe_yz_dims
    mem_stats["activation_checkpointing"] = activation_checkpointing

    attn_footprint = MemoryFootprint(name="attn", device_for=device_for_fp, node=node)
    ffn_moe_footprint = MemoryFootprint(name="ffn_moe", device_for=device_for_fp, node=node)

    attn_tp_size = attn_tp_topology_view.num_chips
    moe_tp_size = moe_tp_topology_view.num_chips

    attn_dp_size = attn_dp_topology_view.num_chips
    moe_dp_size = moe_dp_topology_view.num_chips

    # 1. Weights (use full-precision storage for training, compute dtype for inference)
    weights_fs = (
        workload_spec.full_precision_param_float_size
        if pass_mode == PassMode.TRAINING
        else workload_spec.float_size_bytes
    )
    # Role-aware at-rest sharding for attention (projection) weights. The
    # attention block's weights are exactly the projection weights (W_q/k/v/o),
    # so they shard across the *projection* TP domain — which differs from the
    # attention domain under projection groups (g > 1), where the proj view is
    # g-times smaller and the weights are replicated across groups. Under
    # TP-2D prefill/training with yz=shard_tokens the YZ axis carries tokens
    # and the weights are fully replicated across it, so only the X axis
    # divides the footprint. shard_weights and circulate_weights both keep
    # weights sharded across X*YZ at rest.
    attn_weight_tp_size = proj_tp_topology_view.num_chips
    if (
        pass_mode in (PassMode.TRAINING, PassMode.INFERENCE_PREFILL)
        and attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D
    ):
        roles = workload_spec.tp2d_proj_axis_roles
        if roles is None:
            raise ValueError(
                "TP-2D prefill/train weight footprint requires tp2d_proj_axis_roles "
                "on the workload spec (TrainSpec / InferencePrefillSpec)."
            )
        match roles.yz:
            case TP2DProjAxisRole.SHARD_TOKENS:
                if proj_tp_spec.split_2d is None:
                    raise ValueError(
                        "TP-2D projection requires a split_2d on the projection TP spec "
                        "for the weight footprint."
                    )
                _resolved_proj_w = resolve_tp2d_split(
                    tp_view=proj_tp_topology_view,
                    split_2d=proj_tp_spec.split_2d,
                    view_name="TP_Proj",
                )
                attn_weight_tp_size = _resolved_proj_w.tp_x
            case TP2DProjAxisRole.CIRCULATE_WEIGHTS | TP2DProjAxisRole.SHARD_WEIGHTS:
                pass  # weights sharded across X*YZ at rest: full divisor stands
            case _:
                raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")
    attn_footprint.weights = calculate_attn_weights_memory_footprint(
        model_spec,
        attn_weight_tp_size,
        attn_dp_size,
        data_parallel_variant,
        weights_fs,
    )
    ffn_moe_footprint.weights = calculate_ffn_moe_weights_memory_footprint(
        model_spec,
        moe_config,
        moe_tp_size,
        moe_dp_size,
        data_parallel_variant,
        weights_fs,
    )

    # Gradient buffers (training only): per-parameter gradients in compute dtype.
    # Sharded like the weights at rest (role-aware for TP-2D, see above).
    if pass_mode == PassMode.TRAINING:
        grad_fs = workload_spec.float_size_bytes
        attn_grad_buffers = calculate_attn_weights_memory_footprint(
            model_spec,
            attn_weight_tp_size,
            attn_dp_size,
            data_parallel_variant,
            grad_fs,
        )
        ffn_moe_grad_buffers = calculate_ffn_moe_weights_memory_footprint(
            model_spec,
            moe_config,
            moe_tp_size,
            moe_dp_size,
            data_parallel_variant,
            grad_fs,
        )
        attn_footprint.gradient_buffers = attn_grad_buffers
        ffn_moe_footprint.gradient_buffers = ffn_moe_grad_buffers

    # Forward transient peaks apply to all pass modes
    ws_for_footprint = workload_spec
    if pass_mode == PassMode.INFERENCE_DECODE:
        try:
            ws_for_footprint = workload_spec.model_copy(update={"seq_length": 1})
        except Exception:
            # Fallback: mutate a shallow copy if model_copy isn't available
            ws_for_footprint = workload_spec
            try:
                object.__setattr__(ws_for_footprint, "seq_length", 1)
            except Exception:
                pass

    # Forward transient peaks apply to all pass modes.
    # For TP-2D attention we use a stage-aware, phase-aware sizing that reflects:
    # - Phase 1 QKV output (sharded on X*YZ)
    # - Phase 2 score buffer (non-fused model) with sequence sharding on YZ
    # - Phase 3 output projection staging (gathered E/X and partial E/YZ)
    match attn_tp_variant:
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            if attn_tp_spec.split_2d is None:
                raise ValueError("TP-2D attention requires tp_attn_spec.split_2d for footprint.")
            resolved = resolve_tp2d_split(
                tp_view=attn_tp_topology_view, split_2d=attn_tp_spec.split_2d, view_name="TP_Attn"
            )
            # Phase 1/3 staging buffers live in the projection TP domain, which
            # differs from the attention domain under projection groups (g > 1).
            if proj_tp_spec.split_2d is None:
                raise ValueError(
                    "TP-2D attention footprint requires a split_2d on the projection TP spec."
                )
            resolved_proj = resolve_tp2d_split(
                tp_view=proj_tp_topology_view,
                split_2d=proj_tp_spec.split_2d,
                view_name="TP_Proj",
            )
            match pass_mode:
                case PassMode.INFERENCE_DECODE:
                    assert isinstance(workload_spec, InferenceDecodeSpec)
                    kv_total_len = (
                        workload_spec.prefill_seq_len + workload_spec.num_generated_tokens
                    )
                case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
                    kv_total_len = None
                case _:
                    raise ValueError(f"Unsupported pass_mode for footprint: {pass_mode}")
            attn_footprint.transient_forward_peak = (
                calculate_attn_forward_transient_peak_memory_footprint_tp2d(
                    transformer_config=model_spec,
                    workload_spec=ws_for_footprint,
                    attn_tp_size_x=resolved.tp_x,
                    attn_tp_size_yz=resolved.tp_yz,
                    proj_tp_size_x=resolved_proj.tp_x,
                    proj_tp_size_yz=resolved_proj.tp_yz,
                    pass_mode=pass_mode,
                    kv_total_len=kv_total_len,
                )
            )
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            attn_footprint.transient_forward_peak = (
                calculate_attn_forward_transient_peak_memory_footprint_tp1d(
                    model_spec, ws_for_footprint, attn_tp_size
                )
            )
        case _:
            raise ValueError(f"Unsupported attention TP variant for footprint: {attn_tp_variant}")

    match moe_tp_variant:
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            if moe_tp_spec.split_2d is None:
                raise ValueError("TP-2D MoE requires tp_moe_spec.split_2d for footprint.")
            resolved_moe = resolve_tp2d_split(
                tp_view=moe_tp_topology_view,
                split_2d=moe_tp_spec.split_2d,
                view_name="TP_MoE",
            )
            ffn_moe_footprint.transient_forward_peak = (
                calculate_ffn_moe_forward_transient_peak_memory_footprint_tp2d(
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    workload_spec=ws_for_footprint,
                    moe_tp_size_x=resolved_moe.tp_x,
                    moe_tp_size_yz=resolved_moe.tp_yz,
                    pass_mode=pass_mode,
                )
            )
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            ffn_moe_footprint.transient_forward_peak = (
                calculate_ffn_moe_forward_transient_peak_memory_footprint_tp1d(
                    model_spec, ws_for_footprint, moe_tp_size
                )
            )
        case _:
            raise ValueError(f"Unsupported MoE TP variant for footprint: {moe_tp_variant}")

    if pass_mode == PassMode.TRAINING:
        assert isinstance(workload_spec, TrainSpec)
        # Optimizer States (sharded like the weights at rest; role-aware for TP-2D)
        attn_footprint.optimizer_states = calculate_attn_optimizer_states_memory_footprint(
            model_spec,
            attn_weight_tp_size,
            attn_dp_size,
            data_parallel_variant,
            workload_spec.full_precision_param_float_size,
            workload_spec.optimizer,
        )
        ffn_moe_footprint.optimizer_states = calculate_ffn_moe_optimizer_states_memory_footprint(
            model_spec,
            moe_config,
            moe_tp_size,
            moe_dp_size,
            data_parallel_variant,
            workload_spec.full_precision_param_float_size,
        )

        # Saved Activations (training strategy applies)
        attn_footprint.saved_activations = calculate_attn_saved_activations_memory_footprint(
            model_spec, workload_spec, workload_spec.activation_storage_strategy, attn_tp_size
        )
        ffn_moe_footprint.saved_activations = calculate_ffn_moe_saved_activations_memory_footprint(
            model_spec, workload_spec, workload_spec.activation_storage_strategy, attn_tp_size
        )

        # Backward transient peaks
        attn_footprint.transient_backward_peak = (
            calculate_attn_backward_transient_peak_memory_footprint(
                model_spec, workload_spec, workload_spec.activation_storage_strategy, attn_tp_size
            )
        )
        ffn_moe_footprint.transient_backward_peak = (
            calculate_ffn_moe_backward_transient_peak_memory_footprint(
                model_spec, workload_spec, workload_spec.activation_storage_strategy, moe_tp_size
            )
        )

    # KV cache footprint (inference modes)
    if pass_mode == PassMode.INFERENCE_PREFILL or pass_mode == PassMode.INFERENCE_DECODE:
        num_kv_heads = model_spec.num_kv_heads
        head_dim = model_spec.head_dim
        if pass_mode == PassMode.INFERENCE_PREFILL:
            kv_len = workload_spec.seq_length
        else:
            assert isinstance(workload_spec, InferenceDecodeSpec)
            kv_len = workload_spec.prefill_seq_len + workload_spec.num_generated_tokens
        # KV cache footprint:
        # - 2D TP attn: heads sharded by X, sequence/KV length sharded by YZ (use per-XPU max via ceil)
        # - 1D/None: heads sharded by total TP size, full KV length per XPU
        match attn_tp_variant:
            case TensorParallelVariant.SHARD_WEIGHTS_2D:
                if attn_tp_spec.split_2d is None or attn_x_dims is None or attn_yz_dims is None:
                    raise ValueError(
                        "TP-2D attention requires explicit split_2d and resolved x/y dims for KV-cache footprint."
                    )
                resolved = resolve_tp2d_split(
                    tp_view=attn_tp_topology_view,
                    split_2d=attn_tp_spec.split_2d,
                    view_name="TP_Attn",
                )
                kv_heads_sharded, _kv_rep = resolve_kv_heads_per_partition(
                    num_kv_heads=num_kv_heads, tp_x=resolved.tp_x
                )
                kv_seq_shard = resolved.tp_yz
            case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
                kv_heads_sharded, _kv_rep = resolve_kv_heads_per_partition(
                    num_kv_heads=num_kv_heads, tp_x=attn_tp_size
                )
                kv_seq_shard = 1
            case _:
                raise ValueError(
                    f"Unsupported attention TP variant for KV-cache footprint: {attn_tp_variant}"
                )
        # Persistent KV residency per layer class: sliding-window layers only
        # retain the last `window_size` tokens, so their footprint is capped.
        # Blended by layer fractions (per-layer footprint × n_layers elsewhere).
        kv_bytes_per_layer_per_xpu = 0.0
        for layer_class in attention_layer_classes(model_spec):
            kv_len_per_shard_cls = int(np.ceil(layer_class.kv_len(kv_len) / kv_seq_shard))
            kv_bytes_per_layer_per_xpu += (
                layer_class.layer_fraction
                * 2
                * batch_size_per_dp
                * kv_len_per_shard_cls
                * (kv_heads_sharded * head_dim)
                * workload_spec.float_size_bytes
            )
        attn_footprint.kv_cache = int(kv_bytes_per_layer_per_xpu)

    # Verify memory capacities per tier
    combined = attn_footprint + ffn_moe_footprint
    combined.verify()

    # Combine stats
    footprint_stats = {**attn_footprint.get_stats(), **ffn_moe_footprint.get_stats()}
    mem_stats.update(footprint_stats)

    # Calculate total peak memory
    persistent_attn = (
        attn_footprint.weights
        + attn_footprint.optimizer_states
        + attn_footprint.saved_activations
        + getattr(attn_footprint, "gradient_buffers", 0)
        + attn_footprint.kv_cache
    )
    persistent_ffn_moe = (
        ffn_moe_footprint.weights
        + ffn_moe_footprint.optimizer_states
        + ffn_moe_footprint.saved_activations
        + getattr(ffn_moe_footprint, "gradient_buffers", 0)
    )
    peak_transient_layer = max(
        attn_footprint.transient_forward_peak + ffn_moe_footprint.transient_forward_peak,
        attn_footprint.transient_backward_peak + ffn_moe_footprint.transient_backward_peak,
    )
    total_peak_memory_per_chip = (
        persistent_attn + persistent_ffn_moe
    ) * num_layers_per_pp_stage + peak_transient_layer
    mem_stats["total_peak_memory_per_chip_bytes"] = total_peak_memory_per_chip

    # print("Compute stats forward pass: ", batch_compute_time_forward_pass, batch_flops_forward_pass)
    # print("Memory stats: ", mem_stats)
    batch_mem_time_forward_pass = (
        mem_stats[f"{fwd_traffic_per_layer.name}_time_s"]
        + embedding_mem_time
        + unembedding_mem_time
    )
    batch_mem_forward_pass = (
        mem_stats[f"{fwd_traffic_per_layer.name}_total_bytes"]
        + embedding_mem_vol
        + unembedding_mem_vol
    )
    batch_mem_time_backward_pass = mem_stats[f"{bwd_traffic_per_layer.name}_time_s"]
    batch_mem_backward_pass = mem_stats[f"{bwd_traffic_per_layer.name}_total_bytes"]

    match pass_mode:
        case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
            seq_length = workload_spec.seq_length
            comm_stats = get_transformer_comm_stats(
                attn_dp_topology_view,
                moe_dp_topology_view,
                pp_topology_view,
                attn_tp_topology_view,
                proj_tp_topology_view,  # proj_tp_topology (explicit; defaults to attn if absent)
                moe_tp_topology_view,
                ep_topology_view,
                expert_set_topology_view,
                model_spec,
                moe_config,
                workload_spec,
                seq_length,
                pass_mode,
            )
            # Typical practice avoids all-gather; unembedding_comm_* remain zero
        case PassMode.INFERENCE_DECODE:
            assert isinstance(workload_spec, InferenceDecodeSpec)
            num_decode_steps = workload_spec.num_generated_tokens

            # Initialize comm stats to zero, then explicitly accumulate per decode step.
            # Even though comm is currently modeled as per-step constant, decode is synchronous
            # and we already iterate per step for compute/memory; keep comm explicit too.
            comm_stats = {
                "tp_attn_fwd": CommStats(),
                "tp_attn_fwd_phase1": CommStats(),
                "tp_attn_fwd_phase2": CommStats(),
                "tp_attn_fwd_phase3": CommStats(),
                "tp_moe_fwd": CommStats(),
                "tp_attn_bwd": CommStats(),
                "tp_moe_bwd": CommStats(),
                "dp_attn_fwd": CommStats(),
                "dp_moe_fwd": CommStats(),
                "dp_attn_bwd": CommStats(),
                "dp_moe_bwd": CommStats(),
                "pp_fwd": CommStats(),
                "pp_bwd": CommStats(),
                "moe_a2a_fwd_1": CommStats(),
                "moe_a2a_fwd_2": CommStats(),
            }

            for i in range(num_decode_steps):
                # We use seq_length = 1 for decode steps in TP Megatron
                step_comm_stats = get_transformer_comm_stats(
                    attn_dp_topology_view,
                    moe_dp_topology_view,
                    pp_topology_view,
                    attn_tp_topology_view,
                    proj_tp_topology_view,  # proj_tp_topology (explicit; defaults to attn if absent)
                    moe_tp_topology_view,
                    ep_topology_view,
                    expert_set_topology_view,
                    model_spec,
                    moe_config,
                    workload_spec,
                    n_seq=1,
                    pass_mode=pass_mode,
                    layout_trace=_actual_layout_trace if i == 0 else NULL_LAYOUT_TRACE,
                )
                # Accumulate stats for each decode step
                for key, val in step_comm_stats.items():
                    comm_stats[key] += val
                # Typical practice avoids all-gather; unembedding_comm_* remain zero

            # Emit expected vs actual layout trace comparison
            if enable_decode_layout_trace and _actual_layout_trace is not NULL_LAYOUT_TRACE:
                _emit_decode_layout_trace(
                    model_spec=model_spec,
                    workload_spec=workload_spec,
                    moe_config=moe_config,
                    attn_tp_topology_view=attn_tp_topology_view,
                    proj_tp_topology_view=proj_tp_topology_view,
                    moe_tp_topology_view=moe_tp_topology_view,
                    attn_x_dims=attn_x_dims,
                    attn_yz_dims=attn_yz_dims,
                    proj_x_dims=proj_x_dims,
                    proj_yz_dims=proj_yz_dims,
                    moe_x_dims=moe_x_dims,
                    moe_yz_dims=moe_yz_dims,
                    g=g,
                    router_sync_mode=router_sync_mode,
                    actual_trace=_actual_layout_trace,
                )
        case _:
            raise ValueError(
                f"Unsupported pass mode: {pass_mode}. Supported modes are TRAINING, INFERENCE_PREFILL, and INFERENCE_DECODE."
            )

    # Unpack dictionary for aggregated stats
    tp_attn_fwd = comm_stats["tp_attn_fwd"]
    tp_attn_fwd_phase1 = comm_stats.get("tp_attn_fwd_phase1", CommStats())
    tp_attn_fwd_phase2 = comm_stats.get("tp_attn_fwd_phase2", CommStats())
    tp_attn_fwd_phase3 = comm_stats.get("tp_attn_fwd_phase3", CommStats())
    tp_moe_fwd = comm_stats["tp_moe_fwd"]
    tp_attn_bwd = comm_stats["tp_attn_bwd"]
    tp_moe_bwd = comm_stats["tp_moe_bwd"]
    dp_attn_fwd = comm_stats["dp_attn_fwd"]
    dp_moe_fwd = comm_stats["dp_moe_fwd"]
    dp_attn_bwd = comm_stats["dp_attn_bwd"]
    dp_moe_bwd = comm_stats["dp_moe_bwd"]
    pp_fwd = comm_stats["pp_fwd"]
    pp_bwd = comm_stats["pp_bwd"]
    moe_a2a_fwd_1 = comm_stats["moe_a2a_fwd_1"]
    moe_a2a_fwd_2 = comm_stats["moe_a2a_fwd_2"]

    # Assume that backward comm is the same as the forward comm
    moe_a2a_bwd = moe_a2a_fwd_1 + moe_a2a_fwd_2 if pass_mode == PassMode.TRAINING else CommStats()

    batch_comm_time_forward_pass = (
        tp_attn_fwd.total_comm_time
        + tp_moe_fwd.total_comm_time
        + dp_attn_fwd.total_comm_time
        + dp_moe_fwd.total_comm_time
        + pp_fwd.total_comm_time
        + moe_a2a_fwd_1.total_comm_time
        + moe_a2a_fwd_2.total_comm_time
    )
    batch_comm_time_backward_pass = (
        tp_attn_bwd.total_comm_time
        + tp_moe_bwd.total_comm_time
        + dp_attn_bwd.total_comm_time
        + dp_moe_bwd.total_comm_time
        + pp_bwd.total_comm_time
        + moe_a2a_bwd.total_comm_time
    )
    batch_comm_vol_forward_pass = (
        tp_attn_fwd.comm_vol
        + tp_moe_fwd.comm_vol
        + dp_attn_fwd.comm_vol
        + dp_moe_fwd.comm_vol
        + pp_fwd.comm_vol
        + moe_a2a_fwd_1.comm_vol
        + moe_a2a_fwd_2.comm_vol
    )
    batch_comm_vol_backward_pass = (
        tp_attn_bwd.comm_vol
        + tp_moe_bwd.comm_vol
        + dp_attn_bwd.comm_vol
        + dp_moe_bwd.comm_vol
        + pp_bwd.comm_vol
        + moe_a2a_bwd.comm_vol
    )
    batch_comm_startup_forward_pass = (
        tp_attn_fwd.startup_latency
        + tp_moe_fwd.startup_latency
        + dp_attn_fwd.startup_latency
        + dp_moe_fwd.startup_latency
        + pp_fwd.startup_latency
        + moe_a2a_fwd_1.startup_latency
        + moe_a2a_fwd_2.startup_latency
    )
    batch_comm_startup_backward_pass = (
        tp_attn_bwd.startup_latency
        + tp_moe_bwd.startup_latency
        + dp_attn_bwd.startup_latency
        + dp_moe_bwd.startup_latency
        + pp_bwd.startup_latency
        + moe_a2a_bwd.startup_latency
    )
    batch_comm_data_transfer_forward_pass = (
        tp_attn_fwd.transmission_time
        + tp_moe_fwd.transmission_time
        + dp_attn_fwd.transmission_time
        + dp_moe_fwd.transmission_time
        + pp_fwd.transmission_time
        + moe_a2a_fwd_1.transmission_time
        + moe_a2a_fwd_2.transmission_time
    )
    batch_comm_data_transfer_backward_pass = (
        tp_attn_bwd.transmission_time
        + tp_moe_bwd.transmission_time
        + dp_attn_bwd.transmission_time
        + dp_moe_bwd.transmission_time
        + pp_bwd.transmission_time
        + moe_a2a_bwd.transmission_time
    )

    def _alpha_for_view(view: LogicalTopology) -> float | None:
        alphas = []
        for dim in getattr(view, "dimensions", []):
            if dim.size <= 1:
                continue
            tag = dim.logical_tag.value if hasattr(dim.logical_tag, "value") else dim.logical_tag
            _, alpha_sec, _, _ = view.get_hockney_model_params(
                tag_in_view=tag,
                float_size_bytes=workload_spec.float_size_bytes,
                context=getattr(view, "global_sharing_context", "concurrent_baseline"),
            )
            if alpha_sec > 0:
                alphas.append(alpha_sec)
        if alphas:
            return min(alphas)
        # Some derived views may have size-1 logical dims while comm is priced
        # on the underlying fabric; use physical interconnect latency explicitly.
        phys_alphas = []
        for phys_dim in getattr(view.physical_topology, "dimensions", []):
            ic = view.physical_topology.node_definition.get_interconnect(phys_dim.interconnect_name)
            if ic.latency_sec > 0:
                phys_alphas.append(ic.latency_sec)
        return min(phys_alphas) if phys_alphas else None

    def _startup_phase_count(comm: CommStats, view: LogicalTopology) -> int:
        if getattr(comm, "startup_phase_count", 0) > 0:
            return int(comm.startup_phase_count)
        if comm.startup_latency <= 0:
            return 0
        alpha_sec = _alpha_for_view(view)
        if alpha_sec is None:
            raise ValueError(
                "Unable to infer startup alpha for non-zero startup latency; "
                "topology view has no active communication dimensions."
            )
        return int(round(comm.startup_latency / alpha_sec))

    startup_phase_count_tp_moe_fwd = _startup_phase_count(tp_moe_fwd, moe_tp_topology_view)
    startup_phase_count_a2a_to_fwd = _startup_phase_count(moe_a2a_fwd_1, ep_topology_view)
    startup_phase_count_a2a_from_fwd = _startup_phase_count(moe_a2a_fwd_2, ep_topology_view)
    startup_phase_count_a2a_bwd = _startup_phase_count(moe_a2a_bwd, ep_topology_view)
    startup_phase_count_total_comm_per_batch = (
        _startup_phase_count(
            CommStats(
                startup_latency=batch_comm_startup_forward_pass + batch_comm_startup_backward_pass
            ),
            attn_tp_topology_view,
        )
        if (batch_comm_startup_forward_pass + batch_comm_startup_backward_pass) > 0
        else 0
    )

    # Detailed Memory
    (
        fwd_attn_traffic,
        fwd_attn_proj_traffic,
        fwd_attn_core_traffic,
        fwd_moe_traffic,
        bwd_attn_traffic,
        bwd_moe_traffic,
    ) = get_detailed_mem_stats(
        attn_tp_topology_view=attn_tp_topology_view,
        proj_tp_topology_view=proj_tp_topology_view,
        moe_tp_topology_view=moe_tp_topology_view,
        model_spec=model_spec,
        moe_config=moe_config,
        workload_spec=workload_spec,
        attn_tp_variant=attn_tp_variant,
        moe_tp_variant=moe_tp_variant,
        attn_x_dims=attn_x_dims,
        attn_yz_dims=attn_yz_dims,
        proj_x_dims=proj_x_dims,
        proj_yz_dims=proj_yz_dims,
        moe_x_dims=moe_x_dims,
        moe_yz_dims=moe_yz_dims,
        pass_mode=pass_mode,
        activation_checkpointing=activation_checkpointing,
        device_for=device_for_map,
        node=node,
    )

    # Prepare detailed per-component memory traffic split by block (attn vs ffn_moe)
    # Scale per-layer component bytes by number of layers per PP stage to get per-batch per-chip totals
    layers_per_stage = num_layers_per_pp_stage
    fwd_attn_layer_stats = fwd_attn_traffic.get_stats(1)
    fwd_attn_proj_layer_stats = fwd_attn_proj_traffic.get_stats(1)
    fwd_attn_core_layer_stats = fwd_attn_core_traffic.get_stats(1)
    fwd_moe_layer_stats = fwd_moe_traffic.get_stats(1)
    # Expose detailed per-layer component stats (including read/write times) for forward
    mem_stats.update(fwd_attn_layer_stats)
    mem_stats.update(fwd_attn_proj_layer_stats)
    mem_stats.update(fwd_attn_core_layer_stats)
    mem_stats.update(fwd_moe_layer_stats)
    mem_stats.update(
        {
            # Forward Attention per-batch per-chip
            "attn_fwd_weights_read_bytes": layers_per_stage
            * fwd_attn_layer_stats.get("fwd_attn_weights_read_bytes_per_layer", 0),
            "attn_fwd_activations_read_bytes": layers_per_stage
            * fwd_attn_layer_stats.get("fwd_attn_activations_read_bytes_per_layer", 0),
            "attn_fwd_activations_written_bytes": layers_per_stage
            * fwd_attn_layer_stats.get("fwd_attn_activations_written_bytes_per_layer", 0),
            "attn_fwd_kv_cache_written_bytes": layers_per_stage
            * fwd_attn_layer_stats.get("fwd_attn_kv_cache_written_bytes_per_layer", 0),
            # Bookkeeping split for attention memory traffic:
            # - Projection: QKV + output projection (weights + activation buffers)
            # - Core: KV-cache attention (activation buffers + KV-cache)
            "attn_fwd_mem_time_proj_s": layers_per_stage
            * (fwd_attn_proj_layer_stats.get("fwd_attn_proj_time_s", 0.0) or 0.0),
            "attn_fwd_mem_time_core_s": layers_per_stage
            * (fwd_attn_core_layer_stats.get("fwd_attn_core_time_s", 0.0) or 0.0),
            "attn_fwd_mem_bytes_proj": layers_per_stage
            * (fwd_attn_proj_layer_stats.get("fwd_attn_proj_total_bytes_per_layer", 0) or 0),
            "attn_fwd_mem_bytes_core": layers_per_stage
            * (fwd_attn_core_layer_stats.get("fwd_attn_core_total_bytes_per_layer", 0) or 0),
            # Forward MoE/FFN per-batch per-chip
            "ffn_moe_fwd_weights_read_bytes": layers_per_stage
            * fwd_moe_layer_stats.get("fwd_moe_weights_read_bytes_per_layer", 0),
            "ffn_moe_fwd_activations_read_bytes": layers_per_stage
            * fwd_moe_layer_stats.get("fwd_moe_activations_read_bytes_per_layer", 0),
            "ffn_moe_fwd_activations_written_bytes": layers_per_stage
            * fwd_moe_layer_stats.get("fwd_moe_activations_written_bytes_per_layer", 0),
        }
    )
    if pass_mode == PassMode.TRAINING:
        bwd_attn_layer_stats = bwd_attn_traffic.get_stats(1)
        bwd_moe_layer_stats = bwd_moe_traffic.get_stats(1)
        # Expose detailed per-layer component stats (including read/write times) for backward
        mem_stats.update(bwd_attn_layer_stats)
        mem_stats.update(bwd_moe_layer_stats)
        mem_stats.update(
            {
                # Backward Attention per-batch per-chip
                "attn_bwd_weights_read_bytes": layers_per_stage
                * bwd_attn_layer_stats.get("bwd_attn_weights_read_bytes_per_layer", 0),
                "attn_bwd_gradients_written_bytes": layers_per_stage
                * bwd_attn_layer_stats.get("bwd_attn_gradients_written_bytes_per_layer", 0),
                "attn_bwd_activations_read_bytes": layers_per_stage
                * bwd_attn_layer_stats.get("bwd_attn_activations_read_bytes_per_layer", 0),
                # Backward MoE/FFN per-batch per-chip
                "ffn_moe_bwd_weights_read_bytes": layers_per_stage
                * bwd_moe_layer_stats.get("bwd_moe_weights_read_bytes_per_layer", 0),
                "ffn_moe_bwd_gradients_written_bytes": layers_per_stage
                * bwd_moe_layer_stats.get("bwd_moe_gradients_written_bytes_per_layer", 0),
                "ffn_moe_bwd_activations_read_bytes": layers_per_stage
                * bwd_moe_layer_stats.get("bwd_moe_activations_read_bytes_per_layer", 0),
            }
        )

        # Additional FFN/MoE backward per-layer breakdown
        d_model = model_spec.d_model
        mlp_ratio = model_spec.mlp_ratio
        n_batch_dp = batch_size_per_dp
        n_seq_val = seq_length
        float_size = workload_spec.float_size_bytes
        total_num_routed_experts = model_spec.num_regular_experts
        tokens_per_set = (
            workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq_val
        )
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        moe_tp_size_val = moe_tp_topology_view.num_chips
        num_colocated = moe_config.num_colocated_experts

        # Per-layer components (will be zero if no routed experts)
        regular_input_size = traffic_mult * n_batch_dp * n_seq_val * d_model * float_size
        attn_output_size = n_batch_dp * n_seq_val * d_model * float_size
        if total_num_routed_experts > 0:
            tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
            token_vol_per_expert = tokens_per_routed_expert * traffic_mult * d_model * float_size
            ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / moe_tp_size_val
        else:
            token_vol_per_expert = 0
            ff1_out_sharded_per_expert = 0

        # Reads per layer
        ffn_bwd_reads_original_input = regular_input_size
        ffn_bwd_reads_attn_output = attn_output_size
        ffn_bwd_reads_ff1_out_sharded = num_colocated * ff1_out_sharded_per_expert
        ffn_bwd_reads_expert_input = num_colocated * token_vol_per_expert
        # Weight reads per layer (both FFN matrices), sharded by TP and scaled by colocated experts
        ffn_bwd_reads_weights = (
            num_colocated * model_spec.expert_params_per_expert * float_size / moe_tp_size_val
        )
        # Gradient writes per layer
        ffn_bwd_grad_writes_expert_input = num_colocated * token_vol_per_expert
        ffn_bwd_grad_writes_ff1_out_sharded = num_colocated * ff1_out_sharded_per_expert

        mem_stats.update(
            {
                "ffn_moe_bwd_detail_original_input_read_bytes": layers_per_stage
                * int(ffn_bwd_reads_original_input),
                "ffn_moe_bwd_detail_attn_output_read_bytes": layers_per_stage
                * int(ffn_bwd_reads_attn_output),
                "ffn_moe_bwd_detail_ff1_out_sharded_read_bytes": layers_per_stage
                * int(ffn_bwd_reads_ff1_out_sharded),
                "ffn_moe_bwd_detail_expert_input_read_bytes": layers_per_stage
                * int(ffn_bwd_reads_expert_input),
                "ffn_moe_bwd_detail_weights_read_bytes": layers_per_stage
                * int(ffn_bwd_reads_weights),
                "ffn_moe_bwd_detail_weight_gradients_written_bytes": layers_per_stage
                * int(ffn_bwd_reads_weights),
                "ffn_moe_bwd_detail_gradients_written_expert_input_bytes": layers_per_stage
                * int(ffn_bwd_grad_writes_expert_input),
                "ffn_moe_bwd_detail_gradients_written_ff1_out_sharded_bytes": layers_per_stage
                * int(ffn_bwd_grad_writes_ff1_out_sharded),
            }
        )

    # Overlap/efficiency settings sourced from workload spec
    comp_comm_efficiency = workload_spec.comp_comm_efficiency
    comp_mem_efficiency = workload_spec.comp_mem_efficiency
    comm_mem_efficiency = workload_spec.comm_mem_efficiency

    # Component-level time breakdowns (forward)
    #
    # These are intended for sweep reporting (BEST markers) where you want to optimize a sub-component
    # (e.g. attention projection) without introducing global toggles.
    attn_proj_comp_time_fwd = float(attn_proj_compute_time_fwd or 0.0)
    attn_core_comp_time_fwd = float(attn_core_compute_time_fwd or 0.0)
    attn_proj_mem_time_fwd = float(mem_stats.get("attn_fwd_mem_time_proj_s", 0.0) or 0.0)
    attn_core_mem_time_fwd = float(mem_stats.get("attn_fwd_mem_time_core_s", 0.0) or 0.0)
    attn_proj_comm_time_fwd = float(
        (tp_attn_fwd_phase1.total_comm_time or 0.0) + (tp_attn_fwd_phase3.total_comm_time or 0.0)
    )
    attn_core_comm_time_fwd = float(tp_attn_fwd_phase2.total_comm_time or 0.0)

    attn_proj_overlapped_fwd = calculate_overlapped_time(
        comp_time=attn_proj_comp_time_fwd,
        comm_time=attn_proj_comm_time_fwd,
        mem_time=attn_proj_mem_time_fwd,
        comp_comm_efficiency=comp_comm_efficiency,
        comp_mem_efficiency=comp_mem_efficiency,
        comm_mem_efficiency=comm_mem_efficiency,
    )
    attn_core_overlapped_fwd = calculate_overlapped_time(
        comp_time=attn_core_comp_time_fwd,
        comm_time=attn_core_comm_time_fwd,
        mem_time=attn_core_mem_time_fwd,
        comp_comm_efficiency=comp_comm_efficiency,
        comp_mem_efficiency=comp_mem_efficiency,
        comm_mem_efficiency=comm_mem_efficiency,
    )

    # FFN/MoE block totals (forward) for reporting
    ffn_comp_time_fwd = float(compute_time_moe_fwd or 0.0)
    ffn_mem_time_fwd = float(
        (fwd_moe_layer_stats.get("fwd_moe_time_s", 0.0) or 0.0) * layers_per_stage
    )
    ffn_comm_time_fwd = float(
        (tp_moe_fwd.total_comm_time or 0.0)
        + (moe_a2a_fwd_1.total_comm_time or 0.0)
        + (moe_a2a_fwd_2.total_comm_time or 0.0)
    )
    ffn_overlapped_fwd = calculate_overlapped_time(
        comp_time=ffn_comp_time_fwd,
        comm_time=ffn_comm_time_fwd,
        mem_time=ffn_mem_time_fwd,
        comp_comm_efficiency=comp_comm_efficiency,
        comp_mem_efficiency=comp_mem_efficiency,
        comm_mem_efficiency=comm_mem_efficiency,
    )

    batch_overlapped_time_forward_pass = calculate_overlapped_time(
        comp_time=batch_compute_time_forward_pass,
        comm_time=batch_comm_time_forward_pass,
        mem_time=batch_mem_time_forward_pass,
        comp_comm_efficiency=comp_comm_efficiency,
        comp_mem_efficiency=comp_mem_efficiency,
        comm_mem_efficiency=comm_mem_efficiency,
    )
    batch_time_to_train_sec_forward_pass = batch_overlapped_time_forward_pass["total_time"]

    # Time-To-First-Token (TTFT) for inference prefill: time for one micro-batch to traverse PP
    ttft_forward_sec = None
    if pass_mode == PassMode.INFERENCE_PREFILL:
        batch_size_per_dp_val = batch_size_per_dp
        micro_batch_size_val = workload_spec.effective_micro_batch_size
        micro_factor = micro_batch_size_val / batch_size_per_dp_val
        micro_compute_time = batch_compute_time_forward_pass * micro_factor
        micro_comm_time = batch_comm_time_forward_pass * micro_factor
        micro_mem_time = batch_mem_time_forward_pass * micro_factor
        micro_overlapped = calculate_overlapped_time(
            comp_time=micro_compute_time,
            comm_time=micro_comm_time,
            mem_time=micro_mem_time,
            comp_comm_efficiency=0,  # comp_comm_efficiency,
            comp_mem_efficiency=0,  # comp_mem_efficiency,
            comm_mem_efficiency=0,  # comm_mem_efficiency,
        )
        micro_time_forward = micro_overlapped["total_time"]
        ttft_forward_sec = micro_time_forward * pp_stages

    batch_overlapped_time_backward_pass = calculate_overlapped_time(
        comp_time=batch_compute_time_backward_pass,
        comm_time=batch_comm_time_backward_pass,
        mem_time=batch_mem_time_backward_pass,
        comp_comm_efficiency=comp_comm_efficiency,
        comp_mem_efficiency=comp_mem_efficiency,
        comm_mem_efficiency=comm_mem_efficiency,
    )

    batch_time_to_train_sec_backward_pass = batch_overlapped_time_backward_pass["total_time"]

    batch_time_to_train_sec = (
        batch_time_to_train_sec_forward_pass + batch_time_to_train_sec_backward_pass
    )

    # Aggregate MoE A2A time (forward + backward) and fraction of total overlapped time
    total_moe_a2a_time_fwd_bwd = (
        moe_a2a_fwd_1.total_comm_time + moe_a2a_fwd_2.total_comm_time + moe_a2a_bwd.total_comm_time
    )
    total_moe_a2a_startup_fwd_bwd = (
        moe_a2a_fwd_1.startup_latency + moe_a2a_fwd_2.startup_latency + moe_a2a_bwd.startup_latency
    )
    total_moe_a2a_data_transfer_fwd_bwd = (
        moe_a2a_fwd_1.transmission_time
        + moe_a2a_fwd_2.transmission_time
        + moe_a2a_bwd.transmission_time
    )
    total_fwd_bwd_time = (
        batch_time_to_train_sec_forward_pass + batch_time_to_train_sec_backward_pass
    )
    moe_a2a_time_fraction_fwd_bwd = (
        total_moe_a2a_time_fwd_bwd / total_fwd_bwd_time if total_fwd_bwd_time > 0 else 0.0
    )

    # Total comm fraction per batch (forward + backward) over total overlapped batch time
    total_comm_time_per_batch = batch_comm_time_forward_pass + batch_comm_time_backward_pass
    comm_fraction_per_batch = (
        total_comm_time_per_batch / total_fwd_bwd_time if total_fwd_bwd_time > 0 else 0.0
    )

    # Non-overlapped (raw) per-batch totals and comm fraction
    total_raw_time_forward_pass = (
        batch_compute_time_forward_pass + batch_comm_time_forward_pass + batch_mem_time_forward_pass
    )
    total_raw_time_backward_pass = (
        batch_compute_time_backward_pass
        + batch_comm_time_backward_pass
        + batch_mem_time_backward_pass
    )
    total_raw_time_fwd_bwd = total_raw_time_forward_pass + total_raw_time_backward_pass
    raw_comm_fraction_per_batch = (
        total_comm_time_per_batch / total_raw_time_fwd_bwd if total_raw_time_fwd_bwd > 0 else 0.0
    )

    # pipeline parallelism bubble
    if pass_mode == PassMode.TRAINING:
        pp_strategy = get_pp_strategy(workload_spec.pp_spec.variant)
        effective_micro_batch_size = batch_sizes.effective_micro_batch_size
        if effective_micro_batch_size <= 0:
            raise ValueError(
                f"effective_micro_batch_size must be > 0, got {effective_micro_batch_size}"
            )
        if batch_size_per_dp % effective_micro_batch_size != 0:
            raise ValueError(
                "batch_size_per_dp must be divisible by effective_micro_batch_size for GPipe. "
                f"Got batch_size_per_dp={batch_size_per_dp}, "
                f"effective_micro_batch_size={effective_micro_batch_size}."
            )
        num_microbatches = batch_size_per_dp // effective_micro_batch_size

        bubble_kwargs = {
            "num_stages": pp_stages,
            "num_microbatches": num_microbatches,
            "batch_time": batch_time_to_train_sec,
        }
        pp_overhead_stats = pp_strategy.backward_comm(**bubble_kwargs)
        batch_time_pp_overhead = pp_overhead_stats.total_comm_time
    else:
        batch_time_pp_overhead = 0.0

    batch_time_to_train_sec += batch_time_pp_overhead

    time_to_train_sec = batch_time_to_train_sec * train_steps
    tokens_per_second = total_train_tokens / time_to_train_sec
    total_xpus_val = attn_tp_topology_view.num_chips * attn_dp_topology_view.num_chips * pp_stages
    tokens_per_second_per_xpu = tokens_per_second / total_xpus_val

    # Decode-only TPOT:
    # - System TPOT: wall-time per *system-level* output token (across the whole batch).
    # - User TPOT: wall-time per *user-visible* output token (per sequence). This depends on
    #   batch size by definition (User TPOT = System TPOT * global_batch_size).
    match pass_mode:
        case PassMode.INFERENCE_DECODE:
            num_decode_steps = workload_spec.num_generated_tokens
            global_batch_size = workload_spec.global_batch_size
            tpot_system_sec = (
                (time_to_train_sec / (num_decode_steps * global_batch_size))
                if (num_decode_steps > 0 and global_batch_size > 0)
                else None
            )
            tpot_user_sec = (
                (tpot_system_sec * global_batch_size) if tpot_system_sec is not None else None
            )
        case _:
            tpot_system_sec = None
            tpot_user_sec = None

    # batch_comm_vol_forward_pass = batch_forward_pass_comm_stats.comm_vol
    # batch_comm_vol_backward_pass = batch_backward_pass_comm_stats.comm_vol

    tp_node = tp_topology_view.physical_topology.node_definition
    stats = {
        plotting.tp_node_label: tp_node.node_name,
        plotting.tp_xpu_count_label: attn_tp_topology_view.num_chips,
        plotting.dp_ranks_label: attn_dp_topology_view.num_chips,
        plotting.high_bandwidth_xpu_count_label: attn_tp_topology_view.num_chips
        * moe_config.attn_domains_per_full_expert_set,
        plotting.low_bandwidth_ranks_label: attn_dp_topology_view.num_chips
        // moe_config.attn_domains_per_full_expert_set,
        plotting.total_xpu_count_label: total_xpus_val,
        plotting.batch_size_per_dp_label: batch_size_per_dp,
        plotting.shared_expert_count_label: model_spec.num_shared_experts,
        plotting.regular_expert_count_label: model_spec.num_regular_experts,
        plotting.shared_experts_per_dp_rank_label: moe_config.num_shared_experts_per_dp_rank,
        plotting.regular_experts_per_dp_rank_label: moe_config.num_regular_experts_per_dp_rank,
        plotting.num_selected_experts_label: model_spec.num_selected_experts,
        plotting.num_experts_per_dp_rank_label: moe_config.num_all_experts_per_dp_rank,
        plotting.attn_domains_per_full_expert_set_label: moe_config.attn_domains_per_full_expert_set,
        plotting.num_complete_expert_sets_label: moe_config.num_complete_expert_sets,
        # Expert activation cardinality diagnostics — see
        # workload_spec.ExpertWeightResidencyPolicy for definitions.
        # Computed unconditionally so that switching policies produces auditable diffs.
        **_expert_activation_diagnostics(
            workload_spec=workload_spec,
            model_spec=model_spec,
        ),
        plotting.optimizer_label: getattr(workload_spec, "optimizer", "N/A"),
        plotting.float_size_label: workload_spec.float_size_bytes,
        plotting.mlp_ratio_label: model_spec.mlp_ratio,
        plotting.mlp_kind_label: model_spec.mlp_kind.value,
        plotting.topology_dims_label: attn_tp_topology_view.shape,
        # Backward-compatible single label; encode attn/moe variants explicitly.
        plotting.tensor_parallel_variant_label: f"{attn_tp_variant.name}|{moe_tp_variant.name}",
        plotting.n_layers_label: num_layers_per_pp_stage,
        plotting.pp_stages_label: pp_stages,
        plotting.seq_len_label: seq_length,
        plotting.model_hidden_dim_label: d_model,
        plotting.time_to_train_sec_label: time_to_train_sec,
        plotting.time_to_train_day_label: time_to_train_sec / (3600 * 24),
        plotting.tokens_per_second_label: tokens_per_second,
        plotting.tokens_per_second_per_xpu_label: tokens_per_second_per_xpu,
        plotting.tpot_system_sec_label: tpot_system_sec,
        plotting.tpot_user_sec_label: tpot_user_sec,
        plotting.ttft_forward_sec_label: ttft_forward_sec,
        plotting.flops_forward_pass_label: batch_flops_forward_pass,
        plotting.flops_backward_pass_label: batch_flops_backward_pass,
        plotting.compute_time_forward_pass_label: batch_compute_time_forward_pass,
        plotting.compute_time_backward_pass_label: batch_compute_time_backward_pass,
        # Embedding/Unembedding compute breakdown
        plotting.compute_time_embedding_fwd_label: embedding_compute_time,
        plotting.compute_time_unembedding_fwd_label: unembedding_compute_time,
        plotting.mem_forward_pass_label: batch_mem_forward_pass,
        plotting.mem_backward_pass_label: batch_mem_backward_pass,
        plotting.mem_time_forward_pass_label: batch_mem_time_forward_pass,
        plotting.mem_time_backward_pass_label: batch_mem_time_backward_pass,
        # Embedding/Unembedding memory breakdown
        plotting.embedding_mem_time_fwd_label: embedding_mem_time,
        plotting.embedding_mem_vol_fwd_label: embedding_mem_vol,
        plotting.embedding_mem_time_bwd_label: embedding_mem_time_bwd,
        plotting.embedding_mem_vol_bwd_label: embedding_mem_vol_bwd,
        plotting.unembedding_mem_time_fwd_label: unembedding_mem_time,
        plotting.unembedding_mem_vol_fwd_label: unembedding_mem_vol,
        plotting.unembedding_mem_time_bwd_label: unembedding_mem_time_bwd,
        plotting.unembedding_mem_vol_bwd_label: unembedding_mem_vol_bwd,
        plotting.comm_vol_forward_pass_label: batch_comm_vol_forward_pass,
        plotting.comm_vol_backward_pass_label: batch_comm_vol_backward_pass,
        plotting.comm_time_forward_pass_label: batch_comm_time_forward_pass,
        plotting.comm_time_backward_pass_label: batch_comm_time_backward_pass,
        # Unembedding comm omitted (typical vocab-parallel path)
        # Detailed comm stats
        plotting.comm_vol_tp_attn_fwd_label: tp_attn_fwd.comm_vol,
        plotting.comm_vol_tp_attn_fwd_phase1_label: tp_attn_fwd_phase1.comm_vol,
        plotting.comm_vol_tp_attn_fwd_phase2_label: tp_attn_fwd_phase2.comm_vol,
        plotting.comm_vol_tp_attn_fwd_phase3_label: tp_attn_fwd_phase3.comm_vol,
        plotting.comm_vol_tp_moe_fwd_label: tp_moe_fwd.comm_vol,
        plotting.comm_vol_tp_attn_bwd_label: tp_attn_bwd.comm_vol,
        plotting.comm_vol_tp_moe_bwd_label: tp_moe_bwd.comm_vol,
        plotting.comm_vol_dp_attn_fwd_label: dp_attn_fwd.comm_vol,
        plotting.comm_vol_dp_moe_fwd_label: dp_moe_fwd.comm_vol,
        plotting.comm_vol_dp_attn_bwd_label: dp_attn_bwd.comm_vol,
        plotting.comm_vol_dp_moe_bwd_label: dp_moe_bwd.comm_vol,
        plotting.comm_vol_pp_fwd_label: pp_fwd.comm_vol,
        plotting.comm_vol_pp_bwd_label: pp_bwd.comm_vol,
        plotting.comm_time_tp_attn_fwd_label: tp_attn_fwd.total_comm_time,
        plotting.comm_time_tp_attn_fwd_phase1_label: tp_attn_fwd_phase1.total_comm_time,
        plotting.comm_time_tp_attn_fwd_phase2_label: tp_attn_fwd_phase2.total_comm_time,
        plotting.comm_time_tp_attn_fwd_phase3_label: tp_attn_fwd_phase3.total_comm_time,
        plotting.comm_time_tp_moe_fwd_label: tp_moe_fwd.total_comm_time,
        plotting.comm_time_tp_attn_bwd_label: tp_attn_bwd.total_comm_time,
        plotting.comm_time_tp_moe_bwd_label: tp_moe_bwd.total_comm_time,
        plotting.comm_time_dp_attn_fwd_label: dp_attn_fwd.total_comm_time,
        plotting.comm_time_dp_moe_fwd_label: dp_moe_fwd.total_comm_time,
        plotting.comm_time_dp_attn_bwd_label: dp_attn_bwd.total_comm_time,
        plotting.comm_time_dp_moe_bwd_label: dp_moe_bwd.total_comm_time,
        plotting.comm_time_pp_fwd_label: pp_fwd.total_comm_time,
        plotting.comm_time_pp_bwd_label: pp_bwd.total_comm_time,
        # Backward MoE all-to-all
        plotting.moe_alltoall_bwd_time_label: moe_a2a_bwd.total_comm_time,
        plotting.moe_alltoall_bwd_vol_label: moe_a2a_bwd.comm_vol,
        "Startup_Phase_Count_TP_MoE_Fwd": startup_phase_count_tp_moe_fwd,
        "Startup_Phase_Count_MoE_A2A_To_Fwd": startup_phase_count_a2a_to_fwd,
        "Startup_Phase_Count_MoE_A2A_From_Fwd": startup_phase_count_a2a_from_fwd,
        "Startup_Phase_Count_MoE_A2A_Bwd": startup_phase_count_a2a_bwd,
        "Startup_Phase_Count_MoE_A2A_Total_FwdBwd": (
            startup_phase_count_a2a_to_fwd
            + startup_phase_count_a2a_from_fwd
            + startup_phase_count_a2a_bwd
        ),
        "Startup_Phase_Count_Total_Comm_Per_Batch": startup_phase_count_total_comm_per_batch,
        # MoE A2A forward+backward aggregate and fraction of total overlapped time
        "MoE_A2A_Total_Time_FwdBwd": total_moe_a2a_time_fwd_bwd,
        "MoE_A2A_Startup_Latency_FwdBwd": total_moe_a2a_startup_fwd_bwd,
        "MoE_A2A_Data_Transfer_Time_FwdBwd": total_moe_a2a_data_transfer_fwd_bwd,
        "FwdBwd_Total_Time": total_fwd_bwd_time,
        "MoE_A2A_Time_Fraction_FwdBwd": moe_a2a_time_fraction_fwd_bwd,
        # Communication fraction over overlapped batch time
        "Total_Comm_Time_Per_Batch": total_comm_time_per_batch,
        "Total_Batch_Time": total_fwd_bwd_time,
        "Comm_Fraction_Per_Batch": comm_fraction_per_batch,
        # Raw (non-overlapped) totals and ratios
        "Raw_Total_Time_Per_Batch": total_raw_time_fwd_bwd,
        "Raw_Comm_Fraction_Per_Batch": raw_comm_fraction_per_batch,
        # Detailed compute stats
        plotting.flops_attn_fwd_label: attn_flops_forward_pass,
        plotting.flops_moe_fwd_label: moe_flops_forward_pass,
        plotting.flops_embedding_fwd_label: embedding_flops,
        plotting.flops_unembedding_fwd_label: unembedding_flops,
        plotting.flops_attn_bwd_label: attn_flops_backward_pass,
        plotting.flops_moe_bwd_label: moe_flops_backward_pass,
        plotting.compute_time_attn_fwd_label: compute_time_attn_fwd,
        plotting.compute_time_moe_fwd_label: compute_time_moe_fwd,
        plotting.compute_time_attn_bwd_label: compute_time_attn_bwd,
        plotting.compute_time_moe_bwd_label: compute_time_moe_bwd,
        # Detailed memory stats
        plotting.mem_vol_attn_fwd_label: fwd_attn_traffic.total(),
        plotting.mem_vol_moe_fwd_label: fwd_moe_traffic.total(),
        plotting.mem_vol_attn_bwd_label: bwd_attn_traffic.total(),
        plotting.mem_vol_moe_bwd_label: bwd_moe_traffic.total(),
        plotting.mem_time_attn_fwd_label: fwd_attn_traffic.get_stats(1)[
            f"{fwd_attn_traffic.name}_time_s"
        ],
        plotting.mem_time_moe_fwd_label: fwd_moe_traffic.get_stats(1)[
            f"{fwd_moe_traffic.name}_time_s"
        ],
        plotting.mem_time_attn_bwd_label: bwd_attn_traffic.get_stats(1)[
            f"{bwd_attn_traffic.name}_time_s"
        ],
        plotting.mem_time_moe_bwd_label: bwd_moe_traffic.get_stats(1)[
            f"{bwd_moe_traffic.name}_time_s"
        ],
        plotting.moe_to_experts_vol_label: moe_a2a_fwd_1.comm_vol,
        plotting.moe_from_experts_vol_label: moe_a2a_fwd_2.comm_vol,
        plotting.moe_to_experts_time_label: moe_a2a_fwd_1.total_comm_time,
        plotting.moe_from_experts_time_label: moe_a2a_fwd_2.total_comm_time,
        # --- Component-level timing breakdowns (forward) ---
        # Attention projection (QKV + output projection)
        "Attn_Proj_Fwd_Compute_Time_Sec": attn_proj_comp_time_fwd,
        "Attn_Proj_Fwd_Mem_Time_Sec": attn_proj_mem_time_fwd,
        "Attn_Proj_Fwd_Comm_Time_Sec": attn_proj_comm_time_fwd,
        "Attn_Proj_Fwd_Total_Time_Overlapped_Sec": attn_proj_overlapped_fwd["total_time"],
        # Attention core (KV-cache attention)
        "Attn_Core_Fwd_Compute_Time_Sec": attn_core_comp_time_fwd,
        "Attn_Core_Fwd_Mem_Time_Sec": attn_core_mem_time_fwd,
        "Attn_Core_Fwd_Comm_Time_Sec": attn_core_comm_time_fwd,
        "Attn_Core_Fwd_Total_Time_Overlapped_Sec": attn_core_overlapped_fwd["total_time"],
        # FFN/MoE block
        "FFN_Fwd_Compute_Time_Sec": ffn_comp_time_fwd,
        "FFN_Fwd_Mem_Time_Sec": ffn_mem_time_fwd,
        "FFN_Fwd_Comm_Time_Sec": ffn_comm_time_fwd,
        "FFN_Fwd_Total_Time_Overlapped_Sec": ffn_overlapped_fwd["total_time"],
    }

    # Per-component comm timing triplets.
    # Base keys now consistently represent total communication time.
    comm_components = {
        plotting.comm_time_label: CommStats(
            startup_latency=batch_comm_startup_forward_pass + batch_comm_startup_backward_pass,
            transmission_time=batch_comm_data_transfer_forward_pass
            + batch_comm_data_transfer_backward_pass,
        ),
        plotting.comm_time_forward_pass_label: CommStats(
            startup_latency=batch_comm_startup_forward_pass,
            transmission_time=batch_comm_data_transfer_forward_pass,
        ),
        plotting.comm_time_backward_pass_label: CommStats(
            startup_latency=batch_comm_startup_backward_pass,
            transmission_time=batch_comm_data_transfer_backward_pass,
        ),
        plotting.comm_time_tp_attn_fwd_label: tp_attn_fwd,
        plotting.comm_time_tp_attn_fwd_phase1_label: tp_attn_fwd_phase1,
        plotting.comm_time_tp_attn_fwd_phase2_label: tp_attn_fwd_phase2,
        plotting.comm_time_tp_attn_fwd_phase3_label: tp_attn_fwd_phase3,
        plotting.comm_time_tp_moe_fwd_label: tp_moe_fwd,
        plotting.comm_time_tp_attn_bwd_label: tp_attn_bwd,
        plotting.comm_time_tp_moe_bwd_label: tp_moe_bwd,
        plotting.comm_time_dp_attn_fwd_label: dp_attn_fwd,
        plotting.comm_time_dp_moe_fwd_label: dp_moe_fwd,
        plotting.comm_time_dp_attn_bwd_label: dp_attn_bwd,
        plotting.comm_time_dp_moe_bwd_label: dp_moe_bwd,
        plotting.comm_time_pp_fwd_label: pp_fwd,
        plotting.comm_time_pp_bwd_label: pp_bwd,
        plotting.moe_to_experts_time_label: moe_a2a_fwd_1,
        plotting.moe_from_experts_time_label: moe_a2a_fwd_2,
        plotting.moe_alltoall_bwd_time_label: moe_a2a_bwd,
    }
    for label, comm in comm_components.items():
        stats[f"{label}_Total"] = comm.total_comm_time
        stats[f"{label}_Startup_Latency"] = comm.startup_latency
        stats[f"{label}_Data_Transfer"] = comm.transmission_time

    stats.update(mem_stats)

    # --- Base (non-swept) config fields for reporting ---
    #
    # The sweep `report:` extracts columns from `structure_run_results()` output, which is built
    # from the DataFrame row. If a field is not part of the sweep_case (i.e., not explicitly swept),
    # it won't appear in summary.csv unless we surface it here.
    stats["base.model_spec.num_heads"] = model_spec.num_heads
    stats["base.model_spec.num_kv_heads"] = model_spec.num_kv_heads
    stats["base.workload_spec.comp_comm_efficiency"] = workload_spec.comp_comm_efficiency
    stats["base.workload_spec.comp_mem_efficiency"] = workload_spec.comp_mem_efficiency
    stats["base.workload_spec.comm_mem_efficiency"] = workload_spec.comm_mem_efficiency
    # Decode-only knobs (surface even when not explicitly swept).
    stats["base.workload_spec.decode_tp2d_kv_comm_mode"] = getattr(
        workload_spec, "decode_tp2d_kv_comm_mode", None
    )
    stats["base.workload_spec.decode_moe_router_sync_mode"] = getattr(
        workload_spec, "decode_moe_router_sync_mode", None
    )
    stats["base.workload_spec.tp_moe_a2a_replication_optimization"] = getattr(
        workload_spec, "tp_moe_a2a_replication_optimization", None
    )
    stats["base.workload_spec.moe_a2a_comm_mode"] = getattr(
        workload_spec, "moe_a2a_comm_mode", None
    )

    return stats


def run_sweep(
    sweep_fn,
    base_specs,
    sweep_cfg: SweepConfig,
    pass_mode,
    *,
    enable_decode_layout_trace: bool = True,
):

    # Records are assembled in flight: completed rows are flushed from a dict
    # buffer into compact columnar chunks every RECORD_CHUNK_SIZE points and the
    # raw dicts are freed, so we never hold the full list-of-dicts (the dominant
    # memory peak for large sweeps) alongside the final DataFrame. Chunks are
    # concatenated once at the end.
    RECORD_CHUNK_SIZE = 20_000
    record_frames: list[pd.DataFrame] = []
    record_buffer: list[dict] = []
    first_record_keys: list[str] = []

    def _flush_record_buffer() -> None:
        if record_buffer:
            record_frames.append(pd.DataFrame.from_records(record_buffer))
            record_buffer.clear()

    # Expansion runs every verifier on every cartesian combo before the first
    # point is yielded; for large 2D grids this is a long, silent phase, so we
    # label it and report how long it took once the point count is known.
    print(
        "\nExpanding sweep grid (running verifiers over all combos; "
        "large 2D grids can take a while)...",
        flush=True,
    )
    expand_start = time.perf_counter()
    # The expansion is eager (every candidate combo is verified before the
    # first point is yielded), so a determinate bar is impossible — the total
    # combo count is unknown until it finishes. We surface a live count of
    # verified candidate combos via a callback instead.
    expand_bar = tqdm(
        desc="Expanding grid",
        unit="combo",
        dynamic_ncols=True,
        leave=False,
    )
    points = list(
        generate_cases(
            cfg=sweep_cfg,
            base_specs=base_specs,
            progress_callback=lambda: expand_bar.update(1),
        )
    )
    combos_evaluated = expand_bar.n
    expand_bar.close()
    total_points = len(points)
    print(
        f"Sweep points to evaluate: {total_points} "
        f"(evaluated {combos_evaluated} candidate combos in "
        f"{time.perf_counter() - expand_start:.1f}s)"
    )
    if total_points == 0:
        raise ValueError("Sweep expanded to zero points; check sweep filters/verifiers.")

    sweep_iter = tqdm(
        points,
        total=total_points,
        desc="Sweep progress",
        unit="pt",
        dynamic_ncols=True,
    )
    for i, pt in enumerate(sweep_iter):
        case = pt.case
        current_specs = apply_case_to_specs(case, base_specs=base_specs)
        model_spec = current_specs["model_spec"]
        workload_spec = current_specs["workload_spec"]
        logical_topology = current_specs["logical_topology"]
        physical_topology = current_specs["physical_topology"]

        try:
            # Attach layers_per_pipeline_stage to workload_spec dynamically; keep model_spec unchanged
            pp_stages = logical_topology.get_mapping_for_tag("PP").size
            workload_with_layers = workload_spec.with_layers_per_pp_stage(
                total_layers=model_spec.n_layers, pp_stages=pp_stages
            )

            stats = sweep_fn(
                physical_topology,
                logical_topology,
                model_spec,
                workload_with_layers,
                pass_mode,
                enable_decode_layout_trace=enable_decode_layout_trace,
            )
            # Defensive: ensure we have a plain dict so pandas treats this as a record and we can
            # reliably add sweep/label fields.
            if not isinstance(stats, dict):
                try:
                    stats = dict(stats)  # type: ignore[arg-type]
                except Exception as e:
                    raise TypeError(
                        f"sweep_fn returned non-dict stats of type {type(stats).__name__}, "
                        f"and it could not be converted to dict: {e}"
                    ) from e
            # include sweep params as flat columns
            for k, v in case.items():
                stats[f"sweep.{k}"] = v
            # attach labels (and a flattened view for convenient summary filtering)
            stats["labels"] = dict(pt.labels)
            for lk, lv in pt.labels.items():
                stats[f"label.{lk}"] = lv

            if not first_record_keys and isinstance(stats, dict):
                first_record_keys = list(stats.keys())
            record_buffer.append(stats)
            if len(record_buffer) >= RECORD_CHUNK_SIZE:
                _flush_record_buffer()
        except Exception as e:
            print(f"Error in sweep iteration {i}: {e}")
            sweep_iter.close()
            raise
            # continue

    sweep_iter.close()

    _flush_record_buffer()
    if not record_frames:
        raise ValueError("Sweep produced zero result records.")
    print(
        f"Assembling results dataframe ({total_points} rows, " f"{len(record_frames)} chunk(s))...",
        flush=True,
    )
    assemble_start = time.perf_counter()
    df = (
        record_frames[0] if len(record_frames) == 1 else pd.concat(record_frames, ignore_index=True)
    )
    # Drop chunk references to free memory, but keep `record_frames` bound: it is
    # captured by the `_flush_record_buffer` closure, and `del` would make that
    # free variable undefined.
    record_frames.clear()
    print(f"Dataframe assembled in {time.perf_counter() - assemble_start:.1f}s")
    # TODO(MB) we need to add all-reduce flops here as well
    required = [
        plotting.compute_time_forward_pass_label,
        plotting.compute_time_backward_pass_label,
        plotting.mem_time_forward_pass_label,
        plotting.mem_time_backward_pass_label,
        plotting.comm_time_forward_pass_label,
        plotting.comm_time_backward_pass_label,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sample_keys = first_record_keys[:30]
        raise KeyError(
            "Sweep results are missing required flat columns needed for derived metrics. "
            f"Missing={missing}. Sample keys (first record)={sample_keys}"
        )
    compute_time = (
        df[plotting.compute_time_forward_pass_label] + df[plotting.compute_time_backward_pass_label]
    )
    mem_time = df[plotting.mem_time_forward_pass_label] + df[plotting.mem_time_backward_pass_label]
    comm_time = (
        df[plotting.comm_time_forward_pass_label] + df[plotting.comm_time_backward_pass_label]
    )
    # Join all derived columns at once via concat(axis=1). The assembled frame
    # has many columns (>100 blocks), so per-column inserts (incl. df.assign,
    # which sets columns one at a time) trip pandas' "highly fragmented"
    # PerformanceWarning. concat(axis=1) never calls insert and returns a
    # consolidated (de-fragmented) frame.
    derived = pd.DataFrame(
        {
            plotting.compute_time_label: compute_time,
            plotting.memory_time_label: mem_time,
            plotting.comm_time_label: comm_time,
            plotting.comm_compute_ratio_label: comm_time / compute_time,
            plotting.memory_compute_ratio_label: mem_time / compute_time,
        },
        index=df.index,
    )
    df = pd.concat([df, derived], axis=1)
    return df


def build_results_df(
    base_specs, sweep_cfg: SweepConfig, pass_mode, *, enable_decode_layout_trace: bool = True
):
    df = run_sweep(
        get_model_stats,
        base_specs,
        sweep_cfg,
        pass_mode=pass_mode,
        enable_decode_layout_trace=enable_decode_layout_trace,
    )
    # Join all derived columns at once via concat(axis=1); avoids the per-insert
    # "DataFrame is highly fragmented" PerformanceWarning and returns a
    # consolidated frame (see run_sweep for details).
    comm_compute_ratio = df["Comm_Compute_Ratio"]
    derived = pd.DataFrame(
        {
            "Comm_Ineff": np.where(comm_compute_ratio > 1.0, 1, 0.1),
            "Comp_Util": np.where(comm_compute_ratio > 1.0, 1.0 / comm_compute_ratio, 1),
            "Comp_Comm_Ratio": 1.0 / comm_compute_ratio,
            "Num_Tokens_Per_Batch_Per_TP_Group": (
                df[plotting.seq_len_label] * df[plotting.batch_size_per_dp_label]
            ),
        },
        index=df.index,
    )
    df = pd.concat([df, derived], axis=1)
    return df


def get_detailed_mem_stats(
    attn_tp_topology_view,
    proj_tp_topology_view,
    moe_tp_topology_view,
    model_spec,
    moe_config,
    workload_spec,
    attn_tp_variant,
    moe_tp_variant,
    attn_x_dims,
    attn_yz_dims,
    proj_x_dims,
    proj_yz_dims,
    moe_x_dims,
    moe_yz_dims,
    pass_mode,
    activation_checkpointing,
    device_for,
    node,
):
    """Helper to get detailed attn/moe memory traffic."""

    # Projections may be modeled on a different TP domain (e.g., projection_groups).
    # The caller resolves the projection TP view; no fallback here.
    if proj_tp_topology_view is None:
        raise ValueError("get_detailed_mem_stats requires a resolved proj_tp_topology_view.")

    def _prod_from_shape(shape, indices):
        p = 1
        for idx in indices:
            p *= shape[idx]
        return p

    # Forward Attention (bookkeeping split: projection vs core)
    fwd_attn_proj_traffic = LayerMemoryTraffic("fwd_attn_proj", device_for, node)
    fwd_attn_core_traffic = LayerMemoryTraffic("fwd_attn_core", device_for, node)
    fwd_attn_traffic = LayerMemoryTraffic("fwd_attn", device_for, node)
    match attn_tp_variant:
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            if attn_x_dims is None or attn_yz_dims is None:
                raise ValueError("TP-2D attn requires attn_x_dims and attn_yz_dims to be defined.")
            shape_attn = attn_tp_topology_view.shape
            tp_x_attn = _prod_from_shape(shape_attn, attn_x_dims)
            tp_yz_attn = _prod_from_shape(shape_attn, attn_yz_dims)

            if proj_x_dims is None or proj_yz_dims is None:
                raise ValueError("TP-2D proj requires proj_x_dims and proj_yz_dims to be defined.")
            shape_proj = proj_tp_topology_view.shape
            tp_x_proj = _prod_from_shape(shape_proj, proj_x_dims)
            tp_yz_proj = _prod_from_shape(shape_proj, proj_yz_dims)
            match pass_mode:
                case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
                    _forward_pass_prefill_attention_projection_tp_2d_memory_traffic(
                        traffic=fwd_attn_proj_traffic,
                        transformer_config=model_spec,
                        n_seq=workload_spec.seq_length,
                        proj_tp_size_x=tp_x_proj,
                        proj_tp_size_yz=tp_yz_proj,
                        attn_tp_size_x=tp_x_attn,
                        attn_tp_size_yz=tp_yz_attn,
                        workload_spec=workload_spec,
                    )
                    _forward_pass_prefill_attention_attn_tp_2d_memory_traffic(
                        traffic=fwd_attn_core_traffic,
                        transformer_config=model_spec,
                        n_seq=workload_spec.seq_length,
                        proj_tp_size_x=tp_x_proj,
                        proj_tp_size_yz=tp_yz_proj,
                        attn_tp_size_x=tp_x_attn,
                        attn_tp_size_yz=tp_yz_attn,
                        save_kv_cache=(pass_mode == PassMode.INFERENCE_PREFILL),
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_DECODE:
                    _decode_attention_projection_tp_2d_memory_traffic(
                        traffic=fwd_attn_proj_traffic,
                        transformer_config=model_spec,
                        proj_tp_size_x=tp_x_proj,
                        proj_tp_size_yz=tp_yz_proj,
                        workload_spec=workload_spec,
                    )
                    _decode_attention_attn_tp_2d_memory_traffic(
                        traffic=fwd_attn_core_traffic,
                        transformer_config=model_spec,
                        n_seq_so_far=workload_spec.prefill_seq_len,
                        proj_tp_size_x=tp_x_proj,
                        proj_tp_size_yz=tp_yz_proj,
                        attn_tp_size_x=tp_x_attn,
                        attn_tp_size_yz=tp_yz_attn,
                        workload_spec=workload_spec,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported pass_mode for TP-2D attn detailed traffic: {pass_mode}"
                    )
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            match pass_mode:
                case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
                    _forward_pass_prefill_attention_projection_tp_1d_memory_traffic(
                        traffic=fwd_attn_proj_traffic,
                        transformer_config=model_spec,
                        n_seq=workload_spec.seq_length,
                        proj_tp_size=proj_tp_topology_view.num_chips,
                        workload_spec=workload_spec,
                    )
                    _forward_pass_prefill_attention_attn_tp_1d_memory_traffic(
                        traffic=fwd_attn_core_traffic,
                        transformer_config=model_spec,
                        n_seq=workload_spec.seq_length,
                        proj_tp_size=proj_tp_topology_view.num_chips,
                        attn_tp_size=attn_tp_topology_view.num_chips,
                        save_kv_cache=(pass_mode == PassMode.INFERENCE_PREFILL),
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_DECODE:
                    _decode_attention_projection_tp_1d_memory_traffic(
                        traffic=fwd_attn_proj_traffic,
                        transformer_config=model_spec,
                        proj_tp_size=proj_tp_topology_view.num_chips,
                        workload_spec=workload_spec,
                    )
                    _decode_attention_attn_tp_1d_memory_traffic(
                        traffic=fwd_attn_core_traffic,
                        transformer_config=model_spec,
                        n_seq_so_far=workload_spec.prefill_seq_len,
                        proj_tp_size=proj_tp_topology_view.num_chips,
                        attn_tp_size=attn_tp_topology_view.num_chips,
                        workload_spec=workload_spec,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported pass_mode for 1D attn detailed traffic: {pass_mode}"
                    )
        case _:
            raise ValueError(
                f"Unsupported attention TP variant in detailed traffic: {attn_tp_variant}"
            )

    # Combine the projection/core bookkeeping split into the total "fwd_attn" traffic.
    fwd_attn_traffic = fwd_attn_proj_traffic + fwd_attn_core_traffic

    fwd_moe_traffic = LayerMemoryTraffic("fwd_moe", device_for, node)
    match moe_tp_variant:
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            if moe_x_dims is None or moe_yz_dims is None:
                raise ValueError("TP-2D MoE requires moe_x_dims and moe_yz_dims to be defined.")
            shape = moe_tp_topology_view.shape
            tp_x = _prod_from_shape(shape, moe_x_dims)
            tp_yz = _prod_from_shape(shape, moe_yz_dims)
            moe_seq = workload_spec.seq_length if pass_mode != PassMode.INFERENCE_DECODE else 1
            _forward_pass_prefill_moe_ffn_tp_2d_memory_traffic(
                fwd_moe_traffic,
                model_spec,
                moe_seq,
                moe_tp_size_x=tp_x,
                moe_tp_size_yz=tp_yz,
                workload_spec=workload_spec,
                moe_config=moe_config,
            )
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            moe_seq = workload_spec.seq_length if pass_mode != PassMode.INFERENCE_DECODE else 1
            _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic(
                fwd_moe_traffic,
                model_spec,
                moe_seq,
                moe_config,
                moe_tp_topology_view.num_chips,
                workload_spec,
            )
        case _:
            raise ValueError(f"Unsupported MoE TP variant in detailed traffic: {moe_tp_variant}")

    bwd_attn_traffic = LayerMemoryTraffic("bwd_attn", device_for, node)
    bwd_moe_traffic = LayerMemoryTraffic("bwd_moe", device_for, node)
    if pass_mode == PassMode.TRAINING:
        match attn_tp_variant:
            case TensorParallelVariant.SHARD_WEIGHTS_2D:
                if attn_x_dims is None or attn_yz_dims is None:
                    raise ValueError(
                        "TP-2D attn requires attn_x_dims and attn_yz_dims to be defined."
                    )
                if proj_x_dims is None or proj_yz_dims is None:
                    raise ValueError(
                        "TP-2D proj requires proj_x_dims and proj_yz_dims to be defined."
                    )
                shape = attn_tp_topology_view.shape
                tp_x = _prod_from_shape(shape, attn_x_dims)
                tp_yz = _prod_from_shape(shape, attn_yz_dims)
                shape_proj = proj_tp_topology_view.shape
                tp_x_proj_bwd = _prod_from_shape(shape_proj, proj_x_dims)
                tp_yz_proj_bwd = _prod_from_shape(shape_proj, proj_yz_dims)
                bwd_attn_traffic = _backward_attn_tp_2d_memory_traffic(
                    name="bwd_attn_2d",
                    transformer_config=model_spec,
                    workload_spec=workload_spec,
                    proj_tp_size_x=tp_x_proj_bwd,
                    proj_tp_size_yz=tp_yz_proj_bwd,
                    attn_tp_size_x=tp_x,
                    attn_tp_size_yz=tp_yz,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
                bwd_attn_traffic = _backward_attn_tp_1d_memory_traffic(
                    name="bwd_attn",
                    transformer_config=model_spec,
                    workload_spec=workload_spec,
                    proj_tp_size=proj_tp_topology_view.num_chips,
                    attn_tp_size=attn_tp_topology_view.num_chips,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case _:
                raise ValueError(
                    f"Unsupported attention TP variant in detailed backward: {attn_tp_variant}"
                )

        match moe_tp_variant:
            case TensorParallelVariant.SHARD_WEIGHTS_2D:
                if moe_x_dims is None or moe_yz_dims is None:
                    raise ValueError("TP-2D MoE requires moe_x_dims and moe_yz_dims to be defined.")
                shape = moe_tp_topology_view.shape
                tp_x = _prod_from_shape(shape, moe_x_dims)
                tp_yz = _prod_from_shape(shape, moe_yz_dims)
                bwd_moe_traffic = _backward_moe_tp_2d_memory_traffic(
                    name="bwd_moe_2d",
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    workload_spec=workload_spec,
                    moe_tp_size_x=tp_x,
                    moe_tp_size_yz=tp_yz,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
                bwd_moe_traffic = _backward_moe_tp_1d_memory_traffic(
                    name="bwd_moe",
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    moe_tp_topology=moe_tp_topology_view,
                    workload_spec=workload_spec,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case _:
                raise ValueError(
                    f"Unsupported MoE TP variant in detailed backward: {moe_tp_variant}"
                )

    return (
        fwd_attn_traffic,
        fwd_attn_proj_traffic,
        fwd_attn_core_traffic,
        fwd_moe_traffic,
        bwd_attn_traffic,
        bwd_moe_traffic,
    )


def structure_run_results(row, human_readable=False):
    """Converts a DataFrame row into a structured dictionary for detailed output."""

    def get_val(label):
        val = row.get(label, None)
        if isinstance(val, Enum):
            return val.value
        return val

    # Conditionally format values
    def fmt_b(val):
        return format_bytes(val) if human_readable else val

    def fmt_f(val):
        return format_flops(val) if human_readable else val

    def fmt_t(val):
        return format_time(val) if human_readable else val

    # Build the structured dictionary
    # Pre-compute per-layer and per-PP-stage memory breakdowns where desired
    num_layers_per_stage = get_val(plotting.n_layers_label)
    if num_layers_per_stage is None or num_layers_per_stage <= 0:
        raise ValueError(
            f"Missing or invalid `{plotting.n_layers_label}` in results row: {num_layers_per_stage}"
        )

    def comm_seconds_triplet(label):
        total_s = get_val(f"{label}_Total")
        if total_s is None:
            total_s = get_val(label)
        startup_s = get_val(f"{label}_Startup_Latency")
        data_transfer_s = get_val(f"{label}_Data_Transfer")
        return total_s, startup_s, data_transfer_s

    def comm_triplet(label):
        total_s, startup_s, data_transfer_s = comm_seconds_triplet(label)
        return {
            "total_time": fmt_t(total_s),
            "total_time_seconds": total_s,
            "startup_latency": fmt_t(startup_s),
            "startup_latency_seconds": startup_s,
            "data_transfer_time": fmt_t(data_transfer_s),
            "data_transfer_time_seconds": data_transfer_s,
        }

    def comm_triplet_per_layer(label):
        total_s, startup_s, data_transfer_s = comm_seconds_triplet(label)
        return {
            "total_time": fmt_t((total_s or 0.0) / num_layers_per_stage),
            "total_time_seconds": ((total_s or 0.0) / num_layers_per_stage),
            "startup_latency": fmt_t((startup_s or 0.0) / num_layers_per_stage),
            "startup_latency_seconds": ((startup_s or 0.0) / num_layers_per_stage),
            "data_transfer_time": fmt_t((data_transfer_s or 0.0) / num_layers_per_stage),
            "data_transfer_time_seconds": ((data_transfer_s or 0.0) / num_layers_per_stage),
        }

    attn_time_per_layer = get_val(plotting.mem_time_attn_fwd_label)
    attn_time_per_stage = (attn_time_per_layer or 0.0) * (num_layers_per_stage or 0)
    attn_vol_per_layer = get_val(plotting.mem_vol_attn_fwd_label)
    attn_vol_per_stage = (attn_vol_per_layer or 0) * (num_layers_per_stage or 0)

    moe_time_per_layer = get_val(plotting.mem_time_moe_fwd_label)
    moe_time_per_stage = (moe_time_per_layer or 0.0) * (num_layers_per_stage or 0)
    moe_vol_per_layer = get_val(plotting.mem_vol_moe_fwd_label)
    moe_vol_per_stage = (moe_vol_per_layer or 0) * (num_layers_per_stage or 0)

    # Backward per-layer and per-PP-stage memory breakdowns
    bwd_attn_time_per_layer = get_val(plotting.mem_time_attn_bwd_label)
    bwd_attn_time_per_stage = (bwd_attn_time_per_layer or 0.0) * (num_layers_per_stage or 0)
    bwd_attn_vol_per_layer = get_val(plotting.mem_vol_attn_bwd_label)
    bwd_attn_vol_per_stage = (bwd_attn_vol_per_layer or 0) * (num_layers_per_stage or 0)

    bwd_moe_time_per_layer = get_val(plotting.mem_time_moe_bwd_label)
    bwd_moe_time_per_stage = (bwd_moe_time_per_layer or 0.0) * (num_layers_per_stage or 0)
    bwd_moe_vol_per_layer = get_val(plotting.mem_vol_moe_bwd_label)
    bwd_moe_vol_per_stage = (bwd_moe_vol_per_layer or 0) * (num_layers_per_stage or 0)

    sweep_case = _extract_sweep_case_from_row(row)
    return {
        "sweep_case": sweep_case,
        "labels": get_val("labels") or {},
        "base": {
            "model_spec": {
                "num_heads": get_val("base.model_spec.num_heads"),
                "num_kv_heads": get_val("base.model_spec.num_kv_heads"),
            },
            "workload_spec": {
                "comp_comm_efficiency": get_val("base.workload_spec.comp_comm_efficiency"),
                "comp_mem_efficiency": get_val("base.workload_spec.comp_mem_efficiency"),
                "comm_mem_efficiency": get_val("base.workload_spec.comm_mem_efficiency"),
                "decode_tp2d_kv_comm_mode": get_val("base.workload_spec.decode_tp2d_kv_comm_mode"),
                "decode_moe_router_sync_mode": get_val(
                    "base.workload_spec.decode_moe_router_sync_mode"
                ),
                "tp_moe_a2a_replication_optimization": get_val(
                    "base.workload_spec.tp_moe_a2a_replication_optimization"
                ),
                "moe_a2a_comm_mode": get_val("base.workload_spec.moe_a2a_comm_mode"),
            },
        },
        "configuration": {
            "tp_node": get_val(plotting.tp_node_label),
            "tp_xpu_count": get_val(plotting.tp_xpu_count_label),
            "dp_ranks": get_val(plotting.dp_ranks_label),
            "total_xpu_count": get_val(plotting.total_xpu_count_label),
            "pp_stages": get_val(plotting.pp_stages_label),
            "batch_size_per_dp": get_val(plotting.batch_size_per_dp_label),
            "sequence_length": get_val(plotting.seq_len_label),
            "model_hidden_dim": get_val(plotting.model_hidden_dim_label),
            "layers_per_stage": get_val(plotting.n_layers_label),
            "tensor_parallel_variant": get_val(plotting.tensor_parallel_variant_label),
            "optimizer": get_val(plotting.optimizer_label),
        },
        "model_configuration": {
            "shared_expert_count": get_val(plotting.shared_expert_count_label),
            "regular_expert_count": get_val(plotting.regular_expert_count_label),
            "selected_experts": get_val(plotting.num_selected_experts_label),
            "mlp_ratio": get_val(plotting.mlp_ratio_label),
            "mlp_kind": get_val(plotting.mlp_kind_label),
        },
        "expert_placement": {
            "shared_experts_per_dp_rank": get_val(plotting.shared_experts_per_dp_rank_label),
            "regular_experts_per_dp_rank": get_val(plotting.regular_experts_per_dp_rank_label),
            "experts_per_dp_rank": get_val(plotting.num_experts_per_dp_rank_label),
            "attn_domains_per_full_expert_set": get_val(
                plotting.attn_domains_per_full_expert_set_label
            ),
            "num_complete_expert_sets": get_val(plotting.num_complete_expert_sets_label),
        },
        "time": {
            "total_time": fmt_t(get_val(plotting.time_to_train_sec_label)),
            "total_time_seconds": get_val(plotting.time_to_train_sec_label),
            "tokens_per_second": get_val(plotting.tokens_per_second_label),
            "tokens_per_second_per_xpu": get_val(plotting.tokens_per_second_per_xpu_label),
            "tpot_user": fmt_t(get_val(plotting.tpot_user_sec_label)),
            "tpot_user_seconds": get_val(plotting.tpot_user_sec_label),
            "tpot_system": fmt_t(get_val(plotting.tpot_system_sec_label)),
            "tpot_system_seconds": get_val(plotting.tpot_system_sec_label),
            "ttft_forward": fmt_t(get_val(plotting.ttft_forward_sec_label)),
            "ttft_forward_seconds": get_val(plotting.ttft_forward_sec_label),
        },
        "time_breakdown": {
            "forward": {
                "attn_projection": {
                    "total_time": fmt_t(get_val("Attn_Proj_Fwd_Total_Time_Overlapped_Sec")),
                    "total_time_seconds": get_val("Attn_Proj_Fwd_Total_Time_Overlapped_Sec"),
                    "compute_s": get_val("Attn_Proj_Fwd_Compute_Time_Sec"),
                    "memory_s": get_val("Attn_Proj_Fwd_Mem_Time_Sec"),
                    "communication_s": get_val("Attn_Proj_Fwd_Comm_Time_Sec"),
                },
                "attn_core": {
                    "total_time": fmt_t(get_val("Attn_Core_Fwd_Total_Time_Overlapped_Sec")),
                    "total_time_seconds": get_val("Attn_Core_Fwd_Total_Time_Overlapped_Sec"),
                    "compute_s": get_val("Attn_Core_Fwd_Compute_Time_Sec"),
                    "memory_s": get_val("Attn_Core_Fwd_Mem_Time_Sec"),
                    "communication_s": get_val("Attn_Core_Fwd_Comm_Time_Sec"),
                },
                "ffn": {
                    "total_time": fmt_t(get_val("FFN_Fwd_Total_Time_Overlapped_Sec")),
                    "total_time_seconds": get_val("FFN_Fwd_Total_Time_Overlapped_Sec"),
                    "compute_s": get_val("FFN_Fwd_Compute_Time_Sec"),
                    "memory_s": get_val("FFN_Fwd_Mem_Time_Sec"),
                    "communication_s": get_val("FFN_Fwd_Comm_Time_Sec"),
                },
            }
        },
        "compute_per_batch_per_xpu": {
            "total_time": fmt_t(get_val(plotting.compute_time_label)),
            "total_time_seconds": get_val(plotting.compute_time_label),
            "forward": {
                "time": fmt_t(get_val(plotting.compute_time_forward_pass_label)),
                "time_seconds": get_val(plotting.compute_time_forward_pass_label),
                "total_flops": fmt_f(get_val(plotting.flops_forward_pass_label)),
                "embedding": {
                    "time": fmt_t(get_val(plotting.compute_time_embedding_fwd_label)),
                    "time_seconds": get_val(plotting.compute_time_embedding_fwd_label),
                    "flops": fmt_f(get_val(plotting.flops_embedding_fwd_label)),
                },
                "attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.compute_time_attn_fwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.compute_time_attn_fwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.compute_time_attn_fwd_label)),
                    "flops_per_layer": fmt_f(
                        (get_val(plotting.flops_attn_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "flops_per_pp_stage": fmt_f(get_val(plotting.flops_attn_fwd_label)),
                },
                "moe": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.compute_time_moe_fwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.compute_time_moe_fwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.compute_time_moe_fwd_label)),
                    "flops_per_layer": fmt_f(
                        (get_val(plotting.flops_moe_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "flops_per_pp_stage": fmt_f(get_val(plotting.flops_moe_fwd_label)),
                },
                "unembedding": {
                    "time": fmt_t(get_val(plotting.compute_time_unembedding_fwd_label)),
                    "time_seconds": get_val(plotting.compute_time_unembedding_fwd_label),
                    "flops": fmt_f(get_val(plotting.flops_unembedding_fwd_label)),
                },
            },
            "backward": {
                "time": fmt_t(get_val(plotting.compute_time_backward_pass_label)),
                "time_seconds": get_val(plotting.compute_time_backward_pass_label),
                "total_flops": fmt_f(get_val(plotting.flops_backward_pass_label)),
                "attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.compute_time_attn_bwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.compute_time_attn_bwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.compute_time_attn_bwd_label)),
                    "flops_per_layer": fmt_f(
                        (get_val(plotting.flops_attn_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "flops_per_pp_stage": fmt_f(get_val(plotting.flops_attn_bwd_label)),
                },
                "moe": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.compute_time_moe_bwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.compute_time_moe_bwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.compute_time_moe_bwd_label)),
                    "flops_per_layer": fmt_f(
                        (get_val(plotting.flops_moe_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "flops_per_pp_stage": fmt_f(get_val(plotting.flops_moe_bwd_label)),
                },
            },
        },
        "memory_traffic_per_batch_per_xpu": {
            "total_time": fmt_t(get_val(plotting.memory_time_label)),
            "total_time_seconds": get_val(plotting.memory_time_label),
            "forward": {
                "time": fmt_t(get_val(plotting.mem_time_forward_pass_label)),
                "time_seconds": get_val(plotting.mem_time_forward_pass_label),
                "total_volume_bytes": fmt_b(get_val(plotting.mem_forward_pass_label)),
                "embedding": {
                    "time": fmt_t(get_val(plotting.embedding_mem_time_fwd_label)),
                    "time_seconds": get_val(plotting.embedding_mem_time_fwd_label),
                    "volume_bytes": fmt_b(get_val(plotting.embedding_mem_vol_fwd_label)),
                },
                "attn": {
                    "time_per_layer": fmt_t(attn_time_per_layer),
                    "time_per_pp_stage_seconds": attn_time_per_stage,
                    "time_per_pp_stage": fmt_t(attn_time_per_stage),
                    "volume_bytes_per_layer": fmt_b(attn_vol_per_layer),
                    "volume_bytes_per_pp_stage": fmt_b(attn_vol_per_stage),
                    "projection": {
                        "time_per_pp_stage_seconds": get_val("attn_fwd_mem_time_proj_s"),
                        "time_per_pp_stage": fmt_t(get_val("attn_fwd_mem_time_proj_s")),
                        "volume_bytes_per_pp_stage": fmt_b(get_val("attn_fwd_mem_bytes_proj")),
                    },
                    "attn_core": {
                        "time_per_pp_stage_seconds": get_val("attn_fwd_mem_time_core_s"),
                        "time_per_pp_stage": fmt_t(get_val("attn_fwd_mem_time_core_s")),
                        "volume_bytes_per_pp_stage": fmt_b(get_val("attn_fwd_mem_bytes_core")),
                    },
                    "by_component": {
                        "weights": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_attn_weights_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_attn_weights_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_weights_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_weights_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_attn_weights_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_attn_weights_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_weights_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_weights_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "activations": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_attn_activations_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_attn_activations_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_activations_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_activations_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_attn_activations_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_attn_activations_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_activations_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_activations_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "gradients": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_attn_gradients_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_attn_gradients_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_gradients_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_gradients_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_attn_gradients_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_attn_gradients_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_gradients_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_gradients_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "kv_cache": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_attn_kv_cache_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_attn_kv_cache_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_kv_cache_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_attn_kv_cache_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_attn_kv_cache_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_attn_kv_cache_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_kv_cache_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_attn_kv_cache_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                    },
                },
                "moe": {
                    "time_per_layer": fmt_t(moe_time_per_layer),
                    "time_per_pp_stage_seconds": moe_time_per_stage,
                    "time_per_pp_stage": fmt_t(moe_time_per_stage),
                    "volume_bytes_per_layer": fmt_b(moe_vol_per_layer),
                    "volume_bytes_per_pp_stage": fmt_b(moe_vol_per_stage),
                    "by_component": {
                        "weights": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_moe_weights_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_moe_weights_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_moe_weights_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_moe_weights_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_moe_weights_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_moe_weights_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_moe_weights_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_moe_weights_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "activations": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_moe_activations_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_moe_activations_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_moe_activations_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_moe_activations_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_moe_activations_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_moe_activations_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_moe_activations_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_moe_activations_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "gradients": {
                            "time_read_per_layer": fmt_t(
                                get_val("fwd_moe_gradients_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("fwd_moe_gradients_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("fwd_moe_gradients_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("fwd_moe_gradients_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("fwd_moe_gradients_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("fwd_moe_gradients_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("fwd_moe_gradients_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("fwd_moe_gradients_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                    },
                },
                "unembedding": {
                    "time": fmt_t(get_val(plotting.unembedding_mem_time_fwd_label)),
                    "time_seconds": get_val(plotting.unembedding_mem_time_fwd_label),
                    "volume_bytes": fmt_b(get_val(plotting.unembedding_mem_vol_fwd_label)),
                },
            },
            "backward": {
                "time": fmt_t(get_val(plotting.mem_time_backward_pass_label)),
                "time_seconds": get_val(plotting.mem_time_backward_pass_label),
                "total_volume_bytes": fmt_b(get_val(plotting.mem_backward_pass_label)),
                "embedding": {
                    "time": fmt_t(get_val(plotting.embedding_mem_time_bwd_label)),
                    "time_seconds": get_val(plotting.embedding_mem_time_bwd_label),
                    "volume_bytes": fmt_b(get_val(plotting.embedding_mem_vol_bwd_label)),
                },
                "attn": {
                    "time_per_layer": fmt_t(bwd_attn_time_per_layer),
                    "time_per_pp_stage_seconds": bwd_attn_time_per_stage,
                    "time_per_pp_stage": fmt_t(bwd_attn_time_per_stage),
                    "volume_bytes_per_layer": fmt_b(bwd_attn_vol_per_layer),
                    "volume_bytes_per_pp_stage": fmt_b(bwd_attn_vol_per_stage),
                    "by_component": {
                        "weights": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_attn_weights_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_attn_weights_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_weights_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_weights_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_attn_weights_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_attn_weights_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_weights_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_weights_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "activations": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_attn_activations_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_attn_activations_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_activations_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_activations_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_attn_activations_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_attn_activations_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_activations_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_activations_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "gradients": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_attn_gradients_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_attn_gradients_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_gradients_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_gradients_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_attn_gradients_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_attn_gradients_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_gradients_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_gradients_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "kv_cache": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_attn_kv_cache_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_attn_kv_cache_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_kv_cache_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_attn_kv_cache_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_attn_kv_cache_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_attn_kv_cache_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_kv_cache_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_attn_kv_cache_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                    },
                },
                "moe": {
                    "time_per_layer": fmt_t(bwd_moe_time_per_layer),
                    "time_per_pp_stage_seconds": bwd_moe_time_per_stage,
                    "time_per_pp_stage": fmt_t(bwd_moe_time_per_stage),
                    "volume_bytes_per_layer": fmt_b(bwd_moe_vol_per_layer),
                    "volume_bytes_per_pp_stage": fmt_b(bwd_moe_vol_per_stage),
                    "by_component": {
                        "weights": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_moe_weights_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_moe_weights_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_moe_weights_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_moe_weights_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_moe_weights_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_moe_weights_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_moe_weights_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_moe_weights_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "activations": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_moe_activations_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_moe_activations_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_moe_activations_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_moe_activations_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_moe_activations_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_moe_activations_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_moe_activations_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_moe_activations_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                        "gradients": {
                            "time_read_per_layer": fmt_t(
                                get_val("bwd_moe_gradients_time_read_s_per_layer") or 0.0
                            ),
                            "time_write_per_layer": fmt_t(
                                get_val("bwd_moe_gradients_time_write_s_per_layer") or 0.0
                            ),
                            "time_read_per_pp_stage": fmt_t(
                                (get_val("bwd_moe_gradients_time_read_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "time_write_per_pp_stage": fmt_t(
                                (get_val("bwd_moe_gradients_time_write_s_per_layer") or 0.0)
                                * num_layers_per_stage
                            ),
                            "bytes_read_per_layer": fmt_b(
                                get_val("bwd_moe_gradients_read_bytes_per_layer") or 0
                            ),
                            "bytes_written_per_layer": fmt_b(
                                get_val("bwd_moe_gradients_written_bytes_per_layer") or 0
                            ),
                            "bytes_read_per_pp_stage": fmt_b(
                                (get_val("bwd_moe_gradients_read_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                            "bytes_written_per_pp_stage": fmt_b(
                                (get_val("bwd_moe_gradients_written_bytes_per_layer") or 0)
                                * num_layers_per_stage
                            ),
                        },
                    },
                },
                "unembedding": {
                    "time": fmt_t(get_val(plotting.unembedding_mem_time_bwd_label)),
                    "time_seconds": get_val(plotting.unembedding_mem_time_bwd_label),
                    "volume_bytes": fmt_b(get_val(plotting.unembedding_mem_vol_bwd_label)),
                },
            },
        },
        "communication_per_batch_per_xpu": {
            "total_time": fmt_t(get_val(plotting.comm_time_label)),
            "total_time_seconds": get_val(plotting.comm_time_label),
            "startup_latency": fmt_t(get_val(f"{plotting.comm_time_label}_Startup_Latency")),
            "startup_latency_seconds": get_val(f"{plotting.comm_time_label}_Startup_Latency"),
            "data_transfer_time": fmt_t(get_val(f"{plotting.comm_time_label}_Data_Transfer")),
            "data_transfer_time_seconds": get_val(f"{plotting.comm_time_label}_Data_Transfer"),
            "startup_phase_count": get_val("Startup_Phase_Count_Total_Comm_Per_Batch"),
            "forward": {
                "total_time": fmt_t(get_val(plotting.comm_time_forward_pass_label)),
                "total_time_seconds": get_val(plotting.comm_time_forward_pass_label),
                "startup_latency": fmt_t(
                    get_val(f"{plotting.comm_time_forward_pass_label}_Startup_Latency")
                ),
                "startup_latency_seconds": get_val(
                    f"{plotting.comm_time_forward_pass_label}_Startup_Latency"
                ),
                "data_transfer_time": fmt_t(
                    get_val(f"{plotting.comm_time_forward_pass_label}_Data_Transfer")
                ),
                "data_transfer_time_seconds": get_val(
                    f"{plotting.comm_time_forward_pass_label}_Data_Transfer"
                ),
                "total_volume_bytes": fmt_b(get_val(plotting.comm_vol_forward_pass_label)),
                "unembedding_all_gather": {
                    "time": fmt_t(get_val(plotting.comm_time_unembedding_ag_label)),
                    "time_seconds": get_val(plotting.comm_time_unembedding_ag_label),
                    "volume_bytes": fmt_b(get_val(plotting.comm_vol_unembedding_ag_label)),
                },
                "tensor_parallel_attn": {
                    "timing_per_layer": comm_triplet_per_layer(
                        plotting.comm_time_tp_attn_fwd_label
                    ),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_tp_attn_fwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_attn_fwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_tp_attn_fwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_attn_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_attn_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_tp_attn_fwd_label)
                    ),
                    "phase1_input_proj": {
                        "timing_per_layer": comm_triplet_per_layer(
                            plotting.comm_time_tp_attn_fwd_phase1_label
                        ),
                        "timing_per_pp_stage": comm_triplet(
                            plotting.comm_time_tp_attn_fwd_phase1_label
                        ),
                        "time_per_layer": fmt_t(
                            (get_val(plotting.comm_time_tp_attn_fwd_phase1_label) or 0.0)
                            / num_layers_per_stage
                        ),
                        "time_per_pp_stage_seconds": get_val(
                            plotting.comm_time_tp_attn_fwd_phase1_label
                        ),
                        "time_per_pp_stage": fmt_t(
                            get_val(plotting.comm_time_tp_attn_fwd_phase1_label)
                        ),
                        "volume_bytes_per_layer": fmt_b(
                            (get_val(plotting.comm_vol_tp_attn_fwd_phase1_label) or 0)
                            / num_layers_per_stage
                        ),
                        "volume_bytes_per_pp_stage": fmt_b(
                            get_val(plotting.comm_vol_tp_attn_fwd_phase1_label)
                        ),
                    },
                    "phase2_attn_core": {
                        "timing_per_layer": comm_triplet_per_layer(
                            plotting.comm_time_tp_attn_fwd_phase2_label
                        ),
                        "timing_per_pp_stage": comm_triplet(
                            plotting.comm_time_tp_attn_fwd_phase2_label
                        ),
                        "time_per_layer": fmt_t(
                            (get_val(plotting.comm_time_tp_attn_fwd_phase2_label) or 0.0)
                            / num_layers_per_stage
                        ),
                        "time_per_pp_stage_seconds": get_val(
                            plotting.comm_time_tp_attn_fwd_phase2_label
                        ),
                        "time_per_pp_stage": fmt_t(
                            get_val(plotting.comm_time_tp_attn_fwd_phase2_label)
                        ),
                        "volume_bytes_per_layer": fmt_b(
                            (get_val(plotting.comm_vol_tp_attn_fwd_phase2_label) or 0)
                            / num_layers_per_stage
                        ),
                        "volume_bytes_per_pp_stage": fmt_b(
                            get_val(plotting.comm_vol_tp_attn_fwd_phase2_label)
                        ),
                    },
                    "phase3_output_proj": {
                        "timing_per_layer": comm_triplet_per_layer(
                            plotting.comm_time_tp_attn_fwd_phase3_label
                        ),
                        "timing_per_pp_stage": comm_triplet(
                            plotting.comm_time_tp_attn_fwd_phase3_label
                        ),
                        "time_per_layer": fmt_t(
                            (get_val(plotting.comm_time_tp_attn_fwd_phase3_label) or 0.0)
                            / num_layers_per_stage
                        ),
                        "time_per_pp_stage_seconds": get_val(
                            plotting.comm_time_tp_attn_fwd_phase3_label
                        ),
                        "time_per_pp_stage": fmt_t(
                            get_val(plotting.comm_time_tp_attn_fwd_phase3_label)
                        ),
                        "volume_bytes_per_layer": fmt_b(
                            (get_val(plotting.comm_vol_tp_attn_fwd_phase3_label) or 0)
                            / num_layers_per_stage
                        ),
                        "volume_bytes_per_pp_stage": fmt_b(
                            get_val(plotting.comm_vol_tp_attn_fwd_phase3_label)
                        ),
                    },
                    "projection_total": {
                        "timing_per_pp_stage": {
                            "total_time": fmt_t(
                                (get_val(plotting.comm_time_tp_attn_fwd_phase1_label) or 0.0)
                                + (get_val(plotting.comm_time_tp_attn_fwd_phase3_label) or 0.0)
                            ),
                            "total_time_seconds": (
                                (get_val(plotting.comm_time_tp_attn_fwd_phase1_label) or 0.0)
                                + (get_val(plotting.comm_time_tp_attn_fwd_phase3_label) or 0.0)
                            ),
                            "startup_latency": fmt_t(
                                (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase1_label}_Startup_Latency"
                                    )
                                    or 0.0
                                )
                                + (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase3_label}_Startup_Latency"
                                    )
                                    or 0.0
                                )
                            ),
                            "startup_latency_seconds": (
                                (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase1_label}_Startup_Latency"
                                    )
                                    or 0.0
                                )
                                + (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase3_label}_Startup_Latency"
                                    )
                                    or 0.0
                                )
                            ),
                            "data_transfer_time": fmt_t(
                                (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase1_label}_Data_Transfer"
                                    )
                                    or 0.0
                                )
                                + (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase3_label}_Data_Transfer"
                                    )
                                    or 0.0
                                )
                            ),
                            "data_transfer_time_seconds": (
                                (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase1_label}_Data_Transfer"
                                    )
                                    or 0.0
                                )
                                + (
                                    get_val(
                                        f"{plotting.comm_time_tp_attn_fwd_phase3_label}_Data_Transfer"
                                    )
                                    or 0.0
                                )
                            ),
                        },
                        "time_per_pp_stage_seconds": (
                            (get_val(plotting.comm_time_tp_attn_fwd_phase1_label) or 0.0)
                            + (get_val(plotting.comm_time_tp_attn_fwd_phase3_label) or 0.0)
                        ),
                        "time_per_pp_stage": fmt_t(
                            (get_val(plotting.comm_time_tp_attn_fwd_phase1_label) or 0.0)
                            + (get_val(plotting.comm_time_tp_attn_fwd_phase3_label) or 0.0)
                        ),
                        "volume_bytes_per_pp_stage": fmt_b(
                            (get_val(plotting.comm_vol_tp_attn_fwd_phase1_label) or 0)
                            + (get_val(plotting.comm_vol_tp_attn_fwd_phase3_label) or 0)
                        ),
                    },
                },
                "tensor_parallel_moe": {
                    "startup_phase_count_per_pp_stage": get_val("Startup_Phase_Count_TP_MoE_Fwd"),
                    "timing_per_layer": comm_triplet_per_layer(plotting.comm_time_tp_moe_fwd_label),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_tp_moe_fwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_moe_fwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_tp_moe_fwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_moe_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_moe_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_tp_moe_fwd_label)),
                },
                "data_parallel_attn": {
                    "timing_per_layer": comm_triplet_per_layer(
                        plotting.comm_time_dp_attn_fwd_label
                    ),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_dp_attn_fwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_attn_fwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_dp_attn_fwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_attn_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_attn_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_dp_attn_fwd_label)
                    ),
                },
                "data_parallel_moe": {
                    "timing_per_layer": comm_triplet_per_layer(plotting.comm_time_dp_moe_fwd_label),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_dp_moe_fwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_moe_fwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_dp_moe_fwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_moe_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_moe_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_dp_moe_fwd_label)),
                },
                "moe_all_to_all": {
                    "to_experts_startup_phase_count_per_pp_stage": get_val(
                        "Startup_Phase_Count_MoE_A2A_To_Fwd"
                    ),
                    "to_experts_timing_per_layer": comm_triplet_per_layer(
                        plotting.moe_to_experts_time_label
                    ),
                    "to_experts_timing_per_pp_stage": comm_triplet(
                        plotting.moe_to_experts_time_label
                    ),
                    "to_experts_time_per_layer": fmt_t(
                        (get_val(plotting.moe_to_experts_time_label) or 0.0) / num_layers_per_stage
                    ),
                    "to_experts_time_per_pp_stage": fmt_t(
                        get_val(plotting.moe_to_experts_time_label)
                    ),
                    "to_experts_volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.moe_to_experts_vol_label) or 0) / num_layers_per_stage
                    ),
                    "to_experts_volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.moe_to_experts_vol_label)
                    ),
                    "from_experts_startup_phase_count_per_pp_stage": get_val(
                        "Startup_Phase_Count_MoE_A2A_From_Fwd"
                    ),
                    "from_experts_timing_per_layer": comm_triplet_per_layer(
                        plotting.moe_from_experts_time_label
                    ),
                    "from_experts_timing_per_pp_stage": comm_triplet(
                        plotting.moe_from_experts_time_label
                    ),
                    "from_experts_time_per_layer": fmt_t(
                        (get_val(plotting.moe_from_experts_time_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "from_experts_time_per_pp_stage": fmt_t(
                        get_val(plotting.moe_from_experts_time_label)
                    ),
                    "from_experts_volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.moe_from_experts_vol_label) or 0) / num_layers_per_stage
                    ),
                    "from_experts_volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.moe_from_experts_vol_label)
                    ),
                },
            },
            "backward": {
                "total_time": fmt_t(get_val(plotting.comm_time_backward_pass_label)),
                "total_time_seconds": get_val(plotting.comm_time_backward_pass_label),
                "startup_latency": fmt_t(
                    get_val(f"{plotting.comm_time_backward_pass_label}_Startup_Latency")
                ),
                "startup_latency_seconds": get_val(
                    f"{plotting.comm_time_backward_pass_label}_Startup_Latency"
                ),
                "data_transfer_time": fmt_t(
                    get_val(f"{plotting.comm_time_backward_pass_label}_Data_Transfer")
                ),
                "data_transfer_time_seconds": get_val(
                    f"{plotting.comm_time_backward_pass_label}_Data_Transfer"
                ),
                "total_volume_bytes": fmt_b(get_val(plotting.comm_vol_backward_pass_label)),
                "tensor_parallel_attn": {
                    "timing_per_layer": comm_triplet_per_layer(
                        plotting.comm_time_tp_attn_bwd_label
                    ),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_tp_attn_bwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_attn_bwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_tp_attn_bwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_attn_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_attn_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_tp_attn_bwd_label)
                    ),
                },
                "tensor_parallel_moe": {
                    "timing_per_layer": comm_triplet_per_layer(plotting.comm_time_tp_moe_bwd_label),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_tp_moe_bwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_moe_bwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_tp_moe_bwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_moe_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_moe_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_tp_moe_bwd_label)),
                },
                "data_parallel_attn": {
                    "timing_per_layer": comm_triplet_per_layer(
                        plotting.comm_time_dp_attn_bwd_label
                    ),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_dp_attn_bwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_attn_bwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_dp_attn_bwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_attn_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_attn_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_dp_attn_bwd_label)
                    ),
                },
                "data_parallel_moe": {
                    "timing_per_layer": comm_triplet_per_layer(plotting.comm_time_dp_moe_bwd_label),
                    "timing_per_pp_stage": comm_triplet(plotting.comm_time_dp_moe_bwd_label),
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_moe_bwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage_seconds": get_val(plotting.comm_time_dp_moe_bwd_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_moe_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_moe_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_dp_moe_bwd_label)),
                },
                "moe_all_to_all": {
                    "timing_per_pp_stage": comm_triplet(plotting.moe_alltoall_bwd_time_label),
                    "time_per_pp_stage": fmt_t(get_val(plotting.moe_alltoall_bwd_time_label)),
                },
            },
        },
        "ratios": {
            "comm_to_compute": get_val(plotting.comm_compute_ratio_label),
            "memory_to_compute": get_val(plotting.memory_compute_ratio_label),
            "moe_all2all_time_fraction_fwd_bwd": get_val("MoE_A2A_Time_Fraction_FwdBwd"),
            "comm_fraction_per_batch": get_val("Comm_Fraction_Per_Batch"),
            "raw_comm_fraction_per_batch": get_val("Raw_Comm_Fraction_Per_Batch"),
        },
        "memory_footprint": {
            "attn": {
                "weights_bytes_per_xpu_per_layer": fmt_b(get_val("attn_weights_bytes")),
                "weights_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("attn_weights_bytes") or 0) * num_layers_per_stage
                ),
                "optimizer_states_bytes_per_xpu_per_layer": fmt_b(
                    get_val("attn_optimizer_states_bytes")
                ),
                "optimizer_states_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("attn_optimizer_states_bytes") or 0) * num_layers_per_stage
                ),
                "saved_activations_bytes_per_xpu_per_layer": fmt_b(
                    get_val("attn_saved_activations_bytes")
                ),
                "saved_activations_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("attn_saved_activations_bytes") or 0) * num_layers_per_stage
                ),
                "gradient_buffers_bytes_per_xpu_per_layer": fmt_b(
                    get_val("attn_gradient_buffers_bytes")
                ),
                "gradient_buffers_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("attn_gradient_buffers_bytes") or 0) * num_layers_per_stage
                ),
                "kv_cache_bytes_per_xpu_per_layer": fmt_b(get_val("attn_kv_cache_bytes")),
                "kv_cache_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("attn_kv_cache_bytes") or 0) * num_layers_per_stage
                ),
                "transient_forward_peak_bytes_per_xpu_per_layer": fmt_b(
                    get_val("attn_transient_forward_peak_bytes")
                ),
                "transient_forward_peak_bytes_per_xpu_per_pp_stage": fmt_b(
                    get_val("attn_transient_forward_peak_bytes")
                ),
                "transient_backward_peak_bytes_per_xpu_per_layer": fmt_b(
                    get_val("attn_transient_backward_peak_bytes")
                ),
                "transient_backward_peak_bytes_per_xpu_per_pp_stage": fmt_b(
                    get_val("attn_transient_backward_peak_bytes")
                ),
            },
            "moe": {
                "weights_bytes_per_xpu_per_layer": fmt_b(get_val("ffn_moe_weights_bytes")),
                "weights_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("ffn_moe_weights_bytes") or 0) * num_layers_per_stage
                ),
                "optimizer_states_bytes_per_xpu_per_layer": fmt_b(
                    get_val("ffn_moe_optimizer_states_bytes")
                ),
                "optimizer_states_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("ffn_moe_optimizer_states_bytes") or 0) * num_layers_per_stage
                ),
                "saved_activations_bytes_per_xpu_per_layer": fmt_b(
                    get_val("ffn_moe_saved_activations_bytes")
                ),
                "saved_activations_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("ffn_moe_saved_activations_bytes") or 0) * num_layers_per_stage
                ),
                "gradient_buffers_bytes_per_xpu_per_layer": fmt_b(
                    get_val("ffn_moe_gradient_buffers_bytes")
                ),
                "gradient_buffers_bytes_per_xpu_per_pp_stage": fmt_b(
                    (get_val("ffn_moe_gradient_buffers_bytes") or 0) * num_layers_per_stage
                ),
                "transient_forward_peak_bytes_per_xpu_per_layer": fmt_b(
                    get_val("ffn_moe_transient_forward_peak_bytes")
                ),
                "transient_forward_peak_bytes_per_xpu_per_pp_stage": fmt_b(
                    get_val("ffn_moe_transient_forward_peak_bytes")
                ),
                "transient_backward_peak_bytes_per_xpu_per_layer": fmt_b(
                    get_val("ffn_moe_transient_backward_peak_bytes")
                ),
                "transient_backward_peak_bytes_per_xpu_per_pp_stage": fmt_b(
                    get_val("ffn_moe_transient_backward_peak_bytes")
                ),
            },
            "total_peak_memory_per_xpu_bytes": fmt_b(get_val("total_peak_memory_per_chip_bytes")),
        },
        "memory_traffic_details_per_batch": {
            "forward": {
                "attn": {
                    "weights_read_bytes": fmt_b(get_val("attn_fwd_weights_read_bytes")),
                    "activations_read_bytes": fmt_b(get_val("attn_fwd_activations_read_bytes")),
                    "activations_written_bytes": fmt_b(
                        get_val("attn_fwd_activations_written_bytes")
                    ),
                    "kv_cache_written_bytes": fmt_b(get_val("attn_fwd_kv_cache_written_bytes")),
                },
                "ffn_moe": {
                    "weights_read_bytes": fmt_b(get_val("ffn_moe_fwd_weights_read_bytes")),
                    "activations_read_bytes": fmt_b(get_val("ffn_moe_fwd_activations_read_bytes")),
                    "activations_written_bytes": fmt_b(
                        get_val("ffn_moe_fwd_activations_written_bytes")
                    ),
                },
            },
            "backward": {
                "attn": {
                    "weights_read_bytes": fmt_b(get_val("attn_bwd_weights_read_bytes")),
                    "gradients_written_bytes": fmt_b(get_val("attn_bwd_gradients_written_bytes")),
                    "activations_read_bytes": fmt_b(get_val("attn_bwd_activations_read_bytes")),
                },
                "ffn_moe": {
                    "weights_read_bytes": fmt_b(get_val("ffn_moe_bwd_weights_read_bytes")),
                    "gradients_written_bytes": fmt_b(
                        get_val("ffn_moe_bwd_gradients_written_bytes")
                    ),
                    "activations_read_bytes": fmt_b(get_val("ffn_moe_bwd_activations_read_bytes")),
                    "details": {
                        "original_input_read_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_original_input_read_bytes")
                        ),
                        "attn_output_read_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_attn_output_read_bytes")
                        ),
                        "ff1_out_sharded_read_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_ff1_out_sharded_read_bytes")
                        ),
                        "expert_input_read_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_expert_input_read_bytes")
                        ),
                        "weights_read_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_weights_read_bytes")
                        ),
                        "weight_gradients_written_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_weight_gradients_written_bytes")
                        ),
                        "gradients_written_expert_input_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_gradients_written_expert_input_bytes")
                        ),
                        "gradients_written_ff1_out_sharded_bytes": fmt_b(
                            get_val("ffn_moe_bwd_detail_gradients_written_ff1_out_sharded_bytes")
                        ),
                    },
                },
            },
        },
    }


def produce_plots(df, hlines=None):

    annotation_cols_to_show = [
        "TP_XPU_Count",
        "High_Bandwidth_XPU_Count",
        "Time_To_Train_Day",
        "Time_To_Train_Sec",
        "Tokens_Per_Sec",
    ]
    annotation_labels = [
        "TP XPU Count",
        "High Bandwidth XPU Count",
        "Time_To_Train_Day",
        "Time_To_Train_Sec",
        "Tokens_Per_Sec",
    ]
    plot_flag = str(os.environ.get("SCALESCOPE_PLOT", "")).lower() in ("1", "true", "yes")
    if plot_flag:
        plotting.plot_node_comparison(
            df,
            log_scale=False,
            start_y_with_zero=True,
            title="Training Time Scaling with Cluster Size",
            x_label="Cluster size (XPUs)",
            y_step=1,
            hlines=None,
            show_relative=False,
            annotation_cols=annotation_cols_to_show,
            annotation_labels=annotation_labels,
        )


if __name__ == "__main__":
    main()
