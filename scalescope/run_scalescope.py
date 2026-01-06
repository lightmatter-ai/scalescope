"""CLI orchestrator and end-to-end analysis driver.

Parses arguments, loads hardware/model/workload/topology specs, constructs
logical topology views, invokes compute/comm/memory models, applies overlap and
pipeline bubble models, verifies memory footprints, builds a results DataFrame, and
produces structured YAML summaries (optionally human-formatted) and plots.
"""

import argparse
import os
from enum import Enum

import numpy as np
import pandas as pd
import yaml

import scalescope.plotting as plotting
from scalescope.comm_model import get_transformer_comm_stats
from scalescope.compute_model import (
    backward_pass_flops_per_chip,
    calculate_embedding_compute,
    calculate_unembedding_compute,
    forward_decode_flops_per_chip,
    forward_prefill_flops_per_chip,
    parameter_update_flops,
)
from scalescope.ep_strategy import infer_expert_placement_per_xpu
from scalescope.hardware_spec import HardwareRegistry
from scalescope.memory_model import (
    _backward_attn_tp_1d_memory_traffic,
    _backward_moe_tp_1d_memory_traffic,
    _decode_attention_memory_traffic,
    _forward_pass_prefill_attention_memory_traffic,
    _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic,
    calculate_attn_backward_transient_peak_memory_footprint,
    calculate_attn_forward_transient_peak_memory_footprint,
    calculate_attn_optimizer_states_memory_footprint,
    calculate_attn_saved_activations_memory_footprint,
    calculate_attn_weights_memory_footprint,
    calculate_embedding_memory_traffic_1d_tp_bwd,
    calculate_embedding_memory_traffic_1d_tp_fwd,
    calculate_ffn_moe_backward_transient_peak_memory_footprint,
    calculate_ffn_moe_forward_transient_peak_memory_footprint,
    calculate_ffn_moe_optimizer_states_memory_footprint,
    calculate_ffn_moe_saved_activations_memory_footprint,
    calculate_ffn_moe_weights_memory_footprint,
    calculate_unembedding_memory_traffic_1d_tp_bwd,
    calculate_unembedding_memory_traffic_1d_tp_fwd,
    get_transformer_mem_stats_per_layer,
    LayerMemoryTraffic,
    MemoryFootprint,
)
from scalescope.model_spec import TransformerConfig
from scalescope.pp_strategy import get_pp_strategy
from scalescope.spec_utils import (
    convert_sweep_values,
)
from scalescope.stats import CommStats
from scalescope.sweep_utils import generate_specs_from_sweep
from scalescope.topology_spec import LogicalTopology, PhysicalTopology
from scalescope.tp_strategy import get_2d_tp_dims, validate_2d_tp_invariants
from scalescope.utils import (
    calculate_overlapped_time,
    format_bytes,
    format_flops,
    format_time,
    PassMode,
)
from scalescope.workload_spec import (
    InferenceDecodeSpec,
    InferencePrefillSpec,
    TensorParallelVariant,
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
        "--sweep-config", type=str, required=True, help="Path to the sweep definition YAML."
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

    args = parser.parse_args()

    # Perform basic validation on paths
    paths_to_check = {
        "Hardware library": args.hardware_lib,
        "Physical Topology": args.physical_topology,
        "Logical Topology": args.logical_topology,
        "Model spec": args.model_spec,
        "Workload spec": args.workload_spec,
        "Sweep config": args.sweep_config,
    }
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            parser.error(f"{name} file not found: {path}")

    return args


def main():
    try:
        args = parse_arguments()

        print("--- Parsed Arguments ---")
        print(f"Hardware Library: {args.hardware_lib}")
        print(f"Model Spec:       {args.model_spec}")
        print(f"Workload Spec:    {args.workload_spec}")
        print(f"Physical Topology File: {args.physical_topology}")
        print(f"Logical Topology File: {args.logical_topology}")
        print(f"Sweep Configuration: {args.sweep_config}")

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

        # load sweep config
        with open(args.sweep_config, "r") as f:
            sweep_yaml_data = yaml.safe_load(f)

        # Distinguish empty vs malformed
        if sweep_yaml_data is None:
            print("Sweep YAML is empty. Proceeding with no sweep.")
            processed_sweep_config = {}
        elif not isinstance(sweep_yaml_data, dict):
            raise TypeError("Sweep YAML is malformed: top-level must be a mapping (dictionary).")
        else:
            if "sweep" not in sweep_yaml_data:
                print("Sweep YAML has no 'sweep' key. Proceeding with no sweep.")
                processed_sweep_config = {}
            else:
                raw_sweep_dict = sweep_yaml_data["sweep"]
                if not isinstance(raw_sweep_dict, dict):
                    raise TypeError("Sweep YAML is malformed: 'sweep' value must be a dictionary.")
                # convert sweep values (e.g., strings to Enums) ---
                processed_sweep_config = convert_sweep_values(raw_sweep_dict, base_specs)

        print("\nProcessed sweep config (Python types):")
        print(processed_sweep_config)

    except Exception as e:
        print(f"Error loading configuration files: {e}")
        raise

    # Do not mutate or replace model_spec here; compute per-stage layers later per sweep iteration
    # Build results dataframe (separate from plotting)
    df = build_results_df(base_specs, processed_sweep_config, args.pass_mode)

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

    # Display concise summary to console
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
                # Required final lines per row
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


def get_model_stats(
    cluster_physical_topology,
    cluster_logical_topology,
    model_spec: TransformerConfig,
    workload_spec: WorkloadSpec,
    pass_mode,
):

    d_model = model_spec.d_model
    tensor_parallel_variant = workload_spec.tp_spec.variant
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

    tp_topology_view = cluster_logical_topology.create_view_from_prefix(
        view_name="TP_View", prefix="TP", global_sharing_context="max_exclusive"
    )

    attn_tp_topology_view = tp_topology_view.create_view_from_prefix(
        view_name="TP_Attn_View", prefix="TP_Attn", global_sharing_context="max_exclusive"
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

    # Configure MoE from v2 spec and derive EP placement directly
    # Use the size of the IntraPodFabric physical dimension as the expert high-bandwidth group
    phys_expert_hb_group = cluster_physical_topology.get_dimension("IntraPodFabric").size
    v2_spec = workload_spec.moe_spec_v2
    v2_spec = v2_spec.model_copy(
        update={"expert_high_bandwidth_group_size_xpus": phys_expert_hb_group}
    )

    moe_config = infer_expert_placement_per_xpu(
        model_spec=model_spec,
        moe_v2=v2_spec,
        attn_tp_size=attn_tp_topology_view.num_chips,
        expert_tp_size=moe_tp_topology_view.num_chips,
        ep_chips_per_pp_stage=attn_dp_topology_view.num_chips * attn_tp_topology_view.num_chips,
    )

    # Attach resolved batch sizes to the workload spec (global-first derivation)
    workload_spec = workload_spec.with_resolved_batches(
        attn_dp_topology_view.num_chips,
        moe_config.num_complete_expert_sets,
    )
    batch_sizes = workload_spec.get_resolved_batches()
    batch_size_per_dp = batch_sizes.batch_size_per_dp

    match pass_mode:
        case PassMode.TRAINING:
            assert isinstance(workload_spec, TrainSpec)
            total_train_tokens = workload_spec.total_train_tokens
            seq_length = workload_spec.seq_length
            train_steps, _ = workload_spec.calculate_number_of_train_steps()
            (
                batch_compute_time_forward_pass,
                batch_flops_forward_pass,
                attn_flops_forward_pass,
                moe_flops_forward_pass,
                compute_time_attn_fwd,
                compute_time_moe_fwd,
            ) = forward_prefill_flops_per_chip(
                attn_tp_topology_view, moe_tp_topology_view, model_spec, moe_config, workload_spec
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
                attn_tp_topology_view,
                moe_tp_topology_view,
                model_spec,
                moe_config,
                workload_spec,
            )
            compute_time_param_update, flops_param_update = parameter_update_flops(
                attn_tp_topology_view, model_spec, moe_config, workload_spec
            )

        case PassMode.INFERENCE_PREFILL:
            assert isinstance(workload_spec, InferencePrefillSpec)
            seq_length = workload_spec.seq_length
            # inference does not have train steps
            train_steps = 1
            total_train_tokens = workload_spec.global_batch_size * seq_length * train_steps
            (
                batch_compute_time_forward_pass,
                batch_flops_forward_pass,
                attn_flops_forward_pass,
                moe_flops_forward_pass,
                compute_time_attn_fwd,
                compute_time_moe_fwd,
            ) = forward_prefill_flops_per_chip(
                attn_tp_topology_view, moe_tp_topology_view, model_spec, moe_config, workload_spec
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

            batch_compute_time_forward_pass = 0.0
            batch_flops_forward_pass = 0
            attn_flops_forward_pass = 0
            moe_flops_forward_pass = 0
            compute_time_attn_fwd = 0
            compute_time_moe_fwd = 0

            for i in range(num_decode_steps):
                n_seq_so_far = prefill_seq_len + i
                (
                    step_time,
                    step_flops,
                    step_attn_flops,
                    step_moe_flops,
                    step_attn_time,
                    step_moe_time,
                ) = forward_decode_flops_per_chip(
                    attn_tp_topology_view,
                    moe_tp_topology_view,
                    model_spec,
                    moe_config,
                    workload_spec,
                    n_seq_so_far,
                )
                batch_compute_time_forward_pass += step_time
                batch_flops_forward_pass += step_flops
                attn_flops_forward_pass += step_attn_flops
                moe_flops_forward_pass += step_moe_flops
                compute_time_attn_fwd += step_attn_time
                compute_time_moe_fwd += step_moe_time

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

    # Derive TP 2D dims (X and YZ) when needed
    try:
        x_dims, yz_dims = get_2d_tp_dims(attn_tp_topology_view)
        if workload_spec.tp_spec.variant.name == "SHARD_WEIGHTS_2D":
            validate_2d_tp_invariants(model_spec, attn_tp_topology_view, x_dims, yz_dims)
    except Exception:
        # Fall back to defaults if not 2D-capable topology; 1D paths won’t use these
        x_dims = [0]
        yz_dims = [n for n in range(1, len(attn_tp_topology_view.shape))]

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
                tensor_parallel_variant,
                x_dims,
                yz_dims,
                pass_mode,
                workload_spec.float_size_bytes,
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
                        tensor_parallel_variant,
                        x_dims,
                        yz_dims,
                        pass_mode,
                        workload_spec.float_size_bytes,
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
    mem_stats["tp_variant"] = tensor_parallel_variant
    mem_stats["tp_size_x"] = x_dims
    mem_stats["tp_size_y"] = yz_dims
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
    attn_footprint.weights = calculate_attn_weights_memory_footprint(
        model_spec,
        attn_tp_size,
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

    # Gradient buffers (training only): per-parameter gradients in compute dtype
    if pass_mode == PassMode.TRAINING:
        grad_fs = workload_spec.float_size_bytes
        attn_grad_buffers = calculate_attn_weights_memory_footprint(
            model_spec,
            attn_tp_size,
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

    attn_footprint.transient_forward_peak = calculate_attn_forward_transient_peak_memory_footprint(
        model_spec, ws_for_footprint, attn_tp_size
    )
    ffn_moe_footprint.transient_forward_peak = (
        calculate_ffn_moe_forward_transient_peak_memory_footprint(
            model_spec, ws_for_footprint, moe_tp_size
        )
    )

    if pass_mode == PassMode.TRAINING:
        assert isinstance(workload_spec, TrainSpec)
        # Optimizer States
        attn_footprint.optimizer_states = calculate_attn_optimizer_states_memory_footprint(
            model_spec,
            attn_tp_size,
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
        d_model = model_spec.d_model
        num_heads = model_spec.num_heads
        num_kv_heads = model_spec.num_kv_heads
        head_dim = d_model // num_heads
        batch_size_per_dp = batch_size_per_dp
        if pass_mode == PassMode.INFERENCE_PREFILL:
            kv_len = workload_spec.seq_length
        else:
            assert isinstance(workload_spec, InferenceDecodeSpec)
            kv_len = workload_spec.prefill_seq_len + workload_spec.num_generated_tokens
        kv_heads_sharded = num_kv_heads // attn_tp_size
        kv_bytes_per_layer_per_xpu = (
            2
            * batch_size_per_dp
            * kv_len
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
                attn_tp_topology_view,
                moe_tp_topology_view,
                ep_topology_view,
                model_spec,
                moe_config,
                workload_spec,
                seq_length,
                pass_mode,
            )
            # Typical practice avoids all-gather; unembedding_comm_* remain zero
        case PassMode.INFERENCE_DECODE:
            # Initialize all comm stats to zero for decode
            comm_stats = {
                "tp_attn_fwd": CommStats(),
                "tp_moe_fwd": CommStats(),
                "tp_attn_bwd": CommStats(),
                "tp_moe_bwd": CommStats(),
                "dp_attn_fwd": CommStats(),
                "dp_moe_fwd": CommStats(),
                "dp_attn_bwd": CommStats(),
                "dp_moe_bwd": CommStats(),
                "moe_a2a_fwd_1": CommStats(),
                "moe_a2a_fwd_2": CommStats(),
            }
            assert isinstance(workload_spec, InferenceDecodeSpec)
            num_decode_steps = workload_spec.num_generated_tokens

            for i in range(num_decode_steps):
                # We use seq_length = 1 for decode steps in TP Megatron
                seq_length = 1
                step_comm_stats = get_transformer_comm_stats(
                    attn_dp_topology_view,
                    moe_dp_topology_view,
                    attn_tp_topology_view,
                    moe_tp_topology_view,
                    ep_topology_view,
                    model_spec,
                    moe_config,
                    workload_spec,
                    seq_length,
                    pass_mode,
                )
                # Accumulate stats for each decode step
                for key in comm_stats:
                    comm_stats[key] += step_comm_stats[key]
                # Typical practice avoids all-gather; unembedding_comm_* remain zero
        case _:
            raise ValueError(
                f"Unsupported pass mode: {pass_mode}. Supported modes are TRAINING, INFERENCE_PREFILL, and INFERENCE_DECODE."
            )

    # Unpack dictionary for aggregated stats
    tp_attn_fwd = comm_stats["tp_attn_fwd"]
    tp_moe_fwd = comm_stats["tp_moe_fwd"]
    tp_attn_bwd = comm_stats["tp_attn_bwd"]
    tp_moe_bwd = comm_stats["tp_moe_bwd"]
    dp_attn_fwd = comm_stats["dp_attn_fwd"]
    dp_moe_fwd = comm_stats["dp_moe_fwd"]
    dp_attn_bwd = comm_stats["dp_attn_bwd"]
    dp_moe_bwd = comm_stats["dp_moe_bwd"]
    moe_a2a_fwd_1 = comm_stats["moe_a2a_fwd_1"]
    moe_a2a_fwd_2 = comm_stats["moe_a2a_fwd_2"]

    # Assume that backward comm is the same as the forward comm
    moe_a2a_bwd = moe_a2a_fwd_1 + moe_a2a_fwd_2 if pass_mode == PassMode.TRAINING else CommStats()

    batch_comm_time_forward_pass = (
        tp_attn_fwd.total_comm_time
        + tp_moe_fwd.total_comm_time
        + dp_attn_fwd.total_comm_time
        + dp_moe_fwd.total_comm_time
        + moe_a2a_fwd_1.total_comm_time
        + moe_a2a_fwd_2.total_comm_time
    )
    batch_comm_time_backward_pass = (
        tp_attn_bwd.total_comm_time
        + tp_moe_bwd.total_comm_time
        + dp_attn_bwd.total_comm_time
        + dp_moe_bwd.total_comm_time
        + moe_a2a_bwd.total_comm_time
    )
    batch_comm_vol_forward_pass = (
        tp_attn_fwd.comm_vol
        + tp_moe_fwd.comm_vol
        + dp_attn_fwd.comm_vol
        + dp_moe_fwd.comm_vol
        + moe_a2a_fwd_1.comm_vol
        + moe_a2a_fwd_2.comm_vol
    )
    batch_comm_vol_backward_pass = (
        tp_attn_bwd.comm_vol
        + tp_moe_bwd.comm_vol
        + dp_attn_bwd.comm_vol
        + dp_moe_bwd.comm_vol
        + moe_a2a_bwd.comm_vol
    )

    # Detailed Memory
    fwd_attn_traffic, fwd_moe_traffic, bwd_attn_traffic, bwd_moe_traffic = get_detailed_mem_stats(
        attn_tp_topology_view,
        moe_tp_topology_view,
        model_spec,
        moe_config,
        workload_spec,
        tensor_parallel_variant,
        x_dims,
        yz_dims,
        pass_mode,
        activation_checkpointing,
        device_for_map,
        node,
    )

    # Prepare detailed per-component memory traffic split by block (attn vs ffn_moe)
    # Scale per-layer component bytes by number of layers per PP stage to get per-batch per-chip totals
    layers_per_stage = num_layers_per_pp_stage
    fwd_attn_layer_stats = fwd_attn_traffic.get_stats(1)
    fwd_moe_layer_stats = fwd_moe_traffic.get_stats(1)
    # Expose detailed per-layer component stats (including read/write times) for forward
    mem_stats.update(fwd_attn_layer_stats)
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
        d_ff = d_model * mlp_ratio
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
        ffn1_weight_size = num_colocated * (d_model * d_ff * float_size) / max(moe_tp_size_val, 1)
        ffn2_weight_size = num_colocated * (d_ff * d_model * float_size) / max(moe_tp_size_val, 1)
        ffn_bwd_reads_weights = ffn1_weight_size + ffn2_weight_size
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

        # In a simple bubble model, the overhead is a function of the non-overlapped
        # time of a single stage. Here, we approximate it with the total batch time.
        # A more detailed model would compute this based on the critical path of one stage.
        bubble_kwargs = {"num_stages": pp_stages, "batch_time": batch_time_to_train_sec}
        pp_overhead_stats = pp_strategy.backward_comm(**bubble_kwargs)
        batch_time_pp_overhead = pp_overhead_stats.total_comm_time
    else:
        batch_time_pp_overhead = 0.0

    batch_time_to_train_sec += batch_time_pp_overhead

    time_to_train_sec = batch_time_to_train_sec * train_steps
    tokens_per_second = total_train_tokens / time_to_train_sec
    total_xpus_val = attn_tp_topology_view.num_chips * attn_dp_topology_view.num_chips * pp_stages
    tokens_per_second_per_xpu = tokens_per_second / total_xpus_val

    # batch_comm_vol_forward_pass = batch_forward_pass_comm_stats.comm_vol
    # batch_comm_vol_backward_pass = batch_backward_pass_comm_stats.comm_vol

    tp_node = tp_topology_view.physical_topology.node_definition
    stats = {
        plotting.tp_node_label: tp_node.node_name,
        plotting.tp_xpu_count_label: attn_tp_topology_view.num_chips,
        plotting.dp_ranks_label: attn_dp_topology_view.num_chips,
        plotting.high_bandwidth_xpu_count_label: attn_tp_topology_view.num_chips
        * moe_config.num_dp_ranks_per_complete_expert_set,
        plotting.low_bandwidth_ranks_label: attn_dp_topology_view.num_chips
        // moe_config.num_dp_ranks_per_complete_expert_set,
        plotting.total_xpu_count_label: total_xpus_val,
        plotting.batch_size_per_dp_label: batch_size_per_dp,
        plotting.shared_expert_count_label: model_spec.num_shared_experts,
        plotting.regular_expert_count_label: model_spec.num_regular_experts,
        plotting.shared_experts_per_dp_rank_label: moe_config.num_shared_experts_per_dp_rank,
        plotting.regular_experts_per_dp_rank_label: moe_config.num_regular_experts_per_dp_rank,
        plotting.num_selected_experts_label: model_spec.num_selected_experts,
        plotting.num_experts_per_dp_rank_label: moe_config.num_all_experts_per_dp_rank,
        plotting.num_dp_ranks_per_complete_expert_set_label: moe_config.num_dp_ranks_per_complete_expert_set,
        plotting.num_complete_expert_sets_label: moe_config.num_complete_expert_sets,
        plotting.optimizer_label: getattr(workload_spec, "optimizer", "N/A"),
        plotting.float_size_label: workload_spec.float_size_bytes,
        plotting.mlp_ratio_label: model_spec.mlp_ratio,
        plotting.topology_dims_label: attn_tp_topology_view.shape,
        plotting.tensor_parallel_variant_label: workload_spec.tp_spec.variant,
        plotting.n_layers_label: num_layers_per_pp_stage,
        plotting.pp_stages_label: pp_stages,
        plotting.seq_len_label: seq_length,
        plotting.model_hidden_dim_label: d_model,
        plotting.time_to_train_sec_label: time_to_train_sec,
        plotting.time_to_train_day_label: time_to_train_sec / (3600 * 24),
        plotting.tokens_per_second_label: tokens_per_second,
        plotting.tokens_per_second_per_xpu_label: tokens_per_second_per_xpu,
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
        plotting.comm_vol_tp_moe_fwd_label: tp_moe_fwd.comm_vol,
        plotting.comm_vol_tp_attn_bwd_label: tp_attn_bwd.comm_vol,
        plotting.comm_vol_tp_moe_bwd_label: tp_moe_bwd.comm_vol,
        plotting.comm_vol_dp_attn_fwd_label: dp_attn_fwd.comm_vol,
        plotting.comm_vol_dp_moe_fwd_label: dp_moe_fwd.comm_vol,
        plotting.comm_vol_dp_attn_bwd_label: dp_attn_bwd.comm_vol,
        plotting.comm_vol_dp_moe_bwd_label: dp_moe_bwd.comm_vol,
        plotting.comm_time_tp_attn_fwd_label: tp_attn_fwd.total_comm_time,
        plotting.comm_time_tp_moe_fwd_label: tp_moe_fwd.total_comm_time,
        plotting.comm_time_tp_attn_bwd_label: tp_attn_bwd.total_comm_time,
        plotting.comm_time_tp_moe_bwd_label: tp_moe_bwd.total_comm_time,
        plotting.comm_time_dp_attn_fwd_label: dp_attn_fwd.total_comm_time,
        plotting.comm_time_dp_moe_fwd_label: dp_moe_fwd.total_comm_time,
        plotting.comm_time_dp_attn_bwd_label: dp_attn_bwd.total_comm_time,
        plotting.comm_time_dp_moe_bwd_label: dp_moe_bwd.total_comm_time,
        # Backward MoE all-to-all
        plotting.moe_alltoall_bwd_time_label: moe_a2a_bwd.transmission_time,
        plotting.moe_alltoall_bwd_vol_label: moe_a2a_bwd.comm_vol,
        # MoE A2A forward+backward aggregate and fraction of total overlapped time
        "MoE_A2A_Total_Time_FwdBwd": total_moe_a2a_time_fwd_bwd,
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
        plotting.moe_to_experts_time_label: moe_a2a_fwd_1.transmission_time,
        plotting.moe_from_experts_time_label: moe_a2a_fwd_2.transmission_time,
    }

    stats.update(mem_stats)

    return stats


def run_sweep(sweep_fn, base_specs, sweep_config, pass_mode):

    data = []
    sweep_iterator = generate_specs_from_sweep(base_specs=base_specs, sweep_config=sweep_config)
    for i, (sweep_params, current_specs) in enumerate(sweep_iterator):

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
            )

            data.append(stats)
        except Exception as e:
            print(f"Error in sweep iteration {i}: {e}")
            raise
            # continue

    df = pd.DataFrame(data)
    # TODO(MB) we need to add all-reduce flops here as well
    compute_time = (
        df[plotting.compute_time_forward_pass_label] + df[plotting.compute_time_backward_pass_label]
    )
    mem_time = df[plotting.mem_time_forward_pass_label] + df[plotting.mem_time_backward_pass_label]
    comm_time = (
        df[plotting.comm_time_forward_pass_label] + df[plotting.comm_time_backward_pass_label]
    )
    df[plotting.compute_time_label] = compute_time
    df[plotting.memory_time_label] = mem_time
    df[plotting.comm_time_label] = comm_time
    df[plotting.comm_compute_ratio_label] = comm_time / compute_time
    df[plotting.memory_compute_ratio_label] = mem_time / compute_time
    return df


def build_results_df(base_specs, sweep_config, pass_mode):
    df = run_sweep(get_model_stats, base_specs, sweep_config, pass_mode=pass_mode)
    df["Comm_Ineff"] = np.where(df["Comm_Compute_Ratio"] > 1.0, 1, 0.1)
    df["Comp_Util"] = np.where(df["Comm_Compute_Ratio"] > 1.0, 1.0 / df["Comm_Compute_Ratio"], 1)
    df["Comp_Comm_Ratio"] = 1.0 / df["Comm_Compute_Ratio"]
    df["Num_Tokens_Per_Batch_Per_TP_Group"] = (
        df[plotting.seq_len_label] * df[plotting.batch_size_per_dp_label]
    )
    return df


def get_detailed_mem_stats(
    attn_tp_topology,
    moe_tp_topology,
    model_spec,
    moe_config,
    workload_spec,
    tp_variant,
    x_dims,
    yz_dims,
    pass_mode,
    activation_checkpointing,
    device_for,
    node,
):
    """Helper to get detailed attn/moe memory traffic."""

    # This requires re-calculating the traffic with a different grouping.
    # We can create dummy LayerMemoryTraffic objects and call the sub-functions.

    # Forward Attention
    fwd_attn_traffic = LayerMemoryTraffic("fwd_attn", device_for, node)
    if tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        pass  # TODO: Add 2D support
    else:
        if pass_mode in [PassMode.TRAINING, PassMode.INFERENCE_PREFILL]:
            _forward_pass_prefill_attention_memory_traffic(
                fwd_attn_traffic,
                model_spec,
                workload_spec.seq_length,
                attn_tp_topology.num_chips,
                False,
                workload_spec,
            )
        else:
            _decode_attention_memory_traffic(
                fwd_attn_traffic,
                model_spec,
                workload_spec.prefill_seq_len,
                attn_tp_topology.num_chips,
                workload_spec,
            )

    # Forward MoE
    fwd_moe_traffic = LayerMemoryTraffic("fwd_moe", device_for, node)
    if tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        pass  # TODO: Add 2D support
    else:
        _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic(
            fwd_moe_traffic,
            model_spec,
            workload_spec.seq_length,
            moe_config,
            moe_tp_topology.num_chips,
            workload_spec,
        )

    bwd_attn_traffic = LayerMemoryTraffic("bwd_attn", device_for, node)
    bwd_moe_traffic = LayerMemoryTraffic("bwd_moe", device_for, node)
    if pass_mode == PassMode.TRAINING:
        if tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
            pass  # TODO: Add 2D support
        else:
            bwd_attn_traffic = _backward_attn_tp_1d_memory_traffic(
                "bwd_attn",
                model_spec,
                workload_spec,
                attn_tp_topology.num_chips,
                activation_checkpointing,
                device_for,
                node,
            )
            bwd_moe_traffic = _backward_moe_tp_1d_memory_traffic(
                "bwd_moe",
                model_spec,
                moe_config,
                moe_tp_topology,
                workload_spec,
                activation_checkpointing,
                device_for,
                node,
            )

    return fwd_attn_traffic, fwd_moe_traffic, bwd_attn_traffic, bwd_moe_traffic


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

    return {
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
        },
        "expert_placement": {
            "shared_experts_per_dp_rank": get_val(plotting.shared_experts_per_dp_rank_label),
            "regular_experts_per_dp_rank": get_val(plotting.regular_experts_per_dp_rank_label),
            "experts_per_dp_rank": get_val(plotting.num_experts_per_dp_rank_label),
            "dp_ranks_per_complete_expert_set": get_val(
                plotting.num_dp_ranks_per_complete_expert_set_label
            ),
            "num_complete_expert_sets": get_val(plotting.num_complete_expert_sets_label),
        },
        "time": {
            "total_time": fmt_t(get_val(plotting.time_to_train_sec_label)),
            "total_time_seconds": get_val(plotting.time_to_train_sec_label),
            "tokens_per_second": get_val(plotting.tokens_per_second_label),
            "tokens_per_second_per_xpu": get_val(plotting.tokens_per_second_per_xpu_label),
            "ttft_forward": fmt_t(get_val(plotting.ttft_forward_sec_label)),
            "ttft_forward_seconds": get_val(plotting.ttft_forward_sec_label),
        },
        "compute_per_batch_per_xpu": {
            "total_time": fmt_t(get_val(plotting.compute_time_label)),
            "forward": {
                "time": fmt_t(get_val(plotting.compute_time_forward_pass_label)),
                "total_flops": fmt_f(get_val(plotting.flops_forward_pass_label)),
                "embedding": {
                    "time": fmt_t(get_val(plotting.compute_time_embedding_fwd_label)),
                    "flops": fmt_f(get_val(plotting.flops_embedding_fwd_label)),
                },
                "attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.compute_time_attn_fwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
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
                    "time_per_pp_stage": fmt_t(get_val(plotting.compute_time_moe_fwd_label)),
                    "flops_per_layer": fmt_f(
                        (get_val(plotting.flops_moe_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "flops_per_pp_stage": fmt_f(get_val(plotting.flops_moe_fwd_label)),
                },
                "unembedding": {
                    "time": fmt_t(get_val(plotting.compute_time_unembedding_fwd_label)),
                    "flops": fmt_f(get_val(plotting.flops_unembedding_fwd_label)),
                },
            },
            "backward": {
                "time": fmt_t(get_val(plotting.compute_time_backward_pass_label)),
                "total_flops": fmt_f(get_val(plotting.flops_backward_pass_label)),
                "attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.compute_time_attn_bwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
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
            "forward": {
                "time": fmt_t(get_val(plotting.mem_time_forward_pass_label)),
                "total_volume_bytes": fmt_b(get_val(plotting.mem_forward_pass_label)),
                "embedding": {
                    "time": fmt_t(get_val(plotting.embedding_mem_time_fwd_label)),
                    "volume_bytes": fmt_b(get_val(plotting.embedding_mem_vol_fwd_label)),
                },
                "attn": {
                    "time_per_layer": fmt_t(attn_time_per_layer),
                    "time_per_pp_stage": fmt_t(attn_time_per_stage),
                    "volume_bytes_per_layer": fmt_b(attn_vol_per_layer),
                    "volume_bytes_per_pp_stage": fmt_b(attn_vol_per_stage),
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
                    "volume_bytes": fmt_b(get_val(plotting.unembedding_mem_vol_fwd_label)),
                },
            },
            "backward": {
                "time": fmt_t(get_val(plotting.mem_time_backward_pass_label)),
                "total_volume_bytes": fmt_b(get_val(plotting.mem_backward_pass_label)),
                "embedding": {
                    "time": fmt_t(get_val(plotting.embedding_mem_time_bwd_label)),
                    "volume_bytes": fmt_b(get_val(plotting.embedding_mem_vol_bwd_label)),
                },
                "attn": {
                    "time_per_layer": fmt_t(bwd_attn_time_per_layer),
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
                    "volume_bytes": fmt_b(get_val(plotting.unembedding_mem_vol_bwd_label)),
                },
            },
        },
        "communication_per_batch_per_xpu": {
            "total_time": fmt_t(get_val(plotting.comm_time_label)),
            "forward": {
                "total_time": fmt_t(get_val(plotting.comm_time_forward_pass_label)),
                "total_volume_bytes": fmt_b(get_val(plotting.comm_vol_forward_pass_label)),
                "unembedding_all_gather": {
                    "time": fmt_t(get_val(plotting.comm_time_unembedding_ag_label)),
                    "volume_bytes": fmt_b(get_val(plotting.comm_vol_unembedding_ag_label)),
                },
                "tensor_parallel_attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_attn_fwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_attn_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_attn_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_tp_attn_fwd_label)
                    ),
                },
                "tensor_parallel_moe": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_moe_fwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_moe_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_moe_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_tp_moe_fwd_label)),
                },
                "data_parallel_attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_attn_fwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_attn_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_attn_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_dp_attn_fwd_label)
                    ),
                },
                "data_parallel_moe": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_moe_fwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_moe_fwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_moe_fwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_dp_moe_fwd_label)),
                },
                "moe_all_to_all": {
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
                "total_volume_bytes": fmt_b(get_val(plotting.comm_vol_backward_pass_label)),
                "tensor_parallel_attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_attn_bwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_attn_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_attn_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_tp_attn_bwd_label)
                    ),
                },
                "tensor_parallel_moe": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_tp_moe_bwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_tp_moe_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_tp_moe_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_tp_moe_bwd_label)),
                },
                "data_parallel_attn": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_attn_bwd_label) or 0.0)
                        / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_attn_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_attn_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(
                        get_val(plotting.comm_vol_dp_attn_bwd_label)
                    ),
                },
                "data_parallel_moe": {
                    "time_per_layer": fmt_t(
                        (get_val(plotting.comm_time_dp_moe_bwd_label) or 0.0) / num_layers_per_stage
                    ),
                    "time_per_pp_stage": fmt_t(get_val(plotting.comm_time_dp_moe_bwd_label)),
                    "volume_bytes_per_layer": fmt_b(
                        (get_val(plotting.comm_vol_dp_moe_bwd_label) or 0) / num_layers_per_stage
                    ),
                    "volume_bytes_per_pp_stage": fmt_b(get_val(plotting.comm_vol_dp_moe_bwd_label)),
                },
                "moe_all_to_all": {
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
