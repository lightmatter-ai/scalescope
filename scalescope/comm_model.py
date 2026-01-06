"""Communication model for Transformer workloads.

Computes per-XPU forward/backward communication volumes and times for
tensor-parallel (TP), data-parallel (DP), and MoE all-to-all exchanges,
parameterized by logical topologies and workload/model specs. Outputs a
structured dictionary that aggregates batch-level stats.
"""

from scalescope.dp_strategy import get_dp_strategy
from scalescope.ep_strategy import (
    calculate_moe_comm_first_total_exchange,
    calculate_moe_comm_second_total_exchange,
)
from scalescope.model_spec import (
    attn_params_per_xpu,
    expert_params_per_xpu,
    TransformerConfig,
)
from scalescope.stats import CommStats

# for type checking
from scalescope.topology_spec import LogicalTopology
from scalescope.tp_strategy import (
    get_2d_tp_dims,
    get_tp_strategies,
    validate_2d_tp_invariants,
)
from scalescope.utils import PassMode
from scalescope.workload_spec import TensorParallelVariant, WorkloadSpec


def pipeline_parallel_comm_forward_backward():
    pass


def get_transformer_comm_stats(
    attn_dp_topology,
    moe_dp_topology,
    attn_tp_topology,
    moe_tp_topology,
    ep_topology,
    model_spec,
    moe_config,
    workload_spec,
    n_seq,
    pass_mode,
):

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    # Use layers per pipeline stage from workload spec
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()
    tensor_parallel_variant = workload_spec.tp_spec.variant
    data_parallel_variant = workload_spec.dp_spec.variant

    attn_tp_strategy, moe_tp_strategy = get_tp_strategies(tensor_parallel_variant)

    # Determine sequence length for communication based on pass mode
    comm_n_seq = 1 if pass_mode == PassMode.INFERENCE_DECODE else n_seq

    # Common explicit parameters
    transformer_config = model_spec
    attn_kwargs = {"attn_tp_topology": attn_tp_topology}
    moe_kwargs = {"moe_tp_topology": moe_tp_topology}

    # Add kwargs specific to 2D strategy
    if tensor_parallel_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        # Derive and validate dims separately for attention and MoE (they may differ)
        attn_x_dims, attn_yz_dims = get_2d_tp_dims(attn_tp_topology)
        validate_2d_tp_invariants(model_spec, attn_tp_topology, attn_x_dims, attn_yz_dims)
        attn_kwargs.update(
            {
                "attn_tp_topology": attn_tp_topology,
                "x_dims": attn_x_dims,
                "yz_dims": attn_yz_dims,
            }
        )

        moe_x_dims, moe_yz_dims = get_2d_tp_dims(moe_tp_topology)
        validate_2d_tp_invariants(model_spec, moe_tp_topology, moe_x_dims, moe_yz_dims)
        moe_kwargs.update(
            {
                "moe_tp_topology": moe_tp_topology,
                "x_dims": moe_x_dims,
                "yz_dims": moe_yz_dims,
            }
        )

    # Calculate forward pass TP communication
    tp_attn_fwd_comm = num_layers_per_pp_stage * attn_tp_strategy.forward_comm(
        transformer_config, moe_config, workload_spec, comm_n_seq, **attn_kwargs
    )
    tp_moe_fwd_comm = num_layers_per_pp_stage * moe_tp_strategy.forward_comm(
        transformer_config, moe_config, workload_spec, comm_n_seq, **moe_kwargs
    )

    # Calculate backward pass TP communication
    if pass_mode == PassMode.TRAINING:
        tp_attn_bwd_comm = num_layers_per_pp_stage * attn_tp_strategy.backward_comm(
            transformer_config, moe_config, workload_spec, comm_n_seq, **attn_kwargs
        )
        tp_moe_bwd_comm = num_layers_per_pp_stage * moe_tp_strategy.backward_comm(
            transformer_config, moe_config, workload_spec, comm_n_seq, **moe_kwargs
        )
    else:
        tp_attn_bwd_comm = CommStats()
        tp_moe_bwd_comm = CommStats()

    # MoE communication cost for the forward pass
    num_global_batch_tokens = attn_dp_topology.num_chips * n_batch * n_seq
    moe_stats_1 = calculate_moe_comm_first_total_exchange(
        ep_topology,
        model_spec,
        moe_config,
        num_global_batch_tokens,
        float_size=workload_spec.float_size_bytes,
        workload_spec=workload_spec,
    )
    moe_stats_2 = calculate_moe_comm_second_total_exchange(
        ep_topology_view=ep_topology,
        transformer_config=model_spec,
        moe_config=moe_config,
        num_global_batch_tokens=num_global_batch_tokens,
        float_size=workload_spec.float_size_bytes,
        workload_spec=workload_spec,
    )

    if pass_mode == PassMode.TRAINING:

        non_expert_params_per_layer_per_xpu = attn_params_per_xpu(model_spec, attn_tp_topology)
        expert_weights_per_layer_per_xpu = expert_params_per_xpu(model_spec, moe_config)

        # In the future, we might use different strategies for attn and moe.
        attn_dp_strategy = get_dp_strategy(data_parallel_variant)
        moe_dp_strategy = get_dp_strategy(data_parallel_variant)

        # Forward pass DP communication
        dp_attn_fwd_comm = attn_dp_strategy.forward_comm(
            attn_dp_topology,
            num_layers_per_pp_stage,
            non_expert_params_per_layer_per_xpu,
            workload_spec.float_size_bytes,
        )
        dp_moe_fwd_comm = moe_dp_strategy.forward_comm(
            moe_dp_topology,
            num_layers_per_pp_stage,
            expert_weights_per_layer_per_xpu,
            workload_spec.float_size_bytes,
        )

        # Backward pass DP communication
        dp_attn_bwd_comm = attn_dp_strategy.backward_comm(
            attn_dp_topology,
            num_layers_per_pp_stage,
            non_expert_params_per_layer_per_xpu,
            workload_spec.float_size_bytes,
        )
        dp_moe_bwd_comm = moe_dp_strategy.backward_comm(
            moe_dp_topology,
            num_layers_per_pp_stage,
            expert_weights_per_layer_per_xpu,
            workload_spec.float_size_bytes,
        )
    else:
        dp_attn_fwd_comm = CommStats()
        dp_moe_fwd_comm = CommStats()
        dp_attn_bwd_comm = CommStats()
        dp_moe_bwd_comm = CommStats()

    # Return a dictionary of all comm stats for detailed breakdown
    return {
        "tp_attn_fwd": tp_attn_fwd_comm,
        "tp_moe_fwd": tp_moe_fwd_comm,
        "tp_attn_bwd": tp_attn_bwd_comm,
        "tp_moe_bwd": tp_moe_bwd_comm,
        "dp_attn_fwd": dp_attn_fwd_comm,
        "dp_moe_fwd": dp_moe_fwd_comm,
        "dp_attn_bwd": dp_attn_bwd_comm,
        "dp_moe_bwd": dp_moe_bwd_comm,
        "moe_a2a_fwd_1": moe_stats_1,
        "moe_a2a_fwd_2": moe_stats_2,
    }


def calculate_unembedding_comm(
    tp_topology: "LogicalTopology",
    model_spec: "TransformerConfig",
    workload_spec: "WorkloadSpec",
):
    """
    Typical practice (vocab-parallel loss or distributed top-k) avoids full-logit all-gather,
    so we model no additional unembedding communication by default.
    """
    return CommStats()
