"""Communication model for Transformer workloads.

Computes per-XPU forward/backward communication volumes and times for
tensor-parallel (TP), data-parallel (DP), and MoE all-to-all exchanges,
parameterized by logical topologies and workload/model specs. Outputs a
structured dictionary that aggregates batch-level stats.
"""

from scalescope.decode_layout_trace import NULL_LAYOUT_TRACE
from scalescope.dp_strategy import get_dp_strategy
from scalescope.ep_strategy import (
    calculate_moe_combine_decode,
    calculate_moe_comm_first_total_exchange,
    calculate_moe_comm_second_total_exchange,
    calculate_moe_decode_cross_domain_exchange,
    resolve_moe_token_scopes,
)
from scalescope.model_spec import (
    attn_params_per_xpu,
    expert_params_per_xpu,
    TransformerConfig,
)
from scalescope.network_topology_model import hockney_model_send_recv_on_nd_topology
from scalescope.stats import CommStats

# for type checking
from scalescope.topology_spec import LogicalTopology
from scalescope.tp_strategy import (
    AttnForwardCommPhases,
    get_attn_tp_strategy,
    get_moe_tp_strategy,
    resolve_tp2d_split,
    validate_2d_tp_invariants_attn,
    validate_2d_tp_invariants_moe,
    validate_2d_tp_invariants_proj,
)
from scalescope.utils import PassMode
from scalescope.workload_spec import (
    MoERouterSyncMode,
    TensorParallelVariant,
    WorkloadSpec,
)


def pipeline_parallel_comm_forward_backward(
    *,
    pp_topology: "LogicalTopology",
    attn_tp_topology: "LogicalTopology",
    model_spec: "TransformerConfig",
    workload_spec: "WorkloadSpec",
    comm_n_seq: int,
    pass_mode: PassMode,
) -> tuple[CommStats, CommStats]:
    """Model explicit PP stage-boundary transfers (activations / activation-grads).

    This models the scale-out fabric cost of moving the residual stream tensor between
    pipeline stages. We approximate each boundary transfer as a simple send/recv of a
    single contiguous buffer sized like:
        [batch_size_per_dp, comm_n_seq, d_model]

    Notes / simplifications:
    - This is a bandwidth-oriented model; we treat the whole per-DP batch as one message
      per stage boundary (so latency does not scale with microbatch count).
    - Per-chip volume is approximated by sharding along the attention TP domain size.
      (I.e., each TP rank sends its shard to the corresponding TP rank in the next stage.)
    """
    pp_size = pp_topology.num_chips
    if pp_size <= 1:
        return CommStats(), CommStats()

    tp_size = max(1, attn_tp_topology.num_chips)
    batch_size_per_dp = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size_bytes = workload_spec.float_size_bytes

    # Per-chip activation shard volume for a single stage-boundary transfer.
    bytes_per_chip = (
        batch_size_per_dp * comm_n_seq * model_spec.d_model * float_size_bytes
    ) / tp_size

    # Simple P2P send/recv model on the PP interconnect slice (implemented in low-level net model).
    per_boundary = hockney_model_send_recv_on_nd_topology(
        topology=pp_topology, bytes_per_sender=bytes_per_chip, float_size_bytes=float_size_bytes
    )

    num_boundaries = pp_size - 1
    pp_fwd = per_boundary * num_boundaries

    match pass_mode:
        case PassMode.TRAINING:
            # Activation-gradient transfers mirror activation size.
            pp_bwd = per_boundary * num_boundaries
        case PassMode.INFERENCE_PREFILL | PassMode.INFERENCE_DECODE:
            pp_bwd = CommStats()
        case _:
            raise ValueError(f"Unrecognized pass_mode: {pass_mode}")

    return pp_fwd, pp_bwd


def get_transformer_comm_stats(
    attn_dp_topology,
    moe_dp_topology,
    pp_topology,
    attn_tp_topology,
    proj_tp_topology,
    moe_tp_topology,
    ep_topology,
    expert_set_topology,
    model_spec,
    moe_config,
    workload_spec,
    n_seq,
    pass_mode,
    layout_trace=NULL_LAYOUT_TRACE,
):
    # Use layers per pipeline stage from workload spec
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()
    # Allow different TP variants for Attn vs MoE
    attn_tp_variant = workload_spec.get_attn_tp_spec().variant
    proj_tp_spec = workload_spec.get_proj_tp_spec()
    proj_tp_variant = proj_tp_spec.variant
    moe_tp_variant = workload_spec.get_moe_tp_spec().variant
    data_parallel_variant = workload_spec.dp_spec.variant
    attn_tp_strategy = get_attn_tp_strategy(attn_tp_variant)
    moe_tp_strategy = get_moe_tp_strategy(moe_tp_variant)

    # Determine sequence length for communication based on pass mode
    match pass_mode:
        case PassMode.INFERENCE_DECODE:
            comm_n_seq = 1
        case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
            comm_n_seq = n_seq
        case _:
            raise ValueError(f"Unrecognized pass_mode: {pass_mode}")

    # Explicit PP boundary transfers (scale-out domain).
    pp_fwd_comm, pp_bwd_comm = pipeline_parallel_comm_forward_backward(
        pp_topology=pp_topology,
        attn_tp_topology=attn_tp_topology,
        model_spec=model_spec,
        workload_spec=workload_spec,
        comm_n_seq=comm_n_seq,
        pass_mode=pass_mode,
    )

    # Common explicit parameters
    transformer_config = model_spec
    if proj_tp_topology is None:
        raise ValueError("get_transformer_comm_stats requires proj_tp_topology (explicit).")
    if proj_tp_variant != attn_tp_variant:
        raise ValueError(
            f"tp_proj_spec.variant ({proj_tp_variant}) must match tp_attn_spec.variant ({attn_tp_variant}) for now."
        )
    attn_kwargs = {
        "attn_tp_topology": attn_tp_topology,
        "proj_tp_topology": proj_tp_topology,
        "layout_trace": layout_trace,
    }
    moe_kwargs = {
        "moe_tp_topology": moe_tp_topology,
        "attn_tp_topology": attn_tp_topology,
        "proj_x_topology": None,
        "proj_yz_topology": None,
        "expert_set_topology": expert_set_topology,
        "layout_trace": layout_trace,
    }

    # Add kwargs specific to 2D strategy (dims and optional sub-views)
    attn_spec = workload_spec.get_attn_tp_spec()
    if attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        # split_2d is mandatory for 2D TP (explicit X/YZ).
        if attn_spec.split_2d is None:
            raise ValueError(
                "tp_attn_spec.split_2d must be set when tp_attn_spec.variant is SHARD_WEIGHTS_2D"
            )
        resolved_attn = resolve_tp2d_split(
            tp_view=attn_tp_topology, split_2d=attn_spec.split_2d, view_name="TP_Attn"
        )
        # Projection split: required iff projections use TP-2D (variant=SHARD_WEIGHTS_2D).
        # For TP-1D projections there is no X/YZ split to specify.
        if proj_tp_spec.split_2d is None:
            raise ValueError(
                "tp_proj_spec.split_2d must be set when tp_proj_spec.variant is SHARD_WEIGHTS_2D"
            )
        resolved_proj = resolve_tp2d_split(
            tp_view=proj_tp_topology, split_2d=proj_tp_spec.split_2d, view_name="TP_Proj"
        )
        attn_kwargs.update(
            {
                "x_dims": resolved_attn.x_dims,
                "yz_dims": resolved_attn.yz_dims,
                "x_topology": resolved_attn.x_view,
                "yz_topology": resolved_attn.yz_view,
                "proj_x_dims": resolved_proj.x_dims,
                "proj_yz_dims": resolved_proj.yz_dims,
                "proj_x_topology": resolved_proj.x_view,
                "proj_yz_topology": resolved_proj.yz_view,
            }
        )
        # The MoE consumer needs the projection sub-views to price the
        # output-projection -> MoE layout handoff.  In decode, output
        # projection leaves D sharded over TP_Proj_X x TP_Proj_Y_eff.
        moe_kwargs.update(
            {
                "proj_x_topology": resolved_proj.x_view,
                "proj_yz_topology": resolved_proj.yz_view,
            }
        )
        validate_2d_tp_invariants_attn(
            model_spec, attn_tp_topology, resolved_attn.x_dims, resolved_attn.yz_dims
        )
        validate_2d_tp_invariants_proj(
            model_spec, proj_tp_topology, resolved_proj.x_dims, resolved_proj.yz_dims
        )

    moe_spec = workload_spec.get_moe_tp_spec()
    tp_moe_x_shards = 1
    tp_moe_y_shards = 1
    if moe_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        # split_2d is mandatory for 2D TP (explicit X/YZ).
        if moe_spec.split_2d is None:
            raise ValueError(
                "tp_moe_spec.split_2d must be set when tp_moe_spec.variant is SHARD_WEIGHTS_2D"
            )
        resolved = resolve_tp2d_split(
            tp_view=moe_tp_topology, split_2d=moe_spec.split_2d, view_name="TP_MoE"
        )
        tp_moe_x_shards = resolved.tp_x
        tp_moe_y_shards = resolved.tp_yz
        moe_kwargs.update(
            {
                "x_dims": resolved.x_dims,
                "yz_dims": resolved.yz_dims,
                "x_topology": resolved.x_view,
                "yz_topology": resolved.yz_view,
            }
        )
        validate_2d_tp_invariants_moe(
            model_spec, moe_tp_topology, moe_kwargs["x_dims"], moe_kwargs["yz_dims"]
        )

    # Calculate TP communication (stage-specific strategy methods)
    phases: AttnForwardCommPhases
    match pass_mode:
        case PassMode.TRAINING:
            phases = attn_tp_strategy.forward_comm_training_phases(
                transformer_config, moe_config, workload_spec, comm_n_seq, **attn_kwargs
            )
            tp_attn_fwd_comm = num_layers_per_pp_stage * phases.total
            tp_moe_fwd_comm = num_layers_per_pp_stage * moe_tp_strategy.forward_comm_training(
                transformer_config, moe_config, workload_spec, comm_n_seq, **moe_kwargs
            )
            tp_attn_bwd_comm = num_layers_per_pp_stage * attn_tp_strategy.backward_comm_training(
                transformer_config, moe_config, workload_spec, comm_n_seq, **attn_kwargs
            )
            tp_moe_bwd_comm = num_layers_per_pp_stage * moe_tp_strategy.backward_comm_training(
                transformer_config, moe_config, workload_spec, comm_n_seq, **moe_kwargs
            )
        case PassMode.INFERENCE_PREFILL:
            phases = attn_tp_strategy.forward_comm_prefill_phases(
                transformer_config, moe_config, workload_spec, comm_n_seq, **attn_kwargs
            )
            tp_attn_fwd_comm = num_layers_per_pp_stage * phases.total
            tp_moe_fwd_comm = num_layers_per_pp_stage * moe_tp_strategy.forward_comm_prefill(
                transformer_config, moe_config, workload_spec, comm_n_seq, **moe_kwargs
            )
            tp_attn_bwd_comm = CommStats()
            tp_moe_bwd_comm = CommStats()
        case PassMode.INFERENCE_DECODE:
            phases = attn_tp_strategy.forward_comm_decode_phases(
                transformer_config, moe_config, workload_spec, comm_n_seq, **attn_kwargs
            )
            tp_attn_fwd_comm = num_layers_per_pp_stage * phases.total
            tp_moe_fwd_comm = num_layers_per_pp_stage * moe_tp_strategy.forward_comm_decode(
                transformer_config, moe_config, workload_spec, comm_n_seq, **moe_kwargs
            )
            tp_attn_bwd_comm = CommStats()
            tp_moe_bwd_comm = CommStats()
        case _:
            raise ValueError(f"Unrecognized pass_mode: {pass_mode}")

    # Per-phase breakdown for attention TP comm
    tp_attn_fwd_phase1 = num_layers_per_pp_stage * phases.phase1_input_proj
    tp_attn_fwd_phase2 = num_layers_per_pp_stage * phases.phase2_attn_core
    tp_attn_fwd_phase3 = num_layers_per_pp_stage * phases.phase3_output_proj

    # MoE communication cost for the forward pass
    resolved_batches = workload_spec.get_resolved_batches()
    if resolved_batches.attention_domain_count != attn_dp_topology.num_chips:
        raise ValueError(
            "Resolved attention-domain count does not match the DP topology: "
            f"{resolved_batches.attention_domain_count} != "
            f"{attn_dp_topology.num_chips}"
        )
    if resolved_batches.complete_expert_set_count != moe_config.num_complete_expert_sets:
        raise ValueError(
            "Resolved expert-set count does not match expert placement: "
            f"{resolved_batches.complete_expert_set_count} != "
            f"{moe_config.num_complete_expert_sets}"
        )
    token_scopes = resolve_moe_token_scopes(
        resolved_batches=resolved_batches,
        tokens_per_sequence=comm_n_seq,
        attn_domains_per_full_expert_set=moe_config.attn_domains_per_full_expert_set,
        full_expert_sets_per_attn_domain=moe_config.full_expert_sets_per_attn_domain,
    )

    # EP / MoE A2A: strategy functions return per-layer stats (same convention
    # as TP strategies above); multiply by num_layers_per_pp_stage here.
    if pass_mode == PassMode.INFERENCE_DECODE:
        attn_domains_per_set = moe_config.attn_domains_per_full_expert_set
        full_set_xpus = moe_config.full_expert_set_xpu_count
        projection_groups = workload_spec.projection_groups
        if attn_domains_per_set > 1:
            # Multi-attention-domain decode: the expert set is larger than one
            # attention TP domain, so dispatch is no longer free.  Two sub-cases:
            #   - scale-up-resident wide-EP set: full_set_xpus <= scale_up_domain
            #     size → all feeder domains and experts share one scale-up fabric.
            #     Single-fabric A2A on EP_SU for both dispatch and combine.
            #   - scale-out-spanning wide-EP set: full_set_xpus >
            #     scale_up_domain size → one expert set spans multiple scale-up
            #     domains. The exact handoff prepositions local-owner shards, so
            #     local dispatch/return is a zero-byte EP_SU no-op; only
            #     remote-owner A2Av is charged on EP_SO.
            try:
                scale_up_domain_size = ep_topology.get_mapping_for_tag("EP_SU").size
            except Exception as e:
                raise ValueError(
                    "EP topology must include tag `EP_SU` to evaluate "
                    "multi-attention-domain decode feasibility."
                ) from e
            if full_set_xpus > scale_up_domain_size:
                if moe_config.num_complete_expert_sets != 1:
                    raise NotImplementedError(
                        "Scale-out-spanning wide-EP decode currently requires exactly "
                        "one complete expert set"
                    )
                if moe_config.full_expert_sets_per_attn_domain != 1:
                    raise NotImplementedError(
                        "Scale-out-spanning wide-EP decode does not support multiple "
                        "complete expert sets inside one attention domain"
                    )
                if moe_config.router_sync_mode in (
                    MoERouterSyncMode.SHARDED_ROUTER,
                    MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER,
                ):
                    if moe_config.router_shard_factor != tp_moe_y_shards:
                        raise ValueError(
                            "Feeder-local sharded router requires router_shard_factor "
                            "to equal TP_MoE_Y; got "
                            f"{moe_config.router_shard_factor} != {tp_moe_y_shards}"
                        )
                    ep_so_mapping = ep_topology.get_mapping_for_tag("EP_SO")
                    router_physical_axes = {
                        mapping.maps_to_physical_dimension_name
                        for mapping in resolved.yz_view.dimensions
                        if mapping.size > 1
                    }
                    if (
                        ep_so_mapping.size > 1
                        and ep_so_mapping.maps_to_physical_dimension_name in router_physical_axes
                    ):
                        raise NotImplementedError(
                            "Feeder-local sharded router cannot use the EP_SO physical "
                            "axis; router logits synchronization must remain inside each "
                            "attention domain"
                        )
                if resolved_proj.yz_view.num_chips != tp_moe_y_shards:
                    raise NotImplementedError(
                        "Scale-out-spanning wide-EP decode currently requires "
                        "TP_Proj_Y_eff == TP_MoE_Y; got "
                        f"{resolved_proj.yz_view.num_chips} != {tp_moe_y_shards}"
                    )
                moe_stats_1 = num_layers_per_pp_stage * calculate_moe_decode_cross_domain_exchange(
                    ep_topology_view=ep_topology,
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    workload_spec=workload_spec,
                    token_scopes=token_scopes,
                    float_size=workload_spec.float_size_bytes,
                    tp_moe_y_shards=tp_moe_y_shards,
                    layout_trace=layout_trace,
                    a2a_leg="dispatch",
                )
                moe_stats_2 = num_layers_per_pp_stage * calculate_moe_decode_cross_domain_exchange(
                    ep_topology_view=ep_topology,
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    workload_spec=workload_spec,
                    token_scopes=token_scopes,
                    float_size=workload_spec.float_size_bytes,
                    tp_moe_y_shards=tp_moe_y_shards,
                    layout_trace=layout_trace,
                    a2a_leg="combine",
                )
                local_group_count = moe_config.num_colocated_groups_per_set // attn_domains_per_set
                moe_stats_2 += num_layers_per_pp_stage * calculate_moe_combine_decode(
                    ep_topology_view=ep_topology,
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    token_scopes=token_scopes,
                    attn_tp_xpus=attn_tp_topology.num_chips,
                    float_size=workload_spec.float_size_bytes,
                    tp_moe_x_shards=tp_moe_x_shards,
                    tp_moe_y_shards=tp_moe_y_shards,
                    num_local_colocated_groups=local_group_count,
                    record_layout_output=True,
                    projection_groups=projection_groups,
                    attn_x_topology=resolved_attn.x_view,
                    attn_yz_topology=resolved_attn.yz_view,
                    proj_x_topology=resolved_proj.x_view,
                    proj_yz_topology=resolved_proj.yz_view,
                    moe_x_topology=resolved.x_view,
                    moe_yz_topology=resolved.yz_view,
                    expert_set_topology=None,
                    layout_trace=layout_trace,
                )
                # Exact scale-out-spanning dispatch/return has been fully priced.
                # Do not also charge the approximate B1 hierarchical primitive.
            else:
                # Scale-up-resident wide-EP set: reuse the prefill A2A primitive
                # for both dispatch and combine. When the set fits in one scale-up
                # domain, the second-fastest EP tag carries zero bytes.
                moe_stats_1 = num_layers_per_pp_stage * calculate_moe_comm_first_total_exchange(
                    ep_topology,
                    model_spec,
                    moe_config,
                    token_scopes,
                    float_size=workload_spec.float_size_bytes,
                    workload_spec=workload_spec,
                    tp_moe_y_shards=tp_moe_y_shards,
                    layout_trace=layout_trace,
                )
                moe_stats_2 = num_layers_per_pp_stage * calculate_moe_comm_second_total_exchange(
                    ep_topology_view=ep_topology,
                    transformer_config=model_spec,
                    moe_config=moe_config,
                    token_scopes=token_scopes,
                    float_size=workload_spec.float_size_bytes,
                    workload_spec=workload_spec,
                    tp_moe_y_shards=tp_moe_y_shards,
                    layout_trace=layout_trace,
                )
        else:
            # One attention domain feeds one or more complete expert sets.
            # With Y-aligned sets, token selection is local; combine performs
            # exact per-set reduction followed by cross-set batch gathering.
            if attn_tp_variant != TensorParallelVariant.SHARD_WEIGHTS_2D:
                raise NotImplementedError(
                    "Exact single-domain decode combine requires 2D attention/projection TP."
                )
            if moe_tp_variant != TensorParallelVariant.SHARD_WEIGHTS_2D:
                raise NotImplementedError("Exact single-domain decode combine requires 2D MoE TP.")
            moe_stats_1 = CommStats()
            moe_stats_2 = num_layers_per_pp_stage * calculate_moe_combine_decode(
                ep_topology_view=ep_topology,
                transformer_config=model_spec,
                moe_config=moe_config,
                token_scopes=token_scopes,
                attn_tp_xpus=attn_tp_topology.num_chips,
                float_size=workload_spec.float_size_bytes,
                tp_moe_x_shards=tp_moe_x_shards,
                tp_moe_y_shards=tp_moe_y_shards,
                num_local_colocated_groups=moe_config.num_colocated_groups_per_set,
                record_layout_output=False,
                projection_groups=projection_groups,
                attn_x_topology=resolved_attn.x_view,
                attn_yz_topology=resolved_attn.yz_view,
                proj_x_topology=resolved_proj.x_view,
                proj_yz_topology=resolved_proj.yz_view,
                moe_x_topology=resolved.x_view,
                moe_yz_topology=resolved.yz_view,
                expert_set_topology=expert_set_topology,
                layout_trace=layout_trace,
            )
    else:
        # Training / prefill: symmetric A2A dispatch + combine.
        moe_stats_1 = num_layers_per_pp_stage * calculate_moe_comm_first_total_exchange(
            ep_topology,
            model_spec,
            moe_config,
            token_scopes,
            float_size=workload_spec.float_size_bytes,
            workload_spec=workload_spec,
            tp_moe_y_shards=tp_moe_y_shards,
        )
        moe_stats_2 = num_layers_per_pp_stage * calculate_moe_comm_second_total_exchange(
            ep_topology_view=ep_topology,
            transformer_config=model_spec,
            moe_config=moe_config,
            token_scopes=token_scopes,
            float_size=workload_spec.float_size_bytes,
            workload_spec=workload_spec,
            tp_moe_y_shards=tp_moe_y_shards,
        )

    match pass_mode:
        case PassMode.TRAINING:
            non_expert_params_per_layer_per_xpu = attn_params_per_xpu(model_spec, attn_tp_topology)
            expert_weights_per_layer_per_xpu = expert_params_per_xpu(model_spec, moe_config)
            # Include router/gate parameters in MoE DP gradient synchronization.
            # Router param shape is approximately [D, E_regular], sharded by
            # router_shard_factor when applicable.
            router_shard_factor = moe_config.router_shard_factor
            if router_shard_factor < 1:
                raise ValueError(f"Invalid router_shard_factor: {router_shard_factor}")
            router_weights_per_layer_per_xpu = int(
                (model_spec.d_model * model_spec.num_regular_experts) / router_shard_factor
            )
            moe_params_per_layer_per_xpu = (
                expert_weights_per_layer_per_xpu + router_weights_per_layer_per_xpu
            )

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
                moe_params_per_layer_per_xpu,
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
                moe_params_per_layer_per_xpu,
                workload_spec.float_size_bytes,
            )
        case PassMode.INFERENCE_PREFILL | PassMode.INFERENCE_DECODE:
            dp_attn_fwd_comm = CommStats()
            dp_moe_fwd_comm = CommStats()
            dp_attn_bwd_comm = CommStats()
            dp_moe_bwd_comm = CommStats()
        case _:
            raise ValueError(f"Unrecognized pass_mode: {pass_mode}")

    # Return a dictionary of all comm stats for detailed breakdown
    return {
        "tp_attn_fwd": tp_attn_fwd_comm,
        "tp_attn_fwd_phase1": tp_attn_fwd_phase1,
        "tp_attn_fwd_phase2": tp_attn_fwd_phase2,
        "tp_attn_fwd_phase3": tp_attn_fwd_phase3,
        "tp_moe_fwd": tp_moe_fwd_comm,
        "tp_attn_bwd": tp_attn_bwd_comm,
        "tp_moe_bwd": tp_moe_bwd_comm,
        "dp_attn_fwd": dp_attn_fwd_comm,
        "dp_moe_fwd": dp_moe_fwd_comm,
        "dp_attn_bwd": dp_attn_bwd_comm,
        "dp_moe_bwd": dp_moe_bwd_comm,
        "pp_fwd": pp_fwd_comm,
        "pp_bwd": pp_bwd_comm,
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
