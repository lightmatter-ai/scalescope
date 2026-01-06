"""Expert placement and per-XPU MoE layout inference.

Infers how distinct experts are placed across chips, taling into account
expert collocation and parallelization strategies.
"""

from dataclasses import dataclass
from math import ceil

from scalescope.model_spec import TransformerConfig
from scalescope.network_topology_model import (
    hockney_model_total_exchange_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.workload_spec import MoESpecV2


@dataclass
class ExpertXPUPlacement:
    num_shared_experts_per_dp_rank: int = 0
    num_regular_experts_per_dp_rank: int = 0
    num_all_experts_per_dp_rank: int = 0
    total_distinct_experts_per_set: int = 0
    regular_expert_traffic_multiplier: float = 0.0
    num_dp_ranks_per_complete_expert_set: int = 1
    num_complete_expert_sets: int = 1
    # EP/placement-wide fields needed by comm/compute
    full_expert_set_xpu_count: int = 1
    expert_high_bandwidth_group_size_xpus: int | None = None
    num_colocated_experts: int = 1
    ep_chips_per_pp_stage: int = 0


def infer_expert_placement_per_xpu(
    *,
    model_spec: TransformerConfig,
    moe_v2: MoESpecV2,
    attn_tp_size: int,
    expert_tp_size: int,
    ep_chips_per_pp_stage: int,
) -> ExpertXPUPlacement:
    """Infer per-XPU expert placement within a DP group.

    Compute how many routed and shared experts are hosted per DP rank,
    how many DP ranks are required to hold a complete expert set, and how many complete
    expert sets fit into the EP domain of a single PP stage. It also derives helper
    quantities used by communication and compute models (e.g., traffic multipliers).

    NB: shared experts are not yet supported

    Args:
    - model_spec: Transformer configuration. Uses:
        - num_regular_experts
        - num_shared_experts
        - num_selected_experts (for traffic multiplier)
    - moe_v2: MoE configuration (MoESpecV2). Uses:
        - num_colocated_experts: number of experts colocated per XPU (>= 1)
        - shared_expert_copies: replication factor for shared experts (optional)
        - expert_imbalance: traffic multiplier (>= 1.0)
        - expert_high_bandwidth_group_size_xpus (optional pass-through)
    - attn_tp_size: TP size of the attention path (chips per DP rank for TP_Attn).
      Used to divide a complete expert set across DP ranks.
    - expert_tp_size: expert tensor-parallel degree (chips per expert).
    - ep_chips_per_pp_stage: total EP chips available per PP stage (DP_Attn * TP_Attn).

    Returns
    - ExpertXPUPlacement
    """

    num_shared = model_spec.num_shared_experts
    num_colocated_experts = int(moe_v2.num_colocated_experts)

    num_regular = model_spec.num_regular_experts
    total_distinct_experts = num_regular + num_shared * (moe_v2.shared_expert_copies or 1)
    # Colocation reduces the number of XPUs needed to host a full set of distinct experts,
    # regardless of whether experts are TP-sharded. Account for both expert TP degree and colocation.
    full_expert_set_xpu_count = ceil(
        (total_distinct_experts * expert_tp_size) / num_colocated_experts
    )
    num_dp_ranks_per_expert_group = full_expert_set_xpu_count // attn_tp_size
    num_routed_experts_per_dp_rank = num_regular // num_dp_ranks_per_expert_group
    num_shared_experts_per_dp_rank = (
        num_shared * (moe_v2.shared_expert_copies or 1) // num_dp_ranks_per_expert_group
    )
    num_of_all_experts_per_dp_rank = num_routed_experts_per_dp_rank + num_shared_experts_per_dp_rank

    # Determine how many complete expert sets fit into the available EP domain per PP stage
    if ep_chips_per_pp_stage % full_expert_set_xpu_count != 0:
        raise ValueError(
            f"EP chips ({ep_chips_per_pp_stage}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_complete_expert_sets = ep_chips_per_pp_stage // full_expert_set_xpu_count

    # Regular expert traffic multiplier
    regular_expert_traffic_multiplier = moe_v2.expert_imbalance * model_spec.num_selected_experts

    return ExpertXPUPlacement(
        num_shared_experts_per_dp_rank=num_shared_experts_per_dp_rank,
        num_regular_experts_per_dp_rank=num_routed_experts_per_dp_rank,
        num_all_experts_per_dp_rank=num_of_all_experts_per_dp_rank,
        total_distinct_experts_per_set=int(total_distinct_experts),
        regular_expert_traffic_multiplier=regular_expert_traffic_multiplier,
        num_dp_ranks_per_complete_expert_set=num_dp_ranks_per_expert_group,
        num_complete_expert_sets=int(num_complete_expert_sets),
        # EP/placement-wide fields
        full_expert_set_xpu_count=int(full_expert_set_xpu_count),
        expert_high_bandwidth_group_size_xpus=moe_v2.expert_high_bandwidth_group_size_xpus,
        num_colocated_experts=num_colocated_experts,
        ep_chips_per_pp_stage=int(ep_chips_per_pp_stage),
    )


def calculate_moe_comm_first_total_exchange(
    ep_topology_view,
    transformer_config,
    moe_config,
    num_global_batch_tokens,
    float_size,
    workload_spec=None,
):
    return calculate_moe_comm_hierarchical_total_exchange(
        ep_topology_view,
        transformer_config,
        moe_config,
        num_global_batch_tokens,
        float_size,
        workload_spec=workload_spec,
    )


def calculate_moe_comm_second_total_exchange(
    ep_topology_view,
    transformer_config,
    moe_config,
    num_global_batch_tokens,
    float_size,
    workload_spec=None,
):
    # assume for now that we get the same stats when we are returning tokens after they are processed by teh experts
    return calculate_moe_comm_first_total_exchange(
        ep_topology_view,
        transformer_config,
        moe_config,
        num_global_batch_tokens,
        float_size,
        workload_spec=workload_spec,
    )


def calculate_moe_comm_hierarchical_total_exchange(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    num_global_batch_tokens: int,
    float_size: int,
    workload_spec=None,
):
    if transformer_config.num_regular_experts == 0:
        return CommStats()

    d_model = transformer_config.d_model
    # Use per-PP-stage layers if provided via workload_spec; otherwise fall back to model's total layers
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()

    full_expert_set_xpu_count = int(getattr(moe_config, "full_expert_set_xpu_count"))
    routed_expert_traffic_multiplier = float(
        getattr(moe_config, "regular_expert_traffic_multiplier")
    )
    hb = getattr(moe_config, "expert_high_bandwidth_group_size_xpus", None)
    high_bw_group_size_xpus = int(hb or full_expert_set_xpu_count)

    # Use EP chips per PP stage (DP_Attn * TP_Attn) if provided by caller via merged placement
    ep_total_xpus = int(
        getattr(moe_config, "ep_chips_per_pp_stage", 0) or ep_topology_view.num_chips
    )
    if ep_total_xpus % max(1, full_expert_set_xpu_count) != 0:
        raise ValueError(
            f"EP chips ({ep_total_xpus}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_full_sets = ep_total_xpus // max(1, full_expert_set_xpu_count)

    tokens_per_set = num_global_batch_tokens / max(1, num_full_sets)
    tokens_per_xpu = tokens_per_set / max(1, full_expert_set_xpu_count)

    # Ensure HB group size does not exceed the set footprint
    if high_bw_group_size_xpus > full_expert_set_xpu_count:
        high_bw_group_size_xpus = full_expert_set_xpu_count
    frac_local = min(1.0, high_bw_group_size_xpus / max(1, full_expert_set_xpu_count))
    frac_remote = max(0.0, 1.0 - frac_local)

    bytes_per_token = d_model * float_size
    P_intra_bytes_per_xpu = (
        tokens_per_xpu * routed_expert_traffic_multiplier * frac_local * bytes_per_token
    )
    P_inter_bytes_per_xpu = (
        tokens_per_xpu * routed_expert_traffic_multiplier * frac_remote * bytes_per_token
    )

    comm_stats_intra_per_layer = CommStats()
    comm_stats_inter_per_layer = CommStats()

    topology_view_speed_sorted_tags = ep_topology_view.get_tags_sorted_by_speed()

    # Intra
    if high_bw_group_size_xpus > 1 and P_intra_bytes_per_xpu > 0:
        fastest_tag = topology_view_speed_sorted_tags[0]
        intra_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
            view_name="EP_Intra_Virtual",
            base_tag=fastest_tag,
            virtual_size=high_bw_group_size_xpus,
            global_sharing_context="max_exclusive",
        )
        comm_stats_intra_per_layer = hockney_model_total_exchange_on_nd_topology(
            intra_view,
            P_total_send_bytes_per_node=P_intra_bytes_per_xpu,
            float_size_bytes=float_size,
        )

    # Inter
    inter_size = max(0, full_expert_set_xpu_count - high_bw_group_size_xpus)
    if inter_size > 1 and P_inter_bytes_per_xpu > 0:
        tag_index = 1 if len(topology_view_speed_sorted_tags) > 1 else 0
        inter_tag = topology_view_speed_sorted_tags[tag_index]
        inter_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
            view_name="EP_Inter_Virtual",
            base_tag=inter_tag,
            virtual_size=inter_size,
            global_sharing_context="max_exclusive",
        )
        comm_stats_inter_per_layer = hockney_model_total_exchange_on_nd_topology(
            inter_view,
            P_total_send_bytes_per_node=P_inter_bytes_per_xpu,
            float_size_bytes=float_size,
        )

    per_layer = comm_stats_intra_per_layer + comm_stats_inter_per_layer
    final_comm_stats = CommStats()
    if num_layers_per_pp_stage > 0:
        final_comm_stats.transmission_time = per_layer.transmission_time * num_layers_per_pp_stage
        final_comm_stats.startup_latency = per_layer.startup_latency * num_layers_per_pp_stage
        final_comm_stats.compute_time = per_layer.compute_time * num_layers_per_pp_stage
        final_comm_stats.comm_vol = per_layer.comm_vol * num_layers_per_pp_stage
    return final_comm_stats
