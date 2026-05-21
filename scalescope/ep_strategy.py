"""Expert placement and per-XPU MoE layout inference.

Infers how distinct experts are placed across chips, taling into account
expert collocation and parallelization strategies.
"""

from dataclasses import dataclass
from math import ceil, sqrt

from scalescope.decode_layout_trace import NULL_LAYOUT_TRACE
from scalescope.model_spec import TransformerConfig
from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_reduce_scatter_on_nd_topology,
    hockney_model_total_exchange_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.workload_spec import MoEA2ACommMode, MoERouterSyncMode, MoESpecV2


@dataclass
class ExpertXPUPlacement:
    num_shared_experts_per_dp_rank: int = 0
    num_regular_experts_per_dp_rank: int = 0
    num_all_experts_per_dp_rank: int = 0
    total_distinct_experts_per_set: int = 0
    regular_expert_traffic_multiplier: float = 0.0
    num_complete_expert_sets: int = 1
    # EP/placement-wide fields needed by comm/compute
    full_expert_set_xpu_count: int = 1
    num_colocated_experts: int = 1
    # Number of colocated expert groups in one full expert set.
    # Colocated group = the colocated expert bundle routed together and hosted by one TP group.
    num_colocated_groups_per_set: int = 1
    # Chips per colocated expert group (TP degree inside one group).
    xpus_per_colocated_group: int = 1
    ep_chips_per_pp_stage: int = 0
    # Size relationship between the attention TP domain and the expert
    # set's chip footprint. At most one of these is > 1 at any time:
    #   - attn_domains_per_full_expert_set > 1 ↔ expert set is larger than
    #     one attention domain → multiple attention domains feed into one
    #     expert set ("centralized pool", currently unsupported for decode).
    #   - full_expert_sets_per_attn_domain > 1 ↔ one attention domain
    #     hosts multiple full expert sets ("small experts per attention
    #     domain", supported).
    #   - both == 1 ↔ expert set fills exactly one attention domain.
    # Defaults are None so missing assignment by a constructor surfaces
    # as a TypeError at the first comparison, instead of silently being
    # treated as 1.
    attn_domains_per_full_expert_set: int | None = None
    full_expert_sets_per_attn_domain: int | None = None
    # Decode router shard factor (resolved by orchestration).
    router_shard_factor: int = 1
    # Router synchronization policy consumed by TP comm strategies.
    # - replicated_router: router logits available locally
    # - sharded_router: explicit router-sync collective required
    router_sync_mode: MoERouterSyncMode = MoERouterSyncMode.REPLICATED_ROUTER
    # TP-MoE redistribution optimization toggle:
    # - OFF: regular routed MoE A2A.
    # - ON: send tokens (or token shards) to an expert shard first, then run all-gather
    #   so that shard is available across the expert TP shards.
    tp_moe_a2a_replication_optimization_enabled: bool = True


def _resolve_attn_expert_size_ratio(
    *, full_expert_set_xpu_count: int, attn_tp_size: int
) -> tuple[int, int]:
    """Resolve the size-ratio pair between the expert set and one attention TP domain.

    Returns ``(attn_domains_per_full_expert_set, full_expert_sets_per_attn_domain)``.

    The two returned values capture how the expert set's chip footprint
    relates to one attention TP domain's chip footprint. They are
    mutually exclusive (at most one is > 1):
      - ``attn_domains_per_full_expert_set > 1`` ↔ the expert set spans
        multiple attention TP domains (centralized pool).
      - ``full_expert_sets_per_attn_domain > 1`` ↔ a single attention TP
        domain hosts multiple full expert sets.

    Invariants enforced:
      - Both values are positive integers.
      - The larger of ``full_expert_set_xpu_count`` and ``attn_tp_size``
        must be an exact multiple of the smaller; otherwise we raise
        ``ValueError`` rather than silently rounding (per the
        explicit-only rule).
    """
    if full_expert_set_xpu_count <= 0:
        raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_expert_set_xpu_count}")
    if attn_tp_size <= 0:
        raise ValueError(f"attn_tp_size must be > 0, got {attn_tp_size}")

    if full_expert_set_xpu_count == attn_tp_size:
        return 1, 1
    if full_expert_set_xpu_count > attn_tp_size:
        if full_expert_set_xpu_count % attn_tp_size != 0:
            raise ValueError(
                f"full_expert_set_xpu_count ({full_expert_set_xpu_count}) must be a "
                f"multiple of attn_tp_size ({attn_tp_size}) when the expert set is "
                f"larger than the attention TP domain."
            )
        return full_expert_set_xpu_count // attn_tp_size, 1
    if attn_tp_size % full_expert_set_xpu_count != 0:
        raise ValueError(
            f"attn_tp_size ({attn_tp_size}) must be a multiple of "
            f"full_expert_set_xpu_count ({full_expert_set_xpu_count}) when the "
            f"attention TP domain is larger than the expert set."
        )
    return 1, attn_tp_size // full_expert_set_xpu_count


def damp_expert_imbalance(
    *, expert_imbalance: float, num_regular_experts: int, num_colocated_experts: int
) -> float:
    """Continuously damp expert imbalance as colocation increases.

    Statistical interpretation:
    - Treat each expert's routed token load as a finite population of size E with variance sigma^2.
    - Colocating c experts means each XPU sees the average of c expert loads sampled without replacement.
    - The variance of that average is:
        Var(mean_c) = (sigma^2 / c) * ((E - c) / (E - 1))
      so the standard deviation shrinks by:
        attenuation = sqrt(((E - c) / (E - 1)) / c)

    We apply this attenuation to the excess over uniform load:
        effective_imbalance = 1 + (expert_imbalance - 1) * attenuation
    where E is total regular experts and c is colocated experts per XPU (clamped to [1, E]).

    Properties:
    - c = 1    -> attenuation = 1, effective imbalance = expert_imbalance
    - c = E    -> attenuation = 0, effective imbalance = 1
    - monotonic and continuous for 1 <= c <= E
    """
    if num_regular_experts <= 1:
        return 1.0

    c = min(max(1, int(num_colocated_experts)), int(num_regular_experts))
    e = int(num_regular_experts)
    base = max(0.0, float(expert_imbalance) - 1.0)
    attenuation = sqrt(max(0.0, ((e - c) / (e - 1)) / c))
    return 1.0 + base * attenuation


def infer_expert_placement_per_xpu(
    *,
    model_spec: TransformerConfig,
    moe_v2: MoESpecV2,
    attn_tp_size: int,
    xpus_per_colocated_group: int,
    ep_chips_per_pp_stage: int,
    router_sync_mode: MoERouterSyncMode,
    router_shard_factor: int,
    tp_moe_a2a_replication_optimization_enabled: bool,
) -> ExpertXPUPlacement:
    """Infer per-XPU expert placement within a DP group.

    Compute how many routed and shared experts are hosted per DP rank,
    how many DP ranks are required to hold a complete expert set, and how many complete
    expert sets fit into the EP domain of a single PP stage. It also derives helper
    quantities used by communication and compute models (e.g., traffic multipliers).
    We also define explicit expert-group metadata:
    - expert group: the colocated expert bundle served by one TP expert group
    - num_colocated_groups_per_set: number of such groups in one full expert set
    - xpus_per_colocated_group: chips per colocated expert group

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
        - EP_SU (logical tag size) defines the high-bandwidth expert group size.
    - attn_tp_size: TP size of the attention path (chips per DP rank for TP_Attn).
      Used to divide a complete expert set across DP ranks.
    - xpus_per_colocated_group: expert tensor-parallel degree (chips per colocated group).
    - ep_chips_per_pp_stage: total EP chips available per PP stage (DP_Attn * TP_Attn).
    - router_sync_mode: router synchronization policy for TP comm strategies.
    - router_shard_factor: number of ranks across which the router weight is sharded.
      For replicated router: 1. For sharded router: Y_eff (2D TP) or attn_tp_size (1D TP).
    - tp_moe_a2a_replication_optimization_enabled: enables TP-MoE expert-input redistribution optimization.

    Returns
    - ExpertXPUPlacement
    """

    if router_shard_factor < 1:
        raise ValueError(f"Invalid router_shard_factor: {router_shard_factor}")

    num_shared = model_spec.num_shared_experts
    normalized_xpus_per_colocated_group = max(1, int(xpus_per_colocated_group))
    num_colocated_experts = int(moe_v2.num_colocated_experts)

    num_regular = model_spec.num_regular_experts
    total_distinct_experts = num_regular + num_shared * (moe_v2.shared_expert_copies or 1)
    if total_distinct_experts == 0:
        # No MoE experts in the model: treat expert placement as a no-op so sweeps can focus on
        # attention/projection behavior without needing EP divisibility constraints.
        return ExpertXPUPlacement(
            num_shared_experts_per_dp_rank=0,
            num_regular_experts_per_dp_rank=0,
            num_all_experts_per_dp_rank=0,
            total_distinct_experts_per_set=0,
            regular_expert_traffic_multiplier=0.0,
            num_complete_expert_sets=1,
            full_expert_set_xpu_count=1,
            num_colocated_experts=num_colocated_experts,
            num_colocated_groups_per_set=0,
            xpus_per_colocated_group=normalized_xpus_per_colocated_group,
            ep_chips_per_pp_stage=int(ep_chips_per_pp_stage),
            attn_domains_per_full_expert_set=1,
            full_expert_sets_per_attn_domain=1,
            router_shard_factor=router_shard_factor,
            router_sync_mode=router_sync_mode,
            tp_moe_a2a_replication_optimization_enabled=tp_moe_a2a_replication_optimization_enabled,
        )
    # Colocation reduces the number of XPUs needed to host a full set of distinct experts,
    # regardless of whether experts are TP-sharded. Account for both expert TP degree and colocation.
    full_expert_set_xpu_count = ceil(
        (total_distinct_experts * normalized_xpus_per_colocated_group) / num_colocated_experts
    )
    num_colocated_groups_per_set = ceil(total_distinct_experts / num_colocated_experts)
    attn_domains_per_full_expert_set, full_expert_sets_per_attn_domain = (
        _resolve_attn_expert_size_ratio(
            full_expert_set_xpu_count=full_expert_set_xpu_count,
            attn_tp_size=attn_tp_size,
        )
    )
    num_routed_experts_per_dp_rank = num_regular // attn_domains_per_full_expert_set
    num_shared_experts_per_dp_rank = (
        num_shared * (moe_v2.shared_expert_copies or 1) // attn_domains_per_full_expert_set
    )
    num_of_all_experts_per_dp_rank = num_routed_experts_per_dp_rank + num_shared_experts_per_dp_rank

    # Determine how many complete expert sets fit into the available EP domain per PP stage
    if ep_chips_per_pp_stage % full_expert_set_xpu_count != 0:
        raise ValueError(
            f"EP chips ({ep_chips_per_pp_stage}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_complete_expert_sets = ep_chips_per_pp_stage // full_expert_set_xpu_count

    # Regular expert traffic multiplier with colocation-aware damping.
    effective_imbalance = damp_expert_imbalance(
        expert_imbalance=moe_v2.expert_imbalance,
        num_regular_experts=num_regular,
        num_colocated_experts=num_colocated_experts,
    )
    regular_expert_traffic_multiplier = effective_imbalance * model_spec.num_selected_experts

    return ExpertXPUPlacement(
        num_shared_experts_per_dp_rank=num_shared_experts_per_dp_rank,
        num_regular_experts_per_dp_rank=num_routed_experts_per_dp_rank,
        num_all_experts_per_dp_rank=num_of_all_experts_per_dp_rank,
        total_distinct_experts_per_set=int(total_distinct_experts),
        regular_expert_traffic_multiplier=regular_expert_traffic_multiplier,
        num_complete_expert_sets=int(num_complete_expert_sets),
        # EP/placement-wide fields
        full_expert_set_xpu_count=int(full_expert_set_xpu_count),
        num_colocated_experts=num_colocated_experts,
        num_colocated_groups_per_set=int(num_colocated_groups_per_set),
        xpus_per_colocated_group=normalized_xpus_per_colocated_group,
        ep_chips_per_pp_stage=int(ep_chips_per_pp_stage),
        attn_domains_per_full_expert_set=attn_domains_per_full_expert_set,
        full_expert_sets_per_attn_domain=full_expert_sets_per_attn_domain,
        router_shard_factor=router_shard_factor,
        router_sync_mode=router_sync_mode,
        tp_moe_a2a_replication_optimization_enabled=tp_moe_a2a_replication_optimization_enabled,
    )


def calculate_moe_comm_first_total_exchange(
    ep_topology_view,
    transformer_config,
    moe_config,
    num_global_batch_tokens,
    float_size,
    tp_moe_y_shards: int,
    workload_spec=None,
):
    return calculate_moe_comm_hierarchical_total_exchange(
        ep_topology_view,
        transformer_config,
        moe_config,
        num_global_batch_tokens,
        float_size,
        workload_spec=workload_spec,
        tp_moe_y_shards=tp_moe_y_shards,
    )


def calculate_moe_comm_second_total_exchange(
    ep_topology_view,
    transformer_config,
    moe_config,
    num_global_batch_tokens,
    float_size,
    tp_moe_y_shards: int,
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
        tp_moe_y_shards=tp_moe_y_shards,
    )


def calculate_moe_combine_decode(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    num_global_batch_tokens: int,
    attn_tp_xpus: int,
    float_size: int,
    tp_moe_x_shards: int,
    tp_moe_y_shards: int,
    decode_tp2d_qkv_groups: int,
    layout_trace=NULL_LAYOUT_TRACE,
) -> CommStats:
    """D=1 decode combine: per-layer reduce-scatter across G expert groups.

    Returns **per-layer** comm stats; the caller is responsible for
    multiplying by ``num_layers_per_pp_stage`` (consistent with the TP
    strategy pattern in ``comm_model.py``).

    The combine is a reduce-scatter **along G** (matching ranks across groups).
    There are tp_x × Y_eff ranks per group; each (X, Y_eff) position has one
    independent reduce-scatter with G participants.

    Each rank holds a dense output buffer of shape [B, D_local] where
    D_local = d_model / xpus_per_group.  After expert processing, entries
    for tokens not routed to this group's experts are zero.

    1. **Reduce** — tokens whose top_k experts span multiple groups have
       partial weighted sums in each contributing group; RS reduces them.
    2. **Scatter/Handoff** — RS also shards outputs across participants,
       which can align with next-layer X sharding when group↔X mapping matches.

    RS input volume per rank = B_set × D_local × float_size  (the full dense buffer
    for one expert set, where B_set = B_global / num_complete_expert_sets).
    No traffic_mult scaling: the top_k fan-out is resolved locally inside
    each group (the group sums its local expert outputs per token before
    the inter-group combine).  The RS operates on the dense buffer
    regardless of routing sparsity.

    When G=1 the reduce-scatter is a no-op (zero cost).

    **Cross-set All-Gather** (when combine-domain set count S_attn > 1):
    After the intra-set RS, each rank has [B_set, D_local].  An AG(S)
    across matching ranks in different expert sets reconstructs the full
    [B_attn_domain, D_local] needed by the next layer in the current
    attention token domain.  When S_attn=1 this step is skipped.
    """

    if transformer_config.num_regular_experts == 0:
        return CommStats()

    G = moe_config.num_colocated_groups_per_set

    ep_chips = moe_config.ep_chips_per_pp_stage
    full_set_xpus = moe_config.full_expert_set_xpu_count
    xpus_per_group = moe_config.xpus_per_colocated_group

    if full_set_xpus <= 0:
        raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_set_xpus}")
    if xpus_per_group <= 0:
        raise ValueError(f"xpus_per_colocated_group must be > 0, got {xpus_per_group}")
    if ep_chips <= 0:
        raise ValueError(f"ep_chips_per_pp_stage must be > 0, got {ep_chips}")
    if attn_tp_xpus <= 0:
        raise ValueError(f"attn_tp_xpus must be > 0, got {attn_tp_xpus}")
    if ep_chips % full_set_xpus != 0:
        raise ValueError(
            f"ep_chips_per_pp_stage ({ep_chips}) must be divisible by "
            f"full_expert_set_xpu_count ({full_set_xpus})"
        )
    if G * xpus_per_group != full_set_xpus:
        raise ValueError(
            f"Group decomposition inconsistent: G ({G}) × xpus_per_group ({xpus_per_group}) "
            f"= {G * xpus_per_group}, expected full_expert_set_xpu_count = {full_set_xpus}"
        )
    if tp_moe_x_shards <= 0:
        raise ValueError(f"tp_moe_x_shards must be > 0, got {tp_moe_x_shards}")
    if decode_tp2d_qkv_groups <= 0:
        raise ValueError(f"decode_tp2d_qkv_groups must be > 0, got {decode_tp2d_qkv_groups}")
    if xpus_per_group % tp_moe_y_shards != 0:
        raise ValueError(
            f"xpus_per_colocated_group ({xpus_per_group}) must be divisible by "
            f"tp_moe_y_shards ({tp_moe_y_shards})"
        )
    if xpus_per_group % tp_moe_x_shards != 0:
        raise ValueError(
            f"xpus_per_colocated_group ({xpus_per_group}) must be divisible by "
            f"tp_moe_x_shards ({tp_moe_x_shards})"
        )

    d_model = transformer_config.d_model

    # D_local: each rank's shard of d_model within the group (sharded by X × Y_eff).
    d_local_bytes = (d_model / xpus_per_group) * float_size

    # Dense RS input buffer: every rank materializes [B, D_local] — zeros for
    # tokens not routed to this group's experts.
    num_processing_sets = moe_config.num_complete_expert_sets
    if num_processing_sets <= 0:
        raise ValueError(f"num_complete_expert_sets must be > 0, got {num_processing_sets}")
    if attn_tp_xpus >= full_set_xpus:
        if attn_tp_xpus % full_set_xpus != 0:
            raise ValueError(
                f"attn_tp_xpus ({attn_tp_xpus}) must be divisible by "
                f"full_expert_set_xpu_count ({full_set_xpus})"
            )
        num_combine_sets = attn_tp_xpus // full_set_xpus
    else:
        # Attention domain smaller than one full expert set: no cross-set combine in this domain.
        num_combine_sets = 1
    if num_combine_sets <= 0:
        raise ValueError(
            f"combine-domain set count must be > 0, got {num_combine_sets} "
            f"(attn_tp_xpus={attn_tp_xpus}, full_set_xpus={full_set_xpus})"
        )
    tokens_per_set = num_global_batch_tokens / num_processing_sets

    combine_bytes_per_rank = tokens_per_set * d_local_bytes

    # Variant-aware axis assignment:
    # - 1xN: RS(G) and cross-set AG are modeled on TP_Attn_X.
    # - Nx1: RS(G) and cross-set AG are modeled on TP_Attn_Y.
    # - NxM: keep explicit mixed-mode marker (not fully modeled).
    if tp_moe_x_shards == 1:
        rs_handoff_mode = "1xN->X"
        preferred_base_tag = "TP_Attn_X"
    elif tp_moe_y_shards == 1:
        rs_handoff_mode = "Nx1->Y_eff"
        preferred_base_tag = "TP_Attn_Y"
    else:
        rs_handoff_mode = "NxM-unsupported"
        preferred_base_tag = "TP_Attn_X"

    # EP view contains only EP_* tags; for axis selection we need TP_Attn_* tags.
    # Use the EP view's original topology reference to access the full tag namespace.
    topo_ref = ep_topology_view.original_topology_ref
    axis_note = ""
    combine_base_tag = preferred_base_tag
    try:
        _ = topo_ref.get_mapping_for_tag(combine_base_tag)
    except Exception:
        ep_tags = ep_topology_view.get_tags_sorted_by_speed()
        if not ep_tags:
            raise ValueError("EP topology view has no tags to map combine axis.")
        combine_base_tag = ep_tags[0]
        axis_note = (
            f" NOTE: {preferred_base_tag} not defined in logical topology; "
            f"using EP axis {combine_base_tag} for combine-domain modeling."
        )
    cross_set_base_tag = combine_base_tag
    # The reduce-scatter runs across G groups on a selected attention-domain axis.
    # Create a virtual view with G participants so the Hockney model uses
    # the correct alpha count and bandwidth sharing.
    # TODO(moe-combine): model explicit subgroup membership (set_id, local_shard_id)
    # for RS/AG participants. Current path models axis + virtual participant count,
    # but not exact rank pairing/placement within the axis.
    combine_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
        view_name="Combine_G",
        base_tag=combine_base_tag,
        virtual_size=G,
        global_sharing_context="max_exclusive",
    )

    per_layer_rs = hockney_model_reduce_scatter_on_nd_topology(
        view=combine_view,
        P_total_bytes_before_rs=combine_bytes_per_rank,
        float_size_bytes=float_size,
    )

    rs_issue_note = ""
    if rs_handoff_mode == "NxM-unsupported":
        rs_issue_note = (
            " ISSUE: combine RS(G) handoff for mixed MoE TP (NxM) is not modeled yet "
            f"(tp_moe_x_shards={tp_moe_x_shards}, tp_moe_y_shards={tp_moe_y_shards})."
        )

    layout_trace.record_comm(
        "combine_rs_g",
        "RS",
        f"{combine_base_tag}({G})",
        G,
        combine_bytes_per_rank,
        per_layer_rs,
        note=(
            f"D=1 dense combine/handoff: RS across G={G} groups, "
            f"buffer=[B_set={tokens_per_set:.0f}, D_local={d_model/xpus_per_group:.0f}]"
            f", S_processing={num_processing_sets}, S_attn={num_combine_sets}"
            f", mode={rs_handoff_mode}, rs_axis={combine_base_tag}"
            f"{axis_note}"
            f"{rs_issue_note}"
        ),
    )

    per_layer = per_layer_rs
    per_layer_cross_set = CommStats()
    per_layer_combine_u = CommStats()

    # Cross-set All-Gather: when combine-domain set count > 1, each set
    # processed only tokens_per_set = B_global / S tokens.  After the
    # intra-set RS(G), every rank has its set's [B/S, D_local] shard.  To
    # reconstruct the full [B_attn_domain, D_local] for the next layer we need
    # an AG(S) across matching ranks in different expert sets.
    #
    # AG input per rank  = tokens_per_set × D_local  (one set's slice)
    # AG output per rank = B_attn_domain × D_local   (current attn-domain batch)
    if num_combine_sets > 1:
        tokens_per_attn_domain = tokens_per_set * num_combine_sets
        cross_set_ag_output_bytes = tokens_per_attn_domain * d_local_bytes

        cross_set_domain_note = ""
        cross_set_phys = topo_ref.get_mapping_for_tag(
            cross_set_base_tag
        ).maps_to_physical_dimension_name
        combine_phys = topo_ref.get_mapping_for_tag(
            combine_base_tag
        ).maps_to_physical_dimension_name
        try:
            attn_x_phys = topo_ref.get_mapping_for_tag("TP_Attn_X").maps_to_physical_dimension_name
            attn_y_phys = topo_ref.get_mapping_for_tag("TP_Attn_Y").maps_to_physical_dimension_name
            if cross_set_phys not in (attn_x_phys, attn_y_phys):
                cross_set_domain_note = (
                    " ISSUE: cross-set AG base domain does not map to attention TP axes "
                    f"(base_tag={cross_set_base_tag}, phys={cross_set_phys}, "
                    f"attn_x_phys={attn_x_phys}, attn_y_phys={attn_y_phys})."
                )
            else:
                # Same-axis RS and cross-set AG can be valid: they may use different
                # subgroups/virtual participant sets of the same physical dimension.
                cross_set_domain_note = (
                    f" verified_domain: rs_axis={combine_base_tag}({combine_phys}), "
                    f"cross_set_axis={cross_set_base_tag}({cross_set_phys})."
                )
        except Exception:
            cross_set_domain_note = (
                f" NOTE: TP_Attn_X/Y tags unavailable; rs_axis={combine_base_tag}({combine_phys}), "
                f"cross_set_axis={cross_set_base_tag}({cross_set_phys})."
            )

        cross_set_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
            view_name="Combine_CrossSet",
            base_tag=cross_set_base_tag,
            virtual_size=num_combine_sets,
            global_sharing_context="max_exclusive",
        )

        per_layer_cross_set = hockney_model_all_gather_on_nd_topology(
            view=cross_set_view,
            P_total_bytes_after_ag=cross_set_ag_output_bytes,
            float_size_bytes=float_size,
        )

        layout_trace.record_comm(
            "combine_ag_cross_set",
            "AG",
            f"{cross_set_base_tag}({num_combine_sets})",
            num_combine_sets,
            cross_set_ag_output_bytes,
            per_layer_cross_set,
            note=(
                f"Cross-set AG: S_attn={num_combine_sets} sets, each contributes "
                f"[B/S={tokens_per_set:.0f}, D_local={d_model/xpus_per_group:.0f}] "
                f"→ attn-domain [B={tokens_per_attn_domain:.0f}, D_local={d_model/xpus_per_group:.0f}]"
                f", ag_axis={cross_set_base_tag}"
                f"{cross_set_domain_note}"
            ),
        )

    else:
        tokens_per_attn_domain = tokens_per_set

    # decode_tp2d_qkv_groups>1 means projection-domain work is split into
    # multiple independent YZ groups. After combine RS(G) (and optional cross-set
    # AG), we need an extra AG over that group axis so every rank in the active
    # attention shard sees the full post-combine token domain.
    if decode_tp2d_qkv_groups > 1:
        # decode_tp2d_qkv_groups (`g`) is introduced by shrinking projection-domain
        # YZ tags in run_scalescope. This reconciliation AG therefore stays on the
        # attention Y axis regardless of whether RS(G)/cross-set AG used X or Y.
        # It models restoring full visibility across projection groups in the YZ domain.
        u_base_tag = "TP_Attn_Y"
        try:
            _ = topo_ref.get_mapping_for_tag(u_base_tag)
        except Exception as e:
            raise ValueError(
                f"Missing required logical tag '{u_base_tag}' for projection-group combine AG(U) "
                f"(g={decode_tp2d_qkv_groups}, combine_axis={combine_base_tag})."
            ) from e

        combine_u_ag_output_bytes = tokens_per_attn_domain * d_local_bytes
        combine_u_view = topo_ref.create_virtual_view_from_tag(
            view_name="Combine_U",
            base_tag=u_base_tag,
            virtual_size=decode_tp2d_qkv_groups,
            global_sharing_context="max_exclusive",
        )
        per_layer_combine_u = hockney_model_all_gather_on_nd_topology(
            view=combine_u_view,
            P_total_bytes_after_ag=combine_u_ag_output_bytes,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            "combine_ag_u",
            "AG",
            f"{u_base_tag}({decode_tp2d_qkv_groups})",
            decode_tp2d_qkv_groups,
            combine_u_ag_output_bytes,
            per_layer_combine_u,
            note=(
                f"Projection-group combine AG(U): g={decode_tp2d_qkv_groups}, "
                f"[B={tokens_per_attn_domain:.0f}, D_local={d_model/xpus_per_group:.0f}] "
                "to restore full combine-domain visibility for next-layer handoff."
            ),
        )
    # Keep post-RS gather stages serialized for now.
    per_layer = per_layer + per_layer_cross_set + per_layer_combine_u

    # Conservative NxM handoff remap:
    # For mixed MoE TP (tp_moe_x_shards>1 and tp_moe_y_shards>1), RS(G) does not
    # guarantee alignment with next-layer attention sharding. Model an explicit
    # remap as AG+RS on TP_Attn_X over the full post-combine buffer to avoid
    # optimistic winners while subgroup-exact mapping is not implemented.
    if tp_moe_x_shards > 1 and tp_moe_y_shards > 1:
        remap_base_tag = "TP_Attn_X"
        try:
            remap_size = topo_ref.get_mapping_for_tag(remap_base_tag).size
        except Exception as e:
            raise ValueError(
                f"Missing required logical tag '{remap_base_tag}' for NxM combine remap "
                f"(tp_moe_x_shards={tp_moe_x_shards}, tp_moe_y_shards={tp_moe_y_shards})."
            ) from e
        remap_output_bytes = tokens_per_attn_domain * d_local_bytes
        remap_view = topo_ref.create_virtual_view_from_tag(
            view_name="Combine_Remap_X",
            base_tag=remap_base_tag,
            virtual_size=remap_size,
            global_sharing_context="concurrent_baseline",
        )
        per_layer_remap_ag = hockney_model_all_gather_on_nd_topology(
            view=remap_view,
            P_total_bytes_after_ag=remap_output_bytes,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            "combine_remap_ag_x",
            "AG",
            f"{remap_base_tag}({remap_size})",
            remap_size,
            remap_output_bytes,
            per_layer_remap_ag,
            note=(
                f"Conservative NxM handoff remap (step 1/2): AG on {remap_base_tag}. "
                f"buffer=[B={tokens_per_attn_domain:.0f}, D_local={d_model/xpus_per_group:.0f}]"
            ),
        )
        per_layer_remap_rs = hockney_model_reduce_scatter_on_nd_topology(
            view=remap_view,
            P_total_bytes_before_rs=remap_output_bytes,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            "combine_remap_rs_x",
            "RS",
            f"{remap_base_tag}({remap_size})",
            remap_size,
            remap_output_bytes,
            per_layer_remap_rs,
            note=(
                f"Conservative NxM handoff remap (step 2/2): RS on {remap_base_tag}. "
                "Used until subgroup-exact remap is modeled."
            ),
        )
        per_layer = per_layer + per_layer_remap_ag + per_layer_remap_rs

    return per_layer


def calculate_moe_comm_hierarchical_total_exchange(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    num_global_batch_tokens: int,
    float_size: int,
    tp_moe_y_shards: int,
    workload_spec=None,
):
    """Per-layer MoE A2A exchange cost (hierarchical or fused).

    Returns **per-layer** comm stats; the caller is responsible for
    multiplying by ``num_layers_per_pp_stage``.
    """
    if transformer_config.num_regular_experts == 0:
        return CommStats()

    d_model = transformer_config.d_model

    try:
        full_expert_set_xpu_count = int(moe_config.full_expert_set_xpu_count)
        xpus_per_colocated_group = int(moe_config.xpus_per_colocated_group)
        num_colocated_groups_per_set = int(moe_config.num_colocated_groups_per_set)
        attn_domains_per_full_expert_set = int(moe_config.attn_domains_per_full_expert_set)
    except AttributeError as e:
        raise ValueError(
            "moe_config must be produced by infer_expert_placement_per_xpu and include "
            "`full_expert_set_xpu_count`, `xpus_per_colocated_group`, "
            "`num_colocated_groups_per_set`, `attn_domains_per_full_expert_set`."
        ) from e
    if (
        full_expert_set_xpu_count <= 0
        or xpus_per_colocated_group <= 0
        or num_colocated_groups_per_set < 0
        or attn_domains_per_full_expert_set <= 0
    ):
        raise ValueError(
            "Invalid expert placement fields: expected full_expert_set_xpu_count>0, "
            "xpus_per_colocated_group>0, num_colocated_groups_per_set>=0, "
            "attn_domains_per_full_expert_set>0."
        )
    routed_expert_traffic_multiplier = float(
        getattr(moe_config, "regular_expert_traffic_multiplier")
    )
    # Use EP chips per PP stage (DP_Attn * TP_Attn) if provided by caller via merged placement
    try:
        ep_total_xpus = int(moe_config.ep_chips_per_pp_stage)
    except AttributeError as e:
        raise ValueError(
            "moe_config must include `ep_chips_per_pp_stage` (produced by infer_expert_placement_per_xpu)."
        ) from e
    if ep_total_xpus <= 0:
        raise ValueError(f"Invalid ep_chips_per_pp_stage: {ep_total_xpus}. Expected > 0.")
    if ep_total_xpus % max(1, full_expert_set_xpu_count) != 0:
        raise ValueError(
            f"EP chips ({ep_total_xpus}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_full_sets = ep_total_xpus // max(1, full_expert_set_xpu_count)

    tokens_per_set = num_global_batch_tokens / max(1, num_full_sets)
    # Smooth locality model (independence assumption):
    # - D = attn_domains_per_full_expert_set (feeder attention domains per expert set)
    # - G = num_colocated_groups_per_set (destination expert groups)
    # - remote_fraction = 1 - 1/(D*G)
    #
    # This makes locality a continuous function:
    # - D=1 and G=1 -> remote_fraction=0 (fully local, no MoE exchange volume)
    # - larger D/G -> remote_fraction approaches 1 (more exchange required)
    source_partitions = max(
        1, attn_domains_per_full_expert_set * max(1, num_colocated_groups_per_set)
    )
    remote_fraction = max(0.0, 1.0 - (1.0 / float(source_partitions)))
    # Model routed exchange at expert-group granularity, then map to per-XPU payload
    # by dividing across TP ranks inside the group.
    tokens_per_group = (
        tokens_per_set
        * routed_expert_traffic_multiplier
        * remote_fraction
        / max(1, num_colocated_groups_per_set)
    )
    # EP A2A toggle semantics:
    # - optimization ON: send one grouped copy over EP exchange, then redistribute inside TP group.
    # - optimization OFF: each TP rank carries its own routed copy over EP exchange.
    # This makes EP A2A volume larger in OFF mode, as intended.
    optimization_enabled = bool(getattr(workload_spec, "tp_moe_a2a_replication_optimization", True))
    a2a_replication_factor = (
        1.0 if optimization_enabled else float(max(1, xpus_per_colocated_group))
    )
    tokens_per_xpu = tokens_per_group / max(1, xpus_per_colocated_group) * a2a_replication_factor

    # Resolve this once so hierarchical and fused paths are based on the same EP interpretation.
    # EP_SU is modeled as XPU-domain capacity for experts in the scale-up domain.
    try:
        high_bw_group_size_xpus = int(ep_topology_view.get_mapping_for_tag("EP_SU").size)
    except Exception as e:
        raise ValueError(
            "EP topology must include tag `EP_SU` to resolve high-bandwidth expert group size."
        ) from e
    if high_bw_group_size_xpus <= 0:
        raise ValueError(f"Invalid EP_SU size: {high_bw_group_size_xpus}. Expected > 0.")

    moe_a2a_comm_mode = workload_spec.moe_a2a_comm_mode
    # Shard MoE A2A activation payload by TP_MoE_Y.
    # For 1D MoE TP this value is 1, so behavior is unchanged.
    tp_moe_y_shards = max(1, int(tp_moe_y_shards))
    bytes_per_token = (d_model * float_size) / tp_moe_y_shards

    if moe_a2a_comm_mode == MoEA2ACommMode.FUSED_ND:
        # Experimental path: model MoE exchange as a single fused N-D total-exchange on EP topology.
        payload_bytes_per_xpu = tokens_per_xpu * bytes_per_token
        per_layer = hockney_model_total_exchange_on_nd_topology(
            ep_topology_view,
            P_total_send_bytes_per_node=payload_bytes_per_xpu,
            float_size_bytes=float_size,
        )
    elif moe_a2a_comm_mode == MoEA2ACommMode.HIERARCHICAL_SPLIT:
        # Original hierarchical split path (intra high-bandwidth + inter low-bandwidth).
        # Ensure HB group size does not exceed the set footprint
        if high_bw_group_size_xpus > full_expert_set_xpu_count:
            high_bw_group_size_xpus = full_expert_set_xpu_count
        frac_local = min(1.0, high_bw_group_size_xpus / max(1, full_expert_set_xpu_count))
        frac_remote = max(0.0, 1.0 - frac_local)

        P_intra_bytes_per_xpu = tokens_per_xpu * frac_local * bytes_per_token
        P_inter_bytes_per_xpu = tokens_per_xpu * frac_remote * bytes_per_token

        comm_stats_intra_per_layer = CommStats()
        comm_stats_inter_per_layer = CommStats()
        topology_view_speed_sorted_tags = ep_topology_view.get_tags_sorted_by_speed()

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
    else:
        raise ValueError(f"Unsupported moe_a2a_comm_mode: {moe_a2a_comm_mode}")
    return per_layer
