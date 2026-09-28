"""Software/runtime overheads that are neither FLOPs, HBM traffic, nor network communication."""

from dataclasses import dataclass

from scalescope.ep_strategy import ExpertXPUPlacement
from scalescope.memory_model import (
    _decode_attention_attn_tp_1d_memory_traffic,
    _decode_attention_attn_tp_2d_memory_traffic,
    LayerMemoryTraffic,
)
from scalescope.model_spec import resolve_kv_heads_per_partition, TransformerConfig
from scalescope.topology_spec import LogicalTopologyView
from scalescope.tp_strategy import resolve_tp2d_split
from scalescope.utils import PassMode
from scalescope.workload_spec import (
    InferenceDecodeSpec,
    TensorParallelSpecV2,
    TensorParallelVariant,
    WorkloadSpec,
)

# Empirical clean-panel fit for incremental decode attention-core memory time
# above an explicit reference context. The calibration set has TP-YZ=1.
# Extending inverse local work to 2D uses TP-YZ/local-KV-heads, which recovers
# the fitted expression exactly at TP-YZ=1 but remains an extrapolation until
# matched 2D measurements are available.
DECODE_ATTN_CONTEXT_GAMMA_INTERCEPT = 1.5559183649426251
DECODE_ATTN_CONTEXT_GAMMA_INV_LOCAL_WORK_SLOPE = 2.388579558589879
DECODE_ATTN_CONTEXT_GAMMA_CENTER = 0.5


@dataclass(frozen=True)
class DecodeAttentionContextCorrectionResult:
    """Resolved long-context attention-memory correction and diagnostics.

    The analytical memory model derives attention time from KV-cache bytes and
    peak bandwidth, which assumes context growth preserves achieved memory
    efficiency. Measurements show that incremental attention cost can grow
    faster, especially when sharding leaves little local KV work per rank. The
    correction therefore anchors the analytical estimate at an explicit
    context length and inflates only the additional memory time beyond that
    anchor. It changes modeled time, not memory-traffic bytes.

    When disabled, the timing and topology diagnostics are zero and
    ``reference_tokens`` is ``None``. The object is immutable so the correction
    applied to latency remains identical to the values exported in the result
    row.
    """

    # Extra time added to the attention-core memory leg.
    correction_time_sec: float
    # Unscaled increase in memory time above the explicit context anchor.
    delta_memory_time_sec: float
    # Empirical multiplier applied to delta_memory_time_sec.
    gamma: float
    # KV heads resident on one rank after attention TP-X head sharding.
    local_kv_heads: int
    # Attention TP-YZ degree that shards the KV-cache sequence dimension.
    attention_tp_yz: int
    # Context-length anchor used to compute the reference memory time.
    reference_tokens: int | None


def decode_attention_context_memory_correction(
    *,
    current_memory_time_sec: float,
    reference_memory_time_sec: float,
    local_kv_heads: int,
    attention_tp_yz: int,
) -> tuple[float, float, float]:
    """Return ``(correction, gamma, delta_memory)`` for decode attention.

    ``local_kv_heads`` is resolved only across attention TP-X. TP-YZ shards
    the sequence dimension, so the empirical inverse-local-work coordinate is
    ``attention_tp_yz / local_kv_heads``. The correction inflates time, not
    bytes, and is intended to enter the attention memory overlap leg.
    """
    if current_memory_time_sec < 0:
        raise ValueError("current_memory_time_sec must be >= 0, " f"got {current_memory_time_sec}")
    if reference_memory_time_sec < 0:
        raise ValueError(
            "reference_memory_time_sec must be >= 0, " f"got {reference_memory_time_sec}"
        )
    if local_kv_heads <= 0:
        raise ValueError(f"local_kv_heads must be > 0, got {local_kv_heads}")
    if attention_tp_yz <= 0:
        raise ValueError(f"attention_tp_yz must be > 0, got {attention_tp_yz}")

    inverse_local_work = attention_tp_yz / local_kv_heads
    gamma = DECODE_ATTN_CONTEXT_GAMMA_INTERCEPT + DECODE_ATTN_CONTEXT_GAMMA_INV_LOCAL_WORK_SLOPE * (
        inverse_local_work - DECODE_ATTN_CONTEXT_GAMMA_CENTER
    )
    if gamma <= 0:
        raise ValueError(
            "decode attention context correction gamma must be > 0, "
            f"got {gamma} for local_kv_heads={local_kv_heads}, "
            f"attention_tp_yz={attention_tp_yz}"
        )
    delta_memory_time_sec = current_memory_time_sec - reference_memory_time_sec
    if delta_memory_time_sec <= 0:
        return 0.0, gamma, 0.0
    return gamma * delta_memory_time_sec, gamma, delta_memory_time_sec


def _resolve_decode_attention_context_correction(
    *,
    pass_mode: PassMode,
    workload_spec: WorkloadSpec,
    model_spec: TransformerConfig,
    attn_tp_spec: TensorParallelSpecV2,
    proj_tp_spec: TensorParallelSpecV2,
    attn_tp_topology_view: LogicalTopologyView,
    proj_tp_topology_view: LogicalTopologyView,
    current_memory_time_sec: float,
    layers_per_stage: int,
) -> DecodeAttentionContextCorrectionResult:
    """Resolve the optional empirical decode attention-memory correction.

    The function rebuilds attention-core traffic at an explicit reference
    context using the same projection/attention partition as the current
    point. It then compares the current and reference analytical memory times
    and passes the positive increment to the empirical efficiency model.

    TP-X determines how many KV heads remain local. TP-YZ shards the context
    length, so both values are retained to parameterize the correction and to
    make the applied architecture coordinate visible in output diagnostics.

    Returns a zero-valued result when the correction does not apply. When
    enabled, the returned correction is extra time for the attention-memory
    leg; the caller is responsible for adding it to both detailed and
    aggregate memory accounting.
    """
    # Context growth is a decode-only concept in this correction model.
    if pass_mode != PassMode.INFERENCE_DECODE:
        return DecodeAttentionContextCorrectionResult(
            correction_time_sec=0.0,
            delta_memory_time_sec=0.0,
            gamma=0.0,
            local_kv_heads=0,
            attention_tp_yz=0,
            reference_tokens=None,
        )
    if not isinstance(workload_spec, InferenceDecodeSpec):
        raise TypeError(
            "inference_decode context correction requires InferenceDecodeSpec, "
            f"got {type(workload_spec).__name__}"
        )
    # Preserve analytical byte/bandwidth timing unless explicitly enabled.
    if not workload_spec.decode_attention_context_correction_enabled:
        return DecodeAttentionContextCorrectionResult(
            correction_time_sec=0.0,
            delta_memory_time_sec=0.0,
            gamma=0.0,
            local_kv_heads=0,
            attention_tp_yz=0,
            reference_tokens=None,
        )

    # The anchor is required: inferring one here would change the calibration
    # meaning and make otherwise identical workloads use different baselines.
    reference_tokens = workload_spec.decode_attention_context_reference_tokens
    if reference_tokens is None:
        raise ValueError(
            "decode_attention_context_reference_tokens must be resolved "
            "when the context correction is enabled"
        )

    # Recompute only attention-core traffic at the anchor. Projection traffic
    # does not scale with KV-cache context and is intentionally excluded.
    reference_traffic = LayerMemoryTraffic(
        name="fwd_attn_core_reference",
        device_for=workload_spec.memory_device_map_traffic,
        node=attn_tp_topology_view.physical_topology.node_definition,
    )
    match attn_tp_spec.variant:
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            # Use the exact resolved split from the current point: TP-X shards
            # heads, while TP-YZ shards the sequence/KV-cache span.
            resolved_attn = resolve_tp2d_split(
                tp_view=attn_tp_topology_view,
                split_2d=attn_tp_spec.split_2d,
                view_name="TP_Attn_Context_Correction",
            )
            resolved_proj = resolve_tp2d_split(
                tp_view=proj_tp_topology_view,
                split_2d=proj_tp_spec.split_2d,
                view_name="TP_Proj_Context_Correction",
            )
            attn_context_tp_x = resolved_attn.tp_x
            attn_context_tp_yz = resolved_attn.tp_yz
            _decode_attention_attn_tp_2d_memory_traffic(
                traffic=reference_traffic,
                transformer_config=model_spec,
                n_seq_so_far=reference_tokens,
                proj_tp_size_x=resolved_proj.tp_x,
                proj_tp_size_yz=resolved_proj.tp_yz,
                attn_tp_size_x=attn_context_tp_x,
                attn_tp_size_yz=attn_context_tp_yz,
                workload_spec=workload_spec,
            )
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            # In the 1D schedule all attention TP ranks shard heads; there is
            # no independent sequence-sharding axis.
            attn_context_tp_x = attn_tp_topology_view.num_chips
            attn_context_tp_yz = 1
            _decode_attention_attn_tp_1d_memory_traffic(
                traffic=reference_traffic,
                transformer_config=model_spec,
                n_seq_so_far=reference_tokens,
                proj_tp_size=proj_tp_topology_view.num_chips,
                attn_tp_size=attn_context_tp_x,
                workload_spec=workload_spec,
            )
        case _:
            raise ValueError(
                "decode attention context correction does not support "
                f"attention TP variant {attn_tp_spec.variant}"
            )

    # KV heads are divided only by TP-X. TP-YZ reduces each rank's context
    # span but does not further divide the set of local heads.
    local_kv_heads, _kv_replication = resolve_kv_heads_per_partition(
        num_kv_heads=model_spec.num_kv_heads,
        tp_x=attn_context_tp_x,
    )
    # Convert anchor traffic to time with the same memory-tier mapping and
    # layer count used for the current operating point.
    reference_memory_time_sec = reference_traffic.get_stats(layers_per_stage)[
        f"{reference_traffic.name}_time_s"
    ]
    # The lower-level model applies gamma only to positive incremental time;
    # contexts at or below the anchor therefore receive no correction.
    correction_time_sec, gamma, delta_memory_time_sec = decode_attention_context_memory_correction(
        current_memory_time_sec=current_memory_time_sec,
        reference_memory_time_sec=reference_memory_time_sec,
        local_kv_heads=local_kv_heads,
        attention_tp_yz=attn_context_tp_yz,
    )
    return DecodeAttentionContextCorrectionResult(
        correction_time_sec=correction_time_sec,
        delta_memory_time_sec=delta_memory_time_sec,
        gamma=gamma,
        local_kv_heads=local_kv_heads,
        attention_tp_yz=attn_context_tp_yz,
        reference_tokens=reference_tokens,
    )


def routed_token_pairs_per_hot_rank(
    *,
    model_spec: TransformerConfig,
    moe_config: ExpertXPUPlacement,
    workload_spec: WorkloadSpec,
    n_seq: int,
) -> float:
    """Return routed token-expert pairs processed by the modeled hot rank."""
    if n_seq <= 0:
        raise ValueError(f"n_seq must be > 0, got {n_seq}")

    num_experts = model_spec.num_regular_experts
    if num_experts <= 1:
        return 0.0

    num_colocated = moe_config.num_colocated_experts
    if num_colocated <= 0:
        raise ValueError(f"num_colocated_experts must be > 0, got {num_colocated}")

    tokens_per_set = workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    tokens_per_expert = tokens_per_set / num_experts
    return tokens_per_expert * moe_config.regular_expert_traffic_multiplier * num_colocated


def hot_rank_routed_tokens_for_overlap(
    *,
    tokens_per_full_expert_set: int,
    num_regular_experts: int,
    regular_expert_traffic_multiplier: float,
    num_colocated_experts: int,
) -> float:
    """Local MoE/FFN tokens that drive overlap tiling on the hot rank.

    Dense FFN is one expert (E=1, k=1): this collapses to ownership tokens
    times colocation. ``routed_token_pairs_per_hot_rank`` still returns 0
    for E<=1 because that helper is routing software overhead, not GEMM work.
    """
    if tokens_per_full_expert_set <= 0:
        raise ValueError(
            "tokens_per_full_expert_set must be > 0, " f"got {tokens_per_full_expert_set}"
        )
    if num_regular_experts <= 0:
        raise ValueError(f"num_regular_experts must be > 0, got {num_regular_experts}")
    if regular_expert_traffic_multiplier <= 0:
        raise ValueError(
            "regular_expert_traffic_multiplier must be > 0, "
            f"got {regular_expert_traffic_multiplier}"
        )
    if num_colocated_experts <= 0:
        raise ValueError(f"num_colocated_experts must be > 0, got {num_colocated_experts}")
    return (
        tokens_per_full_expert_set
        / num_regular_experts
        * regular_expert_traffic_multiplier
        * num_colocated_experts
    )


def moe_token_pair_remote_fraction(
    *,
    pass_mode: PassMode,
    attn_domains_per_full_expert_set: int,
    num_colocated_groups_per_set: int,
) -> float:
    """Fraction of hot-rank routed pairs that leave this rank.

    Decode after the exact handoff: tokens already sit on every local
    group, so only the remote-owner share ``(D − 1) / D`` pays pack
    software. Prefill/train: the token lives on one group, so
    ``(G − 1) / G`` must move. ``D = 1`` or ``G = 1`` is zero. This is
    volume, not a boolean gate.
    """
    if attn_domains_per_full_expert_set <= 0:
        raise ValueError(
            "attn_domains_per_full_expert_set must be > 0, "
            f"got {attn_domains_per_full_expert_set}"
        )
    if num_colocated_groups_per_set <= 0:
        raise ValueError(
            "num_colocated_groups_per_set must be > 0, " f"got {num_colocated_groups_per_set}"
        )
    match pass_mode:
        case PassMode.INFERENCE_DECODE:
            return (attn_domains_per_full_expert_set - 1) / float(attn_domains_per_full_expert_set)
        case PassMode.INFERENCE_PREFILL | PassMode.TRAINING:
            return (num_colocated_groups_per_set - 1) / float(num_colocated_groups_per_set)
        case _:
            raise ValueError(f"unsupported pass_mode {pass_mode}")


def moe_token_pair_overhead_per_pass(
    *,
    model_spec: TransformerConfig,
    moe_config: ExpertXPUPlacement,
    workload_spec: WorkloadSpec,
    n_seq: int,
    pass_mode: PassMode,
) -> float:
    """Return serial MoE token-pair software overhead for one forward/backward pass.

    This is pack/dispatch bookkeeping, not HBM permute bytes and not Hockney
    A2A. Volume is ``N_rows × remote_frac`` (see
    ``moe_token_pair_remote_fraction``). Decode ``D = 1`` and prefill
    ``G = 1`` are zero. Dense E<=1 is zero via
    ``routed_token_pairs_per_hot_rank``. The term is serial: it is not
    part of MoE compute C.
    """
    groups_per_set = moe_config.num_colocated_groups_per_set
    attn_domains = moe_config.attn_domains_per_full_expert_set
    if attn_domains is None:
        raise ValueError("attn_domains_per_full_expert_set must be set")
    remote_frac = moe_token_pair_remote_fraction(
        pass_mode=pass_mode,
        attn_domains_per_full_expert_set=attn_domains,
        num_colocated_groups_per_set=groups_per_set,
    )
    if remote_frac == 0.0:
        return 0.0

    pairs_per_hot_rank = routed_token_pairs_per_hot_rank(
        model_spec=model_spec,
        moe_config=moe_config,
        workload_spec=workload_spec,
        n_seq=n_seq,
    )
    return (
        workload_spec.get_layers_per_pp_stage()
        * pairs_per_hot_rank
        * remote_frac
        * workload_spec.moe_token_pair_overhead_sec
    )
