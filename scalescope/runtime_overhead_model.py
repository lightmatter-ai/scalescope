"""Software/runtime overheads that are neither FLOPs, HBM traffic, nor network communication."""

from scalescope.ep_strategy import ExpertXPUPlacement
from scalescope.model_spec import TransformerConfig
from scalescope.utils import PassMode
from scalescope.workload_spec import WorkloadSpec

# Empirical clean-panel fit for incremental decode attention-core memory time
# above an explicit reference context. The calibration set has TP-YZ=1.
# Extending inverse local work to 2D uses TP-YZ/local-KV-heads, which recovers
# the fitted expression exactly at TP-YZ=1 but remains an extrapolation until
# matched 2D measurements are available.
DECODE_ATTN_CONTEXT_GAMMA_INTERCEPT = 1.5559183649426251
DECODE_ATTN_CONTEXT_GAMMA_INV_LOCAL_WORK_SLOPE = 2.388579558589879
DECODE_ATTN_CONTEXT_GAMMA_CENTER = 0.5


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
