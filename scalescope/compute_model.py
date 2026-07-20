"""Analytical compute model for Transformer blocks.

This module computes per-XPU FLOPs and times for attention, MoE/FFN,
embedding/unembedding, and training backward/parameter-update phases across
pass modes (training, inference prefill, decode). It takes topology
views and model/workload specs and returns per-layer and per-batch totals
used by the orchestrator to build results.
"""

import math
import warnings

from scalescope.model_spec import (
    attention_layer_classes,
    attn_params_per_xpu,
    expert_params_per_xpu,
    TransformerConfig,
)
from scalescope.workload_spec import InferenceDecodeSpec, MoERouterSyncMode


def matmul_flops(a_fst, a_snd, b_fst, b_snd):
    if a_snd != b_fst:
        raise ValueError(f"matrix reduction dimension mismatch: {a_snd} != {b_fst}.")
    return 2 * a_fst * a_snd * b_snd


def _calculate_prefill_attention_compute(
    proj_tp_topology,
    tp_attn_topology,
    transformer_config: TransformerConfig,
    workload_spec,
) -> tuple[float, float]:
    """Per-layer attention: return (time_per_chip, flops_per_chip) for prefill."""
    proj_time, proj_flops = _calculate_prefill_attention_projection_compute(
        proj_tp_topology, transformer_config, workload_spec
    )
    attn_time, attn_flops = _calculate_prefill_attention_attn_compute(
        tp_attn_topology, transformer_config, workload_spec
    )
    return proj_time + attn_time, proj_flops + attn_flops


def _calculate_prefill_attention_projection_compute(
    proj_tp_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """
    Prefill attention projection compute:
    - Q/K/V projections
    - output projection

    Split out to allow projections to run on a different TP domain than attention compute.
    """
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Q, K, V projections with GQA. Q maps d_model -> num_heads * head_dim
    # (which differs from d_model under "wide-Q" layouts such as MiniMax-M2.5,
    # gpt_oss_120b, qwen3_30b_a3b, qwen3_235b_a22b). K/V map d_model ->
    # num_kv_heads * head_dim.
    q_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, q_proj_out_dim)
    k_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, num_kv_heads * head_dim)
    v_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, num_kv_heads * head_dim)
    kvq_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops

    # Output projection: num_heads * head_dim -> d_model.
    out_proj_flops = matmul_flops(n_batch * n_seq, q_proj_out_dim, q_proj_out_dim, d_model)

    total_flops = kvq_proj_flops + out_proj_flops
    n_chips = proj_tp_topology.num_chips
    flops_per_chip = total_flops / n_chips
    node = proj_tp_topology.physical_topology.node_definition
    time_per_chip = node.get_time_per_flop(dtype_size=float_size) * flops_per_chip
    return time_per_chip, flops_per_chip


def _calculate_prefill_attention_attn_compute(
    tp_attn_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """
    Prefill attention *core* compute:
    - attention score (Q @ K^T)
    - attention–V multiplication (Attn @ V)
    """
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    num_heads = transformer_config.num_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Attention score (Q @ K.T) and attention–V multiplication. Compute is
    # `num_heads` parallel head-local matmuls of inner dim `head_dim`; for
    # GQA, K/V are broadcast back to `num_heads` so the FLOPS count is the
    # same as MHA at q_proj_out_dim = num_heads * head_dim. This is the
    # post-Q-projection activation dim, NOT d_model (they differ for wide-Q).
    #
    # Blended across attention layer classes. The full-attention class keeps
    # the dense n_seq × n_seq score region (no causal-triangle discount, per
    # the dense-matmul convention used throughout). The sliding-window class
    # uses the consistent analogue: a rectangle clipped to the window,
    # n_seq × min(window, n_seq).
    total_flops = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        score_width = layer_class.kv_len(n_seq)
        kqv_flops = n_batch * matmul_flops(
            n_seq, q_proj_out_dim, q_proj_out_dim, score_width
        )  # Q @ K.T
        kqv_flops += n_batch * matmul_flops(
            n_seq, score_width, score_width, q_proj_out_dim
        )  # Attn @ V
        total_flops += layer_class.layer_fraction * kqv_flops
    n_chips = tp_attn_topology.num_chips
    flops_per_chip = total_flops / n_chips
    node = tp_attn_topology.physical_topology.node_definition
    time_per_chip = node.get_time_per_flop(dtype_size=float_size) * flops_per_chip
    return time_per_chip, flops_per_chip


def _calculate_prefill_moe_compute(
    moe_tp_topology,
    tp_attn_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
) -> tuple[float, float]:
    """Per-layer MoE: return (time_per_chip, flops_per_chip) for prefill.

    This function is intentionally the shared MoE forward-compute path for all pass modes:
    training/prefill call it directly, and decode reuses it via _calculate_decode_moe_compute().

    Uses batch size per full expert set, computes per-expert FLOPs, then divides by TP_MoE chips.
    Shared experts not yet supported in this normalization path.

    Router sharding uses ``moe_config.router_shard_factor`` which the orchestrator
    sets to Y_eff for 2D TP (after AG(X) the router is sharded only across Y_eff)
    or attn_tp_size for 1D TP.
    """
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_hidden_size
    num_routed_experts = transformer_config.num_regular_experts
    k = transformer_config.num_selected_experts

    # Shared experts not supported yet
    if (
        getattr(transformer_config, "num_shared_experts", 0) > 0
        or getattr(moe_config, "num_shared_experts_per_dp_rank", 0) > 0
    ):
        raise NotImplementedError(
            "Shared experts are not supported in _calculate_prefill_moe_compute normalization"
        )

    # Router computation uses total tokens per full expert set (batch × seq_length)
    tokens_per_set = (
        workload_spec.get_resolved_batches().batch_size_per_full_expert_set
        * workload_spec.seq_length
    )
    # gating_flops = matmul_flops(tokens_per_set, d_model, d_model, max(0, num_routed_experts))
    # if k > 1:
    #    topk_flops = tokens_per_set * max(0, num_routed_experts) * math.log2(k)
    # else:
    #    topk_flops = tokens_per_set * max(0, num_routed_experts)
    # router_flops = gating_flops + topk_flops

    # Router computation
    # TODO(MB): refine router FLOPs approximation
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    gating_flops = matmul_flops(n_batch * n_seq, d_model, d_model, num_routed_experts)
    if k > 1:
        topk_flops = n_batch * n_seq * num_routed_experts * math.log2(k)
    else:
        topk_flops = n_batch * n_seq * num_routed_experts
    router_flops = gating_flops + topk_flops

    # Decode router mode policy for compute:
    # - replicated_router: each rank computes full router scores locally.
    # - sharded_router: router compute is sharded by router_shard_factor.
    # router_shard_factor is set by the orchestrator: Y_eff for 2D TP (after
    # AG(X) the router is sharded only across Y_eff), attn_tp_size for 1D TP.
    router_compute_multiplier = 1.0
    if isinstance(workload_spec, InferenceDecodeSpec):
        router_mode = workload_spec.decode_moe_router_sync_mode
        router_shard_factor = int(moe_config.router_shard_factor)
        if router_shard_factor < 1:
            raise ValueError(f"Invalid router_shard_factor: {router_shard_factor}")
        if router_mode == MoERouterSyncMode.REPLICATED_ROUTER:
            router_compute_multiplier = float(router_shard_factor)
        elif router_mode == MoERouterSyncMode.SHARDED_ROUTER:
            router_compute_multiplier = 1.0
        else:
            raise ValueError(f"Unsupported decode_moe_router_sync_mode: {router_mode}")
    router_flops *= router_compute_multiplier

    # Expert computation per routed expert based on tokens_per_set/num_routed_experts and traffic multiplier.
    # Scale per-chip compute by the number of colocated experts on each XPU.
    if num_routed_experts <= 0:
        per_expert_ffn_flops = 0
    else:
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        tokens_per_routed_expert = tokens_per_set / num_routed_experts
        effective_tokens = tokens_per_routed_expert * traffic_mult
        n_up = transformer_config.mlp_num_up_projections
        n_down = transformer_config.mlp_num_down_projections
        n_elem = transformer_config.mlp_num_elementwise_passes
        per_expert_ffn_flops = (
            n_up * matmul_flops(effective_tokens, d_model, d_model, d_ff)
            + n_down * matmul_flops(effective_tokens, d_ff, d_ff, d_model)
            + n_elem * effective_tokens * d_ff
        )
        # Multiply by colocated experts per chip
        per_expert_ffn_flops *= moe_config.num_colocated_experts

    # We assume that we need to add each expert output to the final sum
    total_experts_in_group = max(0, int(moe_config.num_all_experts_per_dp_rank))
    if total_experts_in_group == 0:
        warnings.warn(
            "MoE compute modeling: total_experts_in_group resolved to 0 "
            f"(num_all_experts_per_dp_rank={getattr(moe_config, 'num_all_experts_per_dp_rank', None)}). "
            "This will zero out expert-output combine FLOPs; double-check that MoE is intentionally disabled "
            "(e.g. num_regular_experts=0) for this run.",
            stacklevel=2,
        )
    expert_output_combine_flops = n_batch * n_seq * d_model * max(0, total_experts_in_group - 1)

    # Combine expert outputs cost (sum across experts within the set)
    # total_experts_in_group = max(0, num_routed_experts)
    # expert_output_combine_flops = tokens_per_set * d_model * max(0, total_experts_in_group - 1)

    moe_tp_size = moe_tp_topology.num_chips if moe_tp_topology is not None else 1
    router_shard_denom = moe_config.router_shard_factor
    attn_tp_size = tp_attn_topology.num_chips if tp_attn_topology is not None else 1
    total_flops = (
        router_flops / router_shard_denom
        + per_expert_ffn_flops / moe_tp_size
        + expert_output_combine_flops / attn_tp_size
    )
    flops_per_chip = total_flops  # total_flops / moe_tp_size
    node = (
        moe_tp_topology.physical_topology.node_definition if moe_tp_topology is not None else None
    )
    time_per_chip = (
        (node.get_time_per_flop(dtype_size=float_size) * flops_per_chip) if node else 0.0
    )
    return time_per_chip, flops_per_chip


# Prefill function


def forward_prefill_flops_per_chip(
    proj_tp_topology,
    attn_tp_topology,
    moe_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
):
    """Calculate FLOPs for forward pass (prefill) per chip."""
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()

    att_time_per_layer_chip, att_flops_per_layer_chip = _calculate_prefill_attention_compute(
        proj_tp_topology, attn_tp_topology, transformer_config, workload_spec
    )
    moe_time_per_layer_chip, moe_flops_per_layer_chip = _calculate_prefill_moe_compute(
        moe_tp_topology, attn_tp_topology, transformer_config, moe_config, workload_spec
    )

    attn_flops_per_chip_total = num_layers_per_pp_stage * att_flops_per_layer_chip
    moe_flops_per_chip_total = num_layers_per_pp_stage * moe_flops_per_layer_chip
    flops_per_chip_total = attn_flops_per_chip_total + moe_flops_per_chip_total

    attn_time_per_chip_total = num_layers_per_pp_stage * att_time_per_layer_chip
    moe_time_per_chip_total = num_layers_per_pp_stage * moe_time_per_layer_chip
    time_per_chip_total = attn_time_per_chip_total + moe_time_per_chip_total

    return (
        time_per_chip_total,
        flops_per_chip_total,
        attn_flops_per_chip_total,
        moe_flops_per_chip_total,
        attn_time_per_chip_total,
        moe_time_per_chip_total,
    )


# Decode helpers


def _calculate_decode_attention_compute(
    proj_tp_topology,
    attn_tp_topology,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq_so_far: int,
) -> tuple[float, float]:
    """Per-layer attention: return (time_per_chip, flops_per_chip) for decode step."""
    # Projections (QKV + output projection) may eventually live on a separate TP domain.
    proj_time, proj_flops = _calculate_decode_attention_projection_compute(
        proj_tp_topology, transformer_config, workload_spec
    )
    # Core attention compute (KV-cache GEMMs) may live on a different TP domain.
    attn_time, attn_flops = _calculate_decode_attention_attn_compute(
        attn_tp_topology, transformer_config, workload_spec, n_seq_so_far
    )
    return proj_time + attn_time, proj_flops + attn_flops


def decode_attention_compute_breakdown_per_layer(
    *,
    proj_tp_topology,
    attn_tp_topology,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq_so_far: int,
) -> tuple[float, float, float, float]:
    """Decode attention compute breakdown for a single token (per-layer, per-chip).

    Returns:
        (proj_time_s, proj_flops, core_time_s, core_flops)
    """
    proj_time, proj_flops = _calculate_decode_attention_projection_compute(
        proj_tp_topology, transformer_config, workload_spec
    )
    core_time, core_flops = _calculate_decode_attention_attn_compute(
        attn_tp_topology, transformer_config, workload_spec, n_seq_so_far
    )
    return proj_time, proj_flops, core_time, core_flops


def prefill_attention_compute_breakdown_per_layer(
    *,
    proj_tp_topology,
    attn_tp_topology,
    transformer_config: TransformerConfig,
    workload_spec,
) -> tuple[float, float, float, float]:
    """Prefill attention compute breakdown (per-layer, per-chip).

    Returns:
        (proj_time_s, proj_flops, core_time_s, core_flops)
    """
    proj_time, proj_flops = _calculate_prefill_attention_projection_compute(
        proj_tp_topology, transformer_config, workload_spec
    )
    core_time, core_flops = _calculate_prefill_attention_attn_compute(
        attn_tp_topology, transformer_config, workload_spec
    )
    return proj_time, proj_flops, core_time, core_flops


def _calculate_decode_attention_projection_compute(
    proj_tp_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """
    Decode attention projection compute for a single new token:
    - Q/K/V projections
    - output projection

    This is intentionally separated from the KV-cache attention compute so that a future
    design can use a different TP domain for projections (`proj_tp_topology`) while keeping
    attention compute on `attn_tp_topology`.
    """
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Q, K, V projections for the single new token, with GQA. Q maps
    # d_model -> num_heads * head_dim (differs from d_model under wide-Q).
    q_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, q_proj_out_dim)
    k_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, num_kv_heads * head_dim)
    v_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, num_kv_heads * head_dim)
    kvq_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops

    # Output projection for the single new token: num_heads * head_dim -> d_model.
    out_proj_flops = matmul_flops(n_batch * 1, q_proj_out_dim, q_proj_out_dim, d_model)

    total_flops = kvq_proj_flops + out_proj_flops
    n_chips = proj_tp_topology.num_chips
    flops_per_chip = total_flops / n_chips
    node = proj_tp_topology.physical_topology.node_definition
    time_per_chip = node.get_time_per_flop(dtype_size=float_size) * flops_per_chip
    return time_per_chip, flops_per_chip


def _calculate_decode_attention_attn_compute(
    attn_tp_topology, transformer_config: TransformerConfig, workload_spec, n_seq_so_far: int
) -> tuple[float, float]:
    """
    Decode attention *core* compute for a single new token attending to the KV cache:
    - Q @ K_cache^T
    - softmax (not explicitly modeled; typically lower order vs GEMMs)
    - Attn @ V_cache
    """
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    num_heads = transformer_config.num_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    #  KV cache : Q (new) attends to *K_cache* (history). The Q-side
    # activation dim is num_heads * head_dim (= q_proj_out_dim), NOT d_model
    # — they only coincide for non-wide-Q layouts. The matmul is per-head
    # (B, 1, head_dim) @ (B, head_dim, S_cache) replicated across num_heads
    # heads; for GQA the cached K is broadcast back to num_heads so the
    # FLOPS count is q_proj_out_dim across the reduction.
    #
    # Costs are blended across attention layer classes (full vs sliding
    # window): sliding-window layers attend to at most `window_size` cached
    # tokens, so their KV-cache GEMMs are capped at that length.
    total_flops = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        kv_len = layer_class.kv_len(n_seq_so_far)
        qkt_flops = matmul_flops(n_batch * 1, q_proj_out_dim, q_proj_out_dim, kv_len)
        # Attention-V multiplication: (Attn @ V) -> (B, 1, S_kv) @ (B, S_kv, q_proj_out_dim).
        att_v_flops = matmul_flops(n_batch * 1, kv_len, kv_len, q_proj_out_dim)
        total_flops += layer_class.layer_fraction * (qkt_flops + att_v_flops)
    n_chips = attn_tp_topology.num_chips
    flops_per_chip = total_flops / n_chips
    node = attn_tp_topology.physical_topology.node_definition
    time_per_chip = node.get_time_per_flop(dtype_size=float_size) * flops_per_chip
    return time_per_chip, flops_per_chip


def _calculate_decode_moe_compute(
    moe_tp_topology,
    attn_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
) -> tuple[float, float]:
    """Per-layer MoE: return (time_per_chip, flops_per_chip) for decode step."""
    return _calculate_prefill_moe_compute(
        moe_tp_topology, attn_tp_topology, transformer_config, moe_config, workload_spec
    )


def forward_decode_flops_per_chip(
    proj_tp_topology,
    attn_tp_topology,
    moe_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
    n_seq_so_far: int,
):
    """Calculate FLOPs for a single token decode step per chip."""
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()

    # The Core Concept: Attention vs. MLP in Decoding
    # Attention (The "Stateful" Part): When decoding a new token, the attention mechanism must compare the new token's query
    # vector against the key and value vectors of all previous tokens in the sequence.
    # This is what the KV Cache is for. Therefore, the FLOPs for the attention calculation are directly
    # proportional to the length of the sequence so far (n_seq_so_far). Its cost grows with every token you generate.
    #
    # MLP/MoE (The "Stateless" Part): The MLP (or MoE) layers operate only on the output vector from the attention
    # block for the current token. This is a single vector of shape [batch_size, d_model].
    # The MLP doesn't need to look at previous tokens; it just processes the vector it's given.
    # Therefore, its computational cost is constant for every single token you decode, regardless
    # of how long the sequence already is.
    #

    att_time_per_layer_chip, att_flops_per_layer_chip = _calculate_decode_attention_compute(
        proj_tp_topology, attn_tp_topology, transformer_config, workload_spec, n_seq_so_far
    )
    moe_time_per_layer_chip, moe_flops_per_layer_chip = _calculate_decode_moe_compute(
        moe_tp_topology, attn_tp_topology, transformer_config, moe_config, workload_spec
    )

    att_flops_per_chip_total = num_layers_per_pp_stage * att_flops_per_layer_chip
    moe_flops_per_chip_total = num_layers_per_pp_stage * moe_flops_per_layer_chip
    flops_per_chip_total = att_flops_per_chip_total + moe_flops_per_chip_total

    attn_time_per_chip_total = num_layers_per_pp_stage * att_time_per_layer_chip
    moe_time_per_chip_total = num_layers_per_pp_stage * moe_time_per_layer_chip
    time_per_chip_total = attn_time_per_chip_total + moe_time_per_chip_total

    return (
        time_per_chip_total,
        flops_per_chip_total,
        att_flops_per_chip_total,
        moe_flops_per_chip_total,
        attn_time_per_chip_total,
        moe_time_per_chip_total,
    )


# Linear layer backward helper
# Makes the higher-level logic clearer.
def _calculate_backward_linear_layer_flops(
    effective_batch_size: int, in_dim: int, out_dim: int
) -> int:
    # dL/dW = X.T @ dL/dY -> (in_dim, B_eff) @ (B_eff, out_dim)
    weight_grad_flops = matmul_flops(in_dim, effective_batch_size, effective_batch_size, out_dim)
    # dL/dX = dL/dY @ W.T -> (B_eff, out_dim) @ (out_dim, in_dim)
    input_grad_flops = matmul_flops(effective_batch_size, out_dim, out_dim, in_dim)
    return weight_grad_flops + input_grad_flops


# Backward-pass helpers


def _calculate_backward_attention_compute(
    proj_tp_topology,
    tp_attn_topology,
    transformer_config: TransformerConfig,
    workload_spec,
) -> tuple[float, float]:
    """Per-layer attention backward: return (time_per_chip, flops_per_chip)."""
    proj_time, proj_flops = _calculate_backward_attention_projection_compute(
        proj_tp_topology, transformer_config, workload_spec
    )
    attn_time, attn_flops = _calculate_backward_attention_attn_compute(
        tp_attn_topology, transformer_config, workload_spec
    )
    return proj_time + attn_time, proj_flops + attn_flops


def _calculate_backward_attention_projection_compute(
    proj_tp_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """
    Backward compute attributable to attention projections:
    - gradients for W_o (output projection)
    - gradients for W_q, W_k, W_v (QKV projections)

    Split out to allow projections to run on a different TP domain than attention compute.
    """
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Gradients for W_o (q_proj_out_dim -> d_model).
    w_o_grad_flops = _calculate_backward_linear_layer_flops(
        n_batch * n_seq, q_proj_out_dim, d_model
    )

    # Gradients for W_q (d_model -> q_proj_out_dim) and W_k / W_v
    # (d_model -> num_kv_heads * head_dim).
    w_q_grad_flops = _calculate_backward_linear_layer_flops(
        n_batch * n_seq, d_model, q_proj_out_dim
    )
    w_kv_grad_flops = 2 * _calculate_backward_linear_layer_flops(
        n_batch * n_seq, d_model, num_kv_heads * head_dim
    )
    w_kvq_proj_grad_flops = w_q_grad_flops + w_kv_grad_flops

    total_flops = w_o_grad_flops + w_kvq_proj_grad_flops
    n_chips = proj_tp_topology.num_chips
    flops_per_chip = total_flops / n_chips
    node = proj_tp_topology.physical_topology.node_definition
    time_per_chip = node.get_time_per_flop(dtype_size=float_size) * flops_per_chip
    return time_per_chip, flops_per_chip


def _calculate_backward_attention_attn_compute(
    tp_attn_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """
    Backward compute attributable to the core attention mechanism (excluding projections).
    """
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    num_heads = transformer_config.num_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Gradients for the core attention mechanism. All compute is at the
    # post-Q-projection dim (q_proj_out_dim = num_heads * head_dim): for GQA
    # the K/V activations are broadcast back to num_heads during the
    # attention compute, so the backward FLOPS — including dK and dV — are
    # at that broadcast level (gradient reduction back to num_kv_heads is
    # an O(1) sum, not in the matmul FLOPS count).
    #
    # Blended across attention layer classes: sliding-window layers have a
    # score region clipped to n_seq × min(window, n_seq) (same windowed-
    # rectangle convention as the forward prefill path).
    total_flops = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        score_width = layer_class.kv_len(n_seq)
        K_grad_flops = n_batch * (
            matmul_flops(n_seq, q_proj_out_dim, q_proj_out_dim, score_width)
            + matmul_flops(n_seq, score_width, score_width, q_proj_out_dim)
        )
        Q_grad_flops = n_batch * (
            matmul_flops(n_seq, q_proj_out_dim, q_proj_out_dim, score_width)
            + matmul_flops(n_seq, score_width, score_width, q_proj_out_dim)
        )
        V_grad_flops = n_batch * matmul_flops(n_seq, score_width, score_width, q_proj_out_dim)
        total_flops += layer_class.layer_fraction * (K_grad_flops + Q_grad_flops + V_grad_flops)
    n_chips = tp_attn_topology.num_chips
    flops_per_chip = total_flops / n_chips
    node = tp_attn_topology.physical_topology.node_definition
    time_per_chip = node.get_time_per_flop(dtype_size=float_size) * flops_per_chip
    return time_per_chip, flops_per_chip


def _calculate_backward_moe_compute(
    moe_tp_topology,
    tp_attn_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
) -> tuple[float, float]:
    """Per-layer MoE backward: return (time_per_chip, flops_per_chip).

    Mirrors forward path normalization: router-related grads are normalized by
    ``router_shard_factor`` (resolved in orchestration), expert FFN grads are
    normalized by TP_MoE chips. Uses tokens per full expert set.
    Shared experts not supported in this normalization path.
    """
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_hidden_size

    # Tokens per full expert set
    tokens_per_set = (
        workload_spec.get_resolved_batches().batch_size_per_full_expert_set
        * workload_spec.seq_length
    )

    num_routed_experts = transformer_config.num_regular_experts
    if (
        getattr(transformer_config, "num_shared_experts", 0) > 0
        or getattr(moe_config, "num_shared_experts_per_dp_rank", 0) > 0
    ):
        raise NotImplementedError(
            "Shared experts are not supported in _calculate_backward_moe_compute normalization"
        )

    # Router gradients use per-DP tokens and are normalized by router_shard_factor
    # (same convention as forward router FLOPs).
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    router_grad_flops = _calculate_backward_linear_layer_flops(
        n_batch * n_seq, d_model, max(0, num_routed_experts)
    )
    router_grad_flops += (
        n_batch * n_seq * max(0, num_routed_experts)
    )  # Mask/softmax-related cost approximation
    router_shard_denom = moe_config.router_shard_factor
    if router_shard_denom < 1:
        raise ValueError(f"Invalid router_shard_factor: {router_shard_denom}")
    router_grad_flops_per_chip = router_grad_flops / router_shard_denom

    # Expert gradients (normalized by TP_MoE chips)
    traffic_mult = moe_config.regular_expert_traffic_multiplier
    routed_tokens_per_expert = tokens_per_set / num_routed_experts * traffic_mult
    n_up = transformer_config.mlp_num_up_projections
    n_down = transformer_config.mlp_num_down_projections
    n_elem = transformer_config.mlp_num_elementwise_passes
    expert_grad_total_flops = (
        n_up * _calculate_backward_linear_layer_flops(routed_tokens_per_expert, d_model, d_ff)
        + n_down * _calculate_backward_linear_layer_flops(routed_tokens_per_expert, d_ff, d_model)
        + n_elem * routed_tokens_per_expert * d_ff
    )
    # Scale per-chip expert grads by number of colocated experts on an XPU
    expert_grad_total_flops *= moe_config.num_colocated_experts
    moe_tp_size = moe_tp_topology.num_chips if moe_tp_topology is not None else 1
    expert_grad_flops_per_chip = expert_grad_total_flops / moe_tp_size

    flops_per_chip = router_grad_flops_per_chip + expert_grad_flops_per_chip
    node = (
        moe_tp_topology.physical_topology.node_definition if moe_tp_topology is not None else None
    )
    time_per_chip = (
        (node.get_time_per_flop(dtype_size=float_size) * flops_per_chip) if node else 0.0
    )
    return time_per_chip, flops_per_chip


# Backward pass (using helpers)


def backward_pass_flops_per_chip(
    proj_tp_topology,
    attn_tp_topology,
    moe_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
):
    """Calculate FLOPs for backward pass per chip."""
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()
    d_model = transformer_config.d_model

    attn_tp_size = attn_tp_topology.num_chips

    att_time_per_layer_chip, att_flops_per_layer_chip = _calculate_backward_attention_compute(
        proj_tp_topology, attn_tp_topology, transformer_config, workload_spec
    )
    moe_time_per_layer_chip, moe_flops_per_layer_chip = _calculate_backward_moe_compute(
        moe_tp_topology, attn_tp_topology, transformer_config, moe_config, workload_spec
    )
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    n_seq = workload_spec.seq_length
    layer_norm_grad_flops = n_batch * n_seq * d_model * 6
    layer_norm_grad_flops_per_chip = layer_norm_grad_flops / attn_tp_size
    layer_norm_time_per_chip = (
        attn_tp_topology.physical_topology.node_definition.get_time_per_flop(dtype_size=float_size)
        * layer_norm_grad_flops_per_chip
    )

    attn_time_total = num_layers_per_pp_stage * (
        att_time_per_layer_chip + 2 * layer_norm_time_per_chip
    )
    moe_time_total = num_layers_per_pp_stage * moe_time_per_layer_chip
    time_per_chip_total = attn_time_total + moe_time_total

    attn_flops_total = num_layers_per_pp_stage * (
        att_flops_per_layer_chip + 2 * layer_norm_grad_flops_per_chip
    )
    moe_flops_total = num_layers_per_pp_stage * moe_flops_per_layer_chip
    flops_per_chip_total = attn_flops_total + moe_flops_total

    return (
        time_per_chip_total,
        flops_per_chip_total,
        attn_flops_total,
        moe_flops_total,
        attn_time_total,
        moe_time_total,
    )


def parameter_update_flops(
    tp_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
):

    # Per-XPU parameter counts (attention sharded by TP, experts via placement expert set footprint)
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()
    attn_params = attn_params_per_xpu(transformer_config, tp_topology_view)
    expert_params = expert_params_per_xpu(transformer_config, moe_config)
    n_model_params = num_layers_per_pp_stage * (attn_params + expert_params)

    # Require optimizer to be explicitly provided on the workload spec (TrainSpec)
    optimizer = workload_spec.optimizer
    if optimizer.lower() == "sgd":
        # SGD update rule: param = param - learning_rate * gradient
        # 1 multiplication and 1 subtraction per parameter
        flops_per_param = 2
    elif optimizer.lower() == "adam":
        # Adam update rule:
        # m = β1 * m + (1 - β1) * gradient
        # v = β2 * v + (1 - β2) * gradient^2
        # m_hat = m / (1 - β1^t)
        # v_hat = v / (1 - β2^t)
        # param = param - learning_rate * m_hat / (sqrt(v_hat) + ε)
        # For now approximate 10 operations per parameter
        flops_per_param = 10
    else:
        raise ValueError("Unsupported optimizer. Choose 'sgd' or 'adam'.")

    flops_per_chip = n_model_params * flops_per_param
    node = tp_topology_view.physical_topology.node_definition
    # Parameter updates typically run in full precision (e.g., FP32) even if training compute is BF16/FP16
    dtype_size_for_update = workload_spec.full_precision_param_float_size
    time_per_chip = node.get_time_per_flop(dtype_size=dtype_size_for_update) * flops_per_chip
    return time_per_chip, flops_per_chip


def calculate_unembedding_compute(
    attn_tp_topology,
    model_spec: "TransformerConfig",
    workload_spec,
    seq_len: int | None = None,
):
    """
    Calculate FLOPs and time for the unembedding (decoding) layer.
    This is modeled as a large matrix multiplication.
    """
    node = attn_tp_topology.physical_topology.node_definition
    tp_size = attn_tp_topology.num_chips
    batch_size = workload_spec.get_resolved_batches().batch_size_per_dp
    if seq_len is None:
        seq_len = workload_spec.seq_length

    # FLOPs for a single MatMul using helper: (batch*seq, d_model) @ (d_model, vocab_size)
    flops = matmul_flops(
        batch_size * seq_len, model_spec.d_model, model_spec.d_model, model_spec.vocab_size
    )
    flops_per_chip = flops / tp_size

    time_per_chip = (
        node.get_time_per_flop(dtype_size=workload_spec.float_size_bytes) * flops_per_chip
    )

    return time_per_chip, flops_per_chip


def calculate_embedding_compute(
    attn_tp_topology,
    model_spec: "TransformerConfig",
    workload_spec,
    seq_len: int | None = None,
):
    """Model the embedding layer as a cheap lookup operation."""
    return 0.0, 0.0
