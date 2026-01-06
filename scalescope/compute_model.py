"""Analytical compute model for Transformer blocks.

This module computes per-XPU FLOPs and times for attention, MoE/FFN,
embedding/unembedding, and training backward/parameter-update phases across
pass modes (training, inference prefill, decode). It takes topology
views and model/workload specs and returns per-layer and per-batch totals
used by the orchestrator to build results.
"""

import math

from scalescope.model_spec import (
    attn_params_per_xpu,
    expert_params_per_xpu,
    TransformerConfig,
)


def matmul_flops(a_fst, a_snd, b_fst, b_snd):
    if a_snd != b_fst:
        raise ValueError(f"matrix reduction dimension mismatch: {a_snd} != {b_fst}.")
    return 2 * a_fst * a_snd * b_snd


def _calculate_prefill_attention_compute(
    tp_attn_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """Per-layer attention: return (time_per_chip, flops_per_chip) for prefill."""
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    # Q, K, V projections with GQA
    q_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, d_model)
    k_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, num_kv_heads * head_dim)
    v_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, num_kv_heads * head_dim)
    kvq_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops

    # Attention score (Q @ K.T) and attention–V multiplication.
    kqv_flops = n_batch * matmul_flops(n_seq, d_model, d_model, n_seq)  # Q @ K.T
    kqv_flops += n_batch * matmul_flops(n_seq, n_seq, n_seq, d_model)  # Attn @ V

    # Output projection
    out_proj_flops = matmul_flops(n_batch * n_seq, d_model, d_model, d_model)

    total_flops = kvq_proj_flops + kqv_flops + out_proj_flops
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

    Uses batch size per full expert set, computes per-expert FLOPs, then divides by TP_MoE chips.
    Shared experts not yet supported in this normalization path.

    Args:
        tp_attn_topology:
    """
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_ratio * d_model
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

    # Expert computation per routed expert based on tokens_per_set/num_routed_experts and traffic multiplier.
    # Scale per-chip compute by the number of colocated experts on each XPU.
    if num_routed_experts <= 0:
        per_expert_ffn_flops = 0
    else:
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        tokens_per_routed_expert = tokens_per_set / num_routed_experts
        effective_tokens = tokens_per_routed_expert * traffic_mult
        per_expert_ffn_flops = matmul_flops(
            effective_tokens, d_model, d_model, d_ff
        ) + matmul_flops(effective_tokens, d_ff, d_ff, d_model)
        # Multiply by colocated experts per chip
        per_expert_ffn_flops *= moe_config.num_colocated_experts

    # We assume that we need to add each expert output to the final sum
    total_experts_in_group = moe_config.num_all_experts_per_dp_rank
    expert_output_combine_flops = n_batch * n_seq * d_model * (total_experts_in_group - 1)

    # Combine expert outputs cost (sum across experts within the set)
    # total_experts_in_group = max(0, num_routed_experts)
    # expert_output_combine_flops = tokens_per_set * d_model * max(0, total_experts_in_group - 1)

    moe_tp_size = moe_tp_topology.num_chips if moe_tp_topology is not None else 1
    attn_tp_size = tp_attn_topology.num_chips if tp_attn_topology is not None else 1
    total_flops = (
        router_flops / attn_tp_size
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
    attn_tp_topology,
    moe_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
):
    """Calculate FLOPs for forward pass (prefill) per chip."""
    num_layers_per_pp_stage = workload_spec.get_layers_per_pp_stage()

    att_time_per_layer_chip, att_flops_per_layer_chip = _calculate_prefill_attention_compute(
        attn_tp_topology, transformer_config, workload_spec
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
    attn_tp_topology, transformer_config: TransformerConfig, workload_spec, n_seq_so_far: int
) -> tuple[float, float]:
    """Per-layer attention: return (time_per_chip, flops_per_chip) for decode step."""
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    # Q, K, V projections are for the single new token, with GQA
    q_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, d_model)
    k_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, num_kv_heads * head_dim)
    v_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, num_kv_heads * head_dim)
    kvq_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops

    #  KV cache : Q (new) attends to *K_cache* (history)
    # Q @ K.T -> (B, 1, D) @ (B, D, S_cache) = (B, 1, S_cache)
    qkt_flops = matmul_flops(n_batch * 1, d_model, d_model, n_seq_so_far)

    # KV cache: Attends to V cache
    # Attention-V multiplication: (Attn @ V) -> (B, 1, S_cache) @ (B, S_cache, D)
    att_v_flops = matmul_flops(n_batch * 1, n_seq_so_far, n_seq_so_far, d_model)

    # Output projection is for the single new token
    out_proj_flops = matmul_flops(n_batch * 1, d_model, d_model, d_model)

    total_flops = kvq_proj_flops + qkt_flops + att_v_flops + out_proj_flops
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
    # MoE logic same as prefill with n_seq=1.
    return _calculate_prefill_moe_compute(
        moe_tp_topology, attn_tp_topology, transformer_config, moe_config, workload_spec
    )


def forward_decode_flops_per_chip(
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
        attn_tp_topology, transformer_config, workload_spec, n_seq_so_far
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
    tp_attn_topology, transformer_config: TransformerConfig, workload_spec
) -> tuple[float, float]:
    """Per-layer attention backward: return (time_per_chip, flops_per_chip)."""
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    # Gradients for W_o
    w_o_grad_flops = _calculate_backward_linear_layer_flops(n_batch * n_seq, d_model, d_model)

    # Gradients for the core attention mechanism
    # dL/dK = (Q * V^T) * dL/dAttn etc.
    K_grad_flops = n_batch * (
        matmul_flops(n_seq, d_model, d_model, n_seq) + matmul_flops(n_seq, n_seq, n_seq, d_model)
    )
    Q_grad_flops = n_batch * (
        matmul_flops(n_seq, d_model, d_model, n_seq) + matmul_flops(n_seq, n_seq, n_seq, d_model)
    )
    V_grad_flops = n_batch * matmul_flops(n_seq, n_seq, n_seq, d_model)
    core_attn_grads = K_grad_flops + Q_grad_flops + V_grad_flops

    # Gradients for W_q, W_k, W_v
    w_q_grad_flops = _calculate_backward_linear_layer_flops(n_batch * n_seq, d_model, d_model)
    w_kv_grad_flops = 2 * _calculate_backward_linear_layer_flops(
        n_batch * n_seq, d_model, num_kv_heads * head_dim
    )
    w_kvq_proj_grad_flops = w_q_grad_flops + w_kv_grad_flops

    total_flops = w_o_grad_flops + core_attn_grads + w_kvq_proj_grad_flops
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

    Mirrors forward path normalization: router-related grads normalized by TP_Attn chips,
    expert FFN grads normalized by TP_MoE chips. Uses tokens per full expert set.
    Shared experts not supported in this normalization path.
    """
    float_size = workload_spec.float_size_bytes
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_ratio * d_model

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

    # Router gradients (normalized by TP_Attn chips) use per-DP tokens (old path parity)
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    router_grad_flops = _calculate_backward_linear_layer_flops(
        n_batch * n_seq, d_model, max(0, num_routed_experts)
    )
    router_grad_flops += (
        n_batch * n_seq * max(0, num_routed_experts)
    )  # Mask/softmax-related cost approximation
    n_chips_tp_attn = tp_attn_topology.num_chips if tp_attn_topology is not None else 1
    router_grad_flops_per_chip = router_grad_flops / n_chips_tp_attn

    # Expert gradients (normalized by TP_MoE chips)
    traffic_mult = moe_config.regular_expert_traffic_multiplier
    routed_tokens_per_expert = tokens_per_set / num_routed_experts * traffic_mult
    ffn2_flops = _calculate_backward_linear_layer_flops(routed_tokens_per_expert, d_ff, d_model)
    activation_grad_flops = routed_tokens_per_expert * d_ff
    ffn1_flops = _calculate_backward_linear_layer_flops(routed_tokens_per_expert, d_model, d_ff)
    expert_grad_total_flops = ffn2_flops + activation_grad_flops + ffn1_flops
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
        attn_tp_topology, transformer_config, workload_spec
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
