"""Analytical memory model for traffic and footprint.

Computes per-layer/per-batch memory traffic (bytes/time) and persistent
footprint per XPU for attention and MoE/FFN, including optimizer/saved
activations/gradient buffers and (in inference) KV cache. Also verifies tier
capacities and computes detailed per-component stats.
"""

from dataclasses import dataclass
from typing import Any, Dict

from scalescope.model_spec import (
    attn_params_per_xpu,
    expert_params_per_xpu,
    TransformerConfig,
)
from scalescope.utils import PassMode
from scalescope.workload_spec import (
    ActivationStorageOptimizationStrategy,
    DataParallelVariant,
    TensorParallelVariant,
    TrainSpec,
    WorkloadSpec,
)


@dataclass
class MemoryAccess:
    bytes_read: int = 0
    bytes_written: int = 0

    def total(self):
        return self.bytes_read + self.bytes_written

    def __add__(self, other):
        return MemoryAccess(
            bytes_read=self.bytes_read + other.bytes_read,
            bytes_written=self.bytes_written + other.bytes_written,
        )

    def __eq__(self, other: Any) -> bool:
        """
        Custom equality check for the MemoryAccess class.
        Returns True if both bytes_read and bytes_written are identical.
        """
        if not isinstance(other, MemoryAccess):
            return NotImplemented
        return self.bytes_read == other.bytes_read and self.bytes_written == other.bytes_written


class LayerMemoryTraffic:
    def __init__(self, name: str, device_for: Dict[str, str], node):
        self.name = name
        self.weights: MemoryAccess = MemoryAccess()
        self.activations: MemoryAccess = MemoryAccess()
        self.gradients: MemoryAccess = MemoryAccess()
        self.kv_cache: MemoryAccess = MemoryAccess()
        # Map each component to a memory device tier name (must be provided)
        if device_for is None:
            raise ValueError("device_for mapping must be provided")
        self.device_for: Dict[str, str] = device_for
        self.node = node

    def total(self):
        """Returns the total bytes moved (read + written) for all components."""
        return (
            self.weights.total()
            + self.activations.total()
            + self.gradients.total()
            + self.kv_cache.total()
        )

    def __add__(self, other):
        # When adding, preserve the name of the left operand.
        result = LayerMemoryTraffic(
            name=self.name, device_for=self.device_for.copy(), node=self.node
        )
        result.weights = self.weights + other.weights
        result.activations = self.activations + other.activations
        result.gradients = self.gradients + other.gradients
        result.kv_cache = self.kv_cache + other.kv_cache
        return result

    def get_stats(self, num_layers_per_pp_stage: int) -> dict:
        """
        Returns a dictionary of detailed memory stats, with keys prefixed by the instance's name.
        """
        if self.node is None:
            raise ValueError(
                "LayerMemoryTraffic.node is not set; cannot resolve device memory bandwidths."
            )
        required_components = ["weights", "activations", "gradients", "kv_cache"]
        for comp in required_components:
            if comp not in self.device_for:
                raise ValueError(f"LayerMemoryTraffic.device_for missing mapping for '{comp}'")
        # Construct per-tier bandwidths from node
        tier_to_bw = {
            tier: mem.total_bandwidth for tier, mem in self.node.device_memory_tiers.items()
        }
        stats = {}
        # Per-layer, per-component stats
        total_time_per_layer = 0.0
        total_bytes_per_layer = 0
        for component_name in required_components:
            component_traffic = getattr(self, component_name)
            component_bytes = component_traffic.total()
            device = self.device_for[component_name]
            bw = tier_to_bw[device]
            component_time = component_bytes / bw
            component_time_read = component_traffic.bytes_read / bw if bw > 0 else 0.0
            component_time_write = component_traffic.bytes_written / bw if bw > 0 else 0.0

            stats[f"{self.name}_{component_name}_read_bytes_per_layer"] = (
                component_traffic.bytes_read
            )
            stats[f"{self.name}_{component_name}_written_bytes_per_layer"] = (
                component_traffic.bytes_written
            )
            stats[f"{self.name}_{component_name}_total_bytes_per_layer"] = component_bytes
            stats[f"{self.name}_{component_name}_time_s_per_layer"] = component_time
            stats[f"{self.name}_{component_name}_time_read_s_per_layer"] = component_time_read
            stats[f"{self.name}_{component_name}_time_write_s_per_layer"] = component_time_write
            total_bytes_per_layer += component_bytes
            total_time_per_layer += component_time

        # Total stats for this pass (e.g., forward or backward)
        total_bytes = total_bytes_per_layer * num_layers_per_pp_stage
        total_time_s = total_time_per_layer * num_layers_per_pp_stage

        stats[f"{self.name}_total_bytes_per_layer"] = total_bytes_per_layer
        stats[f"{self.name}_total_bytes"] = total_bytes
        stats[f"{self.name}_time_s"] = total_time_s

        return stats

    def compare_and_assert(self, expected: "LayerMemoryTraffic", msg_prefix: str = ""):
        """
        Asserts that this instance is equal to another LayerMemoryTraffic instance.

        Provides a detailed error message indicating exactly which field and sub-field
        (bytes_read or bytes_written) did not match.

        Args:
            expected: The expected LayerMemoryTraffic object to compare against.
            msg_prefix: An optional message to prefix to the assertion error for context.
        """
        prefix = f"{msg_prefix}: " if msg_prefix else ""

        # We can now use the clean `==` operator defined on MemoryAccess
        assert self.weights == expected.weights, (
            f"{prefix}Mismatch in 'weights'.\n"
            f"  Actual:   (read={self.weights.bytes_read}, written={self.weights.bytes_written})\n"
            f"  Expected: (read={expected.weights.bytes_read}, written={expected.weights.bytes_written})"
        )

        assert self.activations == expected.activations, (
            f"{prefix}Mismatch in 'activations'.\n"
            f"  Actual:   (read={self.activations.bytes_read}, written={self.activations.bytes_written})\n"
            f"  Expected: (read={expected.activations.bytes_read}, written={expected.activations.bytes_written})"
        )

        assert self.gradients == expected.gradients, (
            f"{prefix}Mismatch in 'gradients'.\n"
            f"  Actual:   (read={self.gradients.bytes_read}, written={self.gradients.bytes_written})\n"
            f"  Expected: (read={expected.gradients.bytes_read}, written={expected.gradients.bytes_written})"
        )

        assert self.kv_cache == expected.kv_cache, (
            f"{prefix}Mismatch in 'kv_cache'.\n"
            f"  Actual:   (read={self.kv_cache.bytes_read}, written={self.kv_cache.bytes_written})\n"
            f"  Expected: (read={expected.kv_cache.bytes_read}, written={expected.kv_cache.bytes_written})"
        )


class MemoryFootprint:
    """
    Represents the memory footprint of a model component (e.g., Attention or FFN) on a single chip.
    Tracks per-component byte counts and maps each component to a specific memory tier.
    """

    def __init__(self, name: str, device_for: Dict[str, str], node):
        self.name = name
        # Define and validate required components once
        self.components = [
            "weights",
            "optimizer_states",
            "saved_activations",
            "gradient_buffers",
            "kv_cache",
            "transient_forward_peak",
            "transient_backward_peak",
        ]
        # Required per-component → tier mapping
        for k in self.components:
            if k not in device_for:
                raise ValueError(f"MemoryFootprint.device_for missing mapping for '{k}'")
        self.device_for: Dict[str, str] = device_for
        # Hardware node to resolve tier capacities
        self.node = node

        # Byte counts
        self.weights: int = 0
        self.optimizer_states: int = 0
        self.saved_activations: int = 0
        self.gradient_buffers: int = 0
        self.kv_cache: int = 0
        self.transient_forward_peak: int = 0
        self.transient_backward_peak: int = 0

    def get_stats(self) -> dict:
        """
        Returns a flat dictionary of the footprint's stats, prefixed with its name.
        """
        return {
            f"{self.name}_weights_bytes": self.weights,
            f"{self.name}_optimizer_states_bytes": self.optimizer_states,
            f"{self.name}_saved_activations_bytes": self.saved_activations,
            f"{self.name}_gradient_buffers_bytes": self.gradient_buffers,
            f"{self.name}_kv_cache_bytes": self.kv_cache,
            f"{self.name}_transient_forward_peak_bytes": self.transient_forward_peak,
            f"{self.name}_transient_backward_peak_bytes": self.transient_backward_peak,
        }

    def __add__(self, other: "MemoryFootprint") -> "MemoryFootprint":
        if not isinstance(other, MemoryFootprint):
            return NotImplemented
        if self.device_for != other.device_for:
            raise ValueError("Cannot add MemoryFootprint with different device_for mappings")
        if self.node is not other.node:
            # We require the same node definition to compare capacities coherently
            raise ValueError("Cannot add MemoryFootprint from different nodes")
        combined = MemoryFootprint(
            name=f"{self.name}+{other.name}", device_for=self.device_for.copy(), node=self.node
        )
        for component_name in self.components:
            setattr(
                combined,
                component_name,
                int(getattr(self, component_name)) + int(getattr(other, component_name)),
            )
        return combined

    def verify(self):
        """
        Simplified capacity check: sum bytes per device across all components
        (weights, optimizer, saved activations, forward/backward peaks) and
        verify the total does not exceed the device capacity.
        Raises ValueError listing any violating tiers.
        """
        if self.node is None:
            raise ValueError("MemoryFootprint.node is not set; cannot verify capacities.")
        tiers = self.node.device_memory_tiers
        usage_by_tier: Dict[str, int] = {}

        def add_bytes(tier: str, amount: int):
            usage_by_tier[tier] = usage_by_tier.get(tier, 0) + int(amount)

        for component_name in self.components:
            if component_name not in self.device_for:
                raise ValueError(
                    f"MemoryFootprint.device_for missing mapping for '{component_name}'"
                )
            tier = self.device_for[component_name]
            bytes_val = getattr(self, component_name)
            add_bytes(tier, bytes_val)

        violations = []
        for tier_name, used in usage_by_tier.items():
            mem = tiers.get(tier_name)
            if mem is None:
                violations.append(f"tier '{tier_name}' not found on node")
                continue
            cap = int(mem.total_size)
            if used > cap:
                violations.append(f"tier '{tier_name}' usage {used} > capacity {cap}")

        if violations:
            raise ValueError(
                f"MemoryFootprint capacity verification failed for '{self.name}': "
                + "; ".join(violations)
            )


# Weights memory
def calculate_attn_weights_memory_footprint(
    transformer_config: TransformerConfig,
    tp_size: int,
    dp_size: int,
    dp_variant: str,
    float_size: int,
) -> int:
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    q_weight_size = d_model * d_model
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = d_model * d_model
    total_attn_weights = q_weight_size + kv_weight_size + out_weight_size

    sharded_weights = total_attn_weights / tp_size
    if dp_variant == "ZERO_STAGE_3":
        sharded_weights /= dp_size

    return sharded_weights * float_size


def calculate_ffn_moe_weights_memory_footprint(
    transformer_config: TransformerConfig,
    moe_config,
    moe_tp_size: int,
    moe_dp_size: int,
    dp_variant: str,
    float_size: int,
) -> int:
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_hidden_size

    ffn1_weights = d_model * d_ff
    ffn2_weights = d_ff * d_model
    total_ffn_weights = ffn1_weights + ffn2_weights

    # Account for colocated experts per XPU
    num_colocated = int(getattr(moe_config, "num_colocated_experts", 1) or 1)
    sharded_weights = (total_ffn_weights / moe_tp_size) * num_colocated
    if dp_variant == "ZERO_STAGE_3":
        sharded_weights /= moe_dp_size

    return sharded_weights * float_size


# Optimizer states memory
def calculate_attn_optimizer_states_memory_footprint(
    transformer_config: TransformerConfig,
    tp_size: int,
    dp_size: int,
    dp_variant: str,
    float_size: int,
    optimizer: str,
) -> int:
    # Assuming Adam optimizer (2 states: momentum and variance)
    if optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer for optimizer state footprint: {optimizer}. Only 'adam' is supported."
        )
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    q_weight_size = d_model * d_model
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = d_model * d_model
    total_attn_weights = q_weight_size + kv_weight_size + out_weight_size

    # Weights are always sharded by Tensor Parallelism
    sharded_weights = total_attn_weights / tp_size

    # Optimizer states are sharded across Data Parallelism ranks for ZeRO variants
    if "ZeRO" in dp_variant:
        sharded_weights /= dp_size

    return 2 * sharded_weights * float_size


def calculate_ffn_moe_optimizer_states_memory_footprint(
    transformer_config: TransformerConfig,
    moe_config,
    moe_tp_size: int,
    moe_dp_size: int,
    dp_variant: str,
    float_size: int,
) -> int:
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_hidden_size

    ffn1_weights = d_model * d_ff
    ffn2_weights = d_ff * d_model
    total_ffn_weights = ffn1_weights + ffn2_weights

    # Weights are always sharded by Tensor Parallelism and scaled by colocated experts
    num_colocated = int(getattr(moe_config, "num_colocated_experts", 1) or 1)
    sharded_weights = (total_ffn_weights / moe_tp_size) * num_colocated

    # Optimizer states are sharded across Data Parallelism ranks for ZeRO variants
    if "ZeRO" in dp_variant:
        sharded_weights /= moe_dp_size

    return 2 * sharded_weights * float_size


# Saved activations memory
def calculate_attn_saved_activations_memory_footprint(
    transformer_config: TransformerConfig,
    workload_spec: "TrainSpec",
    activation_storage_strategy: ActivationStorageOptimizationStrategy,
    tp_size: int,
) -> int:
    float_size = workload_spec.float_size_bytes
    if (
        activation_storage_strategy == "RECOMPUTATION"
        or activation_storage_strategy == "GATHER_RECOMPUTATION"
    ):
        # Only save the input to the layer
        return (
            workload_spec.effective_micro_batch_size
            * workload_spec.seq_length
            * transformer_config.d_model
            * float_size
        ) / tp_size

    # For NONE and GATHER, we save intermediate activations.
    # This is a simplified model; a detailed one would sum specific tensors.
    # Here, we approximate it as a multiple of the layer input.
    return (
        3
        * (
            workload_spec.effective_micro_batch_size
            * workload_spec.seq_length
            * transformer_config.d_model
            * float_size
        )
        / tp_size
    )


def calculate_ffn_moe_saved_activations_memory_footprint(
    transformer_config: TransformerConfig,
    workload_spec: "TrainSpec",
    activation_storage_strategy: ActivationStorageOptimizationStrategy,
    attn_tp_size: int,
) -> int:
    float_size = workload_spec.float_size_bytes
    if (
        activation_storage_strategy == "RECOMPUTATION"
        or activation_storage_strategy == "GATHER_RECOMPUTATION"
    ):
        return (
            workload_spec.effective_micro_batch_size
            * workload_spec.seq_length
            * transformer_config.d_model
            * float_size
        ) / attn_tp_size

    # Approximating saved activations for FFN/MoE.
    return (
        2
        * (
            workload_spec.effective_micro_batch_size
            * workload_spec.seq_length
            * transformer_config.d_model
            * float_size
        )
        / attn_tp_size
    )


# Transient memory
def calculate_attn_forward_transient_peak_memory_footprint(
    transformer_config: TransformerConfig, workload_spec: WorkloadSpec, tp_size: int
) -> int:
    float_size = workload_spec.float_size_bytes
    # Peak is likely the attention scores matrix
    return (
        workload_spec.effective_micro_batch_size
        * transformer_config.num_heads
        / tp_size
        * workload_spec.seq_length
        * workload_spec.seq_length
        * float_size
    )


def calculate_ffn_moe_forward_transient_peak_memory_footprint(
    transformer_config: TransformerConfig, workload_spec: WorkloadSpec, attn_tp_size: int
) -> int:
    float_size = workload_spec.float_size_bytes
    # Peak is the output of the first FFN layer
    return (
        workload_spec.effective_micro_batch_size
        * workload_spec.seq_length
        * transformer_config.mlp_hidden_size
        * float_size
    ) / attn_tp_size


def calculate_attn_backward_transient_peak_memory_footprint(
    transformer_config: TransformerConfig,
    workload_spec: "TrainSpec",
    activation_storage_strategy: ActivationStorageOptimizationStrategy,
    tp_size: int,
) -> int:
    float_size = workload_spec.float_size_bytes
    peak = calculate_attn_forward_transient_peak_memory_footprint(
        transformer_config, workload_spec, tp_size
    )  # A simple approximation
    if (
        activation_storage_strategy == "GATHER"
        or activation_storage_strategy == "GATHER_RECOMPUTATION"
    ):
        gathered_activation_size = (
            workload_spec.effective_micro_batch_size
            * workload_spec.seq_length
            * transformer_config.d_model
            * float_size
        )
        peak = max(peak, gathered_activation_size)
    return peak


def calculate_ffn_moe_backward_transient_peak_memory_footprint(
    transformer_config: TransformerConfig,
    workload_spec: WorkloadSpec,
    activation_storage_strategy: ActivationStorageOptimizationStrategy,
    attn_tp_size: int,
) -> int:
    float_size = workload_spec.float_size_bytes
    peak = calculate_ffn_moe_forward_transient_peak_memory_footprint(
        transformer_config, workload_spec, attn_tp_size
    )
    if (
        activation_storage_strategy == "GATHER"
        or activation_storage_strategy == "GATHER_RECOMPUTATION"
    ):
        gathered_activation_size = (
            workload_spec.effective_micro_batch_size
            * workload_spec.seq_length
            * transformer_config.d_model
            * float_size
        )
        peak = max(peak, gathered_activation_size)
    return peak


def _forward_pass_prefill_attention_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    attn_tp_size: int,
    save_kv_cache: bool,
    workload_spec,
):
    """
    Calculates memory traffic for the Attention block in a prefill/training step.

    This function traces memory accesses for a forward pass through a self-attention
    mechanism, supporting both standard Multi-Head Attention (MHA) and Grouped Query
    Attention (GQA). It accounts for QKV projection, the attention mechanism,
    output projection, and optional saving of the KV cache.

        Args:
            traffic: The LayerMemoryTraffic object to update.
        transformer_config: The model's configuration, including GQA parameters.
            n_batch: The batch size.
            n_seq: The sequence length.
            float_size: The size of floating point numbers in bytes.
            tp_shard_factor: The tensor parallelism sharding factor (1 for no sharding).
        save_kv_cache: If True, models writing the K and V tensors to the cache.
    """
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    # Calculate sharded dimensions
    num_q_heads_sharded = num_heads // attn_tp_size
    num_kv_heads_sharded = num_kv_heads // attn_tp_size

    # Infer sizes from workload spec
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights
    # Weights for Q, K, V, and Output projections are read.
    q_weight_size = d_model * d_model
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = d_model * d_model
    traffic.weights.bytes_read += (
        (q_weight_size + kv_weight_size + out_weight_size) * float_size
    ) / attn_tp_size

    # Activations
    # The input to the attention block (from the previous layer or residual stream).
    # This tensor has the full d_model dimension.
    # TODO(MB): This may not be true for 2D TP, check.
    traffic.activations.bytes_read += n_batch * n_seq * d_model * float_size

    # QKV projections
    # The input is multiplied by the W_q, W_k, and W_v weight matrices.
    # The output is three separate tensors (or a combined one) which are sharded.
    q_out_sharded = n_batch * n_seq * d_model * float_size / attn_tp_size
    kv_out_sharded = 2 * n_batch * n_seq * (num_kv_heads * head_dim) * float_size / attn_tp_size
    traffic.activations.bytes_written += q_out_sharded + kv_out_sharded

    if save_kv_cache:
        # For inference prefill, we write K and V tensors to the cache.
        # The size is based on the number of KV heads, sharded by TP.
        traffic.kv_cache.bytes_written += (
            2 * n_batch * n_seq * (num_kv_heads_sharded * head_dim) * float_size
        )

    # Attention calculation
    # Read the sharded Q, K, and V tensors that were just written.
    traffic.activations.bytes_read += q_out_sharded + kv_out_sharded
    # Write the resulting attention scores matrix (Q @ K.T). Its head dimension is sharded.
    attn_scores_sharded = n_batch * num_q_heads_sharded * n_seq * n_seq * float_size
    traffic.activations.bytes_written += attn_scores_sharded
    # Read the sharded attention scores to apply softmax and multiply by V.
    traffic.activations.bytes_read += attn_scores_sharded
    # Write the resulting attention context vector, which is still sharded.
    attn_context_sharded = n_batch * n_seq * (num_q_heads_sharded * head_dim) * float_size
    traffic.activations.bytes_written += attn_context_sharded

    # Output projection
    # Context @ W_o
    # Read the sharded attention context vector to feed into the final linear layer.
    traffic.activations.bytes_read += attn_context_sharded
    # Intermediate output before All-Reduce
    traffic.activations.bytes_written += n_batch * n_seq * d_model * float_size / attn_tp_size
    # Final full-dimension output
    traffic.activations.bytes_written += n_batch * n_seq * d_model * float_size


def _decode_attention_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq_so_far: int,
    attn_tp_size: int,
    workload_spec,
):
    """
    Calculates memory traffic for a single token decode step in the Attention block.

    This function models the memory accesses for a single step of autoregressive
    decoding, which is dominated by KV cache operations. It supports both MHA and GQA.
    The key operations are:
    - Reading the entire K and V history from the KV cache.
    - Appending the new K and V vectors for the current token to the cache.
    - Performing the attention calculation for the single new query token.

    Args:
        traffic: The LayerMemoryTraffic object to update.
        transformer_config: The model's configuration, including GQA parameters.
        n_batch: The batch size.
        n_seq_so_far: The length of the sequence generated so far (size of the cache).
        float_size: The size of floating point numbers in bytes.
        attn_tp_size: The tensor parallelism sharding factor.
    """
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = d_model // num_heads

    # Calculate sharded dimensions
    num_q_heads_sharded = num_heads // attn_tp_size
    num_kv_heads_sharded = num_kv_heads // attn_tp_size

    # Infer sizes from workload spec
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights
    q_weight_size = d_model * d_model
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = d_model * d_model
    traffic.weights.bytes_read += (
        (q_weight_size + kv_weight_size + out_weight_size) * float_size
    ) / attn_tp_size

    # KV cache
    # READ the entire history for all KV heads from the cache.
    traffic.kv_cache.bytes_read += (
        2 * n_batch * n_seq_so_far * (num_kv_heads_sharded * head_dim) * float_size
    )
    # WRITE (append) the single new K and V to the cache.
    traffic.kv_cache.bytes_written += (
        2 * n_batch * 1 * (num_kv_heads_sharded * head_dim) * float_size
    )

    # Activations
    # Activation traffic is for a single new token (n_seq=1).
    # TODO(MB): check that this works with TP 2D
    traffic.activations.bytes_read += n_batch * 1 * d_model * float_size
    q_out_new_sharded = n_batch * 1 * (num_q_heads_sharded * head_dim) * float_size
    kv_out_new_sharded = 2 * n_batch * 1 * (num_kv_heads_sharded * head_dim) * float_size
    traffic.activations.bytes_written += q_out_new_sharded + kv_out_new_sharded
    # Read the new Q, K, V activations for the attention calculation.
    traffic.activations.bytes_read += q_out_new_sharded + kv_out_new_sharded
    # Write the attention scores.
    attn_scores_sharded = n_batch * num_q_heads_sharded * 1 * (n_seq_so_far + 1) * float_size
    traffic.activations.bytes_written += attn_scores_sharded
    # Read the scores back to multiply by V.
    traffic.activations.bytes_read += attn_scores_sharded
    # Write the attention context vector.
    attn_context_sharded = n_batch * 1 * (num_q_heads_sharded * head_dim) * float_size
    traffic.activations.bytes_written += attn_context_sharded
    # Read the context vector for the output projection.
    traffic.activations.bytes_read += attn_context_sharded
    # Write the intermediate output of the projection.
    traffic.activations.bytes_written += n_batch * 1 * d_model * float_size / attn_tp_size
    # Write the final full-dimension output.
    traffic.activations.bytes_written += n_batch * 1 * d_model * float_size


def _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    moe_config,
    moe_tp_size: int,
    workload_spec,
):
    """
    Calculates memory traffic for the MoE FFN block, replicating the original logic.
    This function models the memory traffic on a single chip, which is part of a larger
    TP group. The model accounts for a hierarchical setup where experts can themselves
    be sharded using Expert Parallelism (EP). It handles standard FFNs correctly
    if the moe_config describes a single-expert setup.
    Args:
        traffic: The LayerMemoryTraffic object to update.
    transformer_config:
        n_batch: The batch size.
        n_seq: The sequence length.
        d_model: The hidden dimension of the model.
        mlp_ratio: The ratio to determine the FFN hidden size (d_ff).
        float_size: The size of floating point numbers in bytes.
        moe_config: Configuration object for the Mixture of Experts layer.
        tp_moe_size: The Expert Parallelism size (how many chips one expert is sharded over).
    """
    d_model = transformer_config.d_model
    mlp_ratio = transformer_config.mlp_ratio
    d_ff = d_model * mlp_ratio

    # Infer sizes from workload spec
    batch_size_per_dp = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights read
    # Colocated experts share the same XPU; scale weights read by the number of colocated experts hosted.
    num_colocated = moe_config.num_colocated_experts
    ffn1_weight_size = num_colocated * (d_model * d_ff * float_size) / moe_tp_size
    ffn2_weight_size = num_colocated * (d_ff * d_model * float_size) / moe_tp_size
    traffic.weights.bytes_read += ffn1_weight_size + ffn2_weight_size

    # Activations
    # The input to the MoE block (from the attention block's residual stream) is read.
    # The router on each chip needs to see all tokens to determine their destination expert.
    traffic.activations.bytes_read += batch_size_per_dp * n_seq * d_model * float_size

    # Shared expert logic
    # Shared experts not supported yet
    if (
        getattr(transformer_config, "num_shared_experts", 0) > 0
        or getattr(moe_config, "num_shared_experts_per_dp_rank", 0) > 0
    ):
        raise NotImplementedError(
            "Shared experts are not supported in _calculate_prefill_moe_compute normalization"
        )

    # Regular expert logic
    # This block executes if the chip hosts a "regular expert". A chip can host either
    # a shared or regular expert, not both, but we sum the possibilities as per the original model.
    total_num_routed_experts = transformer_config.num_regular_experts
    if total_num_routed_experts > 0:
        tokens_per_set = workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
        # max volume that an expert could receive, scaled by traffic multiplier
        token_volume_per_routed_expert = (
            tokens_per_routed_expert * traffic_mult * d_model * float_size
        )

        # Calculate the total volume of tokens routed to a single regular expert.
        # The input for this expert's FFN1 is read from the All-to-All output buffer.
        # Scale by number of colocated experts per chip
        traffic.activations.bytes_read += num_colocated * token_volume_per_routed_expert
        # The output of FFN1 is written. It is sharded by EP_size, because the data is per XPU
        ff1_output_sharded_per_regular_expert = int(
            (mlp_ratio * token_volume_per_routed_expert) / moe_tp_size
        )
        traffic.activations.bytes_written += num_colocated * ff1_output_sharded_per_regular_expert
        # The sharded intermediate activation is read back for the FFN2 computation.
        traffic.activations.bytes_read += num_colocated * (
            ff1_output_sharded_per_regular_expert / moe_tp_size
        )
        # The output of this expert is written. This represents the data before the final
        # All-to-All combines it with output from other experts. Its volume is equivalent
        # to the input token volume for this expert.
        traffic.activations.bytes_written += num_colocated * token_volume_per_routed_expert


def _forward_pass_2d_attention_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    pass_mode: PassMode,
    workload_spec,
):
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights: read once per forward pass, sharded in 2D
    qkv_weight_size = (3 * d_model * d_model * float_size) // (attn_tp_size_x * attn_tp_size_yz)
    attn_out_weight_size = (d_model * d_model * float_size) // (attn_tp_size_x * attn_tp_size_yz)
    traffic.weights.bytes_read += qkv_weight_size + attn_out_weight_size

    head_dim = d_model // num_heads
    heads_per_x = num_heads // attn_tp_size_x if attn_tp_size_x else num_heads
    kv_heads_per_x = num_kv_heads // attn_tp_size_x if attn_tp_size_x else num_kv_heads

    step_seq = 1 if pass_mode == PassMode.INFERENCE_DECODE else n_seq

    # Activations: sharded on X for input channel, YZ for projection outputs
    traffic.activations.bytes_read += n_batch * step_seq * d_model / attn_tp_size_x * float_size
    qkv_output_sharded_size = (
        3 * n_batch * step_seq * d_model / (attn_tp_size_x * attn_tp_size_yz) * float_size
    )
    traffic.activations.bytes_written += qkv_output_sharded_size

    # Attention math
    traffic.activations.bytes_read += qkv_output_sharded_size
    attn_score_sharded_size = (
        n_batch * heads_per_x * step_seq * n_seq * float_size
        if pass_mode == PassMode.INFERENCE_DECODE
        else n_batch * heads_per_x * n_seq * n_seq * float_size
    )
    traffic.activations.bytes_written += attn_score_sharded_size
    traffic.activations.bytes_read += attn_score_sharded_size
    traffic.activations.bytes_written += (
        n_batch * step_seq * d_model / (attn_tp_size_x * attn_tp_size_yz) * float_size
    )

    # KV cache
    if pass_mode == PassMode.INFERENCE_PREFILL:
        traffic.kv_cache.bytes_written += (
            2 * n_batch * n_seq * (kv_heads_per_x * head_dim) * float_size
        )
    elif pass_mode == PassMode.INFERENCE_DECODE:
        traffic.kv_cache.bytes_read += (
            2 * n_batch * n_seq * (kv_heads_per_x * head_dim) * float_size
        )
        traffic.kv_cache.bytes_written += 2 * n_batch * 1 * (kv_heads_per_x * head_dim) * float_size


def _forward_pass_2d_moe_ffn_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    moe_tp_size_x: int,
    moe_tp_size_yz: int,
    pass_mode: PassMode,
    workload_spec,
    moe_config,
):
    d_model = transformer_config.d_model
    mlp_ratio = transformer_config.mlp_ratio
    float_size = workload_spec.float_size_bytes

    # Weights (FFN only part; attention weights handled in attention helper)
    ffn1_weight_size = (d_model * mlp_ratio * d_model * float_size) // (
        moe_tp_size_x * moe_tp_size_yz
    )
    ffn2_weight_size = (mlp_ratio * d_model * d_model * float_size) // (
        moe_tp_size_x * moe_tp_size_yz
    )
    traffic.weights.bytes_read += ffn1_weight_size + ffn2_weight_size

    step_seq = 1 if pass_mode == PassMode.INFERENCE_DECODE else n_seq

    # Mirror 1D MoE token-traffic logic for routed experts, adapted to 2D sharding
    num_colocated = moe_config.num_colocated_experts
    total_num_routed_experts = transformer_config.num_regular_experts

    if total_num_routed_experts > 0:
        tokens_per_set = (
            workload_spec.get_resolved_batches().batch_size_per_full_expert_set * step_seq
        )
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
        token_volume_per_routed_expert = (
            tokens_per_routed_expert * traffic_mult * d_model * float_size
        )

        # Read token volume per routed expert (scaled by colocated experts on this chip)
        traffic.activations.bytes_read += num_colocated * token_volume_per_routed_expert

        # FFN1 output is sharded across the YZ dimension of 2D TP
        ff1_output_sharded_per_regular_expert = (mlp_ratio * token_volume_per_routed_expert) / max(
            moe_tp_size_yz, 1
        )
        traffic.activations.bytes_written += num_colocated * int(
            ff1_output_sharded_per_regular_expert
        )

        # Read back the sharded intermediate for FFN2 compute (remain sharded on YZ)
        traffic.activations.bytes_read += num_colocated * (
            ff1_output_sharded_per_regular_expert / max(moe_tp_size_yz, 1)
        )

        # Final FFN2 output before regrouping (equivalent to input token volume)
        traffic.activations.bytes_written += num_colocated * token_volume_per_routed_expert


def forward_memory_traffic_per_layer(
    attn_tp_topology,
    moe_tp_topology,
    transformer_config,
    moe_config,
    n_seq,
    tp_variant,
    tp_size_x,
    tp_size_yz,
    pass_mode: PassMode,
    name: str,
    device_for: dict[str, str],
    node,
    workload_spec,
):
    """
    Calculate memory traffic for the forward pass of a single layer
    using a match/case statement for control flow.
    """
    attn_tp_size = attn_tp_topology.num_chips if attn_tp_topology else 1
    moe_tp_size = moe_tp_topology.num_chips if moe_tp_topology else 1
    memory_traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)

    match tp_variant:
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            # Unified logic for both NONE and 1D parallelism.
            # For NONE, tp_size=1, making sharding calculations no-ops.

            match pass_mode:

                case PassMode.TRAINING:
                    _forward_pass_prefill_attention_memory_traffic(
                        memory_traffic,
                        transformer_config,
                        n_seq,
                        attn_tp_size=attn_tp_size,
                        save_kv_cache=False,
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_PREFILL:
                    _forward_pass_prefill_attention_memory_traffic(
                        memory_traffic,
                        transformer_config,
                        n_seq,
                        attn_tp_size=attn_tp_size,
                        save_kv_cache=True,
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_DECODE:
                    _decode_attention_memory_traffic(
                        traffic=memory_traffic,
                        transformer_config=transformer_config,
                        n_seq_so_far=n_seq,
                        attn_tp_size=attn_tp_size,
                        workload_spec=workload_spec,
                    )

            # n_seq should be 1 for the inference decode phase
            if pass_mode == PassMode.INFERENCE_DECODE:
                n_seq = 1  # For prefill, we only process one token at a time.

            # moe_config determines if this is a standard FFN or MoE.
            _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic(
                memory_traffic,
                transformer_config,
                n_seq,
                moe_config,
                moe_tp_size=moe_tp_size,
                workload_spec=workload_spec,
            )

        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            _forward_pass_2d_attention_memory_traffic(
                memory_traffic,
                transformer_config,
                n_seq,
                attn_tp_size_x=tp_size_x,
                attn_tp_size_yz=tp_size_yz,
                pass_mode=pass_mode,
                workload_spec=workload_spec,
            )
            _forward_pass_2d_moe_ffn_memory_traffic(
                memory_traffic,
                transformer_config,
                n_seq,
                moe_tp_size_x=tp_size_x,
                moe_tp_size_yz=tp_size_yz,
                pass_mode=pass_mode,
                workload_spec=workload_spec,
                moe_config=moe_config,
            )

        case _:
            # Default case for any other variants not explicitly handled.
            raise NotImplementedError(f"Logic for {tp_variant} is not implemented.")

    return memory_traffic


def _backward_attn_tp_1d_memory_traffic(
    name: str,
    transformer_config: TransformerConfig,
    workload_spec,
    tp_size: int,
    activation_checkpointing: bool,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """
    Compute backward-pass memory traffic for the attention sub-block under NONE/1D TP.

    Notes
    - The computation preserves byte-for-byte parity with the previous unified (non-split)
      implementation. It only factors the attention portion out for clarity.
    - Sizes are derived from `transformer_config` and `workload_spec` to avoid duplicating
      arguments upstream.

    Args:
        name: A descriptive name for the traffic accumulator instance.
        transformer_config: Model transformer configuration (provides `d_model`, `num_heads`).
        workload_spec: Workload spec used to infer `batch_size_per_dp`, `seq_length`, `float_size_bytes`.
        tp_size: Size of the tensor-parallel attention domain (1 for NONE, >1 for 1D shard).
        activation_checkpointing: Whether activation checkpointing is enabled.
        device_for: Mapping of memory components to device tiers.
        node: Hardware node definition (provides memory tier details).

    Returns:
        LayerMemoryTraffic: Accumulated attention backward memory traffic for a single layer.
    """
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    d_model_per_tp = d_model // tp_size if tp_size > 1 else d_model
    heads_per_tp = num_heads // tp_size if tp_size > 1 else num_heads
    # Resolve batch/sequence and dtype size from workload config
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    # Read attention weights (QKV, Out) and write corresponding gradients (sharded by TP)
    qkv_weight_size = (3 * d_model * d_model * float_size) / tp_size
    attn_out_weight_size = (d_model * d_model * float_size) / tp_size
    total_w = qkv_weight_size + attn_out_weight_size
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w
    if not activation_checkpointing:
        traffic.activations.bytes_read += 3 * n_batch * n_seq * d_model_per_tp * float_size
        traffic.activations.bytes_read += n_batch * heads_per_tp * n_seq * n_seq * float_size
        traffic.activations.bytes_read += n_batch * n_seq * d_model_per_tp * float_size
    traffic.gradients.bytes_written += n_batch * n_seq * d_model_per_tp * float_size
    return traffic


def _backward_attn_tp_2d_memory_traffic(
    *,
    name: str,
    transformer_config: TransformerConfig,
    workload_spec,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    activation_checkpointing: bool,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes

    # Weights (QKV + Out), sharded in 2D
    qkv_w = (3 * d_model * d_model * float_size) // (attn_tp_size_x * attn_tp_size_yz)
    out_w = (d_model * d_model * float_size) // (attn_tp_size_x * attn_tp_size_yz)
    total_w = qkv_w + out_w
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w

    if not activation_checkpointing:
        heads_per_x = num_heads // attn_tp_size_x if attn_tp_size_x else num_heads
        # Read QKV proj output, attention scores, and context/output intermediates
        qkv_proj_output = (
            3 * n_batch * n_seq * d_model / (attn_tp_size_x * attn_tp_size_yz) * float_size
        )
        attn_scores = n_batch * heads_per_x * n_seq * n_seq * float_size
        final_proj_in = n_batch * n_seq * d_model / (attn_tp_size_x * attn_tp_size_yz) * float_size
        traffic.activations.bytes_read += int(qkv_proj_output + attn_scores + final_proj_in)

    # Gradient write for final output (d_model sharded across X*YZ)
    traffic.gradients.bytes_written += (
        n_batch * n_seq * d_model / (attn_tp_size_x * attn_tp_size_yz) * float_size
    )
    return traffic


def _backward_moe_tp_1d_memory_traffic(
    name: str,
    transformer_config: TransformerConfig,
    moe_config,
    moe_tp_topology,
    workload_spec,
    activation_checkpointing: bool,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """
    Compute backward-pass memory traffic for the MoE/FFN sub-block under NONE/1D TP.

    Notes
    - The computation preserves byte-for-byte parity with the previous unified (non-split)
      implementation. It only factors the MoE/FFN portion out for clarity.
    - Uses EP topology (`tp_moe_topology`) to determine how intermediate MoE activations are sharded.
    - Sizes are derived from `transformer_config` and `workload_spec` to reduce parameter clutter.

    Args:
        name: A descriptive name for the traffic accumulator instance.
        transformer_config: Model transformer configuration (provides `d_model`, `mlp_ratio`).
        moe_config: MoE placement/configuration object (regular/shared experts, traffic multiplier).
        moe_tp_topology: EP topology for MoE partitioning (affects sharding of MoE intermediates).
        workload_spec: Workload spec used to infer `batch_size_per_dp`, `seq_length`, `float_size_bytes`.
        activation_checkpointing: Whether activation checkpointing is enabled.
        device_for: Mapping of memory components to device tiers.
        node: Hardware node definition (provides memory tier details).

    Returns:
        LayerMemoryTraffic: Accumulated MoE/FFN backward memory traffic for a single layer.
    """
    # Shared experts not supported yet
    if (
        getattr(transformer_config, "num_shared_experts", 0) > 0
        or getattr(moe_config, "num_shared_experts_per_dp_rank", 0) > 0
    ):
        raise NotImplementedError(
            "Shared experts are not supported in _calculate_prefill_moe_compute normalization"
        )

    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)
    d_model = transformer_config.d_model
    mlp_ratio = transformer_config.mlp_ratio
    d_ff = d_model * mlp_ratio
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    total_num_routed_experts = transformer_config.num_regular_experts
    tokens_per_set = workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    traffic_mult = moe_config.regular_expert_traffic_multiplier

    moe_tp_size = moe_tp_topology.num_chips if moe_tp_topology else 1

    num_colocated = moe_config.num_colocated_experts
    ffn1_weight_size = num_colocated * (d_model * d_ff * float_size) / moe_tp_size
    ffn2_weight_size = num_colocated * (d_ff * d_model * float_size) / moe_tp_size

    total_w = ffn1_weight_size + ffn2_weight_size
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w
    if not activation_checkpointing:
        # we have to save activations during the forward pass to compute gradients
        if total_num_routed_experts > 0:
            # Read original layer input for the first FFN gradient
            regular_input_size = traffic_mult * n_batch * n_seq * d_model * float_size
            traffic.activations.bytes_read += regular_input_size
            # Read attention block output (serves as FFN input)
            traffic.activations.bytes_read += n_batch * n_seq * d_model * float_size
            # Regular-expert intermediates (token volume per expert; FFN1 output sharded by EP)

            tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
            # max volume that an expert could receive, scaled by traffic multiplier
            token_vol_per_expert = tokens_per_routed_expert * traffic_mult * d_model * float_size

            ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / moe_tp_size
            traffic.activations.bytes_read += num_colocated * ff1_out_sharded_per_expert
            traffic.activations.bytes_read += num_colocated * token_vol_per_expert

    if total_num_routed_experts > 0:
        tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
        # max volume that an expert could receive, scaled by traffic multiplier
        token_vol_per_expert = tokens_per_routed_expert * traffic_mult * d_model * float_size
        ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / moe_tp_size
        traffic.gradients.bytes_written += num_colocated * token_vol_per_expert
        traffic.gradients.bytes_written += num_colocated * ff1_out_sharded_per_expert
    return traffic


def _backward_moe_tp_2d_memory_traffic(
    *,
    name: str,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
    moe_tp_size_x: int,
    moe_tp_size_yz: int,
    activation_checkpointing: bool,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)
    d_model = transformer_config.d_model
    mlp_ratio = transformer_config.mlp_ratio
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes

    # Shared experts not supported yet in this path
    if getattr(moe_config, "num_shared_experts_per_dp_rank", 0) > 0:
        raise NotImplementedError("Shared experts are not supported in 2D TP MoE backward traffic")

    # Weights sharded across 2D TP (FFN only here)
    ffn1_w = (d_model * mlp_ratio * d_model * float_size) // (moe_tp_size_x * moe_tp_size_yz)
    ffn2_w = (mlp_ratio * d_model * d_model * float_size) // (moe_tp_size_x * moe_tp_size_yz)
    total_w = ffn1_w + ffn2_w
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w

    # Mirror 1D token-volume logic for routed experts, adapted to 2D sharding
    num_colocated = moe_config.num_colocated_experts
    total_num_routed_experts = transformer_config.num_regular_experts

    tokens_per_set = workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    traffic_mult = moe_config.regular_expert_traffic_multiplier

    if not activation_checkpointing:
        if total_num_routed_experts > 0:
            # Read original layer input for first FFN gradient; take max across shared/regular paths
            regular_input_size = traffic_mult * n_batch * n_seq * d_model * float_size
            traffic.activations.bytes_read += regular_input_size
            # Read attention block output (serves as FFN input)
            traffic.activations.bytes_read += n_batch * n_seq * d_model * float_size

            tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
            token_vol_per_expert = tokens_per_routed_expert * traffic_mult * d_model * float_size
            # Read FFN1 output sharded on YZ and the expert input token volume
            ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / max(moe_tp_size_yz, 1)
            traffic.activations.bytes_read += num_colocated * ff1_out_sharded_per_expert
            traffic.activations.bytes_read += num_colocated * token_vol_per_expert

    # Gradient writes for routed experts
    if total_num_routed_experts > 0:
        tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
        token_vol_per_expert = tokens_per_routed_expert * traffic_mult * d_model * float_size
        ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / max(moe_tp_size_yz, 1)
        traffic.gradients.bytes_written += num_colocated * token_vol_per_expert
        traffic.gradients.bytes_written += num_colocated * ff1_out_sharded_per_expert
    return traffic


def backward_memory_traffic_per_layer(
    attn_tp_topology,
    moe_tp_topology,
    transformer_config,
    moe_config,
    tp_variant,
    tp_size,
    tp_size_x,
    tp_size_yz,
    activation_checkpointing,
    name: str,
    device_for: dict[str, str],
    node,
    workload_spec,
):
    """
    Calculates memory traffic for the backward pass of a single layer.
    """
    memory_traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)

    # Handle activation checkpointing
    # This logic is applied first, as it changes the nature of subsequent reads.
    if activation_checkpointing:
        # If checkpointing, we recompute the forward pass. This adds all the reads
        # and writes of a forward pass to the backward pass's memory traffic.
        # Use workload_spec to derive sizes for recompute path
        n_seq = workload_spec.seq_length
        fwd_traffic = forward_memory_traffic_per_layer(
            attn_tp_topology,
            moe_tp_topology,
            transformer_config,
            moe_config,
            n_seq,
            tp_variant,
            tp_size_x,
            tp_size_yz,
            PassMode.TRAINING,
            name="fwd_recompute",
            device_for=device_for,
            node=node,
            workload_spec=workload_spec,
        )
        # Note: For recomputation, we add forward reads and writes to the total traffic.
        memory_traffic.activations.bytes_read += (
            fwd_traffic.activations.bytes_read + fwd_traffic.activations.bytes_written
        )
        memory_traffic.weights.bytes_read += fwd_traffic.weights.bytes_read

    # Backward pass traffic

    match tp_variant:
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            # Use split helpers matching the unified implementation
            attn_traffic = _backward_attn_tp_1d_memory_traffic(
                name=f"{name}_attn_bwd",
                transformer_config=transformer_config,
                workload_spec=workload_spec,
                tp_size=tp_size,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for,
                node=node,
            )

            moe_traffic = _backward_moe_tp_1d_memory_traffic(
                name=f"{name}_moe_bwd",
                transformer_config=transformer_config,
                moe_config=moe_config,
                moe_tp_topology=moe_tp_topology,
                workload_spec=workload_spec,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for,
                node=node,
            )

            memory_traffic = memory_traffic + attn_traffic + moe_traffic

        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            # Use split helpers for 2D backward to match forward structure
            attn_bwd = _backward_attn_tp_2d_memory_traffic(
                name=f"{name}_attn_bwd_2d",
                transformer_config=transformer_config,
                workload_spec=workload_spec,
                attn_tp_size_x=tp_size_x,
                attn_tp_size_yz=tp_size_yz,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for,
                node=node,
            )

            moe_bwd = _backward_moe_tp_2d_memory_traffic(
                name=f"{name}_moe_bwd_2d",
                transformer_config=transformer_config,
                moe_config=moe_config,
                workload_spec=workload_spec,
                moe_tp_size_x=tp_size_x,
                moe_tp_size_yz=tp_size_yz,
                activation_checkpointing=activation_checkpointing,
                device_for=device_for,
                node=node,
            )

            memory_traffic = memory_traffic + attn_bwd + moe_bwd

        case _:
            raise NotImplementedError(f"Logic for {tp_variant} is not implemented.")

    return memory_traffic


def get_transformer_mem_stats_per_layer(
    attn_tp_topology,
    moe_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    n_batch,
    n_seq,
    tp_variant,
    x_dims: list[int],
    yz_dims: list[int],
    pass_mode,
    float_size,
    activation_checkpointing=False,
    device_for: dict[str, str] | None = None,
    workload_spec=None,
    attn_dp_size: int | None = None,
    moe_dp_size: int | None = None,
):
    """
    Args:
        pass_mode: PassMode enum specifying the mode of operation (TRAINING, INFERENCE_PREFILL, INFERENCE_DECODE)
        moe_tp_topology:
        moe_config: Mixture of Experts configuration object
        attn_tp_topology: NetworkTopology object
        n_batch: Number of sequences in batch
        n_seq: Length of each sequence
        tp_variant: TensorParallelVariant enum specifying type of tensor parallelism
        x_dims: topology dimensions for 'x' direction
        yz_dims: topology dimensions for 'yz' direction
        activation_checkpointing: Whether to use activation checkpointing
        float_size: Size of float type in bytes
        memory_bandwidth: Memory bandwidth in bytes/second
    """

    # Extract parameters from transformer config
    node = attn_tp_topology.physical_topology.node_definition

    if tp_variant == TensorParallelVariant.SHARD_ACTIVATIONS:
        raise NotImplementedError("Activation stationary tensor parallelism not implemented")

    if tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        # Compute submesh sizes directly from the topology shape and the provided dims
        shape = getattr(attn_tp_topology, "shape", [])

        def _prod(indices):
            p = 1
            for idx in indices:
                p *= shape[idx]
            return p

        tp_size_x = _prod(x_dims) if x_dims else 1
        tp_size_yz = _prod(yz_dims) if yz_dims else 1
        tp_size = tp_size_x * tp_size_yz
        assert (
            tp_size == attn_tp_topology.num_chips
        ), f"TP size mismatch: {tp_size} != {attn_tp_topology.num_chips}"
    elif tp_variant == TensorParallelVariant.SHARD_WEIGHTS_1D:
        tp_size = attn_tp_topology.num_chips
        tp_size_x = None
        tp_size_yz = None
    else:
        raise ValueError(f"Unsupported tensor parallel variant: {tp_variant}")

    if device_for is None:
        raise ValueError(
            "device_for mapping must be provided to get_transformer_mem_stats_per_layer"
        )
    fwd_traffic_per_layer = forward_memory_traffic_per_layer(
        attn_tp_topology,
        moe_tp_topology,
        transformer_config,
        moe_config,
        n_seq,
        tp_variant,
        tp_size_x,
        tp_size_yz,
        pass_mode=pass_mode,
        name="fwd_per_layer",
        device_for=device_for,
        node=node,
        workload_spec=workload_spec,
    )

    if pass_mode == PassMode.TRAINING:
        bwd_traffic_per_layer = backward_memory_traffic_per_layer(
            attn_tp_topology,
            moe_tp_topology,
            transformer_config,
            moe_config,
            tp_variant,
            tp_size,
            tp_size_x,
            tp_size_yz,
            activation_checkpointing,
            name="bwd_per_layer",
            device_for=device_for,
            node=node,
            workload_spec=workload_spec,
        )
        # Add parameter update memory traffic (weights update reads/writes, grads read)
        param_update_traffic = _parameter_update_memory_traffic(
            name="param_update",
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            attn_tp_topology=attn_tp_topology,
            moe_tp_size=moe_tp_topology.num_chips if moe_tp_topology else 1,
            attn_dp_size=attn_dp_size,
            moe_dp_size=moe_dp_size,
            device_for=device_for,
            node=node,
        )
        bwd_traffic_per_layer = bwd_traffic_per_layer + param_update_traffic
    else:
        # For inference, we assume no backward pass traffic
        bwd_traffic_per_layer = LayerMemoryTraffic(
            name="bwd_per_layer", device_for=device_for, node=node
        )

    return fwd_traffic_per_layer, bwd_traffic_per_layer


def calculate_embedding_memory_traffic_1d_tp_fwd(
    name: str,
    model_spec: "TransformerConfig",
    workload_spec: "WorkloadSpec",
    attn_tp_topology,
    activation_checkpointing: bool,
    device_for: Dict[str, str],
    pass_mode: "PassMode",
    seq_len: int | None = None,
):
    """Calculates the forward memory traffic for the embedding layer."""
    node = attn_tp_topology.physical_topology.node_definition
    traffic = LayerMemoryTraffic(name, device_for, node)
    batch_size = workload_spec.get_resolved_batches().batch_size_per_dp
    seq_len = workload_spec.seq_length if seq_len is None else seq_len
    float_size = workload_spec.float_size_bytes

    # Forward pass (vocab-parallel embeddings):
    # Megatron-style vocab-parallelism shards the embedding table across TP ranks by
    # vocabulary columns. Each TP shard only services ~1/TP of the tokens, so
    # the per-chip memory traffic is reduced by tp_size. We model a dense
    # equivalent volume here (lookup is memory-bound), then divide by tp_size.
    tp_size = attn_tp_topology.num_chips if attn_tp_topology else 1
    embedding_read_bytes = (batch_size * seq_len * model_spec.d_model * float_size) / tp_size
    traffic.weights.bytes_read += int(embedding_read_bytes)
    return traffic


def calculate_embedding_memory_traffic_1d_tp_bwd(
    name: str,
    model_spec: "TransformerConfig",
    workload_spec: "WorkloadSpec",
    attn_tp_topology,
    activation_checkpointing: bool,
    device_for: Dict[str, str],
    pass_mode: "PassMode",
    seq_len: int | None = None,
):
    """Calculates the backward memory traffic for the embedding layer."""
    node = attn_tp_topology.physical_topology.node_definition
    traffic = LayerMemoryTraffic(name, device_for, node)
    batch_size = workload_spec.get_resolved_batches().batch_size_per_dp
    seq_len = workload_spec.seq_length if seq_len is None else seq_len
    float_size = workload_spec.float_size_bytes

    # Backward pass (vocab-parallel embeddings):
    # Gradient updates to the embedding table are sharded across TP ranks by vocab,
    # matching forward sharding. We approximate worst-case dense updates with the
    # same volume as forward and divide by tp_size to obtain per-chip bytes.
    tp_size = attn_tp_topology.num_chips if attn_tp_topology else 1
    embedding_grad_write_bytes = (batch_size * seq_len * model_spec.d_model * float_size) / tp_size
    traffic.gradients.bytes_written += int(embedding_grad_write_bytes)
    return traffic


def calculate_unembedding_memory_traffic_1d_tp_fwd(
    name: str,
    model_spec: "TransformerConfig",
    workload_spec: "WorkloadSpec",
    attn_tp_topology,
    activation_checkpointing: bool,
    device_for: Dict[str, str],
    pass_mode: "PassMode",
    seq_len: int | None = None,
):
    """
    Calculates the forward memory traffic for the unembedding (output) layer.

    Assumptions:
    - We use vocabulary-parallel ("parallel unembed"): the output matrix is column-sharded
      across TP ranks by vocabulary. Each rank only handles its vocab shard, so weights
      reads scale with (vocab_size / tp_size), and we do not all-gather full logits.
    - This aligns with Megatron-style vocab-parallel loss/sampling, avoiding full-logit
      materialization; downstream operations (e.g., cross-entropy, distributed top-k)
      use small collectives instead of all-gather.
    """
    node = attn_tp_topology.physical_topology.node_definition
    traffic = LayerMemoryTraffic(name, device_for, node)
    batch_size = workload_spec.get_resolved_batches().batch_size_per_dp
    seq_len = workload_spec.seq_length if seq_len is None else seq_len
    float_size = workload_spec.float_size_bytes

    # Forward pass: Read hidden states and unembedding weight matrix
    hidden_states_read = batch_size * seq_len * model_spec.d_model * float_size
    traffic.activations.bytes_read += hidden_states_read

    tp_size = attn_tp_topology.num_chips if attn_tp_topology else 1
    unembedding_weights_read = (model_spec.vocab_size / tp_size) * model_spec.d_model * float_size
    traffic.weights.bytes_read += unembedding_weights_read
    return traffic


def calculate_unembedding_memory_traffic_1d_tp_bwd(
    name: str,
    model_spec: "TransformerConfig",
    workload_spec: "WorkloadSpec",
    attn_tp_topology,
    activation_checkpointing: bool,
    device_for: Dict[str, str],
    pass_mode: "PassMode",
    seq_len: int | None = None,
):
    """
    Calculates the backward memory traffic for the unembedding (output) layer.

    Assumptions:
    - Vocabulary-parallel ("parallel unembed"): gradients for output weights are local to the
      owning vocab shard on each TP rank; hidden-state gradients are computed locally and
      combined via typical TP collectives (e.g., reduce-scatter or all-reduce) outside this
      per-layer memory model. We do not model a full-logit all-gather here.
    """
    node = attn_tp_topology.physical_topology.node_definition
    traffic = LayerMemoryTraffic(name, device_for, node)
    batch_size = workload_spec.get_resolved_batches().batch_size_per_dp
    seq_len = workload_spec.seq_length if seq_len is None else seq_len
    float_size = workload_spec.float_size_bytes

    # Backward pass:
    # Read hidden states and weights (again), and write gradients.
    hidden_states_read = batch_size * seq_len * model_spec.d_model * float_size
    traffic.activations.bytes_read += hidden_states_read

    tp_size = attn_tp_topology.num_chips if attn_tp_topology else 1
    unembedding_weights_read = (model_spec.vocab_size / tp_size) * model_spec.d_model * float_size
    traffic.weights.bytes_read += unembedding_weights_read

    traffic.gradients.bytes_written += unembedding_weights_read  # Weight gradients
    traffic.gradients.bytes_written += hidden_states_read  # Activation gradients
    return traffic


def _parameter_update_memory_traffic(
    *,
    name: str,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec: WorkloadSpec,
    attn_tp_topology,
    moe_tp_size: int,
    attn_dp_size: int | None,
    moe_dp_size: int | None,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """
    Orchestrator that combines attention and MoE parameter update memory traffic for one layer.
    """
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)

    attn_traffic = _parameter_update_attention_memory_traffic(
        name=f"{name}_attn",
        transformer_config=transformer_config,
        moe_config=moe_config,
        workload_spec=workload_spec,
        attn_tp_topology=attn_tp_topology,
        attn_dp_size=attn_dp_size,
        device_for=device_for,
        node=node,
    )

    moe_traffic = _parameter_update_moe_memory_traffic(
        name=f"{name}_moe",
        transformer_config=transformer_config,
        moe_config=moe_config,
        workload_spec=workload_spec,
        moe_tp_size=moe_tp_size,
        moe_dp_size=moe_dp_size,
        device_for=device_for,
        node=node,
    )

    return traffic + attn_traffic + moe_traffic


def _parameter_update_attention_memory_traffic(
    *,
    name: str,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec: WorkloadSpec,
    attn_tp_topology,
    attn_dp_size: int,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """Compute per-layer attention (non-expert + layernorm) parameter update traffic per chip."""
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)

    # Use helper to compute per-XPU attention parameters (weights sharded by TP, LN replicated)
    params_per_xpu = attn_params_per_xpu(transformer_config, attn_tp_topology)

    # DP ZeRO stage III shards optimizer states and params across DP ranks
    if workload_spec.dp_spec.variant == DataParallelVariant.ZERO_STAGE_3:
        if attn_dp_size is not None and attn_dp_size > 0:
            params_per_xpu /= attn_dp_size

    # Use full precision for weights (TrainSpec), compute precision for gradients
    weights_fs = int(workload_spec.full_precision_param_float_size)
    grads_fs = int(workload_spec.float_size_bytes)
    weights_bytes = params_per_xpu * weights_fs
    grads_bytes = params_per_xpu * grads_fs

    traffic.weights.bytes_read += int(weights_bytes)
    traffic.gradients.bytes_read += int(grads_bytes)
    traffic.weights.bytes_written += int(weights_bytes)

    # Optimizer state traffic (e.g., Adam: 2 states m, v) in full precision
    if workload_spec.optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer for parameter update traffic: {workload_spec.optimizer}. Only 'adam' is supported."
        )
    opt_states_fs = int(workload_spec.full_precision_param_float_size)
    opt_states_read = 2 * params_per_xpu * opt_states_fs
    opt_states_write = 2 * params_per_xpu * opt_states_fs
    traffic.weights.bytes_read += int(opt_states_read)
    traffic.weights.bytes_written += int(opt_states_write)
    return traffic


def _parameter_update_moe_memory_traffic(
    *,
    name: str,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec: WorkloadSpec,
    moe_tp_size: int,
    moe_dp_size: int,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """
    Compute per-layer MoE parameter update traffic per chip using expert set placement:
    - Each chip hosts num_colocated_experts distinct experts (per DP rank)
    - Each expert's parameters are sharded across moe_tp_size
    - ZeRO stage III further shards across moe_dp_size
    """
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)

    # Shared experts not modeled yet in this traffic path
    if getattr(moe_config, "num_shared_experts_per_dp_rank", 0) > 0:
        raise NotImplementedError("Shared experts parameter update traffic is not implemented")

    # Per-XPU expert parameters using placement (accounts for expert TP and colocation)
    params_per_xpu = expert_params_per_xpu(transformer_config, moe_config)

    # ZeRO Stage III shards parameters across DP ranks (MoE DP domain)
    if workload_spec.dp_spec.variant == DataParallelVariant.ZERO_STAGE_3:
        if moe_dp_size is not None and moe_dp_size > 0:
            params_per_xpu /= moe_dp_size

    # Use full precision for weights (TrainSpec), compute precision for gradients
    weights_fs = int(workload_spec.full_precision_param_float_size)
    grads_fs = int(workload_spec.float_size_bytes)
    weights_bytes = params_per_xpu * weights_fs
    grads_bytes = params_per_xpu * grads_fs

    traffic.weights.bytes_read += int(weights_bytes)
    traffic.gradients.bytes_read += int(grads_bytes)
    traffic.weights.bytes_written += int(weights_bytes)

    # Optimizer state traffic (e.g., Adam: 2 states m, v) in full precision
    if workload_spec.optimizer.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer for parameter update traffic: {workload_spec.optimizer}. Only 'adam' is supported."
        )
    opt_states_fs = int(workload_spec.full_precision_param_float_size)
    opt_states_read = 2 * params_per_xpu * opt_states_fs
    opt_states_write = 2 * params_per_xpu * opt_states_fs
    traffic.weights.bytes_read += int(opt_states_read)
    traffic.weights.bytes_written += int(opt_states_write)
    return traffic
