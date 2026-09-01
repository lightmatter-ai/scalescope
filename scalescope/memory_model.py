"""Analytical memory model for traffic and footprint.

Computes per-layer/per-batch memory traffic (bytes/time) and persistent
footprint per XPU for attention and MoE/FFN, including optimizer/saved
activations/gradient buffers and (in inference) KV cache. Also verifies tier
capacities and computes detailed per-component stats.
"""

from dataclasses import dataclass
from typing import Any, Dict

from scalescope.model_spec import (
    attention_layer_classes,
    attn_params_per_xpu,
    expert_params_per_xpu,
    resolve_kv_heads_per_partition,
    TransformerConfig,
)
from scalescope.utils import PassMode
from scalescope.workload_spec import (
    ActivationStorageOptimizationStrategy,
    DataParallelVariant,
    ExpertWeightResidencyPolicy,
    InferenceDecodeSpec,
    MoERouterSyncMode,
    TensorParallelVariant,
    TP2DProjAxisRole,
    TrainSpec,
    WorkloadSpec,
)


def _expert_weight_residency_fraction(
    *,
    workload_spec: WorkloadSpec,
    transformer_config: TransformerConfig,
    tokens_per_set: float,
) -> float:
    """Return the multiplier applied to the resident-weight memory-traffic term.

    Under ``RESIDENT_FULL_SET`` returns ``1.0`` (every colocated expert is
    billed every step — today's behavior).

    Under ``RESIDENT_ACTIVE_SUBSET`` returns
    ``active_expert_fraction = E_active / num_colocated_experts``, derived from
    ``(tokens_per_set, top_k, num_regular_experts, num_colocated_experts,
    expert_imbalance)`` via ``ep_strategy.compute_expert_activation_cardinality``.
    Scope is **expert weights only**; routed-token terms keep their current scaling.
    """
    # Local import to avoid a circular dependency between memory_model and ep_strategy.
    from scalescope.ep_strategy import compute_expert_activation_cardinality

    policy = workload_spec.moe_spec_v2.expert_weight_residency_policy
    match policy:
        case ExpertWeightResidencyPolicy.RESIDENT_FULL_SET:
            return 1.0
        case ExpertWeightResidencyPolicy.RESIDENT_ACTIVE_SUBSET:
            # Attention-only models (no FFN: num_regular_experts == 0) or models with
            # no routing (num_selected_experts == 0) have no "active subset" notion.
            # The policy is moot and must not error — return 1.0 (no scaling), matching
            # the guard in `run_scalescope._expert_activation_diagnostics`. A dense
            # model (num_regular_experts == 1, num_selected_experts == 1) is NOT caught
            # here: it flows through compute_expert_activation_cardinality, which yields
            # active_expert_fraction == 1.0 (its single FFN is always fully resident).
            if (
                transformer_config.num_regular_experts <= 0
                or transformer_config.num_selected_experts <= 0
            ):
                return 1.0
            cardinality = compute_expert_activation_cardinality(
                tokens_per_set=tokens_per_set,
                num_selected_experts=transformer_config.num_selected_experts,
                num_regular_experts=transformer_config.num_regular_experts,
                num_colocated_experts=workload_spec.moe_spec_v2.num_colocated_experts,
                expert_imbalance=workload_spec.moe_spec_v2.expert_imbalance,
                routing_distribution=workload_spec.moe_spec_v2.expert_routing_distribution,
            )
            return cardinality.active_expert_fraction
        case _:
            raise ValueError(f"Unsupported expert_weight_residency_policy: {policy}")


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
        Single-layer capacity smoke test.

        Sums this footprint's persistent per-component bytes (weights,
        optimizer_states, saved_activations, gradient_buffers, kv_cache) by
        memory tier, adds the **maximum** of `transient_forward_peak` and
        `transient_backward_peak` per tier (since only one is live at any
        instant), and checks each tier's total against the device's capacity
        for that tier. Raises ValueError listing any violating tiers.

        Important caveat — what this does NOT check:
        - This is a **per-layer** sum. It does not multiply persistent
          components by `num_layers_per_pp_stage`, so it cannot detect cases
          where a single layer fits but the full per-chip total (all layers
          hosted on this chip) does not. The realistic HBM-fit metric is
          `total_peak_memory_per_chip_bytes` in `run_scalescope.py`, which
          downstream feasibility checks should compare against device HBM.

        Treat this as an early sanity check that catches gross
        per-component-tier misconfigurations, not as the authoritative
        HBM-feasibility check.
        """
        if self.node is None:
            raise ValueError("MemoryFootprint.node is not set; cannot verify capacities.")
        tiers = self.node.device_memory_tiers
        usage_by_tier: Dict[str, int] = {}
        fwd_by_tier: Dict[str, int] = {}
        bwd_by_tier: Dict[str, int] = {}

        def add_bytes(tier: str, amount: int):
            usage_by_tier[tier] = usage_by_tier.get(tier, 0) + int(amount)

        for component_name in self.components:
            if component_name not in self.device_for:
                raise ValueError(
                    f"MemoryFootprint.device_for missing mapping for '{component_name}'"
                )
            tier = self.device_for[component_name]
            bytes_val = int(getattr(self, component_name))
            if component_name == "transient_forward_peak":
                fwd_by_tier[tier] = fwd_by_tier.get(tier, 0) + bytes_val
            elif component_name == "transient_backward_peak":
                bwd_by_tier[tier] = bwd_by_tier.get(tier, 0) + bytes_val
            else:
                add_bytes(tier, bytes_val)

        # Forward and backward transient peaks are mutually exclusive in
        # time — only one is live at any instant. Charge each tier the
        # larger of the two so we don't double-count.
        for tier in set(fwd_by_tier) | set(bwd_by_tier):
            add_bytes(tier, max(fwd_by_tier.get(tier, 0), bwd_by_tier.get(tier, 0)))

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
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    q_weight_size = d_model * q_proj_out_dim
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = q_proj_out_dim * d_model
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
    total_ffn_weights = transformer_config.expert_params_per_expert

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
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    q_weight_size = d_model * q_proj_out_dim
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = q_proj_out_dim * d_model
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
    total_ffn_weights = transformer_config.expert_params_per_expert

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
def _max_attn_kv_span(transformer_config: TransformerConfig, n_seq: int) -> int:
    """Largest KV span visible to any attention layer class.

    Transient peaks are a max across layers (not a sum), so the peak is set
    by the layer class with the widest score region. With any full-attention
    layers present this is `n_seq`; only a pure-SWA model shrinks the peak.
    """
    return max(
        layer_class.kv_len(n_seq) for layer_class in attention_layer_classes(transformer_config)
    )


def calculate_attn_forward_transient_peak_memory_footprint_tp1d(
    transformer_config: TransformerConfig, workload_spec: WorkloadSpec, tp_size: int
) -> int:
    float_size = workload_spec.float_size_bytes
    # Peak is likely the attention scores matrix
    return (
        workload_spec.effective_micro_batch_size
        * transformer_config.num_heads
        / tp_size
        * workload_spec.seq_length
        * _max_attn_kv_span(transformer_config, workload_spec.seq_length)
        * float_size
    )


def calculate_attn_forward_transient_peak_memory_footprint_tp2d(
    *,
    transformer_config: TransformerConfig,
    workload_spec: WorkloadSpec,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    pass_mode: PassMode,
    kv_total_len: int | None = None,
) -> int:
    """Forward transient peak for TP-2D attention, stage-aware and phase-aware.

    Domain split: phase 1/3 staging buffers (QKV projection output, out-proj
    input/partial, weight panels, role staging) belong to the *projection* TP
    domain and shard by `proj_tp_size_*`; the phase 2 score buffer belongs to
    the *attention* domain and shards by `attn_tp_size_*`. The two domains
    match in most specs but differ under projection groups (g > 1), where the
    projection view is g-times smaller.

    This is intentionally a *simple max-of-major-buffers* estimate to make the footprint logic
    easy to audit and consistent with the TP-2D traffic model's Phase 1/2/3 sizing.

    What this includes (major transient buffers):
    - Phase 1: QKV projection output (sharded on X*YZ)
    - Phase 2: attention score matrix for the local KV shard (non-fused model)
    - Phase 3: output projection staging buffers:
      - gathered activation after YZ all-gather (size ~ E/X)
      - partial output before X reduce-scatter (size ~ E/YZ)

    What this does NOT include:
    - KV cache itself (modeled as persistent `MemoryFootprint.kv_cache`)
    - at-rest projection weights: those are persistent and accounted in
      `MemoryFootprint.weights` (see the role-aware `attn_weight_tp_size`
      logic in `run_scalescope.py`). Only *staged copies* of weights show up
      here — e.g. the YZ-gathered weight panel under yz=circulate_weights.
      Conversely, under yz=shard_weights no weight buffer appears here at all
      (weights never move); the transient cost is the full-sequence
      *activation* staging buffers instead. Transient vs persistent is about
      the buffer's lifetime (exists only within a layer's forward pass vs
      resident across the pass), not about whether it holds weights or
      activations.
    - small online-softmax stats scalars (m, ℓ); see comments in traffic model
    """
    if attn_tp_size_x <= 0 or attn_tp_size_yz <= 0:
        raise ValueError(
            "TP-2D transient peak requires positive attn_tp_size_x and attn_tp_size_yz."
        )
    if proj_tp_size_x <= 0 or proj_tp_size_yz <= 0:
        raise ValueError(
            "TP-2D transient peak requires positive proj_tp_size_x and proj_tp_size_yz."
        )

    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim
    float_size = workload_spec.float_size_bytes
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp

    proj_shard = proj_tp_size_x * proj_tp_size_yz
    heads_per_x = num_heads // attn_tp_size_x

    # Role-dependent staging buffers for the projection schedule
    # (prefill/training only):
    # - circulate_weights: each layer all-gathers its X-shard weight panel
    #   along YZ before the GEMM; the gathered panel (W / proj_tp_size_x)
    #   transiently coexists with the activation buffers.
    # - shard_weights (weight-stationary): the pre-reduce-scatter QKV
    #   partials and the YZ-gathered attention output are full-sequence
    #   activation buffers of shape [B, S, feature_dim / proj_tp_size_x] —
    #   feature-sharded on X only, NOT sequence-sharded on YZ — so each is a
    #   factor proj_tp_size_yz larger than the resident sequence-sharded
    #   slabs [B, S / proj_tp_size_yz, feature_dim / proj_tp_size_x].
    weight_panel_peak = 0.0
    role_staging_peak = 0.0

    match pass_mode:
        case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
            step_seq = workload_spec.seq_length
            kv_len_local = _max_attn_kv_span(transformer_config, step_seq) / attn_tp_size_yz
            roles = workload_spec.tp2d_proj_axis_roles
            if roles is None:
                raise ValueError(
                    "TP-2D prefill/train transient peak requires tp2d_proj_axis_roles "
                    "on the workload spec (TrainSpec / InferencePrefillSpec)."
                )
            match roles.yz:
                case TP2DProjAxisRole.SHARD_TOKENS:
                    pass  # weights replicated on YZ: no role staging buffers
                case TP2DProjAxisRole.CIRCULATE_WEIGHTS:
                    kv_proj_out_dim = num_kv_heads * head_dim
                    w_qkv_panel = (
                        d_model
                        * (q_proj_out_dim + 2 * kv_proj_out_dim)
                        / proj_tp_size_x
                        * float_size
                    )
                    w_o_panel = q_proj_out_dim * d_model / proj_tp_size_x * float_size
                    weight_panel_peak = max(w_qkv_panel, w_o_panel)
                case TP2DProjAxisRole.SHARD_WEIGHTS:
                    kv_proj_out_dim = num_kv_heads * head_dim
                    qkv_partial = (
                        n_batch
                        * step_seq
                        * (q_proj_out_dim + 2 * kv_proj_out_dim)
                        / proj_tp_size_x
                        * float_size
                    )
                    o_gathered = n_batch * step_seq * q_proj_out_dim / proj_tp_size_x * float_size
                    role_staging_peak = max(qkv_partial, o_gathered)
                case _:
                    raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")
        case PassMode.INFERENCE_DECODE:
            step_seq = 1
            if kv_total_len is None:
                raise ValueError("TP-2D decode transient peak requires kv_total_len.")
            kv_len_local = _max_attn_kv_span(transformer_config, kv_total_len) / attn_tp_size_yz
        case _:
            raise ValueError(f"Unsupported pass_mode for TP-2D transient peak: {pass_mode}")

    # Phase 1: QKV projection output (GQA-aware: K/V use num_kv_heads, Q uses num_heads × head_dim).
    # Projection-domain buffer: sharded across the projection X*YZ grid.
    qkv_out_elems = q_proj_out_dim + 2 * (num_kv_heads * head_dim)
    qkv_proj_out = n_batch * step_seq * qkv_out_elems / proj_shard * float_size

    # Phase 2: attention scores for local KV segment (non-fused, explicit buffer)
    attn_scores = n_batch * heads_per_x * step_seq * kv_len_local * float_size

    # Phase 3: output projection staging
    # `attn_out_for_out_proj` is the attention output (input to output proj),
    # whose feature dim is num_heads * head_dim = q_proj_out_dim. Under
    # "wide-Q" layouts this is larger than d_model. In decode it is gathered
    # along YZ (size q_proj_out_dim/X for the single step token); in
    # prefill/training the sequence-sharded schedule keeps it local
    # ((S/YZ) tokens x q_proj_out_dim/X features — no YZ activation gather).
    # `partial_out` is the pre-reduce output-proj activation: full d_model
    # for the local tokens, i.e. (B, S/YZ, d_model) in prefill/training and
    # equivalently (B, step_seq, d_model/YZ) in decode.
    match pass_mode:
        case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
            attn_out_for_out_proj = n_batch * step_seq * q_proj_out_dim / proj_shard * float_size
        case _:
            attn_out_for_out_proj = (
                n_batch * step_seq * q_proj_out_dim / proj_tp_size_x * float_size
            )
    partial_out = n_batch * step_seq * d_model / proj_tp_size_yz * float_size  # ~E/YZ

    return int(
        max(
            qkv_proj_out,
            attn_scores,
            attn_out_for_out_proj,
            partial_out,
            weight_panel_peak,
            role_staging_peak,
        )
    )


def calculate_ffn_moe_forward_transient_peak_memory_footprint_tp1d(
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


def calculate_ffn_moe_forward_transient_peak_memory_footprint_tp2d(
    *,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec: WorkloadSpec,
    moe_tp_size_x: int,
    moe_tp_size_yz: int,
    pass_mode: PassMode,
) -> int:
    """Forward transient peak for TP-2D MoE/FFN, stage-aware and phase-aware.

    This mirrors the phase-structured TP-2D MoE algorithm we model in:
    - `TP2DMoEStrategy` (collectives)
    - `_forward_pass_prefill_moe_ffn_tp_2d_memory_traffic` (HBM traffic)

    Major transient buffers included:
    - Input shard: x^{i,j} ~ [tokens, E/(X*YZ)]
    - Gathered input after X all-gather: x̄^j ~ [tokens, E/YZ]
    - Router scores buffer: [tokens, n_experts] (if MoE enabled)
    - FFN1 output: [tokens', H/X]
    - FFN2 partial output before X reduce-scatter: [tokens', E/YZ]
    - Final output shard: [tokens', E/(X*YZ)]

    Notes:
    - This is a simple max-of-major-buffers estimate (auditable, conservative).
    - EP all-to-all token exchange is not a memory footprint here; it's modeled as comm elsewhere.
    """
    if moe_tp_size_x <= 0 or moe_tp_size_yz <= 0:
        raise ValueError(
            "TP-2D MoE transient peak requires positive moe_tp_size_x and moe_tp_size_yz."
        )

    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_hidden_size
    float_size = workload_spec.float_size_bytes
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    shard = moe_tp_size_x * moe_tp_size_yz

    match pass_mode:
        case PassMode.TRAINING | PassMode.INFERENCE_PREFILL:
            step_seq = workload_spec.seq_length
        case PassMode.INFERENCE_DECODE:
            step_seq = 1
        case _:
            raise ValueError(f"Unsupported pass_mode for TP-2D MoE transient peak: {pass_mode}")

    tokens_local = n_batch * step_seq
    input_shard = tokens_local * d_model / shard * float_size
    gathered_x = tokens_local * d_model / moe_tp_size_yz * float_size  # after X all-gather

    n_experts = transformer_config.num_regular_experts
    router_scores = 0.0
    if n_experts > 0:
        router_scores = tokens_local * n_experts * float_size

    # Routed token expansion proxy (matches our traffic/comm models)
    tokens_prime = tokens_local
    if n_experts > 0:
        # Use full-expert-set normalization for routed expert load (matches EP/comm/compute models).
        tokens_per_set = (
            workload_spec.get_resolved_batches().batch_size_per_full_expert_set * step_seq
        )
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        num_colocated = moe_config.num_colocated_experts
        tokens_per_routed_expert = tokens_per_set / n_experts
        tokens_prime = tokens_per_routed_expert * traffic_mult * num_colocated

    ffn1_out = tokens_prime * (d_ff / moe_tp_size_x) * float_size
    ffn2_partial = tokens_prime * (d_model / moe_tp_size_yz) * float_size
    ffn2_out = tokens_prime * (d_model / shard) * float_size

    return int(max(input_shard, gathered_x, router_scores, ffn1_out, ffn2_partial, ffn2_out))


def calculate_attn_backward_transient_peak_memory_footprint(
    transformer_config: TransformerConfig,
    workload_spec: "TrainSpec",
    activation_storage_strategy: ActivationStorageOptimizationStrategy,
    tp_size: int,
) -> int:
    float_size = workload_spec.float_size_bytes
    peak = calculate_attn_forward_transient_peak_memory_footprint_tp1d(
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
    peak = calculate_ffn_moe_forward_transient_peak_memory_footprint_tp1d(
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


def _forward_pass_prefill_attention_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    proj_tp_size: int,
    attn_tp_size: int,
    save_kv_cache: bool,
    workload_spec,
):
    """
    Wrapper: prefill/training attention memory traffic under NONE/TP-1D, split into:
    - projection (QKV + output projection)
    - attention-core (scores + context + optional KV-cache write)

    `proj_tp_size` is the projection TP domain size and `attn_tp_size` the attention
    domain size; they differ under projection groups (g > 1), where the projection
    view is g-times smaller.
    """
    _forward_pass_prefill_attention_projection_tp_1d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        n_seq=n_seq,
        proj_tp_size=proj_tp_size,
        workload_spec=workload_spec,
    )
    _forward_pass_prefill_attention_attn_tp_1d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        n_seq=n_seq,
        proj_tp_size=proj_tp_size,
        attn_tp_size=attn_tp_size,
        save_kv_cache=save_kv_cache,
        workload_spec=workload_spec,
    )


def _forward_pass_prefill_attention_projection_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    proj_tp_size: int,
    workload_spec,
):
    """Prefill/training attention projection traffic (QKV + output projection) under NONE/TP-1D."""
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights for Q, K/V, and output projection (sharded by projection TP).
    q_weight_size = d_model * q_proj_out_dim
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = q_proj_out_dim * d_model
    traffic.weights.bytes_read += (
        (q_weight_size + kv_weight_size + out_weight_size) * float_size
    ) / proj_tp_size

    # Input activation read.
    traffic.activations.bytes_read += n_batch * n_seq * d_model * float_size

    # QKV projection output written (sharded by projection TP, with KV duplication fallback).
    kv_heads_proj, _kv_rep_proj = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=proj_tp_size
    )
    q_out_sharded = n_batch * n_seq * q_proj_out_dim * float_size / proj_tp_size
    kv_out_sharded = 2 * n_batch * n_seq * (kv_heads_proj * head_dim) * float_size
    traffic.activations.bytes_written += q_out_sharded + kv_out_sharded

    # Output projection staging writes (partial + final output).
    traffic.activations.bytes_written += n_batch * n_seq * d_model * float_size / proj_tp_size
    traffic.activations.bytes_written += n_batch * n_seq * d_model * float_size


def _forward_pass_prefill_attention_attn_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    proj_tp_size: int,
    attn_tp_size: int,
    save_kv_cache: bool,
    workload_spec,
):
    """Prefill/training attention core traffic (scores + context + optional KV-cache write) under NONE/TP-1D."""
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    num_q_heads_attn = num_heads // attn_tp_size
    num_kv_heads_attn, _kv_rep_attn = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=attn_tp_size
    )
    num_kv_heads_proj, _kv_rep_proj = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=proj_tp_size
    )

    # Read QKV outputs produced by projections.
    q_out_sharded = n_batch * n_seq * q_proj_out_dim * float_size / proj_tp_size
    kv_out_sharded = 2 * n_batch * n_seq * (num_kv_heads_proj * head_dim) * float_size
    traffic.activations.bytes_read += q_out_sharded + kv_out_sharded

    if save_kv_cache:
        # Sliding-window layers only retain the last `window_size` tokens,
        # so their KV write is capped; blended per layer class.
        kv_write = 0.0
        for layer_class in attention_layer_classes(transformer_config):
            kv_write += (
                layer_class.layer_fraction
                * 2
                * n_batch
                * layer_class.kv_len(n_seq)
                * (num_kv_heads_attn * head_dim)
                * float_size
            )
        traffic.kv_cache.bytes_written += kv_write

    # Attention scores. Full-attention layers use the dense n_seq × n_seq
    # buffer; sliding-window layers clip the score region to
    # n_seq × min(window, n_seq). Blended per layer class.
    attn_scores_sharded = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        attn_scores_sharded += (
            layer_class.layer_fraction
            * n_batch
            * num_q_heads_attn
            * n_seq
            * layer_class.kv_len(n_seq)
            * float_size
        )
    traffic.activations.bytes_written += attn_scores_sharded
    traffic.activations.bytes_read += attn_scores_sharded

    # Attention context.
    attn_context_sharded = n_batch * n_seq * (num_q_heads_attn * head_dim) * float_size
    traffic.activations.bytes_written += attn_context_sharded

    # Read context for output projection.
    traffic.activations.bytes_read += attn_context_sharded


def _decode_attention_projection_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    proj_tp_size: int,
    workload_spec,
):
    """
    Decode attention *projection* memory traffic (NONE/TP-1D):
    - Weights read for Q/K/V projections and output projection
    - Activation read of layer input (single new token)
    - Q/K/V output write (single new token)
    - Output projection staging writes (partial + final output)

    This is split from `_decode_attention_attn_tp_1d_memory_traffic` to enable a future
    separation where projections run on a different TP domain than attention compute.
    """
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim

    # Calculate sharded dimensions
    num_q_heads_sharded = num_heads // proj_tp_size
    num_kv_heads_sharded, _kv_rep_proj = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=proj_tp_size
    )

    # Infer sizes from workload spec
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights
    q_proj_out_dim = num_heads * head_dim
    q_weight_size = d_model * q_proj_out_dim
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    out_weight_size = q_proj_out_dim * d_model
    traffic.weights.bytes_read += (
        (q_weight_size + kv_weight_size + out_weight_size) * float_size
    ) / proj_tp_size

    # Activations
    # Activation traffic is for a single new token (n_seq=1).
    traffic.activations.bytes_read += n_batch * 1 * d_model * float_size
    q_out_new_sharded = n_batch * 1 * (num_q_heads_sharded * head_dim) * float_size
    kv_out_new_sharded = 2 * n_batch * 1 * (num_kv_heads_sharded * head_dim) * float_size
    traffic.activations.bytes_written += q_out_new_sharded + kv_out_new_sharded

    # Output projection staging:
    # - The output projection consumes the attention context; that read and the context write
    #   are accounted in `_decode_attention_attn_tp_1d_memory_traffic`.
    # - Here we only account for the output projection's write buffers (partial + final),
    #   which depend on the projection TP domain.
    traffic.activations.bytes_written += n_batch * 1 * d_model * float_size / proj_tp_size
    traffic.activations.bytes_written += n_batch * 1 * d_model * float_size


def _decode_attention_attn_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq_so_far: int,
    proj_tp_size: int,
    attn_tp_size: int,
    workload_spec,
):
    """
    Decode attention *core* memory traffic (NONE/TP-1D):
    - KV cache read of history and append write of new K/V (KV-cache sharding follows attn TP)
    - Read Q/K/V activations produced by projections (layout follows proj TP)
    - Attention score write/read and context write (layout follows attn TP)
    - Read context for output projection (consumed by projections)
    """
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Sharded dimensions:
    # - Q/K/V activations follow projection TP (future: may differ from attention TP).
    # - KV cache is owned/sharded by the attention TP domain.
    num_q_heads_proj = num_heads // proj_tp_size
    num_kv_heads_proj, _kv_rep_proj = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=proj_tp_size
    )
    num_kv_heads_attn, _kv_rep_attn = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=attn_tp_size
    )

    # KV cache: read local history + append new K/V. History reads are
    # blended across attention layer classes: sliding-window layers read at
    # most `window_size` cached tokens.
    kv_read_history = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        kv_read_history += (
            layer_class.layer_fraction
            * 2
            * n_batch
            * layer_class.kv_len(n_seq_so_far)
            * (num_kv_heads_attn * head_dim)
            * float_size
        )
    traffic.kv_cache.bytes_read += kv_read_history
    traffic.kv_cache.bytes_written += 2 * n_batch * 1 * (num_kv_heads_attn * head_dim) * float_size

    # Read the new Q, K, V activations for the attention calculation.
    q_out_new_sharded = n_batch * 1 * (num_q_heads_proj * head_dim) * float_size
    kv_out_new_sharded = 2 * n_batch * 1 * (num_kv_heads_proj * head_dim) * float_size
    traffic.activations.bytes_read += q_out_new_sharded + kv_out_new_sharded

    # Attention scores + context (layout follows attention TP domain).
    # Score-row length is the visible KV span (+1 for the new token),
    # blended per layer class.
    num_q_heads_attn = num_heads // attn_tp_size
    attn_scores_sharded = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        attn_scores_sharded += (
            layer_class.layer_fraction
            * n_batch
            * num_q_heads_attn
            * 1
            * (layer_class.kv_len(n_seq_so_far) + 1)
            * float_size
        )
    traffic.activations.bytes_written += attn_scores_sharded
    traffic.activations.bytes_read += attn_scores_sharded
    attn_context_sharded = n_batch * 1 * (num_q_heads_attn * head_dim) * float_size
    traffic.activations.bytes_written += attn_context_sharded

    # Read the context vector for the output projection.
    traffic.activations.bytes_read += attn_context_sharded


def _decode_attention_tp_1d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq_so_far: int,
    proj_tp_size: int,
    attn_tp_size: int,
    workload_spec,
):
    """Convenience wrapper: projection + attention-core for TP-1D decode."""
    _decode_attention_projection_tp_1d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        proj_tp_size=proj_tp_size,
        workload_spec=workload_spec,
    )
    _decode_attention_attn_tp_1d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        n_seq_so_far=n_seq_so_far,
        proj_tp_size=proj_tp_size,
        attn_tp_size=attn_tp_size,
        workload_spec=workload_spec,
    )


def _decode_router_weight_read_bytes(
    *,
    workload_spec,
    moe_config,
    d_model: int,
    total_num_routed_experts: int,
    float_size: int,
) -> int:
    """Per-rank decode router-state read bytes for current router mode.

    Uses ``moe_config.router_shard_factor`` which the orchestrator sets to
    Y_eff for 2D TP (after AG(X) the router weight is sharded only across
    Y_eff) or attn_tp_size for 1D TP. The normalization-deferred mode also
    reads its precomputed correction vectors G and C.
    """
    if not isinstance(workload_spec, InferenceDecodeSpec):
        if moe_config.router_sync_mode == MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER:
            raise ValueError(
                "normalization_deferred_sharded_router is only valid for inference decode"
            )
        return 0

    router_mode = workload_spec.decode_moe_router_sync_mode
    router_weight_full = d_model * total_num_routed_experts * float_size
    router_shard_factor = int(moe_config.router_shard_factor)
    match router_mode:
        case MoERouterSyncMode.REPLICATED_ROUTER:
            # Each rank holds the full W_gate and reads it locally.
            return int(router_weight_full)
        case MoERouterSyncMode.SHARDED_ROUTER:
            # W_gate is partitioned across router_shard_factor ranks.
            if router_shard_factor < 1:
                raise ValueError(f"Invalid router_shard_factor: {router_shard_factor}")
            return int(router_weight_full / router_shard_factor)
        case MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER:
            # Same sharded W_gate as SHARDED_ROUTER, plus the precomputed
            # correction vectors G and C (one pair per routed expert).
            if router_shard_factor <= 1:
                raise ValueError(
                    "normalization_deferred_sharded_router requires router_shard_factor > 1"
                )
            correction_vectors = 2 * total_num_routed_experts * float_size
            return int(router_weight_full / router_shard_factor + correction_vectors)
        case _:
            raise ValueError(f"Unsupported decode_moe_router_sync_mode: {router_mode}")


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

    # Infer sizes from workload spec
    batch_size_per_dp = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes

    # Weights read
    # Colocated experts share the same XPU; scale weights read by the number of colocated experts hosted.
    # Under RESIDENT_ACTIVE_SUBSET, this term is additionally scaled by
    # active_expert_fraction to model tiny-batch selective expert activation
    # (see workload_spec.ExpertWeightResidencyPolicy).
    num_colocated = moe_config.num_colocated_experts
    tokens_per_set_for_residency = (
        workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    )
    expert_weight_residency_fraction = _expert_weight_residency_fraction(
        workload_spec=workload_spec,
        transformer_config=transformer_config,
        tokens_per_set=tokens_per_set_for_residency,
    )
    total_w_bytes = (
        num_colocated
        * transformer_config.expert_params_per_expert
        * float_size
        / moe_tp_size
        * expert_weight_residency_fraction
    )
    traffic.weights.bytes_read += total_w_bytes

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

        # Permute/unpermute staging for grouped expert GEMMs (c > 1).
        # Tokens are sorted into per-expert contiguous segments: dispatch
        # permutes the slab into expert order (read + write), combine
        # unpermutes back to token order (read + write). A regular GEMM
        # (c = 1, including a lone routed expert) has one segment; sort
        # is identity. Dense E = 1 is also c = 1.
        if num_colocated > 1:
            permute_leg_bytes = num_colocated * token_volume_per_routed_expert
            # ×2 = permute (dispatch) + unpermute (combine). Each leg
            # copies the full local slab (one read and one write).
            traffic.activations.bytes_read += 2 * permute_leg_bytes
            traffic.activations.bytes_written += 2 * permute_leg_bytes

        # Calculate the total volume of tokens routed to a single regular expert.
        # The input for this expert's FFN1 is read from the All-to-All output buffer.
        # Scale by number of colocated experts per chip
        traffic.activations.bytes_read += num_colocated * token_volume_per_routed_expert
        # The output of FFN1 is written. It is sharded by EP_size, because the data is per XPU
        ff1_output_sharded_per_regular_expert = int(
            (mlp_ratio * token_volume_per_routed_expert) / moe_tp_size
        )
        traffic.activations.bytes_written += num_colocated * ff1_output_sharded_per_regular_expert
        # FFN2 consumes the same local FFN1 shard [tokens, H/TP] that was just written.
        traffic.activations.bytes_read += num_colocated * ff1_output_sharded_per_regular_expert
        # The output of this expert is written. This represents the data before the final
        # All-to-All combines it with output from other experts. Its volume is equivalent
        # to the input token volume for this expert.
        traffic.activations.bytes_written += num_colocated * token_volume_per_routed_expert

        # Decode router mode can change router-weight reads over the attention TP domain.
        traffic.weights.bytes_read += _decode_router_weight_read_bytes(
            workload_spec=workload_spec,
            moe_config=moe_config,
            d_model=d_model,
            total_num_routed_experts=total_num_routed_experts,
            float_size=float_size,
        )


def _forward_pass_prefill_attention_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    save_kv_cache: bool,
    workload_spec,
):
    """
    Wrapper: TP-2D attention memory traffic for training/prefill, split into:
    - projection (phase 1 + phase 3 + weights)
    - attention-core (phase 2 + optional KV-cache write)

    `proj_tp_size_*` are the projection TP domain splits and `attn_tp_size_*` the
    attention domain splits; they differ under projection groups (g > 1), where the
    projection view is g-times smaller.
    """
    _forward_pass_prefill_attention_projection_tp_2d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        n_seq=n_seq,
        proj_tp_size_x=proj_tp_size_x,
        proj_tp_size_yz=proj_tp_size_yz,
        attn_tp_size_x=attn_tp_size_x,
        attn_tp_size_yz=attn_tp_size_yz,
        workload_spec=workload_spec,
    )
    _forward_pass_prefill_attention_attn_tp_2d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        n_seq=n_seq,
        proj_tp_size_x=proj_tp_size_x,
        proj_tp_size_yz=proj_tp_size_yz,
        attn_tp_size_x=attn_tp_size_x,
        attn_tp_size_yz=attn_tp_size_yz,
        save_kv_cache=save_kv_cache,
        workload_spec=workload_spec,
    )


def _forward_pass_prefill_attention_projection_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    workload_spec,
):
    """TP-2D prefill/training projection traffic: weights + phase 1 + phase 3."""
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Weights: read once per forward pass. What each rank reads depends on
    # the YZ projection-axis role (X always shards weights):
    # - shard_weights (weight-stationary): local shard W/(X*YZ).
    # - shard_tokens: weights replicated across YZ; each rank reads its full
    #   X shard W/X.
    # - circulate_weights: the GEMM consumes the gathered X-shard panel W/X;
    #   the received (W/X - W/(X*YZ)) panel bytes are additionally staged
    #   (written) to HBM before the GEMM reads them.
    q_weight_bytes = d_model * q_proj_out_dim * float_size
    kv_weight_bytes = 2 * (d_model * num_kv_heads * head_dim) * float_size
    out_weight_bytes = q_proj_out_dim * d_model * float_size
    total_weight_bytes = q_weight_bytes + kv_weight_bytes + out_weight_bytes
    shard_proj = proj_tp_size_x * proj_tp_size_yz
    roles = workload_spec.tp2d_proj_axis_roles
    if roles is None:
        raise ValueError(
            "TP-2D prefill/train projection traffic requires tp2d_proj_axis_roles "
            "on the workload spec (TrainSpec / InferencePrefillSpec)."
        )
    match roles.yz:
        case TP2DProjAxisRole.SHARD_TOKENS:
            traffic.weights.bytes_read += total_weight_bytes // proj_tp_size_x
        case TP2DProjAxisRole.CIRCULATE_WEIGHTS:
            traffic.weights.bytes_read += total_weight_bytes // proj_tp_size_x
            traffic.weights.bytes_written += (
                total_weight_bytes // proj_tp_size_x - total_weight_bytes // shard_proj
            )
        case TP2DProjAxisRole.SHARD_WEIGHTS:
            # Weight-stationary: each rank reads only its pinned X x YZ panel.
            traffic.weights.bytes_read += total_weight_bytes // shard_proj
        case _:
            raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")

    # Prefill/training processes full sequence.
    step_seq = n_seq

    # Sequence-sharded schedule (matches `TP2DAttnStrategy` prefill/train comm):
    # tokens live as [B, S/YZ, *] slabs on every rank throughout the layer.
    #
    # -------------------------
    # Phase 1: input projection
    # -------------------------
    # The QKV GEMM consumes the full-d_model token slab for the local
    # sequence shard (reconstructed by the Megatron-SP all-gather along X):
    # [B, S/YZ, d_model], i.e. S*d/YZ bytes.
    traffic.activations.bytes_read += int(
        n_batch * step_seq * d_model / proj_tp_size_yz * float_size
    )

    # QKV projection output. GQA-aware: Q output is num_heads * head_dim;
    # K/V output is num_kv_heads * head_dim.
    # - shard_tokens / circulate_weights: local tokens (S/YZ), output
    #   features sharded on X -> [B, S/YZ, qkv/X].
    # - shard_weights (weight-stationary): the GEMM emits the full-sequence
    #   partial [B, S, qkv/X] (a factor YZ larger), which the YZ
    #   reduce-scatter then shrinks back to the sequence shard.
    qkv_out_elems = q_proj_out_dim + 2 * (num_kv_heads * head_dim)
    match roles.yz:
        case TP2DProjAxisRole.SHARD_WEIGHTS:
            qkv_output_written = n_batch * step_seq * qkv_out_elems / proj_tp_size_x * float_size
        case TP2DProjAxisRole.SHARD_TOKENS | TP2DProjAxisRole.CIRCULATE_WEIGHTS:
            qkv_output_written = n_batch * step_seq * qkv_out_elems / shard_proj * float_size
        case _:
            raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")
    traffic.activations.bytes_written += int(qkv_output_written)

    # --------------------------
    # Phase 3: output projection
    # --------------------------
    # - Input is the attention output, [*, E_in/X] where E_in = num_heads *
    #   head_dim = q_proj_out_dim (the attention output dim, NOT d_model —
    #   they differ under wide-Q layouts). For shard_tokens /
    #   circulate_weights it is the local sequence shard [B, S/YZ, E_in/X]
    #   (no YZ activation gather); for shard_weights (weight-stationary) it
    #   is YZ-gathered to the full sequence [B, S, E_in/X] before the W_O
    #   panel GEMM.
    # - Local matmul produces the partial output O_partial of size
    #   [B, S/YZ, d_model] (equivalently [B, S, d_model/YZ] under
    #   shard_weights — same bytes), written and re-read for the X
    #   reduce-scatter.
    # - Reduce-scatter on X produces the final output shard.
    shard_attn = attn_tp_size_x * attn_tp_size_yz
    match roles.yz:
        case TP2DProjAxisRole.SHARD_WEIGHTS:
            attn_out_in_read = n_batch * step_seq * q_proj_out_dim / attn_tp_size_x * float_size
        case TP2DProjAxisRole.SHARD_TOKENS | TP2DProjAxisRole.CIRCULATE_WEIGHTS:
            attn_out_in_read = n_batch * step_seq * q_proj_out_dim / shard_attn * float_size
        case _:
            raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")
    traffic.activations.bytes_read += int(attn_out_in_read)
    # Partial output (post-output-proj) written and read for the X reduce-scatter.
    partial_out = n_batch * step_seq * d_model / proj_tp_size_yz * float_size
    traffic.activations.bytes_written += int(partial_out)
    traffic.activations.bytes_read += int(partial_out)
    # Final output shard written (projection TP domain).
    out_shard = n_batch * step_seq * d_model / shard_proj * float_size
    traffic.activations.bytes_written += int(out_shard)


def _forward_pass_prefill_attention_attn_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    save_kv_cache: bool,
    workload_spec,
):
    """TP-2D prefill/training attention-core traffic: phase 2 + optional KV-cache write."""
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    shard_proj = proj_tp_size_x * proj_tp_size_yz
    shard_attn = attn_tp_size_x * attn_tp_size_yz

    step_seq = n_seq

    # KV heads per X rank: allow duplication when attn_tp_size_x > num_kv_heads.
    if not attn_tp_size_x:
        kv_heads_per_x = num_kv_heads
    elif attn_tp_size_x <= num_kv_heads:
        if num_kv_heads % attn_tp_size_x != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by TP-X ({attn_tp_size_x})"
            )
        kv_heads_per_x = num_kv_heads // attn_tp_size_x
    else:
        if attn_tp_size_x % num_kv_heads != 0:
            raise ValueError(
                f"TP-X ({attn_tp_size_x}) must be divisible by num_kv_heads ({num_kv_heads}) for KV duplication"
            )
        kv_heads_per_x = 1
    heads_per_x = num_heads // attn_tp_size_x if attn_tp_size_x else num_heads

    # Read QKV for attention (produced by projections).
    qkv_out_elems = q_proj_out_dim + 2 * (num_kv_heads * head_dim)
    qkv_output_sharded_size = n_batch * step_seq * qkv_out_elems / shard_proj * float_size
    traffic.activations.bytes_read += int(qkv_output_sharded_size)

    # KV cache side effects (prefill only). Sliding-window layers only
    # retain the last `window_size` tokens; blended per layer class.
    if save_kv_cache:
        kv_write = 0.0
        for layer_class in attention_layer_classes(transformer_config):
            kv_write += (
                layer_class.layer_fraction
                * 2
                * n_batch
                * (layer_class.kv_len(n_seq) / attn_tp_size_yz)
                * (kv_heads_per_x * head_dim)
                * float_size
            )
        traffic.kv_cache.bytes_written += int(kv_write)

    # Attention scores for local KV segment (sequence sharded on YZ).
    # (Non-fused / explicit buffer model) Sliding-window layers clip the
    # score region to n_seq × min(window, n_seq); blended per layer class.
    attn_score_sharded_size = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        kv_len_local_cls = layer_class.kv_len(n_seq) / attn_tp_size_yz
        attn_score_sharded_size += (
            layer_class.layer_fraction
            * n_batch
            * heads_per_x
            * kv_len_local_cls
            * n_seq
            * float_size
        )
    traffic.activations.bytes_written += int(attn_score_sharded_size)
    traffic.activations.bytes_read += int(attn_score_sharded_size)

    # Write context/output (the post-attention activation, shape
    # (B, S, num_heads * head_dim) = q_proj_out_dim, sharded on X*YZ in the
    # attention TP domain). Differs from d_model under wide-Q layouts.
    traffic.activations.bytes_written += int(
        n_batch * step_seq * q_proj_out_dim / shard_attn * float_size
    )


def _decode_attention_projection_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    workload_spec,
):
    """
    TP-2D decode attention *projection* memory traffic:
    - Weights read (Q, K/V, Out) sharded in 2D (projection TP domain)
    - Phase 1: input projection activation read + QKV out write (projection TP domain)
    - Phase 3: output projection staging (projection TP domain)

    The KV-cache read/append and attention score/context buffers are accounted in
    `_decode_attention_attn_tp_2d_memory_traffic`.
    """
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    # Weights (Q, K/V, Out) read, sharded in 2D (GQA-aware).
    q_weight_bytes = d_model * q_proj_out_dim * float_size
    kv_weight_bytes = 2 * (d_model * num_kv_heads * head_dim) * float_size
    out_weight_bytes = q_proj_out_dim * d_model * float_size
    shard = proj_tp_size_x * proj_tp_size_yz
    traffic.weights.bytes_read += (q_weight_bytes + kv_weight_bytes + out_weight_bytes) // shard

    # Decode processes one new token.
    step_seq = 1

    # Phase 1: input projection (activation read + QKV out write)
    traffic.activations.bytes_read += int(n_batch * step_seq * d_model / shard * float_size)
    qkv_out_elems = q_proj_out_dim + 2 * (num_kv_heads * head_dim)
    qkv_output_sharded_size = n_batch * step_seq * qkv_out_elems / shard * float_size
    traffic.activations.bytes_written += int(qkv_output_sharded_size)

    # Phase 3: output projection staging (explicit, non-fused; matches decode TP comm).
    # Phase 2 AR(Y) leaves the attention output replicated on Y and sharded only on X,
    # so W_O reads one local [B, q_proj_out_dim/X] activation. There is no decode AG(Y).
    # `partial_out` is the post-output-proj activation = d_model/Y; RS(X) then writes
    # the final d_model/(X*Y) residual-stream shard.
    attn_out_after_ar_y = n_batch * step_seq * q_proj_out_dim / proj_tp_size_x * float_size
    traffic.activations.bytes_read += int(attn_out_after_ar_y)
    partial_out = n_batch * step_seq * d_model / proj_tp_size_yz * float_size
    traffic.activations.bytes_written += int(partial_out)
    traffic.activations.bytes_read += int(partial_out)
    final_out_shard = n_batch * step_seq * d_model / shard * float_size
    traffic.activations.bytes_written += int(final_out_shard)


def _decode_attention_attn_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq_so_far: int,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    workload_spec,
):
    """
    TP-2D decode attention *core* memory traffic:
    - KV cache read of local history + append write (KV-cache sharding follows attn TP YZ)
    - Read QKV outputs produced by projections (layout follows proj TP)
    - Phase 2: attention score buffer write/read + attention output write (layout follows attn TP)
    """
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    float_size = workload_spec.float_size_bytes
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    step_seq = 1

    # QKV outputs (produced by projections).
    shard_proj = proj_tp_size_x * proj_tp_size_yz
    qkv_out_elems = q_proj_out_dim + 2 * (num_kv_heads * head_dim)
    qkv_output_sharded_size = n_batch * step_seq * qkv_out_elems / shard_proj * float_size

    # KV cache sharding follows attention TP.
    # KV heads per X rank: allow duplication when attn_tp_size_x > num_kv_heads.
    if not attn_tp_size_x:
        kv_heads_per_x_attn = num_kv_heads
    elif attn_tp_size_x <= num_kv_heads:
        if num_kv_heads % attn_tp_size_x != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by TP-X ({attn_tp_size_x})"
            )
        kv_heads_per_x_attn = num_kv_heads // attn_tp_size_x
    else:
        if attn_tp_size_x % num_kv_heads != 0:
            raise ValueError(
                f"TP-X ({attn_tp_size_x}) must be divisible by num_kv_heads ({num_kv_heads}) for KV duplication"
            )
        kv_heads_per_x_attn = 1
    # Visible KV span per layer class (sliding-window layers cap at window),
    # sequence-sharded on YZ; blended by layer fractions.
    kv_read_history = 0.0
    blended_kv_len_local = 0.0
    for layer_class in attention_layer_classes(transformer_config):
        kv_len_local_cls = layer_class.kv_len(n_seq_so_far) / attn_tp_size_yz
        blended_kv_len_local += layer_class.layer_fraction * kv_len_local_cls
        kv_read_history += (
            layer_class.layer_fraction
            * 2
            * n_batch
            * kv_len_local_cls
            * (kv_heads_per_x_attn * head_dim)
            * float_size
        )
    traffic.kv_cache.bytes_read += int(kv_read_history)
    traffic.kv_cache.bytes_written += int(
        2 * n_batch * 1 * (kv_heads_per_x_attn * head_dim) * float_size
    )

    # Phase 2: attention computation (scores + context).
    # Note on online softmax stats (m, ℓ):
    # - Communication is modeled in the TP strategy (YZ all-reduce with a custom op).
    # - HBM traffic for these scalars is omitted here (assumed register/shared-memory).
    traffic.activations.bytes_read += int(qkv_output_sharded_size)
    heads_per_x_attn = num_heads // attn_tp_size_x if attn_tp_size_x else num_heads
    # Score buffer is linear in the local KV span, so the blended span is exact here.
    attn_score_sharded_size = (
        n_batch * heads_per_x_attn * step_seq * blended_kv_len_local * float_size
    )
    traffic.activations.bytes_written += int(attn_score_sharded_size)
    traffic.activations.bytes_read += int(attn_score_sharded_size)

    # Phase 2 AR(Y) produces the exact attention output replicated across Y and
    # sharded only across X heads. The output projection consumes this layout
    # directly; decode does not perform another AG(Y).
    traffic.activations.bytes_written += int(
        n_batch * step_seq * q_proj_out_dim / attn_tp_size_x * float_size
    )


def _decode_attention_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq_so_far: int,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    workload_spec,
):
    """Convenience wrapper: projection (phases 1+3) + attention-core (phase 2 + KV cache) for TP-2D decode."""
    _decode_attention_projection_tp_2d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        proj_tp_size_x=proj_tp_size_x,
        proj_tp_size_yz=proj_tp_size_yz,
        workload_spec=workload_spec,
    )
    _decode_attention_attn_tp_2d_memory_traffic(
        traffic=traffic,
        transformer_config=transformer_config,
        n_seq_so_far=n_seq_so_far,
        proj_tp_size_x=proj_tp_size_x,
        proj_tp_size_yz=proj_tp_size_yz,
        attn_tp_size_x=attn_tp_size_x,
        attn_tp_size_yz=attn_tp_size_yz,
        workload_spec=workload_spec,
    )


#
# NOTE: No 2D wrapper dispatcher is provided by design.
# Callers should explicitly select between:
# - `_forward_pass_prefill_attention_tp_2d_memory_traffic(...)` for training/prefill
# - `_decode_attention_tp_2d_memory_traffic(...)` for decode
#


def _forward_pass_prefill_moe_ffn_tp_2d_memory_traffic(
    traffic: "LayerMemoryTraffic",
    transformer_config: TransformerConfig,
    n_seq: int,
    moe_tp_size_x: int,
    moe_tp_size_yz: int,
    workload_spec,
    moe_config,
):
    d_model = transformer_config.d_model
    float_size = workload_spec.float_size_bytes
    d_ff = transformer_config.mlp_hidden_size
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp

    # Weights (FFN only part; attention weights handled in attention helper)
    # Each XPU hosts `num_colocated_experts` experts; in TP-2D each expert weight is sharded
    # by (X * YZ), so per-XPU weight volume scales as num_colocated_experts / (X * YZ).
    # This keeps parity with the TP-1D path and avoids undercounting when colocation increases.
    # Under RESIDENT_ACTIVE_SUBSET, this term is additionally scaled by
    # active_expert_fraction to model tiny-batch selective expert activation
    # (see workload_spec.ExpertWeightResidencyPolicy).
    num_colocated = int(getattr(moe_config, "num_colocated_experts", 1) or 1)
    tokens_per_set_for_residency = (
        workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    )
    expert_weight_residency_fraction = _expert_weight_residency_fraction(
        workload_spec=workload_spec,
        transformer_config=transformer_config,
        tokens_per_set=tokens_per_set_for_residency,
    )
    total_w_bytes = (
        num_colocated
        * transformer_config.expert_params_per_expert
        * float_size
        / (moe_tp_size_x * moe_tp_size_yz)
        * expert_weight_residency_fraction
    )
    traffic.weights.bytes_read += total_w_bytes

    step_seq = n_seq

    # Phase-structured MoE memory traffic proxy for TP-2D.
    #
    # This is designed to be consistent with the explicit TP-2D MoE algorithm we model in
    # `TP2DMoEStrategy`:
    # 1) (comm) all-gather on X to reconstruct E/tp_yz input for router + expert matmuls
    # 2) (comm) all-reduce on YZ to complete router scores
    # 3) (comm) all-reduce on YZ for up-projection partial sums (E sharded on YZ)
    # 4) (comm) reduce-scatter on X for down-projection partial sums (H sharded on X)
    #
    # Here we focus on the major HBM activation buffers. We do not attempt to model the EP all-to-all
    # token exchange; that is accounted in comm_model (ep_strategy).

    shard = moe_tp_size_x * moe_tp_size_yz
    tokens_local = n_batch * step_seq

    # Phase 0/1: input activation on this GPU is sharded E/(X*YZ).
    traffic.activations.bytes_read += int(tokens_local * d_model / shard * float_size)

    # After X all-gather, router/expert matmuls conceptually consume x̄ of size E/tp_yz.
    gathered_x = tokens_local * d_model / moe_tp_size_yz * float_size
    # We treat the gathered tensor as a transient buffer produced by comm and consumed by matmuls.
    traffic.activations.bytes_written += int(gathered_x)
    traffic.activations.bytes_read += int(gathered_x)

    # Router scores (logits) buffer: [tokens, n_experts]
    # (we ignore softmax/top-k compute details, but keep the major activation buffer explicit)
    total_num_routed_experts = transformer_config.num_regular_experts
    if total_num_routed_experts > 0:
        router_scores = tokens_local * total_num_routed_experts * float_size
        traffic.activations.bytes_written += int(router_scores)
        traffic.activations.bytes_read += int(router_scores)
        if moe_config.router_sync_mode == MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER:
            if not isinstance(workload_spec, InferenceDecodeSpec):
                raise ValueError(
                    "normalization_deferred_sharded_router is only valid for inference decode"
                )
            # The P portion reuses the router-score buffer above. Charge the
            # two additional packed fields per token for sum and sum-of-squares.
            packed_normalization_stats = 2 * tokens_local * float_size
            traffic.activations.bytes_written += int(packed_normalization_stats)
            traffic.activations.bytes_read += int(packed_normalization_stats)

        # Decode router mode can change router-weight reads over the attention TP domain.
        # Router weight matrix is approximated as [d_model, n_experts].
        traffic.weights.bytes_read += _decode_router_weight_read_bytes(
            workload_spec=workload_spec,
            moe_config=moe_config,
            d_model=d_model,
            total_num_routed_experts=total_num_routed_experts,
            float_size=float_size,
        )

    # Mirror 1D MoE token-traffic logic for routed experts, adapted to 2D sharding.
    num_colocated = moe_config.num_colocated_experts

    if total_num_routed_experts > 0:
        tokens_per_set = (
            workload_spec.get_resolved_batches().batch_size_per_full_expert_set * step_seq
        )
        traffic_mult = moe_config.regular_expert_traffic_multiplier
        tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
        tokens_processed = tokens_per_routed_expert * traffic_mult

        # Expert input activations are model-dimension sharded on YZ in TP-2D.
        # Each rank reads only its local E/TP_MoE_Y feature shard.
        expert_input_sharded = tokens_processed * (d_model / moe_tp_size_yz) * float_size

        # Permute/unpermute for grouped expert GEMMs (c > 1) on the
        # rank's local buffer [tokens, E/YZ]. Regular GEMM (c = 1)
        # has one segment; no sort.
        if num_colocated > 1:
            permute_leg_bytes = num_colocated * expert_input_sharded
            # ×2 = permute (dispatch) + unpermute (combine). Each leg
            # copies the full local slab (one read and one write).
            traffic.activations.bytes_read += 2 * permute_leg_bytes
            traffic.activations.bytes_written += 2 * permute_leg_bytes

        traffic.activations.bytes_read += num_colocated * expert_input_sharded

        # FFN1 output: size [tokens_processed, H/X] where H=d_ff and X=moe_tp_size_x.
        ff1_out = tokens_processed * (d_ff / moe_tp_size_x) * float_size
        traffic.activations.bytes_written += num_colocated * int(ff1_out)
        # Read FFN1 output for FFN2.
        traffic.activations.bytes_read += num_colocated * int(ff1_out)

        # FFN2 (down-proj) produces a partial output of size [tokens_processed, E/YZ].
        ffn2_partial = tokens_processed * (d_model / moe_tp_size_yz) * float_size
        traffic.activations.bytes_written += num_colocated * int(ffn2_partial)
        # Read partial output for X reduce-scatter / combine.
        traffic.activations.bytes_read += num_colocated * int(ffn2_partial)

        # Final sharded output after X reduce-scatter: [tokens_processed, E/(X*YZ)].
        ffn2_out = tokens_processed * (d_model / shard) * float_size
        traffic.activations.bytes_written += num_colocated * int(ffn2_out)


def _backward_attn_tp_1d_memory_traffic(
    name: str,
    transformer_config: TransformerConfig,
    workload_spec,
    proj_tp_size: int,
    attn_tp_size: int,
    activation_checkpointing: bool,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """
    Compute backward-pass memory traffic for the attention sub-block under NONE/1D TP.

    Notes
    - K/V weight and activation terms are GQA-aware (d_model x (num_kv_heads * head_dim));
      Q/O terms are wide-Q-aware (d_model x q_proj_out_dim). For MHA models
      (num_kv_heads == num_heads, q_proj_out_dim == d_model) totals collapse to
      d_model x d_model.
    - Sizes are derived from `transformer_config` and `workload_spec` to avoid duplicating
      arguments upstream.
    - Domain split mirrors the forward 1D path: projection-owned terms (weights, weight
      gradients, QKV projection outputs, out-proj input/output activations) shard by
      `proj_tp_size`; the attention-core score re-read shards by `attn_tp_size`. The two
      domains differ under projection groups (g > 1), where the projection view is
      g-times smaller.

    Args:
        name: A descriptive name for the traffic accumulator instance.
        transformer_config: Model transformer configuration (provides `d_model`, `num_heads`,
            `num_kv_heads`, `head_dim`).
        workload_spec: Workload spec used to infer `batch_size_per_dp`, `seq_length`, `float_size_bytes`.
        proj_tp_size: Size of the projection TP domain (1 for NONE, >1 for 1D shard).
        attn_tp_size: Size of the attention TP domain (1 for NONE, >1 for 1D shard).
        activation_checkpointing: Whether activation checkpointing is enabled.
        device_for: Mapping of memory components to device tiers.
        node: Hardware node definition (provides memory tier details).

    Returns:
        LayerMemoryTraffic: Accumulated attention backward memory traffic for a single layer.
    """
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim
    heads_per_attn = num_heads // attn_tp_size
    # K/V heads per projection TP rank: handles the KV-replication regime
    # (proj_tp_size > num_kv_heads) the same way the forward 1D projection path does.
    num_kv_heads_per_proj, _kv_rep = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=proj_tp_size
    )
    # Resolve batch/sequence and dtype size from workload config
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    # Read attention weights (Q, K/V, Out) and write corresponding gradients
    # (sharded by the projection TP domain — these are the projection weights).
    # GQA-aware K/V sizing matches the forward 1D path
    # (`_forward_pass_prefill_attention_projection_tp_1d_memory_traffic`) and the
    # 2D backward path (`_backward_attn_tp_2d_memory_traffic`). Q uses
    # q_proj_out_dim = num_heads · head_dim to support wide-Q layouts
    # (MiniMax-M2.5, gpt-oss-120b, qwen3 series); reduces to d_model when
    # num_heads · head_dim == d_model.
    q_weight_size = d_model * q_proj_out_dim
    kv_weight_size = 2 * (d_model * num_kv_heads * head_dim)
    attn_out_weight_size = q_proj_out_dim * d_model
    total_w = ((q_weight_size + kv_weight_size + attn_out_weight_size) * float_size) / proj_tp_size
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w
    if not activation_checkpointing:
        # Read QKV projection outputs (GQA-aware; mirrors q_out_sharded + kv_out_sharded
        # in the forward 1D path, sharded by the projection TP domain).
        q_proj_output = n_batch * n_seq * q_proj_out_dim * float_size / proj_tp_size
        kv_proj_output = 2 * (num_kv_heads_per_proj * head_dim) * n_batch * n_seq * float_size
        traffic.activations.bytes_read += q_proj_output + kv_proj_output
        # Attention scores re-read (attention-core buffer, sharded by the attention
        # TP domain): sliding-window layers have a score region clipped to
        # n_seq × min(window, n_seq); blended per class.
        for layer_class in attention_layer_classes(transformer_config):
            traffic.activations.bytes_read += (
                layer_class.layer_fraction
                * n_batch
                * heads_per_attn
                * n_seq
                * layer_class.kv_len(n_seq)
                * float_size
            )
        # Read context (input to output proj): dim = q_proj_out_dim/proj_tp.
        traffic.activations.bytes_read += (
            n_batch * n_seq * q_proj_out_dim * float_size / proj_tp_size
        )
    # Gradient write for the post-output-proj activation: dim = d_model/proj_tp.
    traffic.gradients.bytes_written += n_batch * n_seq * d_model * float_size / proj_tp_size
    return traffic


def _backward_attn_tp_2d_memory_traffic(
    *,
    name: str,
    transformer_config: TransformerConfig,
    workload_spec,
    proj_tp_size_x: int,
    proj_tp_size_yz: int,
    attn_tp_size_x: int,
    attn_tp_size_yz: int,
    activation_checkpointing: bool,
    device_for: dict[str, str],
    node,
) -> LayerMemoryTraffic:
    """Backward attention traffic under TP-2D.

    Domain split mirrors the forward 2D path: projection-owned terms (weights,
    weight gradients, QKV projection outputs, out-proj input/output activations)
    shard across the *projection* X*YZ grid; the attention-core score re-read
    shards by the *attention* X (heads). The two domains differ under projection
    groups (g > 1), where the projection view is g-times smaller.
    """
    traffic = LayerMemoryTraffic(name=name, device_for=device_for, node=node)
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes

    # Weights (QKV + Out), sharded across the projection 2D grid. GQA-aware sizing.
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim
    q_w = d_model * q_proj_out_dim * float_size
    kv_w = 2 * (d_model * transformer_config.num_kv_heads * head_dim) * float_size
    out_w = q_proj_out_dim * d_model * float_size
    proj_shard = proj_tp_size_x * proj_tp_size_yz
    total_w = (q_w + kv_w + out_w) // proj_shard
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w

    if not activation_checkpointing:
        heads_per_x = num_heads // attn_tp_size_x
        # Read QKV proj output, attention scores, and context/output intermediates (GQA-aware QKV size).
        qkv_out_elems = q_proj_out_dim + 2 * (transformer_config.num_kv_heads * head_dim)
        qkv_proj_output = n_batch * n_seq * qkv_out_elems / proj_shard * float_size
        # Sliding-window layers clip the score region to
        # n_seq × min(window, n_seq); blended per layer class.
        attn_scores = 0.0
        for layer_class in attention_layer_classes(transformer_config):
            attn_scores += (
                layer_class.layer_fraction
                * n_batch
                * heads_per_x
                * n_seq
                * layer_class.kv_len(n_seq)
                * float_size
            )
        # `final_proj_in` is the input to the output projection — the
        # post-attention activation, dim = q_proj_out_dim (NOT d_model
        # under wide-Q layouts).
        final_proj_in = n_batch * n_seq * q_proj_out_dim / proj_shard * float_size
        traffic.activations.bytes_read += int(qkv_proj_output + attn_scores + final_proj_in)

    # Gradient write for final output (post-output-proj activation, dim = d_model,
    # sharded across the projection X*YZ grid).
    traffic.gradients.bytes_written += n_batch * n_seq * d_model / proj_shard * float_size
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
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    n_seq = workload_spec.seq_length
    float_size = workload_spec.float_size_bytes
    total_num_routed_experts = transformer_config.num_regular_experts
    tokens_per_set = workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    traffic_mult = moe_config.regular_expert_traffic_multiplier

    moe_tp_size = moe_tp_topology.num_chips if moe_tp_topology else 1

    # Under RESIDENT_ACTIVE_SUBSET, the resident-weight read term is scaled by
    # active_expert_fraction (model tiny-batch selective activation). Gradient
    # writes match the read pattern — only experts that received tokens have
    # gradients to materialize — so the same fraction is applied symmetrically.
    num_colocated = moe_config.num_colocated_experts
    expert_weight_residency_fraction = _expert_weight_residency_fraction(
        workload_spec=workload_spec,
        transformer_config=transformer_config,
        tokens_per_set=tokens_per_set,
    )
    total_w = (
        num_colocated
        * transformer_config.expert_params_per_expert
        * float_size
        / moe_tp_size
        * expert_weight_residency_fraction
    )
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

    # Mirror 1D token-volume logic for routed experts, adapted to 2D sharding
    num_colocated = moe_config.num_colocated_experts
    total_num_routed_experts = transformer_config.num_regular_experts

    tokens_per_set = workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
    traffic_mult = moe_config.regular_expert_traffic_multiplier

    # Weights sharded across 2D TP (FFN only here).
    # Per-expert FFN weight bytes scale with mlp_num_weight_matrices (2 for
    # DENSE_2MATRIX, 3 for GATED_3MATRIX), routed via expert_params_per_expert.
    # Under RESIDENT_ACTIVE_SUBSET, scale by active_expert_fraction symmetrically
    # on read and gradient-write (gradients exist only for experts that fired).
    expert_weight_residency_fraction = _expert_weight_residency_fraction(
        workload_spec=workload_spec,
        transformer_config=transformer_config,
        tokens_per_set=tokens_per_set,
    )
    total_w = (
        transformer_config.expert_params_per_expert
        * float_size
        / (moe_tp_size_x * moe_tp_size_yz)
        * expert_weight_residency_fraction
    )
    traffic.weights.bytes_read += total_w
    traffic.gradients.bytes_written += total_w

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
            ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / moe_tp_size_yz
            traffic.activations.bytes_read += num_colocated * ff1_out_sharded_per_expert
            traffic.activations.bytes_read += num_colocated * token_vol_per_expert

    # Gradient writes for routed experts
    if total_num_routed_experts > 0:
        tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
        token_vol_per_expert = tokens_per_routed_expert * traffic_mult * d_model * float_size
        ff1_out_sharded_per_expert = (mlp_ratio * token_vol_per_expert) / moe_tp_size_yz
        traffic.gradients.bytes_written += num_colocated * token_vol_per_expert
        traffic.gradients.bytes_written += num_colocated * ff1_out_sharded_per_expert
    return traffic


def get_transformer_mem_stats_per_layer(
    attn_tp_topology,
    moe_tp_topology,
    transformer_config: TransformerConfig,
    moe_config,
    n_batch,
    n_seq,
    attn_tp_variant: TensorParallelVariant,
    moe_tp_variant: TensorParallelVariant,
    attn_x_dims: list[int] | None,
    attn_yz_dims: list[int] | None,
    moe_x_dims: list[int] | None,
    moe_yz_dims: list[int] | None,
    pass_mode,
    float_size,
    *,
    proj_tp_topology,
    proj_x_dims: list[int] | None,
    proj_yz_dims: list[int] | None,
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

    if (
        attn_tp_variant == TensorParallelVariant.SHARD_ACTIVATIONS
        or moe_tp_variant == TensorParallelVariant.SHARD_ACTIVATIONS
    ):
        raise NotImplementedError("Activation stationary tensor parallelism not implemented")

    def _prod_from_shape(shape, indices):
        p = 1
        for idx in indices:
            p *= shape[idx]
        return p

    # Resolve TP sizes for projection, attention, and MoE independently.
    # The caller resolves the projection TP domain (run_scalescope reuses the
    # attention view when the spec has no explicit TP_Proj); no fallback here.
    if proj_tp_topology is None:
        raise ValueError(
            "get_transformer_mem_stats_per_layer requires a resolved proj_tp_topology; "
            "the caller must pass the projection TP view explicitly."
        )

    attn_shape = getattr(attn_tp_topology, "shape", [])
    proj_shape = getattr(proj_tp_topology, "shape", [])
    moe_shape = getattr(moe_tp_topology, "shape", [])

    # Projection TP sizes (for attention projections only)
    if attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        tp_size_x_proj = _prod_from_shape(proj_shape, proj_x_dims or []) if proj_x_dims else 1
        tp_size_yz_proj = _prod_from_shape(proj_shape, proj_yz_dims or []) if proj_yz_dims else 1
        tp_size_proj = tp_size_x_proj * tp_size_yz_proj
        assert (
            tp_size_proj == proj_tp_topology.num_chips
        ), f"Proj TP size mismatch: {tp_size_proj} != {proj_tp_topology.num_chips}"
    else:
        tp_size_proj = proj_tp_topology.num_chips
        tp_size_x_proj = None
        tp_size_yz_proj = None

    if attn_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        tp_size_x_attn = _prod_from_shape(attn_shape, attn_x_dims or []) if attn_x_dims else 1
        tp_size_yz_attn = _prod_from_shape(attn_shape, attn_yz_dims or []) if attn_yz_dims else 1
        tp_size_attn = tp_size_x_attn * tp_size_yz_attn
        assert (
            tp_size_attn == attn_tp_topology.num_chips
        ), f"Attn TP size mismatch: {tp_size_attn} != {attn_tp_topology.num_chips}"
    else:
        tp_size_attn = attn_tp_topology.num_chips if attn_tp_topology else 1
        tp_size_x_attn = None
        tp_size_yz_attn = None

    if moe_tp_variant == TensorParallelVariant.SHARD_WEIGHTS_2D:
        tp_size_x_moe = _prod_from_shape(moe_shape, moe_x_dims or []) if moe_x_dims else 1
        tp_size_yz_moe = _prod_from_shape(moe_shape, moe_yz_dims or []) if moe_yz_dims else 1
        tp_size_moe = tp_size_x_moe * tp_size_yz_moe
        assert (
            tp_size_moe == moe_tp_topology.num_chips
        ), f"MoE TP size mismatch: {tp_size_moe} != {moe_tp_topology.num_chips}"
    else:
        tp_size_moe = moe_tp_topology.num_chips if moe_tp_topology else 1
        tp_size_x_moe = None
        tp_size_yz_moe = None

    if device_for is None:
        raise ValueError(
            "device_for mapping must be provided to get_transformer_mem_stats_per_layer"
        )
    # Forward traffic: compute attention and MoE separately to allow different TP variants.
    fwd_traffic_per_layer = LayerMemoryTraffic(
        name="fwd_per_layer", device_for=device_for, node=node
    )

    # Attention forward
    match attn_tp_variant:
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            match pass_mode:
                case PassMode.TRAINING:
                    _forward_pass_prefill_attention_tp_1d_memory_traffic(
                        fwd_traffic_per_layer,
                        transformer_config,
                        n_seq,
                        proj_tp_size=tp_size_proj,
                        attn_tp_size=tp_size_attn,
                        save_kv_cache=False,
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_PREFILL:
                    _forward_pass_prefill_attention_tp_1d_memory_traffic(
                        fwd_traffic_per_layer,
                        transformer_config,
                        n_seq,
                        proj_tp_size=tp_size_proj,
                        attn_tp_size=tp_size_attn,
                        save_kv_cache=True,
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_DECODE:
                    _decode_attention_tp_1d_memory_traffic(
                        traffic=fwd_traffic_per_layer,
                        transformer_config=transformer_config,
                        n_seq_so_far=n_seq,
                        proj_tp_size=tp_size_proj,
                        attn_tp_size=tp_size_attn,
                        workload_spec=workload_spec,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported pass_mode for attention forward traffic: {pass_mode}"
                    )
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            match pass_mode:
                case PassMode.TRAINING:
                    _forward_pass_prefill_attention_tp_2d_memory_traffic(
                        fwd_traffic_per_layer,
                        transformer_config,
                        n_seq,
                        proj_tp_size_x=tp_size_x_proj,
                        proj_tp_size_yz=tp_size_yz_proj,
                        attn_tp_size_x=tp_size_x_attn,
                        attn_tp_size_yz=tp_size_yz_attn,
                        save_kv_cache=False,
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_PREFILL:
                    _forward_pass_prefill_attention_tp_2d_memory_traffic(
                        fwd_traffic_per_layer,
                        transformer_config,
                        n_seq,
                        proj_tp_size_x=tp_size_x_proj,
                        proj_tp_size_yz=tp_size_yz_proj,
                        attn_tp_size_x=tp_size_x_attn,
                        attn_tp_size_yz=tp_size_yz_attn,
                        save_kv_cache=True,
                        workload_spec=workload_spec,
                    )
                case PassMode.INFERENCE_DECODE:
                    _decode_attention_tp_2d_memory_traffic(
                        fwd_traffic_per_layer,
                        transformer_config,
                        n_seq_so_far=n_seq,
                        proj_tp_size_x=tp_size_x_proj,
                        proj_tp_size_yz=tp_size_yz_proj,
                        attn_tp_size_x=tp_size_x_attn,
                        attn_tp_size_yz=tp_size_yz_attn,
                        workload_spec=workload_spec,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported pass_mode for TP-2D attn forward traffic: {pass_mode}"
                    )
        case _:
            raise ValueError(f"Unsupported attention TP variant: {attn_tp_variant}")

    # MoE/FFN forward (note: for decode, we model it as n_seq=1 internally by callers)
    if pass_mode == PassMode.INFERENCE_DECODE:
        moe_seq = 1
    else:
        moe_seq = n_seq

    match moe_tp_variant:
        case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
            _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic(
                fwd_traffic_per_layer,
                transformer_config,
                moe_seq,
                moe_config,
                moe_tp_size=tp_size_moe,
                workload_spec=workload_spec,
            )
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            _forward_pass_prefill_moe_ffn_tp_2d_memory_traffic(
                fwd_traffic_per_layer,
                transformer_config,
                moe_seq,
                moe_tp_size_x=tp_size_x_moe,
                moe_tp_size_yz=tp_size_yz_moe,
                workload_spec=workload_spec,
                moe_config=moe_config,
            )
        case _:
            raise ValueError(f"Unsupported MoE TP variant: {moe_tp_variant}")

    if pass_mode == PassMode.TRAINING:
        # Backward traffic: attention and MoE separately for independent variants.
        bwd_traffic_per_layer = LayerMemoryTraffic(
            name="bwd_per_layer", device_for=device_for, node=node
        )

        if activation_checkpointing:
            # If checkpointing, we recompute the forward pass and add it to backward traffic.
            fwd_recompute = LayerMemoryTraffic(
                name="fwd_recompute", device_for=device_for, node=node
            )
            # Reuse the same logic as forward, but force TRAINING mode.
            if attn_tp_variant in (
                TensorParallelVariant.NONE,
                TensorParallelVariant.SHARD_WEIGHTS_1D,
            ):
                _forward_pass_prefill_attention_tp_1d_memory_traffic(
                    fwd_recompute,
                    transformer_config,
                    n_seq,
                    proj_tp_size=tp_size_proj,
                    attn_tp_size=tp_size_attn,
                    save_kv_cache=False,
                    workload_spec=workload_spec,
                )
            else:
                _forward_pass_prefill_attention_tp_2d_memory_traffic(
                    fwd_recompute,
                    transformer_config,
                    n_seq,
                    proj_tp_size_x=tp_size_x_proj,
                    proj_tp_size_yz=tp_size_yz_proj,
                    attn_tp_size_x=tp_size_x_attn,
                    attn_tp_size_yz=tp_size_yz_attn,
                    save_kv_cache=False,
                    workload_spec=workload_spec,
                )
            if moe_tp_variant in (
                TensorParallelVariant.NONE,
                TensorParallelVariant.SHARD_WEIGHTS_1D,
            ):
                _forward_pass_prefill_moe_ffn_tp_1d_memory_traffic(
                    fwd_recompute,
                    transformer_config,
                    n_seq,
                    moe_config,
                    moe_tp_size=tp_size_moe,
                    workload_spec=workload_spec,
                )
            else:
                _forward_pass_prefill_moe_ffn_tp_2d_memory_traffic(
                    fwd_recompute,
                    transformer_config,
                    n_seq,
                    moe_tp_size_x=tp_size_x_moe or 1,
                    moe_tp_size_yz=tp_size_yz_moe or 1,
                    workload_spec=workload_spec,
                    moe_config=moe_config,
                )
            bwd_traffic_per_layer.activations.bytes_read += (
                fwd_recompute.activations.bytes_read + fwd_recompute.activations.bytes_written
            )
            bwd_traffic_per_layer.weights.bytes_read += fwd_recompute.weights.bytes_read

        # Attention backward
        match attn_tp_variant:
            case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
                bwd_traffic_per_layer += _backward_attn_tp_1d_memory_traffic(
                    name="attn_bwd",
                    transformer_config=transformer_config,
                    workload_spec=workload_spec,
                    proj_tp_size=tp_size_proj,
                    attn_tp_size=tp_size_attn,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case TensorParallelVariant.SHARD_WEIGHTS_2D:
                bwd_traffic_per_layer += _backward_attn_tp_2d_memory_traffic(
                    name="attn_bwd_2d",
                    transformer_config=transformer_config,
                    workload_spec=workload_spec,
                    proj_tp_size_x=tp_size_x_proj,
                    proj_tp_size_yz=tp_size_yz_proj,
                    attn_tp_size_x=tp_size_x_attn,
                    attn_tp_size_yz=tp_size_yz_attn,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case _:
                raise ValueError(f"Unsupported attention TP variant in backward: {attn_tp_variant}")

        # MoE backward
        match moe_tp_variant:
            case TensorParallelVariant.NONE | TensorParallelVariant.SHARD_WEIGHTS_1D:
                bwd_traffic_per_layer += _backward_moe_tp_1d_memory_traffic(
                    name="moe_bwd",
                    transformer_config=transformer_config,
                    moe_config=moe_config,
                    moe_tp_topology=moe_tp_topology,
                    workload_spec=workload_spec,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case TensorParallelVariant.SHARD_WEIGHTS_2D:
                bwd_traffic_per_layer += _backward_moe_tp_2d_memory_traffic(
                    name="moe_bwd_2d",
                    transformer_config=transformer_config,
                    moe_config=moe_config,
                    workload_spec=workload_spec,
                    moe_tp_size_x=tp_size_x_moe,
                    moe_tp_size_yz=tp_size_yz_moe,
                    activation_checkpointing=activation_checkpointing,
                    device_for=device_for,
                    node=node,
                )
            case _:
                raise ValueError(f"Unsupported MoE TP variant in backward: {moe_tp_variant}")

        # Add parameter update memory traffic (weights update reads/writes, grads read)
        param_update_traffic = _parameter_update_memory_traffic(
            name="param_update",
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            attn_tp_topology=attn_tp_topology,
            moe_tp_size=tp_size_moe,
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
    weights_fs = workload_spec.full_precision_param_float_size
    grads_fs = workload_spec.float_size_bytes
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
    opt_states_fs = workload_spec.full_precision_param_float_size
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
    weights_fs = workload_spec.full_precision_param_float_size
    grads_fs = workload_spec.float_size_bytes
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
    opt_states_fs = workload_spec.full_precision_param_float_size
    opt_states_read = 2 * params_per_xpu * opt_states_fs
    opt_states_write = 2 * params_per_xpu * opt_states_fs
    traffic.weights.bytes_read += int(opt_states_read)
    traffic.weights.bytes_written += int(opt_states_write)
    return traffic
