"""Workload specification and model placement.

Describes how a transformer model is placed onto a cluster by specifying
parallelization strategies (TP/DP/PP and MoE/EP-related options), together
with training and inference parameters (e.g., global and micro batch sizes,
sequence length, numeric precision/efficiency knobs).
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Type, TypeVar

import yaml

# Pydantic imports
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)

from scalescope.utils import parse_enum_by_name_or_value, StrEnum


class TensorParallelVariant(StrEnum):
    NONE = "None"
    SHARD_WEIGHTS_1D = "1D weight stationary (Megatron)"
    SHARD_WEIGHTS_2D = "2D weight stationary"
    SHARD_ACTIVATIONS = "Activation stationary"


class DataParallelVariant(StrEnum):
    NONE = "None"
    DDP = "DDP"
    ZERO_STAGE_1 = "ZeRO stage I"
    ZERO_STAGE_2 = "ZeRO stage II"
    ZERO_STAGE_3 = "ZeRO stage III"


class PipelineParallelVariant(StrEnum):
    NONE = "None"
    GPIPE = "GPIPE"
    INTERLEAVED_1F1B = "INTERLEAVED_1F1B"


class ActivationStorageOptimizationStrategy(StrEnum):
    NONE = "none"
    RECOMPUTATION = "recomputation"
    GATHER = "gather"
    GATHER_RECOMPUTATION = "gather_recomputation"


class MoERouterSyncMode(StrEnum):
    REPLICATED_ROUTER = "replicated_router"
    SHARDED_ROUTER = "sharded_router"


class MoEA2ACommMode(StrEnum):
    HIERARCHICAL_SPLIT = "hierarchical_split"
    FUSED_ND = "fused_nd"


class ExpertWeightResidencyPolicy(StrEnum):
    """Policy for how MoE expert weights are accounted for in memory traffic.

    Selects how the resident-weight read term in `memory_model.py` is billed.
    Routed-token traffic (compute FLOPs, EP A2A volume, TP MoE collectives) is
    already scaled by `rho_eff * top_k` via `regular_expert_traffic_multiplier`
    and is unaffected by this policy.

    Values:
    - `resident_full_set`: every colocated expert weight is read every step
      (today's behavior). Appropriate for training and high-concurrency decode.
    - `resident_active_subset`: weight-read term is scaled by
      `active_expert_fraction = E_active / num_colocated_experts`, modeling that
      at small batch only a subset of the chip's experts is touched per step. See
      `ep_strategy.compute_expert_activation_cardinality` for the formula.

    Scope is intentionally *expert* weights only — attention QKV / output
    projections, embeddings, and LM head weights are unaffected.
    """

    RESIDENT_FULL_SET = "resident_full_set"
    RESIDENT_ACTIVE_SUBSET = "resident_active_subset"


class ExpertRoutingDistribution(StrEnum):
    """Shape of the per-expert routing probability distribution used to compute
    the per-chip weight-read fraction under `RESIDENT_ACTIVE_SUBSET`.

    The formula returns the **per-chip** weight-traffic multiplier (what
    fraction of a chip's resident expert weights are read in the step).
    Step time is set by the worst-loaded chip, not the average.

    Values:
    - `bottleneck_group_collector`: per-chip multiplier evaluated on the
      worst-loaded colocated group, computed skew-aware over the global Zipf:
          p_k      = k^{-s} / H_{E,s},  s solved from E / H_{E,s} = rho
          E_active = max(1, sum_{k=1..c} (1 - (1 - p_k)^m))
          frac     = E_active / c
      where `c = num_colocated_experts`, `m = batch * top_k` is the per-set
      routing-event count, and the worst chip is modeled (worst case) as
      hosting the `c` hottest experts. Because the memory model reads
      `resident * frac = c·(w/TP_MoE)·(E_active/c) = E_active·(w/TP_MoE)`, the
      per-shard size `w/TP_MoE` is explicit: 2D TP_MoE (finer sharding) shrinks
      an outlier expert's bottleneck while `E_active` stays ≈ 1. Correct
      asymptote (→ 1 as `m → ∞`), correct `c = 1` semantics (always 1.0, the
      chip's one expert must load if touched), and the `c = E` corner is the
      full Zipf active count — within-chip skew is preserved (no collapse to
      uniform). This is the physically-grounded choice for step-time modeling.

      Modeling assumption: worst-case placement (the `c` hottest experts
      colocated) is a deliberate upper bound on the realized worst chip; under
      random placement the worst chip is "hottest expert + (c-1) average
      experts," numerically close for steep Zipf.

    - `bottleneck_group_collector_random`: the expected (rather than worst-case)
      straggler model. Under load-agnostic (random/hash) placement the worst-loaded
      chip holds the single hottest expert (rank 1) plus `c-1` experts drawn
      uniformly from the remaining `E-1`, so the fillers contribute the cold-tail
      average rather than the next-hottest experts:
          a_k      = 1 - (1 - p_k)^m              # P(expert k hit by >= 1 of m events)
          E_active = a_1 + (c - 1) * mean_{k=2..E} a_k
          frac     = E_active / c
      Using the tail average `mean_{k=2..E} a_k` for the fillers (instead of the
      next-hottest `a_2..a_c` of the worst-case model) makes `frac` strictly
      smaller in the interior `1 < c < E`: the straggler reads fewer distinct
      experts. The model is corner-exact against the worst-case bound:
      `c = 1` → `a_1`; `c = E` → `sum_{k=1..E} a_k` (the full Zipf active count);
      `rho = 1` → plain coupon collector. It carries the same `w/TP_MoE` shard
      term, so 2D `TP_MoE` keeps shrinking an outlier expert's bottleneck
      (`E_active → 1` for steep Zipf at small `m`).

      No floor on `E_active` (the key difference from the worst-case model).
      The worst-case variant takes `E_active = max(1, ...)`: it is an *upper
      bound*, and a bound on the worst chip that is assumed to host the hottest
      expert should never claim fewer than that one expert's weight read. This
      model is instead an *expectation* over placements and routing, so a value
      below 1.0 is meaningful: at small `m` each routing event touches only a few
      of the `E` experts, so `a_1 = 1 - (1 - p_1)^m ≈ m·p_1` can be well under 1,
      and `E_active` (the expected distinct-active count on the straggler) is
      genuinely fractional — the straggler reads less than one full expert's
      weights per step on average. Flooring at 1.0 would force every step to pay a
      full per-expert HBM read even in that regime, systematically over-counting
      weight traffic and inflating decode step time at small batch / low top-k.
      Leaving it unfloored preserves the correct small-`m` asymptote (linear in
      `m`) while still rising to the full Zipf active count as `m` grows.
    """

    BOTTLENECK_GROUP_COLLECTOR = "bottleneck_group_collector"
    BOTTLENECK_GROUP_COLLECTOR_RANDOM = "bottleneck_group_collector_random"


# Pydantic Spec Classes


class TensorParallelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    variant: TensorParallelVariant = TensorParallelVariant.NONE

    # Validator for 'variant'
    _parse_variant = field_validator("variant", mode="before")(
        parse_enum_by_name_or_value(TensorParallelVariant)
    )


class TP2DSplitSpec(BaseModel):
    """
    Explicit 2D-TP split specification.

    We keep the nomenclature X vs YZ:
    - X shards heads (Q/K/V heads)
    - YZ shards sequence/KV-cache length (context/sequence parallelism)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    x_tags: list[str] = Field(
        ..., min_length=1, description="Logical TP dimension tags that form the X axis"
    )
    yz_tags: list[str] = Field(
        ..., min_length=1, description="Logical TP dimension tags that form the YZ axis"
    )

    @model_validator(mode="after")
    def _validate_disjoint(self) -> "TP2DSplitSpec":
        x = set(self.x_tags)
        yz = set(self.yz_tags)
        inter = x.intersection(yz)
        if inter:
            raise ValueError(
                f"TP2DSplitSpec x_tags and yz_tags must be disjoint; overlap={sorted(inter)}"
            )
        return self


class TensorParallelSpecV2(BaseModel):
    """
    Tensor-parallel specification with optional explicit 2D split.

    This exists to allow Attn and MoE to have different TP variants and (for 2D) different X/YZ splits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    variant: TensorParallelVariant = TensorParallelVariant.NONE
    split_2d: TP2DSplitSpec | None = Field(
        default=None, description="Required when variant is SHARD_WEIGHTS_2D"
    )

    _parse_variant = field_validator("variant", mode="before")(
        parse_enum_by_name_or_value(TensorParallelVariant)
    )

    @model_validator(mode="after")
    def _validate_2d_requires_split(self) -> "TensorParallelSpecV2":
        if self.variant == TensorParallelVariant.SHARD_WEIGHTS_2D and self.split_2d is None:
            raise ValueError(
                "TensorParallelSpecV2.split_2d must be set when variant is SHARD_WEIGHTS_2D"
            )
        return self


class DataParallelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    variant: DataParallelVariant = DataParallelVariant.NONE

    # Validator for 'variant'
    _parse_variant = field_validator("variant", mode="before")(
        parse_enum_by_name_or_value(DataParallelVariant)
    )


class MoESpecV2(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Required
    expert_imbalance: float = Field(..., ge=1.0)
    expert_weight_residency_policy: ExpertWeightResidencyPolicy = Field(
        ...,
        description=(
            "How MoE expert-weight memory traffic is billed. "
            "'resident_full_set' charges every colocated expert weight per step (training and "
            "high-concurrency decode); 'resident_active_subset' scales the weight-read term by "
            "the expected fraction of experts actually touched in the step "
            "(modeling tiny-batch selective activation). Scope is expert weights only; attention, "
            "embeddings, and LM-head weights are unaffected."
        ),
    )
    expert_routing_distribution: ExpertRoutingDistribution = Field(
        ...,
        description=(
            "Shape of the per-expert routing probability distribution used to compute "
            "the expected fraction of experts active per step. "
            "'bottleneck_group_collector' evaluates the worst-loaded colocated group "
            "skew-aware over the global Zipf solved from rho (correct large-batch "
            "asymptote and correct c=1 semantics). Only consulted when "
            "expert_weight_residency_policy = resident_active_subset."
        ),
    )
    # Optional
    num_colocated_experts: int = Field(1, ge=1, description="Number of colocated experts")
    shared_expert_copies: Optional[int] = Field(
        default=None,
        ge=1,
        description="Replication factor per shared expert; if None, may be inferred later",
    )

    _parse_expert_weight_residency_policy = field_validator(
        "expert_weight_residency_policy", mode="before"
    )(parse_enum_by_name_or_value(ExpertWeightResidencyPolicy))
    _parse_expert_routing_distribution = field_validator(
        "expert_routing_distribution", mode="before"
    )(parse_enum_by_name_or_value(ExpertRoutingDistribution))


class PipelineParallelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    variant: PipelineParallelVariant = PipelineParallelVariant.NONE

    # Validator for 'variant'
    _parse_variant = field_validator("variant", mode="before")(
        parse_enum_by_name_or_value(PipelineParallelVariant)
    )

    @model_validator(mode="before")
    @classmethod
    def convert_variant_to_enum(cls, data):
        if isinstance(data, dict) and "variant" in data and isinstance(data["variant"], str):
            data["variant"] = PipelineParallelVariant(data["variant"].upper())
        return data


T_WorkloadSpec = TypeVar("T_WorkloadSpec", bound="WorkloadSpec")


class WorkloadSpec(BaseModel):
    """
    Base specification for a model workload, defining the execution environment
    and core model parameters like context length. This is IMMUTABLE.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Tensor-parallel specifications are explicitly separated by block type.
    # We intentionally do NOT provide a global `tp_spec` default, because there is no single
    # "good default" across workloads (and implicit fallbacks can be confusing).
    tp_attn_spec: TensorParallelSpecV2
    # Optional V2 override for projection TP (QKV + output projection). If omitted, code falls back
    # to `tp_attn_spec`.
    tp_proj_spec: TensorParallelSpecV2 | None = None
    tp_moe_spec: TensorParallelSpecV2
    dp_spec: DataParallelSpec = Field(default_factory=DataParallelSpec)
    # New primary spec
    moe_spec_v2: "MoESpecV2"
    pp_spec: PipelineParallelSpec = Field(default_factory=PipelineParallelSpec)
    # Required: global batch size per training/inference step for the whole cluster
    global_batch_size: int = Field(..., gt=0)
    # Floating-point size in bytes (e.g., 0.5 for FP4, 1 for FP8,
    # 2 for BF16/FP16, 4 for FP32). Float-typed to allow sub-byte
    # precisions; constrained to be strictly positive.
    float_size_bytes: float = Field(..., gt=0)
    # Memory device mappings (required):
    # - traffic components (LayerMemoryTraffic)
    memory_device_map_traffic: Dict[str, str] = Field(
        ...,
        description="Mapping for traffic components to memory tiers (weights, activations, gradients, kv_cache)",
    )
    # - footprint components (MemoryFootprint)
    memory_device_map_footprint: Dict[str, str] = Field(
        ...,
        description="Mapping for footprint components to memory tiers (weights, optimizer_states, saved_activations, gradient_buffers, transient_forward_peak, transient_backward_peak)",
    )

    # Overlap/efficiency settings for time overlap model
    comp_comm_efficiency: float = Field(..., ge=0.0, le=1.0)
    comp_mem_efficiency: float = Field(..., ge=0.0, le=1.0)
    comm_mem_efficiency: float = Field(..., ge=0.0, le=1.0)
    kernel_launch_latency_per_layer_sec: float = Field(
        ...,
        ge=0,
        description=(
            "Per-layer host kernel-launch/dispatch floor (seconds), added once per "
            "transformer layer per pass (TP-independent, not overlapped): per decode step in "
            "decode, the single forward pass in prefill, and each of the forward and backward "
            "passes in training. Empirical runtime/framework overhead, not a model-architecture "
            "or hardware property. Set 0.0 where this floor is not modeled."
        ),
    )
    # TP-MoE expert-input redistribution optimization toggle (mode-agnostic).
    tp_moe_a2a_replication_optimization: bool = Field(
        ...,
        description=(
            "Controls TP-MoE token redistribution strategy. "
            "When enabled, routed traffic is first placed on expert shards and then "
            "redistributed across expert TP shards via all-gather (lower routed A2A bytes, "
            "extra collective startup). "
            "When disabled, routed MoE all-to-all carries the direct non-optimized payload."
        ),
    )
    moe_a2a_comm_mode: MoEA2ACommMode = Field(
        default=MoEA2ACommMode.HIERARCHICAL_SPLIT,
        description=(
            "MoE all-to-all communication model: "
            "'hierarchical_split' uses the original intra/inter split logic; "
            "'fused_nd' uses a fused N-D total-exchange model."
        ),
    )

    # Optional: resolved batch sizes attached at runtime (immutable via copy)
    resolved_batches: Optional["ResolvedBatchSizes"] = None
    # Optional: layers per pipeline stage, attached at runtime (do not define in YAML)
    layers_per_pipeline_stage: Optional[int] = Field(default=None)

    _parse_moe_a2a_comm_mode = field_validator("moe_a2a_comm_mode", mode="before")(
        parse_enum_by_name_or_value(MoEA2ACommMode)
    )

    @classmethod
    def load_from_yaml(cls: Type[T_WorkloadSpec], filepath: str) -> T_WorkloadSpec:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise TypeError(f"YAML did not load as dict: {filepath}")
            base_config_dict = {k: v for k, v in config_dict.items() if k != "sweep"}
            model = cls.model_validate(base_config_dict)
            return model
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {filepath}: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Config validation failed: {filepath}:\n{e}") from e
        except Exception as e:
            raise ValueError(f"Error loading spec from: {filepath}: {e}") from e

    def infer_global_batch_size(self, dp_attn_chips: int) -> int:
        """Return the explicit global batch size (required field)."""
        return self.global_batch_size

    def infer_batch_size_per_dp(self, dp_attn_chips: int) -> int:
        """
        Returns per-DP batch size from global batch size and dp_attn_chips.
        Enforces divisibility of the global batch by dp_attn_chips.
        """
        gb = self.global_batch_size
        if gb % dp_attn_chips != 0:
            raise ValueError(f"Global batch size {gb} not divisible by DP chips {dp_attn_chips}.")
        return gb // dp_attn_chips

    def infer_batch_size_per_full_expert_set(
        self, dp_attn_chips: int, num_complete_expert_sets: int
    ) -> int:
        """
        Returns batch size allocated to each full expert set, derived from global batch size.
        Enforces divisibility by num_complete_expert_sets.
        """
        gb = self.infer_global_batch_size(dp_attn_chips)
        if num_complete_expert_sets <= 0:
            raise ValueError("num_complete_expert_sets must be >= 1")
        if gb % num_complete_expert_sets != 0:
            raise ValueError(
                f"Global batch size {gb} not divisible by num_complete_expert_sets {num_complete_expert_sets}."
            )
        return gb // num_complete_expert_sets

    def infer_effective_micro_batch_size(self, dp_attn_chips: int) -> int:
        """
        Returns effective micro-batch size. micro_batch_size is required.
        Validates that per-DP batch is divisible by micro_batch_size.
        """
        per_dp = self.infer_batch_size_per_dp(dp_attn_chips)
        mb = getattr(self, "micro_batch_size", None)
        if mb is None:
            raise ValueError("micro_batch_size must be set.")
        if per_dp % mb != 0:
            raise ValueError(f"Per-DP batch {per_dp} not divisible by micro_batch_size {mb}.")
        return mb

    def with_resolved_batches(
        self, dp_attn_chips: int, num_complete_expert_sets: int
    ) -> "WorkloadSpec":
        """
        Returns a new WorkloadSpec with resolved batch sizes attached (field 'resolved_batches').
        This preserves immutability of the original spec.
        """
        rb = resolve_batch_sizes_for_workload(self, dp_attn_chips, num_complete_expert_sets)
        return self.model_copy(update={"resolved_batches": rb})

    def get_resolved_batches(self) -> "ResolvedBatchSizes":
        if self.resolved_batches is None:
            raise ValueError(
                "Resolved batch sizes are not attached. Call with_resolved_batches(...) first."
            )
        return self.resolved_batches

    def with_layers_per_pp_stage(self, total_layers: int, pp_stages: int) -> "WorkloadSpec":
        """Attach layers_per_pipeline_stage computed from total model layers and PP stages (from logical topology)."""
        num_pp_stages = pp_stages
        if total_layers % num_pp_stages != 0:
            raise ValueError(
                f"n_layers ({total_layers}) must be divisible by pp stages ({num_pp_stages})"
            )
        layers = total_layers // num_pp_stages
        if layers <= 0:
            raise ValueError("layers_per_pipeline_stage must be > 0")
        return self.model_copy(update={"layers_per_pipeline_stage": int(layers)})

    def get_layers_per_pp_stage(self) -> int:
        layers = getattr(self, "layers_per_pipeline_stage", None)
        if layers is None:
            raise ValueError(
                "layers_per_pipeline_stage is not attached. Call with_layers_per_pp_stage(...) first."
            )
        return int(layers)

    # ---- TP helpers ----
    def get_attn_tp_spec(self) -> TensorParallelSpecV2:
        """Return the attention TP spec (V2)."""
        return self.tp_attn_spec

    def get_moe_tp_spec(self) -> TensorParallelSpecV2:
        """Return the MoE/FFN TP spec (V2)."""
        return self.tp_moe_spec

    def get_proj_tp_spec(self) -> TensorParallelSpecV2:
        """
        Return the projection TP spec (V2), falling back to the attention TP spec.

        If `tp_proj_spec` is omitted, projections run on the same TP domain/variant
        as attention.
        """
        if self.tp_proj_spec is not None:
            return self.tp_proj_spec
        return self.get_attn_tp_spec()


@dataclass(frozen=True)
class ResolvedBatchSizes:
    global_batch_size: int
    batch_size_per_dp: int
    effective_micro_batch_size: int
    batch_size_per_full_expert_set: int


def resolve_batch_sizes_for_workload(
    workload_spec: WorkloadSpec, dp_attn_chips: int, num_complete_expert_sets: int
) -> ResolvedBatchSizes:
    gb = workload_spec.infer_global_batch_size(dp_attn_chips)
    per_dp = workload_spec.infer_batch_size_per_dp(dp_attn_chips)
    eff_micro = workload_spec.infer_effective_micro_batch_size(dp_attn_chips)
    per_set = workload_spec.infer_batch_size_per_full_expert_set(
        dp_attn_chips, num_complete_expert_sets
    )
    return ResolvedBatchSizes(
        global_batch_size=gb,
        batch_size_per_dp=per_dp,
        effective_micro_batch_size=eff_micro,
        batch_size_per_full_expert_set=per_set,
    )


class TrainSpec(WorkloadSpec):
    """
    Specification for a training workload. This is IMMUTABLE.
    """

    global_batch_size: int = Field(..., gt=0)
    # Full-precision parameter float size (bytes) used to store weights during training (e.g., 4 for FP32)
    full_precision_param_float_size: int = Field(..., gt=0)
    # Optimizer used for training (e.g., 'adam')
    optimizer: str = Field(...)
    micro_batch_size: int = Field(..., ge=1)
    total_train_tokens: int = Field(..., ge=0)
    activation_storage_strategy: ActivationStorageOptimizationStrategy = Field(
        default=ActivationStorageOptimizationStrategy.NONE
    )

    seq_length: int = Field(..., gt=0, description="The context length to train the model.")

    @computed_field
    @property
    def tokens_per_global_step(self) -> int:
        return self.global_batch_size * self.seq_length

    @computed_field
    @property
    def effective_micro_batch_size(self) -> int:
        if self.micro_batch_size is not None:
            return self.micro_batch_size
        else:
            raise ValueError("micro_batch_size is not set. Please set it explicitly.")

    def calculate_number_of_train_steps(self):
        tokens_per_batch = self.global_batch_size * self.seq_length
        train_steps = self.total_train_tokens // tokens_per_batch
        return train_steps, tokens_per_batch


class InferencePrefillSpec(WorkloadSpec):
    """
    Specification for an analytical inference workload. This is IMMUTABLE.
    """

    micro_batch_size: int = Field(..., ge=1)

    @computed_field
    @property
    def effective_micro_batch_size(self) -> int:
        if self.micro_batch_size is not None:
            return self.micro_batch_size
        else:
            raise ValueError("micro_batch_size is not set. Please set it explicitly.")

    seq_length: int = Field(..., gt=0, description="The context length to run prefill.")


class InferenceDecodeSpec(WorkloadSpec):
    """
    Manages the state of a single analytical run based on token counts.
    It is configured with an immutable InferenceSpec and holds the mutable state.
    """

    # batch_size: int = Field(..., gt=0)
    # prefill_seq_len: int = Field(..., gt=0)
    num_generated_tokens: int = Field(
        ..., gt=0, description="Number of tokens to generate in the decode phase."
    )

    micro_batch_size: int = Field(..., ge=1)

    def __init__(self, **data: Any):
        super().__init__(**data)

    prefill_seq_len: int = Field(..., gt=0, description="The length of prefill.")
    # For decode mode, use seq_length=1 (per-step) unless specified otherwise
    seq_length: int = Field(
        default=1, gt=0, description="Sequence length per decode step (typically 1)"
    )
    projection_groups: int = Field(
        default=1,
        ge=1,
        description=(
            "Projection grouping factor `g` for the projection path (Q/K/V projections and output "
            "projection). The TP collective participant group used for projection work is split into "
            "`g` independent subgroups, each running its own smaller collective; per-chip projection "
            "compute and projection memory traffic grow by `g` because the work is effectively "
            "replicated across the subgroups, while collective latency shrinks because each subgroup "
            "has fewer participants. Attention-core sharding (and KV-cache footprint) is unaffected.\n"
            "\n"
            "Concept generalizes to both 1D and 2D TP. The current implementation only supports the "
            "TP-2D variant: it shrinks the *projection-domain* YZ degree by `g` (validated in "
            "`run_scalescope.py` — non-TP-2D configs raise on g > 1). Extending to 1D TP is a "
            "natural follow-up.\n"
            "\n"
            "Constraints: `g >= 1` (1 means no grouping). Under the TP-2D implementation, `g` must "
            "divide the projection TP-2D YZ size and be distributable across the projection "
            "`yz_tags`. Decode-only (lives on `InferenceDecodeSpec`)."
        ),
    )

    # Decode-only: how to place / communicate K/V under TP2D.
    #
    # Semantics:
    # - reduce_to_owner: K/V are reduced along the YZ axis to a single owner rank (sharded KV-cache).
    # - allreduce_and_drop: K/V participate in the YZ all-reduce (replicated), and non-owner ranks
    #   "drop" them afterward. This can reduce the number of distinct collectives (latency),
    #   but increases bandwidth relative to reduce-to-owner.
    decode_tp2d_kv_comm_mode: Literal["reduce_to_owner", "allreduce_and_drop"] = Field(
        default="reduce_to_owner",
        description=(
            "Decode-only TP2D knob for K/V placement/communication. "
            "'reduce_to_owner' models sharded KV-cache via reduce-to-owner along YZ. "
            "'allreduce_and_drop' models a latency-optimized alternative where K/V are replicated "
            "via the YZ all-reduce and then discarded on non-owner ranks."
        ),
    )

    decode_moe_router_sync_mode: MoERouterSyncMode = Field(
        ...,
        description=(
            "Decode-only MoE router synchronization mode over the attention TP domain. "
            "'replicated_router' computes full router logits locally on each attention-TP rank "
            "(no router all-reduce). "
            "'sharded_router' shards router logits across attention-TP ranks and requires an all-reduce."
        ),
    )
    _parse_decode_moe_router_sync_mode = field_validator(
        "decode_moe_router_sync_mode", mode="before"
    )(parse_enum_by_name_or_value(MoERouterSyncMode))
    # Note: K/V reduce-to-owner is always modeled as a single fused reduce on (K,V) concatenated
    # payload when decode_tp2d_kv_comm_mode='reduce_to_owner'. We do not expose this as a knob.

    @property
    def effective_micro_batch_size(self) -> int:
        if self.micro_batch_size is not None:
            return self.micro_batch_size
        else:
            raise ValueError("micro_batch_size is not set. Please set it explicitly.")
