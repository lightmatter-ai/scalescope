"""Workload specification and model placement.

Describes how a transformer model is placed onto a cluster by specifying
parallelization strategies (TP/DP/PP and MoE/EP-related options), together
with training and inference parameters (e.g., global and micro batch sizes,
sequence length, numeric precision/efficiency knobs).
"""

import os
from dataclasses import dataclass
from typing import Annotated, Any, Dict, Literal, Optional, Type, TypeVar, Union

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
    NORMALIZATION_DEFERRED_SHARDED_ROUTER = "normalization_deferred_sharded_router"


class TP2DProjAxisRole(StrEnum):
    """What a TP-2D projection axis shards *at rest* in prefill/training.

    Applies to the attention projection GEMMs (W_qkv, W_O) only; the
    attention core (phase 2) is always sequence-sharded on YZ in
    prefill/training. Decode has its own fixed schedule and does not
    consult these roles.

    Values:
    - `shard_weights`: classic Megatron / weight-stationary — the axis
      shards weight columns/rows and the weights never move; activations
      are gathered/reduce-scattered across the axis around each GEMM
      (S-proportional volume); weight memory divides by the axis size. On
      YZ this is the decode-style regime: cheapest for short sequences,
      loses to `circulate_weights` beyond B*S ~ W_panel/d_model tokens.
    - `shard_tokens`: sequence/context parallelism — the axis shards
      tokens; weights are fully replicated across the axis (no GEMM
      collectives, but weight footprint does NOT divide by the axis size).
    - `circulate_weights`: FSDP/SUMMA-style — both weights and tokens are
      sharded at rest; each layer all-gathers its weight panel across the
      axis (S-independent volume), applies it to the local tokens, and
      discards it. Weight footprint divides by the axis size; a transient
      panel buffer appears at the gather peak.
    """

    SHARD_WEIGHTS = "shard_weights"
    SHARD_TOKENS = "shard_tokens"
    CIRCULATE_WEIGHTS = "circulate_weights"


class SeqParallelAttnMode(StrEnum):
    """Sequence-parallel schedule for the prefill/training attention core
    (phase 2) on the YZ axis of TP-2D attention.

    - `ring`: RingAttention-style KV circulation, priced as an all-gather
      proxy of the K/V buffers along YZ. Every rank receives the full
      sequence's K/V, so per-rank volume scales with the whole context S
      and stays roughly constant as the YZ group grows.
    - `ulysses`: DeepSpeed-Ulysses — two all-to-all collectives reshard
      Q/K/V from sequence-sharded to head-sharded and the attention output
      back. Each rank exchanges only its local S/YZ token shard (all Q/K/V/O
      heads), so per-rank volume shrinks as the YZ group grows — cheaper
      than ring once YZ exceeds ~num_heads/num_kv_heads + 1. Requires
      enough query heads per (X, YZ) shard.
    - `hybrid`: USP-style composition (arXiv:2405.07719) — the YZ axis
      factors as P_ulysses x P_ring via a tag partition
      (`seq_parallel_ulysses_tags` names the Ulysses subgroup tags; the
      remaining yz tags form the ring axis). Ulysses a2a runs inside each
      P_ulysses subgroup, ring circulation across the P_ring subgroups with
      the circulated KV shrunk by P_ulysses. Only the Ulysses factor is
      head-count-bounded.
    """

    RING = "ring"
    ULYSSES = "ulysses"
    HYBRID = "hybrid"


class TP2DProjAxisRoles(BaseModel):
    """Per-axis operand roles for the TP-2D projection legs (prefill/train)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    x: TP2DProjAxisRole = Field(..., description="Role of the X (head-sharding) axis.")
    yz: TP2DProjAxisRole = Field(..., description="Role of the YZ (sequence-sharding) axis.")

    _parse_x = field_validator("x", mode="before")(parse_enum_by_name_or_value(TP2DProjAxisRole))
    _parse_yz = field_validator("yz", mode="before")(parse_enum_by_name_or_value(TP2DProjAxisRole))

    @model_validator(mode="after")
    def _validate_supported_combination(self) -> "TP2DProjAxisRoles":
        # X shards heads for the attention core, so the projection GEMMs
        # must keep weights sharded (columns/rows) on X. Token sharding or
        # circulation on X is not modeled.
        if self.x != TP2DProjAxisRole.SHARD_WEIGHTS:
            raise ValueError(
                f"tp2d_proj_axis_roles.x must be 'shard_weights' (got '{self.x}'); "
                "head sharding on X requires weight-sharded projections on X."
            )
        # All three roles are valid on YZ. The attention core stays
        # sequence-sharded on YZ in every case:
        # - shard_tokens / circulate_weights: tokens are sequence-sharded at
        #   rest on YZ.
        # - shard_weights (weight-stationary, decode-style): the residual
        #   stream is feature-sharded on YZ at rest; phase 1 reduce-scatters
        #   the QKV partials over the *sequence* dim, so phase 2 sees the
        #   same sequence-sharded layout.
        return self


class MoEA2ACommMode(StrEnum):
    HIERARCHICAL_SPLIT = "hierarchical_split"
    FUSED_ND = "fused_nd"


class DecodeCrossDomainEPMode(StrEnum):
    """Decode exchange contract when one expert set spans attention domains."""

    EXPLICIT_OWNER_A2AV = "explicit_owner_a2av"


class ExpertDomainPlacementKind(StrEnum):
    """Supported mappings from colocated expert groups to EP domains."""

    BALANCED_EXPERT_GROUPS = "balanced_expert_groups"


class ExpertDomainPlacementSpec(BaseModel):
    """Explicit ownership policy for expert groups spanning attention domains."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ExpertDomainPlacementKind = Field(...)
    domain_tag: str = Field(..., min_length=1)

    _parse_kind = field_validator("kind", mode="before")(
        parse_enum_by_name_or_value(ExpertDomainPlacementKind)
    )


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
    - `bottleneck_group_collector_random`: the expected straggler model under
      load-agnostic (random/hash) placement. The worst-loaded chip holds the
      single hottest expert (rank 1) plus `c-1` experts drawn uniformly from the
      remaining `E-1`, so the fillers contribute the cold-tail average:
          p_k      = k^{-s} / H_{E,s},  s solved from E / H_{E,s} = rho
          a_k      = 1 - (1 - p_k)^m              # P(expert k hit by >= 1 of m events)
          E_active = a_1 + (c - 1) * mean_{k=2..E} a_k
          frac     = E_active / c
      where `c = num_colocated_experts` and `m = batch * top_k` is the per-set
      routing-event count. Because the memory model reads
      `resident * frac = c·(w/TP_MoE)·(E_active/c) = E_active·(w/TP_MoE)`, the
      per-shard size `w/TP_MoE` is explicit: 2D `TP_MoE` (finer sharding) shrinks
      an outlier expert's bottleneck (`E_active → 1` for steep Zipf at small `m`).
      Corners: `c = 1` → `a_1`; `c = E` → `sum_{k=1..E} a_k` (the full Zipf active
      count); `rho = 1` → plain coupon collector.

      `E_active` is unfloored: it is an *expectation* over placements and routing,
      so a value below 1.0 is meaningful. At small `m` each routing event touches
      only a few of the `E` experts, so `a_1 = 1 - (1 - p_1)^m ≈ m·p_1` can be
      well under 1, and `E_active` (the expected distinct-active count on the
      straggler) is genuinely fractional — the straggler reads less than one full
      expert's weights per step on average. Flooring at 1.0 would force every step
      to pay a full per-expert HBM read even in that regime, systematically
      over-counting weight traffic and inflating decode step time at small batch /
      low top-k. Leaving it unfloored preserves the correct small-`m` asymptote
      (linear in `m`) while still rising to the full Zipf active count as `m` grows.
    """

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
            "'bottleneck_group_collector_random' evaluates the expected worst-loaded "
            "colocated group under random placement (hottest expert + cold-tail-average "
            "fillers), skew-aware over the global Zipf solved from rho. Only consulted "
            "when expert_weight_residency_policy = resident_active_subset."
        ),
    )
    # Optional
    num_colocated_experts: int = Field(1, ge=1, description="Number of colocated experts")
    shared_expert_copies: Optional[int] = Field(
        default=None,
        ge=1,
        description="Replication factor per shared expert; if None, may be inferred later",
    )
    expert_domain_placement: Optional[ExpertDomainPlacementSpec] = Field(
        default=None,
        description=(
            "Explicit colocated-expert-group ownership policy for an expert set spanning "
            "multiple attention domains. Required by the exact cross-domain decode path."
        ),
    )

    _parse_expert_weight_residency_policy = field_validator(
        "expert_weight_residency_policy", mode="before"
    )(parse_enum_by_name_or_value(ExpertWeightResidencyPolicy))
    _parse_expert_routing_distribution = field_validator(
        "expert_routing_distribution", mode="before"
    )(parse_enum_by_name_or_value(ExpertRoutingDistribution))


class OverlapModelKind(StrEnum):
    FIXED = "fixed"
    TOKEN_SATURATING = "token_saturating"


class OverlapPairCurve(BaseModel):
    """One pairwise pipeline overlap curve: e(N) = e_max * max(0, 1 - N_wave / N)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_efficiency: float = Field(..., ge=0.0, le=1.0)
    wave_tokens: int = Field(..., gt=0)


@dataclass(frozen=True)
class OverlapPairSpec:
    """One pairwise curve for token-saturating overlap.

    e(N) = max_efficiency * max(0, 1 - wave_tokens / N).
    """

    # Asymptotic fraction of the shorter resource that may hide (0..1).
    max_efficiency: float
    # Tokens in one pipeline tile. N == wave_tokens → e = 0;
    # N == 2 * wave_tokens → e = max_efficiency / 2.
    wave_tokens: int


@dataclass(frozen=True)
class ResolvedOverlapEfficiencies:
    comp_comm_efficiency: float
    comp_mem_efficiency: float
    comm_mem_efficiency: float


def resolve_overlap_pair_efficiency(*, n_tokens: float, pair_spec: OverlapPairSpec) -> float:
    """Pipeline pairwise efficiency: e(N) = e_max * max(0, 1 - N_wave / N).

    N_wave is one tile of work. A single wave cannot hide anything; two
    waves reach e_max / 2; large N approaches e_max.
    """
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")
    if pair_spec.wave_tokens <= 0:
        raise ValueError(f"wave_tokens must be > 0, got {pair_spec.wave_tokens}")
    if not 0.0 <= pair_spec.max_efficiency <= 1.0:
        raise ValueError(
            "max_efficiency must be between 0.0 and 1.0, " f"got {pair_spec.max_efficiency}"
        )
    hidden_fraction = 1.0 - pair_spec.wave_tokens / n_tokens
    if hidden_fraction <= 0.0:
        return 0.0
    return pair_spec.max_efficiency * hidden_fraction


def resolve_overlap_efficiencies(
    *,
    n_tokens: float,
    compute_communication: OverlapPairSpec,
    compute_memory: OverlapPairSpec,
    communication_memory: OverlapPairSpec,
) -> ResolvedOverlapEfficiencies:
    return ResolvedOverlapEfficiencies(
        comp_comm_efficiency=resolve_overlap_pair_efficiency(
            n_tokens=n_tokens, pair_spec=compute_communication
        ),
        comp_mem_efficiency=resolve_overlap_pair_efficiency(
            n_tokens=n_tokens, pair_spec=compute_memory
        ),
        comm_mem_efficiency=resolve_overlap_pair_efficiency(
            n_tokens=n_tokens, pair_spec=communication_memory
        ),
    )


class FixedOverlapModel(BaseModel):
    """Fixed pairwise overlap efficiencies, independent of local token count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[OverlapModelKind.FIXED]
    comp_comm_efficiency: float = Field(..., ge=0.0, le=1.0)
    comp_mem_efficiency: float = Field(..., ge=0.0, le=1.0)
    comm_mem_efficiency: float = Field(..., ge=0.0, le=1.0)


class TokenSaturatingOverlapModel(BaseModel):
    """Pairwise overlap efficiencies that grow with local pipeline waves.

    Attention/residual use tokens_per_attention_domain. MoE/FFN use hot-rank
    routed tokens: (tokens_per_set / E) * traffic_multiplier * colocated.
    wave_tokens is in those same units for each block.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[OverlapModelKind.TOKEN_SATURATING]
    compute_communication: OverlapPairCurve
    compute_memory: OverlapPairCurve
    communication_memory: OverlapPairCurve


OverlapModel = Annotated[
    Union[FixedOverlapModel, TokenSaturatingOverlapModel],
    Field(discriminator="kind"),
]


def efficiencies_for_overlap_model(
    *, overlap_model: OverlapModel, n_tokens: float
) -> ResolvedOverlapEfficiencies:
    """Resolve block efficiencies from an explicit overlap_model and local N."""
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")
    match overlap_model.kind:
        case OverlapModelKind.FIXED:
            return ResolvedOverlapEfficiencies(
                comp_comm_efficiency=overlap_model.comp_comm_efficiency,
                comp_mem_efficiency=overlap_model.comp_mem_efficiency,
                comm_mem_efficiency=overlap_model.comm_mem_efficiency,
            )
        case OverlapModelKind.TOKEN_SATURATING:
            return resolve_overlap_efficiencies(
                n_tokens=n_tokens,
                compute_communication=OverlapPairSpec(
                    max_efficiency=overlap_model.compute_communication.max_efficiency,
                    wave_tokens=overlap_model.compute_communication.wave_tokens,
                ),
                compute_memory=OverlapPairSpec(
                    max_efficiency=overlap_model.compute_memory.max_efficiency,
                    wave_tokens=overlap_model.compute_memory.wave_tokens,
                ),
                communication_memory=OverlapPairSpec(
                    max_efficiency=overlap_model.communication_memory.max_efficiency,
                    wave_tokens=overlap_model.communication_memory.wave_tokens,
                ),
            )
        case _:
            raise ValueError(f"Unsupported overlap_model.kind: {overlap_model.kind}")


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

    overlap_model: OverlapModel
    fold_residual_into_moe_overlap: bool = Field(
        default=False,
        description=(
            "When true, leftover residual C/K/M is added to the MoE overlap "
            "pot on both the forward and backward passes, and residual is "
            "not a separate latency block. Isolated residual C/K/M and "
            "isolated residual overlapped time remain diagnostics. Default "
            "false keeps residual as its own overlapped block."
        ),
    )
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
    moe_token_pair_overhead_sec: float = Field(
        ...,
        ge=0,
        description=(
            "MoE software/runtime overhead in seconds per routed token-expert "
            "pair that leaves this rank, per MoE layer per pass. Volume is "
            "N_rows × remote_frac: decode uses (D−1)/D after the handoff, "
            "prefill/train use (G−1)/G. D=1 decode and G=1 prefill/train "
            "are zero. Dense E<=1 pays zero. Applied serially outside "
            "C/K/M overlap, like the kernel-launch floor. Uses the modeled "
            "hot-rank routed load (top-k and colocation-diluted imbalance). "
            "This is pack/dispatch bookkeeping, not Hockney communication "
            "or HBM bytes. Set 0.0 where not modeled."
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
            "'hierarchical_split' prices two A2As (EP_SU intra, then EP_SO inter); "
            "'fused_nd' prices one A2A on the full EP mesh (flatten the SU/SO "
            "split; not kernel fusion)."
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
        Return the per-attention-domain batch (legacy name: per-DP batch).

        Each attention DP replica is one attention domain.
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
        Validates that the per-attention-domain batch is divisible by
        micro_batch_size.
        """
        per_dp = self.infer_batch_size_per_dp(dp_attn_chips)
        mb = getattr(self, "micro_batch_size", None)
        if mb is None:
            raise ValueError("micro_batch_size must be set.")
        if per_dp % mb != 0:
            raise ValueError(f"Per-DP batch {per_dp} not divisible by micro_batch_size {mb}.")
        return mb

    def with_resolved_batches(
        self,
        dp_attn_chips: int,
        num_complete_expert_sets: int,
        *,
        allow_sparse_attention_domains: bool,
    ) -> "WorkloadSpec":
        """
        Returns a new WorkloadSpec with resolved batch sizes attached (field 'resolved_batches').
        This preserves immutability of the original spec.
        """
        rb = resolve_batch_sizes_for_workload(
            self,
            dp_attn_chips,
            num_complete_expert_sets,
            allow_sparse_attention_domains=allow_sparse_attention_domains,
        )
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
    """Canonical workload batch partition used by attention and MoE.

    One attention DP replica is one physical attention domain. Normally
    ``batch_size_per_dp`` is the uniform batch on every domain. The narrow
    sparse wide-EP decode case instead assigns one request to each active
    domain and none to the remaining physical domains; in that case
    ``batch_size_per_dp`` is the batch on an active domain.
    """

    global_batch_size: int
    attention_domain_count: int
    # Number of physical attention domains assigned at least one request. It
    # equals attention_domain_count for ordinary uniform partitioning and can
    # be smaller only for the supported sparse wide-EP decode case.
    active_attention_domain_count: int
    complete_expert_set_count: int
    batch_size_per_dp: int
    effective_micro_batch_size: int
    batch_size_per_full_expert_set: int


def resolve_batch_sizes_for_workload(
    workload_spec: WorkloadSpec,
    dp_attn_chips: int,
    num_complete_expert_sets: int,
    *,
    allow_sparse_attention_domains: bool,
) -> ResolvedBatchSizes:
    gb = workload_spec.infer_global_batch_size(dp_attn_chips)
    if dp_attn_chips <= 0:
        raise ValueError(f"dp_attn_chips must be > 0, got {dp_attn_chips}")
    if gb < dp_attn_chips:
        if not allow_sparse_attention_domains:
            raise ValueError(
                f"Global batch size {gb} is smaller than DP chips {dp_attn_chips}, "
                "but sparse attention domains are not enabled for this workload."
            )
        if num_complete_expert_sets != 1:
            raise ValueError(
                "Sparse attention domains require exactly one complete expert set, "
                f"got {num_complete_expert_sets}."
            )
        per_dp = 1
        active_attention_domain_count = gb
        micro_batch_size = getattr(workload_spec, "micro_batch_size", None)
        if micro_batch_size is None:
            raise ValueError("micro_batch_size must be set.")
        if micro_batch_size != 1:
            raise ValueError(
                "Sparse attention domains require micro_batch_size=1, got " f"{micro_batch_size}."
            )
        eff_micro = micro_batch_size
    else:
        per_dp = workload_spec.infer_batch_size_per_dp(dp_attn_chips)
        active_attention_domain_count = dp_attn_chips
        eff_micro = workload_spec.infer_effective_micro_batch_size(dp_attn_chips)
    per_set = workload_spec.infer_batch_size_per_full_expert_set(
        dp_attn_chips, num_complete_expert_sets
    )
    return ResolvedBatchSizes(
        global_batch_size=gb,
        attention_domain_count=dp_attn_chips,
        active_attention_domain_count=active_attention_domain_count,
        complete_expert_set_count=num_complete_expert_sets,
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

    tp2d_proj_axis_roles: TP2DProjAxisRoles | None = Field(
        default=None,
        description=(
            "Per-axis operand roles for the TP-2D projection legs. Required when "
            "the projection TP variant is SHARD_WEIGHTS_2D; must be omitted otherwise."
        ),
    )
    seq_parallel_attn_mode: SeqParallelAttnMode | None = Field(
        default=None,
        description=(
            "Sequence-parallel schedule for the attention core on YZ (ring, "
            "ulysses, or hybrid). Required when the attention TP variant is "
            "SHARD_WEIGHTS_2D; must be omitted otherwise."
        ),
    )
    seq_parallel_ulysses_tags: list[str] | None = Field(
        default=None,
        description=(
            "For seq_parallel_attn_mode=hybrid only: the attention yz_tags that "
            "form the Ulysses subgroup (P_ulysses = product of their sizes). The "
            "remaining yz_tags form the ring axis. Must be a non-empty proper "
            "subset of tp_attn_spec.split_2d.yz_tags."
        ),
    )

    _parse_seq_parallel_attn_mode = field_validator("seq_parallel_attn_mode", mode="before")(
        parse_enum_by_name_or_value(SeqParallelAttnMode)
    )

    @model_validator(mode="after")
    def _validate_tp2d_prefill_train_fields(self) -> "TrainSpec":
        _validate_tp2d_prefill_train_schedule_fields(self)
        return self


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

    tp2d_proj_axis_roles: TP2DProjAxisRoles | None = Field(
        default=None,
        description=(
            "Per-axis operand roles for the TP-2D projection legs. Required when "
            "the projection TP variant is SHARD_WEIGHTS_2D; must be omitted otherwise."
        ),
    )
    seq_parallel_attn_mode: SeqParallelAttnMode | None = Field(
        default=None,
        description=(
            "Sequence-parallel schedule for the attention core on YZ (ring, "
            "ulysses, or hybrid). Required when the attention TP variant is "
            "SHARD_WEIGHTS_2D; must be omitted otherwise."
        ),
    )
    seq_parallel_ulysses_tags: list[str] | None = Field(
        default=None,
        description=(
            "For seq_parallel_attn_mode=hybrid only: the attention yz_tags that "
            "form the Ulysses subgroup (P_ulysses = product of their sizes). The "
            "remaining yz_tags form the ring axis. Must be a non-empty proper "
            "subset of tp_attn_spec.split_2d.yz_tags."
        ),
    )

    _parse_seq_parallel_attn_mode = field_validator("seq_parallel_attn_mode", mode="before")(
        parse_enum_by_name_or_value(SeqParallelAttnMode)
    )

    @model_validator(mode="after")
    def _validate_tp2d_prefill_train_fields(self) -> "InferencePrefillSpec":
        _validate_tp2d_prefill_train_schedule_fields(self)
        return self


def _validate_tp2d_prefill_train_schedule_fields(spec: WorkloadSpec) -> None:
    """Require the TP-2D prefill/train schedule fields exactly when TP-2D is used.

    `tp2d_proj_axis_roles` governs the projection legs (proj TP domain);
    `seq_parallel_attn_mode` governs the attention core (attn TP domain).
    Both are meaningless under 1D TP, so their presence there is rejected to
    keep specs honest.
    """
    proj_is_2d = spec.get_proj_tp_spec().variant == TensorParallelVariant.SHARD_WEIGHTS_2D
    attn_is_2d = spec.get_attn_tp_spec().variant == TensorParallelVariant.SHARD_WEIGHTS_2D

    roles = spec.tp2d_proj_axis_roles
    if proj_is_2d and roles is None:
        raise ValueError(
            "tp2d_proj_axis_roles is required when the projection TP variant is "
            "SHARD_WEIGHTS_2D (prefill/training)."
        )
    if not proj_is_2d and roles is not None:
        raise ValueError(
            "tp2d_proj_axis_roles must be omitted unless the projection TP variant "
            "is SHARD_WEIGHTS_2D."
        )

    mode = spec.seq_parallel_attn_mode
    if attn_is_2d and mode is None:
        raise ValueError(
            "seq_parallel_attn_mode is required when the attention TP variant is "
            "SHARD_WEIGHTS_2D (prefill/training)."
        )
    if not attn_is_2d and mode is not None:
        raise ValueError(
            "seq_parallel_attn_mode must be omitted unless the attention TP variant "
            "is SHARD_WEIGHTS_2D."
        )

    ulysses_tags = spec.seq_parallel_ulysses_tags
    if mode == SeqParallelAttnMode.HYBRID:
        if ulysses_tags is None:
            raise ValueError(
                "seq_parallel_ulysses_tags is required when seq_parallel_attn_mode "
                "is 'hybrid' (it names the yz_tags forming the Ulysses subgroup)."
            )
        if len(ulysses_tags) == 0:
            raise ValueError("seq_parallel_ulysses_tags must be non-empty.")
        if len(set(ulysses_tags)) != len(ulysses_tags):
            raise ValueError(f"seq_parallel_ulysses_tags contains duplicates: {ulysses_tags}")
        attn_split = spec.get_attn_tp_spec().split_2d
        # split_2d presence is already guaranteed by TensorParallelSpecV2 for 2D.
        yz_tags = set(attn_split.yz_tags)
        unknown = set(ulysses_tags) - yz_tags
        if unknown:
            raise ValueError(
                f"seq_parallel_ulysses_tags contains tags not in "
                f"tp_attn_spec.split_2d.yz_tags: {sorted(unknown)}"
            )
        if set(ulysses_tags) == yz_tags:
            raise ValueError(
                "seq_parallel_ulysses_tags must be a proper subset of yz_tags: with "
                "all yz tags in the Ulysses subgroup there is no ring axis — use "
                "seq_parallel_attn_mode: ulysses instead."
            )
    elif ulysses_tags is not None:
        raise ValueError(
            "seq_parallel_ulysses_tags must be omitted unless seq_parallel_attn_mode "
            "is 'hybrid'."
        )


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
            "'sharded_router' shards router logits across attention-TP ranks and requires separate "
            "normalization-statistics and router-logit all-reduces. "
            "'normalization_deferred_sharded_router' computes partial logits from unnormalized "
            "activations and packs them with normalization statistics into one all-reduce."
        ),
    )

    _parse_decode_moe_router_sync_mode = field_validator(
        "decode_moe_router_sync_mode", mode="before"
    )(parse_enum_by_name_or_value(MoERouterSyncMode))
    decode_cross_domain_ep_mode: Optional[DecodeCrossDomainEPMode] = Field(
        default=None,
        description=(
            "Exact decode exchange contract for an expert set spanning multiple attention "
            "domains. Required for cross-scale-up-domain decode; omission retains fail-fast "
            "behavior."
        ),
    )
    _parse_decode_cross_domain_ep_mode = field_validator(
        "decode_cross_domain_ep_mode", mode="before"
    )(parse_enum_by_name_or_value(DecodeCrossDomainEPMode))
    # Note: K/V reduce-to-owner is always modeled as a single fused reduce on (K,V) concatenated
    # payload when decode_tp2d_kv_comm_mode='reduce_to_owner'. We do not expose this as a knob.

    @property
    def effective_micro_batch_size(self) -> int:
        if self.micro_batch_size is not None:
            return self.micro_batch_size
        else:
            raise ValueError("micro_batch_size is not set. Please set it explicitly.")
