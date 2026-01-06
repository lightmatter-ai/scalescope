"""Workload specification and model placement.

Describes how a transformer model is placed onto a cluster by specifying
parallelization strategies (TP/DP/PP and MoE/EP-related options), together
with training and inference parameters (e.g., global and micro batch sizes,
sequence length, numeric precision/efficiency knobs).
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

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


# Pydantic Spec Classes


class TensorParallelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    variant: TensorParallelVariant = TensorParallelVariant.NONE

    # Validator for 'variant'
    _parse_variant = field_validator("variant", mode="before")(
        parse_enum_by_name_or_value(TensorParallelVariant)
    )


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
    # Optional
    num_colocated_experts: int = Field(1, ge=1, description="Number of colocated experts")
    shared_expert_copies: Optional[int] = Field(
        default=None,
        ge=1,
        description="Replication factor per shared expert; if None, may be inferred later",
    )
    expert_high_bandwidth_group_size_xpus: Optional[int] = Field(
        default=None,
        ge=1,
        description="High-bandwidth expert group size in XPUs; if None, can be inferred from EP topology later",
    )


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

    tp_spec: TensorParallelSpec = Field(default_factory=TensorParallelSpec)
    dp_spec: DataParallelSpec = Field(default_factory=DataParallelSpec)
    # New primary spec
    moe_spec_v2: "MoESpecV2"
    pp_spec: PipelineParallelSpec = Field(default_factory=PipelineParallelSpec)
    # Required: global batch size per training/inference step for the whole cluster
    global_batch_size: int = Field(..., gt=0)
    # Floating-point size in bytes (e.g., 2 for BF16/FP16, 4 for FP32)
    float_size_bytes: int = Field(..., ge=1)
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
    # Optional: resolved batch sizes attached at runtime (immutable via copy)
    resolved_batches: Optional["ResolvedBatchSizes"] = None
    # Optional: layers per pipeline stage, attached at runtime (do not define in YAML)
    layers_per_pipeline_stage: Optional[int] = Field(default=None)

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
        If no global batch size is defined, falls back to micro_batch_size * GA.
        Enforces divisibility when global batch size is provided.
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

    @property
    def effective_micro_batch_size(self) -> int:
        if self.micro_batch_size is not None:
            return self.micro_batch_size
        else:
            raise ValueError("micro_batch_size is not set. Please set it explicitly.")
