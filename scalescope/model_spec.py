"""Transformer model specification.

Defines TransformerConfig` - a model configuration loaded from YAML.
It specifies exclusively transformer-level parameters (the model itself),
and intentionally abstracts away workload settings, topology,
and parallelization policy, which are defined in other specs.
"""

import os
from enum import Enum
from typing import Optional, Type, TypeVar

import yaml

# Pydantic imports
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
    Field,
    model_validator,
    PositiveInt,
    ValidationError,
)


class MlpKind(str, Enum):
    """MLP block topology.

    Applied uniformly to the dense FFN (modeled as a single expert,
    ``num_regular_experts == 1``) and to every expert in an MoE configuration
    (``num_regular_experts > 1``).

    Variants:
        DENSE_2MATRIX:
            Classic FFN — ``y = down(act(up(x)))``. Two weight matrices:
            ``W_up: [d_model, d_ff]`` and ``W_down: [d_ff, d_model]``.

        GATED_3MATRIX:
            SwiGLU/GeGLU — ``y = down(silu(gate(x)) ⊙ up(x))``. Three
            weight matrices: ``W_gate: [d_model, d_ff]``,
            ``W_up: [d_model, d_ff]``, ``W_down: [d_ff, d_model]``.
    """

    DENSE_2MATRIX = "dense_2matrix"
    GATED_3MATRIX = "gated_3matrix"


try:
    from typing import Self
except ImportError:
    from typing_extensions import (
        Self,  # Requires pip install typing-extensions for older Python
    )


T_TransformerConfig = TypeVar("T_TransformerConfig", bound="TransformerConfig")


def resolve_kv_heads_per_partition(*, num_kv_heads: int, tp_x: int) -> tuple[int, int]:
    """Resolve per-rank KV heads and replication factor for head-sharded TP.

    Semantics:
    - If tp_x <= num_kv_heads: shard KV heads across TP ranks (requires divisibility).
    - If tp_x > num_kv_heads: duplicate KV heads across TP ranks (requires tp_x % num_kv_heads == 0),
      so each rank holds one KV head and replication factor is tp_x / num_kv_heads.
    """
    if tp_x <= 0:
        raise ValueError(f"tp_x must be positive, got {tp_x}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")

    if tp_x <= num_kv_heads:
        if num_kv_heads % tp_x != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by TP-X size ({tp_x})"
            )
        return num_kv_heads // tp_x, 1

    if tp_x % num_kv_heads != 0:
        raise ValueError(
            f"TP-X size ({tp_x}) must be divisible by num_kv_heads ({num_kv_heads}) for KV duplication"
        )
    return 1, tp_x // num_kv_heads


class TransformerConfig(BaseModel):
    """Pydantic model for Transformer configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core Parameters (required)
    d_model: PositiveInt = Field(..., description="Dimension of hidden layers")
    n_layers: PositiveInt = Field(
        ..., alias="n_layers", description="Total number of transformer layers"
    )
    num_heads: PositiveInt = Field(..., description="Number of attention heads")
    head_dim: PositiveInt = Field(
        ...,
        description=(
            "Per-head attention dimension. Required. The Q projection "
            "outputs `num_heads × head_dim` activations (which need NOT "
            "equal `d_model` — modern 'wide-Q' layouts like MiniMax-M2.5, "
            "gpt-oss-120b, Qwen3 have `num_heads × head_dim > d_model`). "
            "KV projections output `num_kv_heads × head_dim`. Always pass "
            "the value from the model's HuggingFace config verbatim."
        ),
    )
    num_kv_heads: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of key/value heads for Grouped Query Attention. If None, defaults to num_heads.",
    )
    mlp_ratio: float = Field(..., gt=0, description="Ratio of mlp hidden dim to embedding dim")
    mlp_kind: MlpKind = Field(
        ...,
        description=(
            "MLP topology applied to dense FFN and every MoE expert. "
            "Required: every model spec must declare this explicitly."
        ),
    )
    max_sequence_length: PositiveInt = Field(..., description="Maximum sequence length")
    vocab_size: PositiveInt = Field(..., description="Vocabulary size")

    # MoE Parameters (optional, default to dense)
    num_regular_experts: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of 'regular' experts per layer. ScaleScope convention: "
            "1=dense (single FFN), >1=MoE, 0=attention-only (no FFN compute)."
        ),
    )
    num_selected_experts: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of experts selected per token. 1 for dense, top-k for MoE, "
            "0 for attention-only."
        ),
    )
    num_shared_experts: int = Field(
        default=0, ge=0, description="Number of shared experts per layer"
    )

    # Validators
    @model_validator(mode="after")
    def set_kv_heads_and_validate(self) -> Self:
        if self.num_kv_heads is None:
            # Pydantic v2 forbids direct mutation, so we use object.__setattr__
            object.__setattr__(self, "num_kv_heads", self.num_heads)

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        return self

    @model_validator(mode="after")
    def check_model_consistency(self) -> Self:
        # `d_model` and `num_heads × head_dim` are independent: the Q
        # projection maps `d_model -> num_heads × head_dim` and the O
        # projection maps `num_heads × head_dim -> d_model`. These need
        # not be equal. We surface "wide-Q" layouts (where the two
        # differ) as a one-line note at config load so the choice is
        # visible in run logs, but do not reject the config — wide-Q is
        # an explicit modern architectural choice (MiniMax-M2.5,
        # gpt-oss-120b, Qwen3, DeepSeek-V4 all use it).
        q_proj_out_dim = self.num_heads * self.head_dim
        if q_proj_out_dim != self.d_model:
            print(
                f"note: wide-Q attention layout in {self.__class__.__name__}: "
                f"num_heads × head_dim = {self.num_heads} × {self.head_dim} = "
                f"{q_proj_out_dim} ≠ d_model = {self.d_model} "
                f"(ratio {q_proj_out_dim / self.d_model:.2f}x)"
            )
        is_moe = self.num_regular_experts > 0
        if is_moe:
            if self.num_selected_experts <= 0:
                raise ValueError("MoE requires num_selected_experts > 0.")
            if self.num_selected_experts > self.num_regular_experts:
                raise ValueError(
                    f"num_selected ({self.num_selected_experts}) > num_regular ({self.num_regular_experts})"
                )
        else:
            if self.num_selected_experts != 0:
                print(
                    f"Warning: Dense model but num_selected_experts={self.num_selected_experts}. Consider setting to 0."
                )
                # object.__setattr__(self, 'num_selected_experts', 0) # Optional: Auto-correct (use with caution on frozen models)
        return self

    @computed_field
    @property
    def mlp_hidden_size(self) -> int:
        return int(self.d_model * self.mlp_ratio)

    @computed_field
    @property
    def is_moe(self) -> bool:
        return self.num_regular_experts > 0

    @computed_field(repr=False)
    @property
    def mlp_num_up_projections(self) -> int:
        """Number of ``[d_model, d_ff]`` matrices in one MLP block.

        DENSE_2MATRIX has 1 (the up-projection ``W_up``).
        GATED_3MATRIX has 2 (the gate ``W_gate`` and the up ``W_up``,
        which have identical shape and are run in parallel before the
        element-wise gate-multiply).
        """
        if self.mlp_kind is MlpKind.DENSE_2MATRIX:
            return 1
        if self.mlp_kind is MlpKind.GATED_3MATRIX:
            return 2
        raise ValueError(f"Unhandled MlpKind: {self.mlp_kind!r}")

    @computed_field(repr=False)
    @property
    def mlp_num_down_projections(self) -> int:
        """Number of ``[d_ff, d_model]`` matrices in one MLP block."""
        if self.mlp_kind is MlpKind.DENSE_2MATRIX:
            return 1
        if self.mlp_kind is MlpKind.GATED_3MATRIX:
            return 1
        raise ValueError(f"Unhandled MlpKind: {self.mlp_kind!r}")

    @computed_field(repr=False)
    @property
    def mlp_num_elementwise_passes(self) -> int:
        """Number of elementwise passes over a ``[tokens, d_ff]`` tensor
        in one MLP block.

        DENSE_2MATRIX has 1: the activation function (silu / gelu /
        relu) applied once between up and down.
        GATED_3MATRIX has 2: the silu on the gate output, plus the
        element-wise multiply ``silu(gate) ⊙ up`` that fuses gate with
        up before the down-projection.

        Used symmetrically on forward and backward FFN compute (one
        gradient-pass per forward elementwise op). The cost is small
        relative to the matmuls but is modelled explicitly on both
        sides — no asymmetric backward-only term.
        """
        if self.mlp_kind is MlpKind.DENSE_2MATRIX:
            return 1
        if self.mlp_kind is MlpKind.GATED_3MATRIX:
            return 2
        raise ValueError(f"Unhandled MlpKind: {self.mlp_kind!r}")

    @computed_field(repr=False)
    @property
    def mlp_num_weight_matrices(self) -> int:
        """Total weight-matrix count = up + down."""
        return self.mlp_num_up_projections + self.mlp_num_down_projections

    @computed_field(repr=False)
    @property
    def expert_params_per_expert(self) -> int:
        return self.mlp_num_weight_matrices * self.d_model * self.mlp_hidden_size

    @computed_field(repr=False)
    @property
    def total_regular_expert_params_per_layer(self) -> int:
        return self.expert_params_per_expert * self.num_regular_experts

    @computed_field(repr=False)
    @property
    def total_shared_expert_params_per_layer(self) -> int:
        return self.expert_params_per_expert * self.num_shared_experts

    @computed_field(repr=False)
    @property
    def router_params_per_layer(self) -> int:
        return self.d_model * self.num_regular_experts if self.num_regular_experts > 0 else 0

    def __repr__(self):
        moe_str = ""
        if self.is_moe:
            moe_str = (
                f", experts={self.num_regular_experts}, "
                f"selected={self.num_selected_experts}, "
                f"shared={self.num_shared_experts}"
            )
        gqa_str = ""
        if self.num_kv_heads != self.num_heads:
            gqa_str = f", kv_heads={self.num_kv_heads}"

        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(d_model={self.d_model},"
            f" n_layers={self.n_layers},"
            f" n_heads={self.num_heads}{gqa_str},"
            f" mlp_ratio={self.mlp_ratio:.2f},"
            f" mlp_kind={self.mlp_kind.value},"
            f" seq_len={self.max_sequence_length},"
            f" vocab={self.vocab_size}{moe_str})"
        )

    @classmethod
    def load_from_yaml(cls: Type[T_TransformerConfig], filepath: str) -> T_TransformerConfig:
        """Loads and validates a TransformerConfig from a YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transformer config file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise TypeError(f"YAML file '{filepath}' did not load as a dictionary.")
            # Use cls (which refers to TransformerConfig or a subclass)
            model = cls.model_validate(config_dict)
            return model
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{filepath}': {e}") from e
        except ValidationError as e:
            raise ValueError(
                f"{cls.__name__} configuration validation failed for '{filepath}':\n{e}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error loading {cls.__name__} spec from '{filepath}': {e}") from e


def attn_params_per_xpu(transformer_config: TransformerConfig, tp_topology) -> int:
    """
    Returns per-layer parameter count on a single XPU for the attention path.
    Assumptions:
    - Attention weight matrices (Q,K,V,Out) are sharded across the TP-attn group.
    - LayerNorm parameters are replicated (not sharded) across TP ranks.
    """
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    tp_size = tp_topology.num_chips
    if tp_size <= 0:
        raise ValueError("TP topology must have num_chips >= 1")

    # Q maps d_model -> (num_heads × head_dim); O maps the inverse.
    # Asymmetric under wide-Q layouts (num_heads × head_dim ≠ d_model,
    # e.g. MiniMax-M2.5, gpt-oss-120b, qwen3 series).
    q_weights_per_xpu = (d_model * q_proj_out_dim) / tp_size
    o_weights_per_xpu = (q_proj_out_dim * d_model) / tp_size
    # K/V use GQA-aware head sharding with duplication fallback.
    kv_heads_per_rank, _kv_rep = resolve_kv_heads_per_partition(
        num_kv_heads=num_kv_heads, tp_x=tp_size
    )
    kv_weights_per_xpu = d_model * (kv_heads_per_rank * head_dim)
    attn_weights_per_xpu = q_weights_per_xpu + 2 * kv_weights_per_xpu + o_weights_per_xpu

    # LayerNorm parameters are replicated
    layernorm_params = 2 * (2 * d_model)  # Two LNs per layer, each with gamma/beta of size d_model
    return int(attn_weights_per_xpu + layernorm_params)


def expert_params_per_xpu(transformer_config: TransformerConfig, placement) -> int:
    """
    Returns per-layer expert parameters per XPU using placement-derived quantities:
    - total_distinct_experts_per_set: number of distinct experts in a complete expert set
    - full_expert_set_xpu_count: XPUs required to host one complete expert set
    """
    params_per_expert = transformer_config.expert_params_per_expert
    total_params_in_set = params_per_expert * placement.total_distinct_experts_per_set
    return int(total_params_in_set / placement.full_expert_set_xpu_count)
