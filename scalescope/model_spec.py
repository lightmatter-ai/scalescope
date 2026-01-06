"""Transformer model specification.

Defines TransformerConfig` - a model configuration loaded from YAML.
It specifies exclusively transformer-level parameters (the model itself),
and intentionally abstracts away workload settings, topology,
and parallelization policy, which are defined in other specs.
"""

import os
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

try:
    from typing import Self
except ImportError:
    from typing_extensions import (
        Self,  # Requires pip install typing-extensions for older Python
    )


T_TransformerConfig = TypeVar("T_TransformerConfig", bound="TransformerConfig")


class TransformerConfig(BaseModel):
    """Pydantic model for Transformer configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core Parameters (required)
    d_model: PositiveInt = Field(..., description="Dimension of hidden layers")
    n_layers: PositiveInt = Field(
        ..., alias="n_layers", description="Total number of transformer layers"
    )
    num_heads: PositiveInt = Field(..., description="Number of attention heads")
    num_kv_heads: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of key/value heads for Grouped Query Attention. If None, defaults to num_heads.",
    )
    mlp_ratio: float = Field(..., gt=0, description="Ratio of mlp hidden dim to embedding dim")
    max_sequence_length: PositiveInt = Field(..., description="Maximum sequence length")
    vocab_size: PositiveInt = Field(..., description="Vocabulary size")

    # MoE Parameters (optional, default to dense)
    num_regular_experts: int = Field(
        default=0, ge=0, description="Number of 'regular' experts per layer (0=dense)"
    )
    num_selected_experts: int = Field(
        default=0, ge=0, description="Number of experts selected per token (0=dense)"
    )
    num_shared_experts: int = Field(
        default=1, ge=0, description="Number of shared experts per layer"
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
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
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
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

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
    def expert_params_per_expert(self) -> int:
        return 2 * self.d_model * self.mlp_hidden_size

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
    head_dim = d_model // num_heads

    # Q, K, V, Out with GQA accounted for in K and V
    q_weights = d_model * d_model
    k_weights = d_model * (num_kv_heads * head_dim)
    v_weights = d_model * (num_kv_heads * head_dim)
    o_weights = d_model * d_model
    attn_weights_total = q_weights + k_weights + v_weights + o_weights

    tp_size = tp_topology.num_chips
    if tp_size <= 0:
        raise ValueError("TP topology must have num_chips >= 1")
    attn_weights_per_xpu = attn_weights_total / tp_size

    # LayerNorm parameters are replicated
    layernorm_params = 2 * (2 * d_model)  # Two LNs per layer, each with gamma/beta of size d_model
    return int(attn_weights_per_xpu + layernorm_params)


def expert_params_per_xpu(transformer_config: TransformerConfig, placement) -> int:
    """
    Returns per-layer expert parameters per XPU using placement-derived quantities:
    - total_distinct_experts_per_set: number of distinct experts in a complete expert set
    - full_expert_set_xpu_count: XPUs required to host one complete expert set
    """
    d_model = transformer_config.d_model
    d_ff = transformer_config.mlp_hidden_size
    params_per_expert = d_model * d_ff + d_ff * d_model  # FFN1 + FFN2
    total_params_in_set = params_per_expert * placement.total_distinct_experts_per_set
    return int(total_params_in_set / placement.full_expert_set_xpu_count)
