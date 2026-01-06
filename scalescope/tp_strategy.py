"""Tensor-parallel (TP) communication strategies for attention and MoE.

Implements modeling for forward/backward TP collectives for 1D/2D weight sharding variants.
"""

from abc import ABC, abstractmethod

from scalescope.model_spec import TransformerConfig
from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_all_reduce_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.topology_spec import LogicalTopology
from scalescope.workload_spec import TensorParallelVariant


class TensorParallelStrategy(ABC):
    """Abstract base class for tensor parallel communication strategies."""

    @abstractmethod
    def forward_comm(
        self, transformer_config: TransformerConfig, moe_config, workload_spec, n_seq: int, **kwargs
    ) -> CommStats:
        """Calculates forward pass communication. Common parameters are explicit; variant-specific are kwargs."""
        pass

    @abstractmethod
    def backward_comm(
        self, transformer_config: TransformerConfig, moe_config, workload_spec, n_seq: int, **kwargs
    ) -> CommStats:
        """Calculates backward pass communication. Common parameters are explicit; variant-specific are kwargs."""
        pass


class TP1DAttnStrategy(TensorParallelStrategy):
    """Models 1D TP for the attention block."""

    def forward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology = None,
        **kwargs,
    ) -> CommStats:
        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        volume = float_size * n_batch * n_seq * d_model
        return hockney_model_all_reduce_on_nd_topology(
            attn_tp_topology, volume, float_size_bytes=float_size
        )

    def backward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology = None,
        **kwargs,
    ) -> CommStats:
        # Backward gradient comm is same as forward activation comm for this strategy.
        fwd_comm = self.forward_comm(
            transformer_config, moe_config, workload_spec, n_seq, attn_tp_topology, **kwargs
        )
        # Layernorm communication is part of the attention block's backward pass
        layernorm_comm_vol = 2 * 2 * workload_spec.float_size_bytes
        ln_comm = hockney_model_all_reduce_on_nd_topology(
            attn_tp_topology,
            layernorm_comm_vol,
            float_size_bytes=workload_spec.float_size_bytes,
        )
        return fwd_comm + ln_comm


class TP1DMoEStrategy(TensorParallelStrategy):
    """Models 1D TP for the MoE/FFN block."""

    def forward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        **kwargs,
    ) -> CommStats:
        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes

        total_comm_stats = CommStats(0, 0, 0, 0)
        num_shared = moe_config.num_shared_experts_per_dp_rank
        if num_shared > 0:
            raise NotImplementedError("Shared experts not implemented in 1D TP MoE strategy.")

        total_num_routed_experts = transformer_config.num_regular_experts
        if total_num_routed_experts > 0:
            tokens_per_set = (
                workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
            )
            traffic_mult = moe_config.regular_expert_traffic_multiplier
            tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
            # max volume that an expert could receive, scaled by traffic multiplier
            token_volume_per_routed_expert = (
                tokens_per_routed_expert * traffic_mult * d_model * float_size
            )
            # Scale comm by number of collocated experts per chip
            token_volume_per_routed_expert *= moe_config.num_colocated_experts

            total_comm_stats += hockney_model_all_reduce_on_nd_topology(
                moe_tp_topology, token_volume_per_routed_expert, float_size_bytes=float_size
            )
            # A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize
            # Mixture-of-Experts Training
            # https://arxiv.org/pdf/2303.06318
            total_comm_stats += hockney_model_all_gather_on_nd_topology(
                moe_tp_topology, token_volume_per_routed_expert, float_size_bytes=float_size
            )
        return total_comm_stats

    def backward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        **kwargs,
    ) -> CommStats:
        return self.forward_comm(
            transformer_config, moe_config, workload_spec, n_seq, moe_tp_topology, **kwargs
        )


class TP2DAttnStrategy(TensorParallelStrategy):
    """Models 2D TP for the attention block."""

    def forward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        **kwargs,
    ) -> CommStats:
        # Logic from _calculate_2d_prefill_attention_comm_per_layer
        tp_topology = attn_tp_topology
        x_dims = x_dims or [0]
        yz_dims = yz_dims or [1]
        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        volume = float_size * n_batch * n_seq * d_model

        shape = getattr(tp_topology, "shape", [])

        def _prod(indices):
            p = 1
            for idx in indices:
                p *= shape[idx]
            return p

        tp_x = _prod(x_dims) if x_dims else 1
        tp_yz = _prod(yz_dims) if yz_dims else 1

        comm1 = hockney_model_all_reduce_on_nd_topology(
            tp_topology, volume / tp_x, float_size_bytes=float_size
        )
        comm2 = hockney_model_all_reduce_on_nd_topology(
            tp_topology, volume / tp_yz, float_size_bytes=float_size
        )
        return comm1 + comm2

    def backward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        **kwargs,
    ) -> CommStats:
        return self.forward_comm(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            attn_tp_topology,
            x_dims,
            yz_dims,
            **kwargs,
        )


class TP2DMoEStrategy(TensorParallelStrategy):
    """Models 2D TP for the MoE/FFN block."""

    def forward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        **kwargs,
    ) -> CommStats:
        # Logic from _calculate_2d_prefill_moe_comm_per_layer
        tp_topology = moe_tp_topology
        x_dims = x_dims or [0]
        yz_dims = yz_dims or [1]
        d_model = transformer_config.d_model
        mlp_ratio = transformer_config.mlp_ratio
        float_size = workload_spec.float_size_bytes

        shape = getattr(tp_topology, "shape", [])

        def _prod(indices):
            p = 1
            for idx in indices:
                p *= shape[idx]
            return p

        tp_x = _prod(x_dims) if x_dims else 1
        tp_yz = _prod(yz_dims) if yz_dims else 1

        # Use tokens-per-routed-expert volume and scale by colocation (parity with 1D path)
        total_comm = CommStats(0, 0, 0, 0)
        if moe_config.num_shared_experts_per_dp_rank > 0:
            raise NotImplementedError("Shared experts not implemented in 2D TP MoE strategy.")

        total_num_routed_experts = transformer_config.num_regular_experts
        if total_num_routed_experts > 0:
            tokens_per_set = (
                workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
            )
            traffic_mult = moe_config.regular_expert_traffic_multiplier
            tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
            token_volume = tokens_per_routed_expert * traffic_mult * d_model * float_size
            token_volume *= moe_config.num_colocated_experts

            comm_yz = hockney_model_all_reduce_on_nd_topology(
                tp_topology, token_volume / tp_x, float_size_bytes=float_size
            )
            comm_x = hockney_model_all_reduce_on_nd_topology(
                tp_topology, (mlp_ratio * token_volume) / tp_yz, float_size_bytes=float_size
            )
            total_comm = comm_yz + comm_x

        return total_comm

    def backward_comm(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        **kwargs,
    ) -> CommStats:
        return self.forward_comm(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            moe_tp_topology,
            x_dims,
            yz_dims,
            **kwargs,
        )


def get_tp_strategies(
    variant: TensorParallelVariant,
) -> tuple[TensorParallelStrategy, TensorParallelStrategy]:
    """Factory to create the TP strategies for Attn and MoE based on a variant."""
    match variant:
        case TensorParallelVariant.SHARD_WEIGHTS_1D:
            return TP1DAttnStrategy(), TP1DMoEStrategy()
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            return TP2DAttnStrategy(), TP2DMoEStrategy()
        case TensorParallelVariant.SHARD_ACTIVATIONS | TensorParallelVariant.NONE:
            raise NotImplementedError(f"TP variant {variant} is not implemented.")
        case _:
            raise ValueError(f"Unknown TP variant: {variant}")


def get_2d_tp_dims(tp_topology) -> tuple[list[int], list[int]]:
    """Returns (x_dims, yz_dims) for 2D TP, defaulting to first dim as X and the rest as YZ.

    This is a convention used across comm/memory to split TP into two groups.
    """
    shape = getattr(tp_topology, "shape", None)
    if not shape or len(shape) < 2:
        raise ValueError("2D TP requires a TP topology with at least 2 dimensions")
    # First dim -> X, remaining dims -> YZ
    x_dims = [0]
    yz_dims = [i for i in range(1, len(shape))]
    return x_dims, yz_dims


def validate_2d_tp_invariants(model_spec, tp_topology, x_dims: list[int], yz_dims: list[int]):
    """Validate basic divisibility constraints for 2D TP.

    - d_model divisible by TP total size
    - num_heads divisible by TP X size
    - num_kv_heads divisible by TP X size
    """
    shape = getattr(tp_topology, "shape", [])

    def _prod(indices):
        p = 1
        for idx in indices:
            p *= shape[idx]
        return p

    tp_x = _prod(x_dims) if x_dims else 1
    tp_yz = _prod(yz_dims) if yz_dims else 1
    tp_total = tp_x * tp_yz

    d_model = model_spec.d_model
    num_heads = model_spec.num_heads
    num_kv_heads = getattr(model_spec, "num_kv_heads", num_heads)

    if d_model % tp_total != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by 2D TP total ({tp_total})")
    if num_heads % tp_x != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by TP-X size ({tp_x})")
    if num_kv_heads % tp_x != 0:
        raise ValueError(f"num_kv_heads ({num_kv_heads}) must be divisible by TP-X size ({tp_x})")

    # Additional FFN divisibility for 2D sharding: d_ff splits across YZ
    mlp_ratio = getattr(model_spec, "mlp_ratio", None)
    if mlp_ratio is not None:
        try:
            d_ff = int(model_spec.d_model * mlp_ratio)
        except Exception:
            d_ff = None
        if d_ff is not None and tp_yz > 0 and d_ff % tp_yz != 0:
            raise ValueError(
                f"FFN hidden size d_ff ({d_ff}) must be divisible by TP-YZ size ({tp_yz}) for 2D TP"
            )
