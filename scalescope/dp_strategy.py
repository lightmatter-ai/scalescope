"""DP (data-parallel) gradient communication strategies.

Defines abstract and concrete DP strategies (DDP, ZeRO stages 1–3) that
estimate per-layer and per-stage traffic and time.
"""

from abc import ABC, abstractmethod

from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_all_reduce_on_nd_topology,
    hockney_model_reduce_scatter_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.topology_spec import LogicalTopology
from scalescope.workload_spec import DataParallelVariant


def _reduce_scatter_followed_by_all_gather(topology, rs_data_volume_B, float_size):
    """Helper to model Reduce-Scatter followed by All-Gather."""
    rs_comm_stats = hockney_model_reduce_scatter_on_nd_topology(
        topology, rs_data_volume_B, float_size_bytes=float_size
    )
    n_chips = topology.num_chips
    ag_data_volume = rs_data_volume_B / n_chips
    ag_comm_stats = hockney_model_all_gather_on_nd_topology(
        topology, ag_data_volume, float_size_bytes=float_size
    )
    return rs_comm_stats + ag_comm_stats


class DataParallelStrategy(ABC):
    """Abstract base class for data parallel communication strategies."""

    @abstractmethod
    def forward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_per_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        """Calculates forward pass communication for a model component."""
        pass

    @abstractmethod
    def backward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_per_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        """Calculates backward pass communication for a model component."""
        pass


class DDPStrategy(DataParallelStrategy):
    """Models communication for Distributed Data Parallel (DDP)."""

    def forward_comm(self, *args, **kwargs) -> CommStats:
        return CommStats(0, 0, 0, 0)

    def backward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_per_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        data_volume_per_layer_per_xpu = float_size * layer_params_per_xpu
        return num_layers_per_pp_stage * hockney_model_all_reduce_on_nd_topology(
            dp_topology, data_volume_per_layer_per_xpu, float_size_bytes=float_size
        )


class ZeROStage1Strategy(DataParallelStrategy):
    """Models communication for ZeRO Stage 1."""

    def forward_comm(self, *args, **kwargs) -> CommStats:
        return CommStats(0, 0, 0, 0)

    def backward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_per_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        data_volume = float_size * layer_params_per_xpu
        return num_layers_per_pp_stage * _reduce_scatter_followed_by_all_gather(
            dp_topology, data_volume, float_size
        )


class ZeROStage2Strategy(DataParallelStrategy):
    """Models communication for ZeRO Stage 2."""

    def forward_comm(self, *args, **kwargs) -> CommStats:
        return CommStats(0, 0, 0, 0)

    def backward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_per_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        data_volume = float_size * layer_params_per_xpu
        return num_layers_per_pp_stage * _reduce_scatter_followed_by_all_gather(
            dp_topology, data_volume, float_size
        )


class ZeROStage3Strategy(DataParallelStrategy):
    """Models communication for ZeRO Stage 3."""

    def forward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_per_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        total_params = num_layers_per_pp_stage * layer_params_per_xpu
        weights_volume = float_size * total_params
        return hockney_model_all_gather_on_nd_topology(
            dp_topology, weights_volume, float_size_bytes=float_size
        )

    def backward_comm(
        self,
        dp_topology: LogicalTopology,
        num_layers_pp_stage: int,
        layer_params_per_xpu: int,
        float_size: int,
    ) -> CommStats:
        # All-gather weights
        total_params_per_xpu = num_layers_pp_stage * layer_params_per_xpu
        weights_volume = float_size * total_params_per_xpu
        comm_stats = hockney_model_all_gather_on_nd_topology(
            dp_topology, weights_volume, float_size_bytes=float_size
        )
        # Reduce-scatter gradients
        gradient_volume_per_layer = float_size * layer_params_per_xpu
        comm_stats += num_layers_pp_stage * hockney_model_reduce_scatter_on_nd_topology(
            dp_topology, gradient_volume_per_layer, float_size_bytes=float_size
        )
        return comm_stats


def get_dp_strategy(variant: DataParallelVariant) -> DataParallelStrategy:
    """Factory to create a DP strategy based on a variant."""
    match variant:
        case DataParallelVariant.DDP:
            return DDPStrategy()
        case DataParallelVariant.ZERO_STAGE_1:
            return ZeROStage1Strategy()
        case DataParallelVariant.ZERO_STAGE_2:
            return ZeROStage2Strategy()
        case DataParallelVariant.ZERO_STAGE_3:
            return ZeROStage3Strategy()
        case _:
            raise ValueError(f"Unsupported DP variant: {variant}")
