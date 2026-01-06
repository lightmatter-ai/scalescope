"""Defines low-level network performance models and topology-aware collectives.
Contains Hockney-style analytic models (alpha/beta) for collective operations
on N-D logical topologies and related helpers to create topology views for use with TP/DP/EP.
"""

import math
from typing import Callable, List, Optional

from scalescope.stats import CommStats
from scalescope.topology_spec import (
    LogicalTopology,
    PhysicalTopology1DType,
)

# All functions in this file assume bandwidth is provided in BYTES PER SECOND (Bps).
# Any conversion from bits per second (bps) must happen before calling these models.

# 1D collective models (Bps)


def hockney_model_all_reduce_on_ring_1d(
    N: int,
    P_bytes: float,
    alpha_sec: float,
    bandwidth_Bps: float,
    gamma_sec_per_byte: Optional[float],
) -> CommStats:
    """Calculates time for 1D Ring All-Reduce.

    Args:
        N: Number of GPUs/nodes in the ring.
        P_bytes: Total data size per node (Bytes).
        alpha_sec: Startup latency per message (seconds).
        bandwidth_Bps: Unidirectional bandwidth per GPU (Bytes per second).
        gamma_sec_per_byte: Computation time per byte (seconds/Byte), or None.

    Returns:
        CommStats object with performance metrics.
    """
    if N <= 1:
        return CommStats()

    startup_latency = 2 * (N - 1) * alpha_sec
    comm_vol_bytes = 2 * (N - 1) / N * P_bytes
    transmission_time = comm_vol_bytes / bandwidth_Bps if bandwidth_Bps > 0 else float("inf")

    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = (N - 1) * gamma_sec_per_byte * (P_bytes / N)

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time)


def hockney_model_all_reduce_on_fully_connected_1d(
    N: int,
    P_bytes: float,
    alpha_sec: float,
    bandwidth_Bps: float,
    gamma_sec_per_byte: Optional[float],
) -> CommStats:
    """Calculates time for 1D Fully Connected All-Reduce.

    Args:
        N: Number of GPUs/nodes.
        P_bytes: Total data size per node (Bytes).
        alpha_sec: Startup latency per message (seconds).
        bandwidth_Bps: Unidirectional bandwidth per GPU (Bytes per second).
        gamma_sec_per_byte: Computation time per byte (seconds/Byte), or None.

    Returns:
        CommStats object with performance metrics.
    """
    if N <= 1:
        return CommStats()

    startup_latency = 2 * alpha_sec
    comm_vol_bytes = 2 * (N - 1) / N * P_bytes
    transmission_time = comm_vol_bytes / bandwidth_Bps if bandwidth_Bps > 0 else float("inf")

    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = (N - 1) * gamma_sec_per_byte * (P_bytes / N)

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time)


# Auxiliary 1D step functions for N-D models (Bps)
# These model a single phase (e.g., reduce-scatter) of a larger algorithm.


def aux_ring_reduce_scatter_nd(
    d_size: int,
    P_dim_before_rs_bytes: float,
    alpha_sec: float,
    B_dim_Bps: float,
    gamma_sec_per_byte: Optional[float],
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    startup_latency = (d_size - 1) * alpha_sec
    comm_vol_bytes = P_dim_before_rs_bytes * (d_size - 1) / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = (d_size - 1) * gamma_sec_per_byte * (P_dim_before_rs_bytes / d_size)
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time)


def aux_ring_all_gather_nd(
    d_size: int, P_dim_after_ag_bytes: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    startup_latency = (d_size - 1) * alpha_sec
    comm_vol_bytes = P_dim_after_ag_bytes * (d_size - 1) / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0)


def aux_fully_connected_reduce_scatter_nd(
    d_size: int,
    P_dim_before_rs_bytes: float,
    alpha_sec: float,
    B_dim_Bps: float,
    gamma_sec_per_byte: Optional[float],
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    startup_latency = alpha_sec
    comm_vol_bytes = P_dim_before_rs_bytes * (d_size - 1) / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = (d_size - 1) * gamma_sec_per_byte * (P_dim_before_rs_bytes / d_size)
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time)


def aux_fully_connected_all_gather_nd(
    d_size: int, P_dim_after_ag_bytes: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    startup_latency = alpha_sec
    comm_vol_bytes = P_dim_after_ag_bytes * (d_size - 1) / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0)


# N-D topology collectives (Bps)


def hockney_model_reduce_scatter_on_nd_topology(
    view: LogicalTopology, P_total_bytes_before_rs: float, float_size_bytes: int
) -> CommStats:
    """
    Calculates ReduceScatter time on an N-D topology by modeling all possible
    execution pipelines ("colors") and returning the performance of the slowest one.

    """
    if not view.dimensions:
        return CommStats()
    num_colors = len(view.dimensions)
    if num_colors == 0:
        return CommStats()

    all_pipeline_stats: List[CommStats] = []

    # Iterate through each possible starting dimension to model each "color" or pipeline.
    for i in range(num_colors):
        # Create a cyclically shifted list of dimensions for this specific pipeline.
        reordered_dims = view.dimensions[i:] + view.dimensions[:i]

        # Calculate performance for this single pipeline
        pipeline_stats = CommStats()
        # Calculate the data size per color for this pipeline.
        current_P_rs = P_total_bytes_before_rs / num_colors
        for l_dim_map in reordered_dims:
            #  bandwidth is shared between all colors
            d_size, alpha_sec, effective_B_Bps, gamma = view.get_hockney_model_params(
                l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
            )
            # no comm is needed
            if d_size <= 1:
                if d_size > 0:
                    current_P_rs /= d_size
                continue

            topo_type = view.get_physical_dimension_for_tag(l_dim_map.logical_tag).topology_type
            step_stats: CommStats
            match topo_type:
                case PhysicalTopology1DType.RING:
                    step_stats = aux_ring_reduce_scatter_nd(
                        d_size, current_P_rs, alpha_sec, effective_B_Bps, gamma
                    )

                # Use the | (OR) operator to handle multiple matching cases
                case PhysicalTopology1DType.FULLY_CONNECTED | PhysicalTopology1DType.SWITCHED:
                    step_stats = aux_fully_connected_reduce_scatter_nd(
                        d_size, current_P_rs, alpha_sec, effective_B_Bps, gamma
                    )

                # The wildcard case `_` handles all other possibilities (like an else block)
                case _:
                    raise ValueError(f"RS: Unsupported 1D topology type: {topo_type}")
            pipeline_stats += step_stats
            if d_size > 0:
                current_P_rs /= d_size

        all_pipeline_stats.append(pipeline_stats)

    # The operation is only as fast as its slowest pipeline.
    max_startup = max(s.startup_latency for s in all_pipeline_stats)
    max_transmission = max(s.transmission_time for s in all_pipeline_stats)
    # Compute and volume are constant across pipelines.
    total_compute = all_pipeline_stats[0].compute_time
    total_volume = all_pipeline_stats[0].comm_vol

    return CommStats(max_startup, max_transmission, total_volume, total_compute)


def hockney_model_all_gather_on_nd_topology(
    view: LogicalTopology, P_total_bytes_after_ag: float, float_size_bytes: int
) -> CommStats:
    """
    Calculates AllGather time on an N-D topology by modeling all possible
    execution pipelines ("colors") and returning the performance of the slowest one.
    """
    if not view.dimensions:
        return CommStats()
    num_colors = len(view.dimensions)
    if num_colors == 0:
        return CommStats()

    all_pipeline_stats: List[CommStats] = []

    # Iterate through each possible starting dimension to model each "color" or pipeline.
    for i in range(num_colors):
        # Create a cyclically shifted list of dimensions for this specific pipeline.
        reordered_dims = view.dimensions[i:] + view.dimensions[:i]

        # Calculate performance for this single pipeline
        pipeline_stats = CommStats()
        product_of_dims = math.prod(d.size for d in reordered_dims if d.size > 0)
        current_P_ag = P_total_bytes_after_ag / product_of_dims if product_of_dims > 0 else 0
        current_P_ag /= num_colors  # Data per color for this pipeline.

        # Process in REVERSE order for All-Gather.
        for l_dim_map in reversed(reordered_dims):
            #  bandwidth is shared between all colors
            d_size, alpha_sec, effective_B_Bps, _ = view.get_hockney_model_params(
                l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
            )

            if d_size > 0:
                current_P_ag *= d_size

            # no comm is needed
            if d_size <= 1:
                continue

            topo_type = view.get_physical_dimension_for_tag(l_dim_map.logical_tag).topology_type
            step_stats: CommStats
            match topo_type:
                case PhysicalTopology1DType.RING:
                    step_stats = aux_ring_all_gather_nd(
                        d_size, current_P_ag, alpha_sec, effective_B_Bps
                    )

                # Use the | (OR) operator to handle multiple matching cases
                case PhysicalTopology1DType.FULLY_CONNECTED | PhysicalTopology1DType.SWITCHED:
                    step_stats = aux_fully_connected_all_gather_nd(
                        d_size, current_P_ag, alpha_sec, effective_B_Bps
                    )

                # The wildcard case `_` handles all other possibilities
                case _:
                    raise ValueError(f"AG: Unsupported 1D topology type: {topo_type}")

            pipeline_stats += step_stats

        all_pipeline_stats.append(pipeline_stats)

    # The operation is only as fast as its slowest pipeline.
    max_startup = max(s.startup_latency for s in all_pipeline_stats)
    max_transmission = max(s.transmission_time for s in all_pipeline_stats)
    # Volume is constant across pipelines. All-Gather has no compute time.
    total_volume = all_pipeline_stats[0].comm_vol

    return CommStats(
        startup_latency=max_startup, transmission_time=max_transmission, comm_vol=total_volume
    )


def hockney_model_all_reduce_on_nd_topology(
    view: LogicalTopology, P_total_bytes: float, float_size_bytes: int
) -> CommStats:
    """
    Calculates All-Reduce time by composing the full, pipeline-aware Reduce-Scatter
    and All-Gather models.

    This function represents All-Reduce as a simple, sequential composition of
    two phases. All the complexity of modeling different execution pipelines ("colors")
    is now encapsulated within the `ReduceScatter` and `AllGather` functions themselves.

    Returns:
        A CommStats object representing the total performance of the N-D All-Reduce.
    """
    # Step 1: Calculate the performance of the Reduce-Scatter phase, which internally
    # models all possible pipelines and returns the stats for the slowest one.
    reduce_scatter_stats = hockney_model_reduce_scatter_on_nd_topology(
        view=view, P_total_bytes_before_rs=P_total_bytes, float_size_bytes=float_size_bytes
    )

    # Step 2: Calculate the performance of the All-Gather phase, which also
    # models all pipelines and returns the stats for its slowest one.
    all_gather_stats = hockney_model_all_gather_on_nd_topology(
        view=view, P_total_bytes_after_ag=P_total_bytes, float_size_bytes=float_size_bytes
    )

    # Step 3: The total performance is the sum of the two phases.
    return reduce_scatter_stats + all_gather_stats


# Rotal Exchange (All-to-All) Models


def aux_ring_total_exchange_nd(
    d_size: int, P_data_per_node_B: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    """1D Ring Total Exchange (All-to-All Personalized)."""
    if d_size <= 1:
        return CommStats()
    startup_latency = (d_size - 1) * alpha_sec
    # the first 2.0 account for average case of message size being half of total data.
    # as each node takes its part of the message and sends the rest to the next node
    # the second 2.0 is for sending message in both directions
    comm_vol_bytes = P_data_per_node_B * (d_size - 1) / 2.0 / 2.0
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0)


def aux_fully_connected_total_exchange_nd(
    d_size: int, P_data_per_node_B: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    """1D Fully Connected Total Exchange (All-to-All Personalized)."""
    if d_size <= 1:
        return CommStats()
    startup_latency = alpha_sec
    comm_vol_bytes = P_data_per_node_B * (d_size - 1) / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0)


def hockney_model_total_exchange_on_nd_topology(
    view: LogicalTopology, P_total_send_bytes_per_node: float, float_size_bytes: int
) -> CommStats:
    """
    Calculate time for total exchange (All-to-All) by modeling all possible
    execution pipelines ("colors") and returning the performance of the slowest one.

    This model recognizes that an N-D All-to-All can be initiated along any of
    its N dimensions. It calculates the performance for each of these N distinct
    algorithmic pipelines and returns the statistics for the bottleneck pipeline.

    Note: this is a standard algorithm to do All-to-All in N-D topologies,
    for advanced algorithm see All-port Total Exchange in Cartesian Product Networks
    by Vassilios V. Dimakopoulos
    https://paragroup.cse.uoi.gr/publications/files/82jpdc05.pdf

    Args:
        view: The LogicalTopology for the All-to-All operation.
        P_total_send_bytes_per_node: Total data a single node sends (Bytes).
        float_size_bytes: Size of floating-point numbers (used for parameter fetching).

    Returns:
        A CommStats object representing the performance of the slowest All-to-All pipeline.
    """
    if not view.dimensions:
        return CommStats()
    num_colors = len(view.dimensions)
    if num_colors == 0:
        return CommStats()

    # Pre-calculate the total product of dimension sizes, as it's constant across all pipelines.
    product_of_dims = math.prod(d.size for d in view.dimensions if d.size > 0)
    if product_of_dims == 0:
        return CommStats()

    all_pipeline_stats: list[CommStats] = []

    P_total_send_bytes_per_node_per_color = P_total_send_bytes_per_node / num_colors

    # Iterate through each possible starting dimension to model each "color" or pipeline.
    for i in range(num_colors):
        # Create a cyclically shifted list of dimensions for this specific pipeline's execution order.
        reordered_dims = view.dimensions[i:] + view.dimensions[:i]

        pipeline_stats = CommStats()

        # Calculate the performance for this single pipeline by processing its dimensions in order.
        for l_dim_map in reordered_dims:
            d_size = l_dim_map.size
            if d_size <= 1:
                continue

            # For sequential A2A, each 1D step gets exclusive use of its bandwidth slice,
            # but because we use color we allow concurrent use of the bandwidth
            _, alpha_sec, effective_B_Bps, _ = view.get_hockney_model_params(
                l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
            )

            # Data processed in a 1D A2A along dimension `d` is the total data
            # divided by the product of all *other* dimension sizes.
            other_dims_product = product_of_dims / d_size
            P_data_for_1D_step = P_total_send_bytes_per_node_per_color / other_dims_product

            topo_type = view.get_physical_dimension_for_tag(l_dim_map.logical_tag).topology_type

            op_fn: Callable
            match topo_type:
                case PhysicalTopology1DType.RING:
                    op_fn = aux_ring_total_exchange_nd
                case PhysicalTopology1DType.FULLY_CONNECTED | PhysicalTopology1DType.SWITCHED:
                    op_fn = aux_fully_connected_total_exchange_nd
                case _:
                    raise ValueError(f"TotalExchange: Unsupported 1D topology type: {topo_type}")

            # Accumulate the stats for each step in this pipeline.
            pipeline_stats += op_fn(d_size, P_data_for_1D_step, alpha_sec, effective_B_Bps)

        all_pipeline_stats.append(pipeline_stats)

    if not all_pipeline_stats:
        return CommStats()

    # The operation is only as fast as its slowest pipeline. Find the max time components.
    max_startup = max(s.startup_latency for s in all_pipeline_stats)
    max_transmission = max(s.transmission_time for s in all_pipeline_stats)

    # Communication volume is constant across all valid pipelines.
    total_volume = all_pipeline_stats[0].comm_vol

    # All-to-All is assumed to have no compute time in this model.
    return CommStats(max_startup, max_transmission, total_volume, compute_time=0)
