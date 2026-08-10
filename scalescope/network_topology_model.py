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

    num_steps = N - 1
    num_peers = N - 1
    num_phases = 2 * num_steps  # reduce-scatter + all-gather over ring
    startup_latency = num_phases * alpha_sec
    comm_vol_bytes = 2 * num_peers / N * P_bytes
    transmission_time = comm_vol_bytes / bandwidth_Bps if bandwidth_Bps > 0 else float("inf")

    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = num_peers * gamma_sec_per_byte * (P_bytes / N)

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time, num_phases)


# ---------------------------------------------------------------------------
# LayerNorm stats All-Reduce: topology-aware small-message variants
#
# LayerNorm requires (mean, variance) across the full d_model when d_model is
# sharded.  The payload is tiny (2 floats per token = 4–8 bytes in fp16/fp32),
# so startup latency dominates.
#
# Decomposition: Allgather + local reduce  (NOT the standard RS + AG).
#
# Conceptually each node broadcasts/multicasts its stats to all peers
# (= Allgather), and every node reduces the received values locally.
# The Allgather can be overlapped with reduction on-the-fly, so no extra
# memory for N copies is needed.
#
# On ring:  Allgather is (N-1) forwarding steps → halves alpha vs RS+AG.
# On FC:    Allgather is 1 multicast phase → same alpha as standard AG.
#           Standard RS+AG (2α) is used since the extra 1α saved by
#           single-phase multicast is negligible and costs N× bandwidth.
# ---------------------------------------------------------------------------


def layernorm_stats_allgather_reduce_on_ring_1d(
    N: int,
    P_bytes: float,
    alpha_sec: float,
    bandwidth_Bps: float,
    gamma_sec_per_byte: Optional[float] = None,
) -> CommStats:
    """LayerNorm stats on ring: Allgather + local reduce (single ring pass).

    Each node's original stats buffer (P bytes) ripples around the ring.
    At each hop every node receives one peer's original data and reduces
    it into a local accumulator.  After N-1 hops every node holds the
    full sum — equivalent to Allgather + reduce, fused into one pass.

    Cost model (Hockney):
        alpha  = (N-1) × α           — one startup per hop
        beta   = (N-1) × P / B       — full P forwarded at each hop
        volume = (N-1) × P

    Comparison with standard RS+AG:
        alpha  = 2(N-1) × α          — two full ring passes
        beta   = 2(N-1)/N × P / B    — bandwidth-optimal
        volume = 2(N-1)/N × P

    Allgather+reduce halves alpha at the cost of ~N/2× more bandwidth.
    Crossover (Allgather+reduce wins):
        P < α × B × N/(N-2)
        For N=32, α=150ns, B=3.2TB/s  ⟹  P_cross ≈ 512 bytes.
    LN stats (4–8 bytes) are well below this threshold.
    """
    if N <= 1:
        return CommStats()

    num_steps = N - 1
    startup_latency = num_steps * alpha_sec
    comm_vol_bytes = num_steps * P_bytes
    transmission_time = comm_vol_bytes / bandwidth_Bps if bandwidth_Bps > 0 else float("inf")

    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = num_steps * gamma_sec_per_byte * P_bytes

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time, num_steps)


def layernorm_stats_allgather_reduce_on_fc_1d(
    N: int,
    P_bytes: float,
    alpha_sec: float,
    bandwidth_Bps: float,
    gamma_sec_per_byte: Optional[float] = None,
) -> CommStats:
    """LayerNorm stats on FC: forward to standard 2-phase AR (RS+AG).

    On FC the standard 2-phase AR has minimal alpha (2α); Allgather+reduce
    is not beneficial (see router_a2a_integration_note.md). Same cost model.
    """
    return hockney_model_all_reduce_on_fully_connected_1d(
        N, P_bytes, alpha_sec, bandwidth_Bps, gamma_sec_per_byte
    )


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

    # FC all-reduce is modeled as 2 communication phases (startup), while
    # volume scales with the number of peer exchanges (N-1)/N.
    num_peers = N - 1
    startup_latency = 2 * alpha_sec
    comm_vol_bytes = 2 * num_peers / N * P_bytes
    transmission_time = comm_vol_bytes / bandwidth_Bps if bandwidth_Bps > 0 else float("inf")

    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = num_peers * gamma_sec_per_byte * (P_bytes / N)

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time, 2)


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
    num_steps = d_size - 1
    num_peers = d_size - 1
    startup_latency = num_steps * alpha_sec
    comm_vol_bytes = P_dim_before_rs_bytes * num_peers / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = num_peers * gamma_sec_per_byte * (P_dim_before_rs_bytes / d_size)
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time, num_steps)


def aux_ring_all_gather_nd(
    d_size: int, P_dim_after_ag_bytes: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    num_steps = d_size - 1
    num_peers = d_size - 1
    startup_latency = num_steps * alpha_sec
    comm_vol_bytes = P_dim_after_ag_bytes * num_peers / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0, num_steps)


def aux_fully_connected_reduce_scatter_nd(
    d_size: int,
    P_dim_before_rs_bytes: float,
    alpha_sec: float,
    B_dim_Bps: float,
    gamma_sec_per_byte: Optional[float],
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    num_peers = d_size - 1
    startup_latency = alpha_sec
    comm_vol_bytes = P_dim_before_rs_bytes * num_peers / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    compute_time = 0.0
    if gamma_sec_per_byte is not None and gamma_sec_per_byte != float("inf"):
        compute_time = num_peers * gamma_sec_per_byte * (P_dim_before_rs_bytes / d_size)
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, compute_time, 1)


def aux_fully_connected_all_gather_nd(
    d_size: int, P_dim_after_ag_bytes: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    if d_size <= 1:
        return CommStats()
    num_peers = d_size - 1
    startup_latency = alpha_sec
    comm_vol_bytes = P_dim_after_ag_bytes * num_peers / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0, 1)


def aux_ring_gather_nd(
    d_size: int,
    P_dim_after_gather_bytes: float,
    alpha_sec: float,
    B_dim_Bps: float,
) -> CommStats:
    """1D Ring Gather (to a root) for use inside N-D "color" models.

    Semantics:
    - This is a *gather*, not all-gather: only a single root ends up with the full buffer.
    - The input is assumed to be evenly sharded across the `d_size` ranks in this 1D subgroup,
      so each rank holds `P_dim_after_gather_bytes / d_size` bytes.

    Modeling notes:
    - Bandwidth semantics match the rest of the N-D models: `B_dim_Bps` is a **fixed total
      per-node bandwidth budget** for this 1D step (not per ring direction).
    - We assume a **bidirectional ring** for root-centric operations, so the maximum hop
      distance to the root is ~ceil((d_size-1)/2). This affects the startup term only.
    - The root must still receive the total non-root volume:
        root_rx_bytes = (d_size - 1) / d_size * P_dim_after_gather_bytes
      so the transmission term uses that total volume divided by the fixed bandwidth budget.

    This is intentionally latency-oriented: it captures the root's receive bottleneck without
    modeling finer-grained pipelining effects.
    """
    if d_size <= 1:
        return CommStats()
    num_hops = int(math.ceil((d_size - 1) / 2))
    startup_latency = num_hops * alpha_sec
    num_non_root_ranks = d_size - 1
    root_rx_bytes = P_dim_after_gather_bytes * num_non_root_ranks / d_size
    transmission_time = root_rx_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, root_rx_bytes, 0.0, num_hops)


def aux_fully_connected_gather_nd(
    d_size: int,
    P_dim_after_gather_bytes: float,
    alpha_sec: float,
    B_dim_Bps: float,
) -> CommStats:
    """1D Fully Connected Gather (to a root) for use inside N-D "color" models.

    Assumed algorithm:
    - Single round where all non-root ranks send their shard directly to the root.

    Bottleneck:
    - Root ingress: the root must receive (d_size-1)/d_size * P_dim_after_gather_bytes bytes
      total in this step.
    """
    if d_size <= 1:
        return CommStats()
    startup_latency = alpha_sec
    num_non_root_ranks = d_size - 1
    root_rx_bytes = P_dim_after_gather_bytes * num_non_root_ranks / d_size
    transmission_time = root_rx_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, root_rx_bytes, 0.0, 1)


# N-D topology collectives (Bps)


def _active_collective_dimensions(view: LogicalTopology):
    """Return dimensions that contribute real participants to an N-D collective.

    A size-one dimension sends no packets. It must not create an additional
    color or receive a fraction of the payload; otherwise an ``N x 1`` view is
    incorrectly priced as an N-D collective instead of the equivalent 1D one.
    """
    return [dimension for dimension in view.dimensions if dimension.size > 1]


def hockney_model_reduce_on_nd_topology(
    view: LogicalTopology, P_total_bytes: float, float_size_bytes: float
) -> CommStats:
    """Calculates Reduce (to a single root) time on an N-D topology.

    We model Reduce as a sequential composition:
      Reduce = ReduceScatter + Gather

    Rationale:
    - ReduceScatter performs the reductions while sharding the result across ranks.
    - Gather then assembles the sharded reduced result onto a single root rank.
    - This mirrors `hockney_model_all_reduce_on_nd_topology` (RS + AllGather), but replaces
      the second phase with a root-centric gather.

    Important modeling detail:
    - The N-D models use the "colors" concurrency model internally. `ReduceScatter` and
      `Gather` both model the slowest pipeline under that concurrency assumption, and we sum
      those phase costs here.
    """
    reduce_scatter_stats = hockney_model_reduce_scatter_on_nd_topology(
        view=view, P_total_bytes_before_rs=P_total_bytes, float_size_bytes=float_size_bytes
    )
    gather_stats = hockney_model_gather_on_nd_topology(
        view=view, P_total_bytes_after_gather=P_total_bytes, float_size_bytes=float_size_bytes
    )
    return reduce_scatter_stats + gather_stats


def hockney_model_reduce_scatter_on_nd_topology(
    view: LogicalTopology, P_total_bytes_before_rs: float, float_size_bytes: float
) -> CommStats:
    """
    Calculates ReduceScatter time on an N-D topology by modeling all possible
    execution pipelines ("colors") and returning the performance of the slowest one.

    """
    active_dimensions = _active_collective_dimensions(view)
    if not active_dimensions:
        return CommStats()
    num_colors = len(active_dimensions)

    all_pipeline_stats: List[CommStats] = []

    # Iterate through each possible starting dimension to model each "color" or pipeline.
    for i in range(num_colors):
        # Only active dimensions define colors. Including a size-one dimension
        # here would divide the payload without adding a communication path.
        reordered_dims = active_dimensions[i:] + active_dimensions[:i]

        # Calculate performance for this single pipeline
        pipeline_stats = CommStats()
        # Calculate the data size per color for this pipeline.
        current_P_rs = P_total_bytes_before_rs / num_colors
        for l_dim_map in reordered_dims:
            #  bandwidth is shared between all colors
            d_size, alpha_sec, effective_B_Bps, gamma = view.get_hockney_model_params(
                l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
            )
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
    # Colors carry disjoint payload partitions and execute concurrently. Time is
    # therefore the slowest pipeline, while wire volume is the sum over colors.
    # Aggregate volume cannot be divided by one bandwidth to reconstruct time
    # when colors use different fabrics.
    total_compute = all_pipeline_stats[0].compute_time
    total_volume = sum(stats.comm_vol for stats in all_pipeline_stats)

    max_phase_count = max(s.startup_phase_count for s in all_pipeline_stats)
    return CommStats(max_startup, max_transmission, total_volume, total_compute, max_phase_count)


def hockney_model_all_gather_on_nd_topology(
    view: LogicalTopology, P_total_bytes_after_ag: float, float_size_bytes: float
) -> CommStats:
    """
    Calculates AllGather time on an N-D topology by modeling all possible
    execution pipelines ("colors") and returning the performance of the slowest one.
    """
    active_dimensions = _active_collective_dimensions(view)
    if not active_dimensions:
        return CommStats()
    num_colors = len(active_dimensions)

    all_pipeline_stats: List[CommStats] = []

    # Iterate through each possible starting dimension to model each "color" or pipeline.
    for i in range(num_colors):
        # Degenerate axes are not colors and must not receive payload.
        reordered_dims = active_dimensions[i:] + active_dimensions[:i]

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
    # Each color gathers a disjoint payload partition. Preserve critical-path
    # timing above, but report aggregate per-rank wire traffic across colors.
    total_volume = sum(stats.comm_vol for stats in all_pipeline_stats)

    max_phase_count = max(s.startup_phase_count for s in all_pipeline_stats)
    return CommStats(
        startup_latency=max_startup,
        transmission_time=max_transmission,
        comm_vol=total_volume,
        startup_phase_count=max_phase_count,
    )


def hockney_model_gather_on_nd_topology(
    view: LogicalTopology, P_total_bytes_after_gather: float, float_size_bytes: float
) -> CommStats:
    """Calculates Gather (to a single root) time on an N-D topology.

    Implementation strategy ("colors"):
    - Mirrors `hockney_model_all_gather_on_nd_topology`'s N-D pipeline ("color") model:
      we model all possible cyclic dimension orderings and take the slowest pipeline.
    - Conceptually, gather is the "root-only" analogue of all-gather: the payload expands
      as we traverse dimensions in reverse order, but only the root needs to receive the
      assembled buffer.

    Bandwidth semantics:
    - Uses `view.get_hockney_model_params(..., context="concurrent_baseline")` so bandwidth is
      already divided across colors.
    - Per 1D step, we model the root's required receive volume as:
        root_rx_bytes = (d_size - 1) / d_size * P_dim_after_gather_bytes
      and divide by the fixed per-node bandwidth budget for that dimension.
    """
    active_dimensions = _active_collective_dimensions(view)
    if not active_dimensions:
        return CommStats()
    num_colors = len(active_dimensions)

    all_pipeline_stats: List[CommStats] = []

    for i in range(num_colors):
        # Match RS/AG semantics: a size-one axis neither communicates nor
        # creates a separate color.
        reordered_dims = active_dimensions[i:] + active_dimensions[:i]
        pipeline_stats = CommStats()

        # Start from the per-rank shard size (per color), then expand as we gather.
        product_of_dims = math.prod(d.size for d in reordered_dims if d.size > 0)
        current_P_gather = (
            P_total_bytes_after_gather / product_of_dims if product_of_dims > 0 else 0
        )
        current_P_gather /= num_colors

        # Gather expands in reverse dimension order (like all-gather).
        for l_dim_map in reversed(reordered_dims):
            d_size, alpha_sec, effective_B_Bps, _ = view.get_hockney_model_params(
                l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
            )

            if d_size > 0:
                current_P_gather *= d_size
            topo_type = view.get_physical_dimension_for_tag(l_dim_map.logical_tag).topology_type
            step_stats: CommStats
            match topo_type:
                case PhysicalTopology1DType.RING:
                    step_stats = aux_ring_gather_nd(
                        d_size, current_P_gather, alpha_sec, effective_B_Bps
                    )
                case PhysicalTopology1DType.FULLY_CONNECTED | PhysicalTopology1DType.SWITCHED:
                    step_stats = aux_fully_connected_gather_nd(
                        d_size, current_P_gather, alpha_sec, effective_B_Bps
                    )
                case _:
                    raise ValueError(f"Gather: Unsupported 1D topology type: {topo_type}")

            pipeline_stats += step_stats

        all_pipeline_stats.append(pipeline_stats)

    max_startup = max(s.startup_latency for s in all_pipeline_stats)
    max_transmission = max(s.transmission_time for s in all_pipeline_stats)
    # Gather colors carry disjoint payload partitions, so their wire bytes add
    # even though their execution time overlaps.
    total_volume = sum(stats.comm_vol for stats in all_pipeline_stats)
    max_phase_count = max(s.startup_phase_count for s in all_pipeline_stats)
    return CommStats(
        startup_latency=max_startup,
        transmission_time=max_transmission,
        comm_vol=total_volume,
        startup_phase_count=max_phase_count,
    )


def hockney_model_all_reduce_on_nd_topology(
    view: LogicalTopology, P_total_bytes: float, float_size_bytes: float
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
    num_steps = d_size - 1
    num_peers = d_size - 1
    startup_latency = num_steps * alpha_sec
    # the first 2.0 account for average case of message size being half of total data.
    # as each node takes its part of the message and sends the rest to the next node
    # the second 2.0 is for sending message in both directions
    comm_vol_bytes = P_data_per_node_B * num_peers / 2.0 / 2.0
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")

    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0, num_steps)


def aux_fully_connected_total_exchange_nd(
    d_size: int, P_data_per_node_B: float, alpha_sec: float, B_dim_Bps: float
) -> CommStats:
    """1D Fully Connected Total Exchange (All-to-All Personalized)."""
    if d_size <= 1:
        return CommStats()
    num_peers = d_size - 1
    startup_latency = alpha_sec
    comm_vol_bytes = P_data_per_node_B * num_peers / d_size
    transmission_time = comm_vol_bytes / B_dim_Bps if B_dim_Bps > 0 else float("inf")
    return CommStats(startup_latency, transmission_time, comm_vol_bytes, 0, 1)


def hockney_model_total_exchange_on_nd_topology(
    view: LogicalTopology, P_total_send_bytes_per_node: float, float_size_bytes: float
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

    all_pipeline_stats: list[CommStats] = []

    # For total exchange, each pipeline models a complete algorithm ordering.
    # Do not split payload by number of colors; colors are alternative orderings,
    # not concurrent partitions of the sender's payload.
    P_total_send_bytes_per_node_per_pipeline = P_total_send_bytes_per_node

    # Iterate through each possible starting dimension to model each "color" or pipeline.
    for i in range(num_colors):
        # Create a cyclically shifted list of dimensions for this specific pipeline's execution order.
        reordered_dims = view.dimensions[i:] + view.dimensions[:i]

        pipeline_stats = CommStats()

        # Calculate the performance for this single pipeline by processing dimensions in order.
        # For personalized exchange in this model, each dimension reflects a full destination split
        # over that axis. We therefore do not shrink the step payload by previously traversed dims;
        # otherwise slow-domain traffic gets undercounted.
        for l_dim_map in reordered_dims:
            d_size = l_dim_map.size
            if d_size <= 1:
                continue

            # For sequential A2A, each 1D step gets exclusive use of its bandwidth slice,
            # but because we use color we allow concurrent use of the bandwidth
            _, alpha_sec, effective_B_Bps, _ = view.get_hockney_model_params(
                l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
            )

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
            pipeline_stats += op_fn(
                d_size, P_total_send_bytes_per_node_per_pipeline, alpha_sec, effective_B_Bps
            )

        all_pipeline_stats.append(pipeline_stats)

    if not all_pipeline_stats:
        return CommStats()

    # The operation is only as fast as its slowest pipeline. Find the max time components.
    max_startup = max(s.startup_latency for s in all_pipeline_stats)
    max_transmission = max(s.transmission_time for s in all_pipeline_stats)

    # Communication volume is constant across all valid pipelines.
    total_volume = all_pipeline_stats[0].comm_vol

    # All-to-All is assumed to have no compute time in this model.
    max_phase_count = max(s.startup_phase_count for s in all_pipeline_stats)
    return CommStats(
        max_startup,
        max_transmission,
        total_volume,
        compute_time=0,
        startup_phase_count=max_phase_count,
    )


# Point-to-point (P2P) Models


def hockney_model_send_recv_on_nd_topology(
    topology: LogicalTopology, bytes_per_sender: float, float_size_bytes: float
) -> CommStats:
    """Estimate a simple point-to-point send/recv on a (typically 1D) topology view.

    This is intentionally a lightweight primitive for modeling *stage-boundary* transfers
    (e.g., pipeline-parallel activation sends) rather than collectives.

    Semantics / assumptions:
    - One sender sends a contiguous buffer of size `bytes_per_sender` to one receiver.
    - We model this as `alpha + P/B`, where `alpha` and `B` are taken from the
      interconnect backing the view.
    - If the topology has no dimensions or all sizes are 1, returns zero.
    - If the topology has multiple dimensions, we conservatively use the *slowest* dimension
      (minimum effective bandwidth, maximum latency) as the bottleneck.
    """
    if not topology.dimensions:
        return CommStats()

    # Identify bottleneck dimension in the view.
    worst_alpha = 0.0
    worst_B = float("inf")
    any_multi = False

    for l_dim_map in topology.dimensions:
        if l_dim_map.size <= 1:
            continue
        any_multi = True
        _, alpha_sec, effective_B_Bps, _ = topology.get_hockney_model_params(
            l_dim_map.logical_tag, float_size_bytes, context="concurrent_baseline"
        )
        worst_alpha = max(worst_alpha, alpha_sec)
        worst_B = min(worst_B, effective_B_Bps)

    if not any_multi:
        return CommStats()

    transmission_time = bytes_per_sender / worst_B if worst_B > 0 else float("inf")
    return CommStats(
        startup_latency=worst_alpha,
        transmission_time=transmission_time,
        comm_vol=bytes_per_sender,
        compute_time=0.0,
    )
