"""Network-wide sender-side communication-volume accounting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkWideCommunicationVolume:
    """Bytes injected by all endpoints for one modeled batch/request.

    Collective fields count each sender-to-receiver transfer once at the
    sender. Pipeline fields use the same convention, but remain separate
    because ScaleScope's per-lane PP statistic already sums all PP boundaries.
    """

    collective_forward_bytes: float
    collective_backward_bytes: float
    pipeline_forward_bytes: float
    pipeline_backward_bytes: float

    @property
    def forward_bytes(self) -> float:
        return self.collective_forward_bytes + self.pipeline_forward_bytes

    @property
    def backward_bytes(self) -> float:
        return self.collective_backward_bytes + self.pipeline_backward_bytes

    @property
    def collective_bytes(self) -> float:
        return self.collective_forward_bytes + self.collective_backward_bytes

    @property
    def pipeline_bytes(self) -> float:
        return self.pipeline_forward_bytes + self.pipeline_backward_bytes

    @property
    def total_bytes(self) -> float:
        return self.forward_bytes + self.backward_bytes


def calculate_network_wide_communication_volume(
    *,
    collective_forward_bytes_per_xpu: float,
    collective_backward_bytes_per_xpu: float,
    pipeline_forward_bytes_per_lane: float,
    pipeline_backward_bytes_per_lane: float,
    total_xpu_count: int,
    pp_stages: int,
) -> NetworkWideCommunicationVolume:
    """Expand per-endpoint statistics into sender-side cluster totals.

    TP, DP, and EP collective statistics are per XPU, so their network-wide
    volume scales with every XPU. PP statistics are per TP/DP lane and already
    include all ``pp_stages - 1`` boundaries, so they scale with the number of
    lanes (``total_xpu_count / pp_stages``), not with every PP-stage endpoint.
    """

    if total_xpu_count <= 0:
        raise ValueError(f"total_xpu_count must be > 0, got {total_xpu_count}")
    if pp_stages <= 0:
        raise ValueError(f"pp_stages must be > 0, got {pp_stages}")
    if total_xpu_count % pp_stages != 0:
        raise ValueError(
            "total_xpu_count must be divisible by pp_stages; got "
            f"{total_xpu_count} and {pp_stages}"
        )

    volumes = {
        "collective_forward_bytes_per_xpu": collective_forward_bytes_per_xpu,
        "collective_backward_bytes_per_xpu": collective_backward_bytes_per_xpu,
        "pipeline_forward_bytes_per_lane": pipeline_forward_bytes_per_lane,
        "pipeline_backward_bytes_per_lane": pipeline_backward_bytes_per_lane,
    }
    for name, value in volumes.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    pipeline_lane_count = total_xpu_count // pp_stages
    return NetworkWideCommunicationVolume(
        collective_forward_bytes=collective_forward_bytes_per_xpu * total_xpu_count,
        collective_backward_bytes=collective_backward_bytes_per_xpu * total_xpu_count,
        pipeline_forward_bytes=pipeline_forward_bytes_per_lane * pipeline_lane_count,
        pipeline_backward_bytes=pipeline_backward_bytes_per_lane * pipeline_lane_count,
    )
