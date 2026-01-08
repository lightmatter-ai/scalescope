"""Lightweight containers and arithmetic for time/volume statistics.

Defines `CommStats` and related helpers to sum communication times/volumes
and support clean aggregation in the models.
"""


class CommStats:
    def __init__(
        self,
        startup_latency: float = 0,
        transmission_time: float = 0,
        comm_vol: float = 0,
        compute_time: float = 0,
        startup_phase_count: int = 0,
    ):
        self.startup_latency = startup_latency  # Initial latency before communication starts
        self.transmission_time = transmission_time  # Time spent for data transmission
        self.comm_vol = comm_vol  # Communication volume in bytes
        self.compute_time = compute_time  # Time spent in computation
        # Topology/algorithm phase count independent of alpha.
        self.startup_phase_count = startup_phase_count

    def __add__(self, other):
        return CommStats(
            self.startup_latency + other.startup_latency,
            self.transmission_time + other.transmission_time,
            self.comm_vol + other.comm_vol,
            self.compute_time + other.compute_time,
            self.startup_phase_count + other.startup_phase_count,
        )

    def __iadd__(self, other):
        self.startup_latency += other.startup_latency
        self.transmission_time += other.transmission_time
        self.comm_vol += other.comm_vol
        self.compute_time += other.compute_time
        self.startup_phase_count += other.startup_phase_count
        return self

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented  # Let Python try other.__rmul__(self) or raise TypeError

        # When multiplying by a scalar (e.g., number of layers),
        # - startup_latency: This is tricky. If operations are sequential, latency adds up.
        #   If they can be perfectly pipelined or batched, the startup might be paid once
        #   or a few times. For simplicity here, we'll scale it, assuming sequential or
        #   that the scalar represents distinct operations where latency applies each time.
        # - transmission_time: Scales directly.
        # - comm_vol: Scales directly.
        # - compute_time: Scales directly.
        return CommStats(
            self.startup_latency * scalar,
            self.transmission_time * scalar,
            self.comm_vol * scalar,
            self.compute_time * scalar,
            int(round(self.startup_phase_count * scalar)),
        )

    def __rmul__(self, scalar):
        # Handles cases like `3 * comm_stats_object`
        return self.__mul__(scalar)

    def __imul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        # Similar considerations for latency as in __mul__
        self.startup_latency *= scalar
        self.transmission_time *= scalar
        self.comm_vol *= scalar
        self.compute_time *= scalar
        self.startup_phase_count = int(round(self.startup_phase_count * scalar))
        return self

    def combine_color(self, other):
        self.startup_latency = max(self.startup_latency, other.startup_latency)
        self.transmission_time = max(self.transmission_time, other.transmission_time)
        self.comm_vol += other.comm_vol
        self.compute_time += other.compute_time
        self.startup_phase_count = max(self.startup_phase_count, other.startup_phase_count)
        return self

    def __str__(self):
        return (
            f"CommStats(startup_latency={self.startup_latency * 1e6:.3f} us, "
            # f"transmission_time={self.transmission_time*1e6:.3f} us, "
            f"transmission_time={self.transmission_time:.3f} s, "
            f"comm_vol={self.comm_vol / (1024*1024):.3f} MB, "
            f"compute_time={self.compute_time*1e6:.3f} us, "
            f"startup_phase_count={self.startup_phase_count})"
        )

    @property
    def total_comm_time(self):
        return self.startup_latency + self.transmission_time
