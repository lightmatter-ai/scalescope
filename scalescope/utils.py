"""Generic utilities shared across the project (enums, formatting, helpers)."""

from enum import Enum
from typing import Any, Type

KILO_B = 1024
MEGA_B = KILO_B * KILO_B
GIGA_B = KILO_B * MEGA_B
TERA_B = KILO_B * GIGA_B
PETA_B = KILO_B * TERA_B
MILLI = 10**-3
MICRO = MILLI * MILLI
NANO = MILLI * MICRO
PICO = MILLI * NANO
KILO = 1000
MEGA = KILO * KILO
GIGA = KILO * MEGA
TERA = KILO * GIGA
PETA = KILO * TERA

INFINITY = 10**32


def calculate_overlapped_time(
    comp_time: float,
    comm_time: float,
    mem_time: float,
    comp_comm_efficiency: float,
    comp_mem_efficiency: float,
    comm_mem_efficiency: float,
) -> dict:
    """Minimum makespan for divisible compute/comm/memory with pairwise overlap caps.

    Each efficiency is the maximum fraction of the shorter member of that pair
    that may overlap. The returned total is

        max(C, K, M,
            C+K-aCK, C+M-aCM, K+M-aKM,
            C+K+M-aCK-aCM-aKM)

    with aij = eij * min(i, j). The inner max is required: subtracting all
    pairwise credits alone can over-credit overlaps that compete for the same
    interval.
    """
    for name, value in (
        ("comp_time", comp_time),
        ("comm_time", comm_time),
        ("mem_time", mem_time),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    for name, eff in (
        ("comp_comm_efficiency", comp_comm_efficiency),
        ("comp_mem_efficiency", comp_mem_efficiency),
        ("comm_mem_efficiency", comm_mem_efficiency),
    ):
        if not 0.0 <= eff <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {eff}")

    overlap_ck = comp_comm_efficiency * min(comp_time, comm_time)
    overlap_cm = comp_mem_efficiency * min(comp_time, mem_time)
    overlap_km = comm_mem_efficiency * min(comm_time, mem_time)
    bounds = {
        "compute": comp_time,
        "communication": comm_time,
        "memory": mem_time,
        "compute_communication": comp_time + comm_time - overlap_ck,
        "compute_memory": comp_time + mem_time - overlap_cm,
        "communication_memory": comm_time + mem_time - overlap_km,
        "total_work": (comp_time + comm_time + mem_time - overlap_ck - overlap_cm - overlap_km),
    }
    total_time = max(bounds.values())
    binding_constraints = [name for name, value in bounds.items() if value == total_time]

    if total_time == 0:
        return {
            "total_time": 0.0,
            "compute_utilization": 0.0,
            "communication_utilization": 0.0,
            "memory_utilization": 0.0,
            "binding_constraints": binding_constraints,
        }

    return {
        "total_time": total_time,
        "compute_utilization": (comp_time / total_time) * 100,
        "communication_utilization": (comm_time / total_time) * 100,
        "memory_utilization": (mem_time / total_time) * 100,
        "binding_constraints": binding_constraints,
    }


def residual_after_accounted(*, total: float, accounted: float, name: str) -> float:
    """Return total-accounted, failing if accounted exceeds total beyond float noise."""
    if total < 0:
        raise ValueError(f"{name} total must be >= 0, got {total}")
    if accounted < 0:
        raise ValueError(f"{name} accounted must be >= 0, got {accounted}")
    residual = total - accounted
    scale = max(abs(total), abs(accounted), 1.0)
    if abs(residual) <= 1e-12 * scale:
        return 0.0
    if residual < 0:
        raise ValueError(f"{name} accounted exceeds total: total={total}, accounted={accounted}")
    return residual


def format_with_units(key: str, value: float) -> str:
    """
    Format value with appropriate units based on key suffix.
    Key must end with underscore and unit identifier.

    Examples:
        memory_ + b -> Gb/Tb (bits)
        memory_ + B -> GB/TB (bytes)
        time_ + s -> hours/minutes/seconds
        bandwidth_ + bps -> Gb/s, Tb/s

    Args:
        key: String key ending with underscore and unit (memory_B, time_s, etc)
        value: Numeric value to format
    """

    parts = key.split("_")
    if len(parts) < 2:
        raise ValueError(f"Key '{key}' must have format 'name_unit_'")
    unit = parts[-1]

    match unit:
        # Memory/storage units
        case "b" | "bit" | "bits":
            return _format_bits(value)
        case "B" | "byte" | "bytes":
            return _format_bytes(value)

        # Time units
        case "s" | "sec" | "second" | "seconds":
            return _format_time(value)

        # Bandwidth units
        case "bps" | "bits/s" | "bits/sec":
            return _format_bits(value) + "/s"
        case "bps" | "bytes/s" | "bytes/sec":
            return _format_bytes(value) + "/s"

        case _:
            raise ValueError(f"Unknown unit format: {unit}")


def _format_bits(bits: float) -> str:
    """Format bits into human readable string with appropriate unit"""
    units = ["b", "Kb", "Mb", "Gb", "Tb", "Pb"]
    unit_idx = 0

    while bits >= 1000 and unit_idx < len(units) - 1:
        bits /= 1000
        unit_idx += 1

    formatted = f"{bits:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} {units[unit_idx]}"


def _format_bytes(bytes: float) -> str:
    """Format bytes into human readable string with appropriate unit"""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_idx = 0

    while bytes >= 1024 and unit_idx < len(units) - 1:
        bytes /= 1024
        unit_idx += 1

    formatted = f"{bytes:.2f}".rstrip("0").rstrip(".")
    return f"{formatted} {units[unit_idx]}"


def _format_time(seconds: float) -> str:
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s".rstrip("0").rstrip(".")

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m".rstrip("0").rstrip(".")

    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f}h".rstrip("0").rstrip(".")

    days = hours / 24
    return f"{days:.2f}d".rstrip("0").rstrip(".")


def format_bytes(b):
    """Converts a number of bytes to a human-readable string."""
    if b is None or not isinstance(b, (int, float)) or b < 0:
        return b
    if b < 1024:
        return f"{b} B"
    for i, unit in enumerate(["KiB", "MiB", "GiB", "TiB", "PiB"]):
        if b < 1024 ** (i + 2):
            return f"{b / (1024 ** (i + 1)):.2f} {unit}"
    return f"{b / (1024 ** 5):.2f} EiB"


def format_flops(f):
    """Converts a number of FLOPs to a human-readable string."""
    if f is None or not isinstance(f, (int, float)) or f < 0:
        return f
    if f < 1e3:
        return f"{f:g} FLOPs"
    for i, unit in enumerate(["KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs"]):
        if f < 1e3 ** (i + 2):
            return f"{f / (1e3 ** (i + 1)):.2f} {unit}"
    return f"{f / 1e18:.2f} ZFLOPs"


def format_time(s):
    """Converts a time in seconds to a human-readable string."""
    if s is None or not isinstance(s, (int, float)) or s < 0:
        return s
    if s == 0:
        return "0.00 s"
    if s < 1e-6:
        return f"{s * 1e9:.2f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} us"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    if s < 60:
        return f"{s:.2f} s"
    if s < 3600:
        return f"{s / 60:.2f} min"
    if s < 86400:
        return f"{s / 3600:.2f} hr"
    return f"{s / 86400:.2f} d"


if __name__ == "__main__":
    # Perfect overlap between all pairs
    result = calculate_overlapped_time(10, 5, 3, 1.0, 1.0, 1.0)
    print("\nPerfect overlap all pairs:")
    print(f"Total time: {result['total_time']:.1f}s")  # 10.0s
    print(f"Compute utilization: {result['compute_utilization']:.1f}%")
    print(f"Communication utilization: {result['communication_utilization']:.1f}%")
    print(f"Memory utilization: {result['memory_utilization']:.1f}%")

    # Perfect compute-comm overlap, partial memory overlap
    result = calculate_overlapped_time(10, 5, 3, 1.0, 0.5, 0.5)
    print("\nPerfect compute-comm, partial memory overlap:")
    print(f"Total time: {result['total_time']:.1f}s")
    print(f"Compute utilization: {result['compute_utilization']:.1f}%")
    print(f"Communication utilization: {result['communication_utilization']:.1f}%")
    print(f"Memory utilization: {result['memory_utilization']:.1f}%")

    # No overlap between any pairs
    result = calculate_overlapped_time(10, 5, 3, 0.0, 0.0, 0.0)
    print("\nNo overlap:")
    print(f"Total time: {result['total_time']:.1f}s")  # 18.0s
    print(f"Compute utilization: {result['compute_utilization']:.1f}%")
    print(f"Communication utilization: {result['communication_utilization']:.1f}%")
    print(f"Memory utilization: {result['memory_utilization']:.1f}%")

    # Example usage:
    examples = [
        ("memory_bits", 1500),
        ("memory_bits", 1_500_000),
        ("memory_bits", 1_500_000_000),
        ("memory_bits", 1_500_000_000_000),
        ("memory_B", 1536),
        ("memory_B", 1536 * 1024),
        ("time_s", 30),
        ("time_s", 3600),
        ("bandwidth_bps", 1_000_000_000),
    ]

    for key, value in examples:
        print(f"{key}: {value} -> {format_with_units(key, value)}")


class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"


# Helper function for reusable Enum parsing
def parse_enum_by_name_or_value(enum_cls: Type[Enum]):
    """Creates a Pydantic 'before' validator for parsing Enums by name or value."""

    def validator(v: Any) -> Any:
        if isinstance(v, str):
            try:
                # Try parsing by name (key) first
                return enum_cls[v]
            except KeyError:
                try:
                    # Fallback: Try parsing by value
                    return enum_cls(v)
                except ValueError:
                    # Failed both ways
                    valid_names = [e.name for e in enum_cls]
                    valid_values = [e.value for e in enum_cls]
                    raise ValueError(
                        f"Invalid value '{v}' for {enum_cls.__name__}. "
                        f"Valid names: {valid_names}. Valid values: {valid_values}"
                    )
        # If not a string (e.g., already an Enum), pass it through
        return v

    return validator


class PassMode(Enum):
    """Defines the distinct modes of a forward pass for both training and inference."""

    TRAINING = "training"
    INFERENCE_PREFILL = "inference_prefill"
    INFERENCE_DECODE = "inference_decode"

    def __str__(self):
        return self.value
