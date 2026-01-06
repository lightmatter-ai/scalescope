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
    comp_comm_efficiency: float = 1.0,
    comp_mem_efficiency: float = 1.0,
    comm_mem_efficiency: float = 1.0,
) -> dict:
    """
    Calculate execution time and utilization with pairwise overlap efficiencies between operations.

    The total time is returned in the same units as the input times.

    Args:
        comp_time: Time taken for computation
        comm_time: Time taken for communication
        mem_time: Time taken for memory operations
        comp_comm_efficiency: Overlap efficiency between compute and communication (0.0 to 1.0)
        comp_mem_efficiency: Overlap efficiency between compute and memory (0.0 to 1.0)
        comm_mem_efficiency: Overlap efficiency between communication and memory (0.0 to 1.0)

    Returns:
        dict: Dictionary containing:
            - total_time: Total execution time
            - compute_utilization: Percentage of total time spent computing (0-100%)
            - communication_utilization: Percentage of total time spent communicating (0-100%)
            - memory_utilization: Percentage of total time spent in memory operations (0-100%)

    """
    for eff in [comp_comm_efficiency, comp_mem_efficiency, comm_mem_efficiency]:
        if not 0.0 <= eff <= 1.0:
            raise ValueError("Efficiency values must be between 0.0 and 1.0")

    # Start with the longest time as base
    times = [comp_time, comm_time, mem_time]
    max_time = max(times)
    total_time = max_time

    # Add non-overlapped portions based on pairwise efficiencies
    if max_time == comp_time:
        total_time += comm_time * (1.0 - comp_comm_efficiency)
        total_time += mem_time * (1.0 - comp_mem_efficiency)
    elif max_time == comm_time:
        total_time += comp_time * (1.0 - comp_comm_efficiency)
        total_time += mem_time * (1.0 - comm_mem_efficiency)
    else:  # memory_time is max
        total_time += comp_time * (1.0 - comp_mem_efficiency)
        total_time += comm_time * (1.0 - comm_mem_efficiency)

    # Calculate utilizations
    if total_time == 0:
        return {
            "total_time": 0.0,
            "compute_utilization": 0.0,
            "communication_utilization": 0.0,
            "memory_utilization": 0.0,
        }

    compute_utilization = (comp_time / total_time) * 100
    communication_utilization = (comm_time / total_time) * 100
    memory_utilization = (mem_time / total_time) * 100

    return {
        "total_time": total_time,
        "compute_utilization": compute_utilization,
        "communication_utilization": communication_utilization,
        "memory_utilization": memory_utilization,
    }


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
