"""Hardware component and node specifications with registry and validation.

Defines Pydantic models for compute, memory tiers, interconnects that constitute
network nodes used in network topologies. Defines a `HardwareRegistry` loader that resolves components from
YAML and validates node blueprints.
"""

import os
from typing import Any, Dict, Optional, TypeVar

import yaml

# Pydantic imports
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
    Field,
    ValidationError,
)

# Component models


class Compute(BaseModel):
    """Defines the computational capabilities of a device."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    fp8_nominal: Optional[float] = Field(
        default=None, description="Nominal FP8 performance (FLOPS)", gt=0
    )
    bf16_nominal: Optional[float] = Field(
        default=None, description="Nominal BF16 performance (FLOPS)", gt=0
    )
    fp32_nominal: Optional[float] = Field(
        default=None, description="Nominal FP32 performance (FLOPS)", gt=0
    )
    efficiency: float = Field(..., description="Operational efficiency (0.0 to 1.0)", gt=0, le=1.0)

    @computed_field(repr=False)
    @property
    def fp8(self) -> Optional[float]:
        """Effective FP8 performance (FLOPS)"""
        return self.fp8_nominal * self.efficiency if self.fp8_nominal else None

    @computed_field(repr=False)
    @property
    def bf16(self) -> Optional[float]:
        """Effective BF16 performance (FLOPS)"""
        return self.bf16_nominal * self.efficiency if self.bf16_nominal else None

    @computed_field(repr=False)
    @property
    def fp32(self) -> Optional[float]:
        """Effective FP32 performance (FLOPS)"""
        return self.fp32_nominal * self.efficiency if self.fp32_nominal else None


class Memory(BaseModel):
    """Defines the memory subsystem characteristics."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    bandwidth_per_stack_nominal: float = Field(
        ...,
        description="Nominal unidirectional bandwidth per memory stack (Bps - bytes per second)",
        gt=0,
    )
    size_per_stack: float = Field(..., description="Capacity per memory stack B", gt=0)
    num_stack: int = Field(..., description="Number of memory stacks", gt=0)
    efficiency: float = Field(..., description="Operational efficiency (0.0 to 1.0)", gt=0, le=1.0)

    @computed_field(repr=False)
    @property
    def bandwidth_per_stack(self) -> float:
        """Effective unidirectional bandwidth per stack Bps (bytes per second)"""
        return self.bandwidth_per_stack_nominal * self.efficiency

    @computed_field
    @property
    def total_bandwidth(self) -> float:
        """Total effective unidirectional memory bandwidth across all stacks (Bps)"""
        return self.bandwidth_per_stack * self.num_stack

    @computed_field
    @property
    def total_size(self) -> float:
        """Total memory capacity across all stacks (B)"""
        return self.size_per_stack * self.num_stack


class Interconnect(BaseModel):
    """
    Defines a specific type of network interconnect hardware component available on a node.
    Represents a group of links/ports providing a certain bandwidth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(..., description="Unique identifier for this interconnect")
    type: str = Field(..., description="Technology type (e.g., NVLink, Ethernet, Infiniband, PCIe)")
    bandwidth_nominal: float = Field(
        ...,
        description="Total nominal unidirectional bandwidth provided by this component (Bps - bytes per second)",
        alias="bandwidth_nominal",
        gt=0,
    )
    efficiency: float = Field(..., description="Operational efficiency (0.0 to 1.0)", gt=0, le=1.0)
    # Optional fields you might add later:
    latency_sec: float = Field(
        default=0.0, description="Base latency for this interconnect type (seconds)"
    )

    # Currently we don't work with links directly. Adding it here for future extensibility.
    num_links: Optional[int] = Field(
        default=None, description="Number of physical links composing this total bandwidth", ge=1
    )

    @computed_field(repr=False)
    @property
    def link_bandwidth_nominal(self) -> float:
        return self.total_nominal_bandwidth / self.num_links if self.num_links > 0 else 0

    @computed_field(repr=False)
    @property
    def link_bandwidth(self) -> float:
        return self.link_bandwidth_nominal * self.efficiency

    @computed_field
    @property
    def bandwidth_Bps(self) -> float:
        """Total effective unidirectional bandwidth provided by this component (bps)"""
        return self.bandwidth_nominal * self.efficiency


# Network node model
T_NetworkNode = TypeVar("T_NetworkNode", bound="NetworkNode")


class NetworkNode(BaseModel):
    """
    Represents the definition of a type of  node , for the purposes of modeling it can be
    thorough to be an GPU or an XPU.
    Contains compute, memory, and an inventory of named interconnect components.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    node_name: str = Field(..., description="Unique identifier for this type of node")
    device_compute: Compute = Field(..., description="Compute device specification")
    # New tiered memory mapping; required going forward
    device_memory_tiers: Dict[str, Memory] = Field(
        ..., description="Tiered memory subsystems keyed by tier name, e.g. 'xpu_memory'"
    )
    device_interconnects: Dict[str, Interconnect] = Field(
        ..., description="Dictionary of available interconnect components, keyed by unique names"
    )

    def get_interconnect(self, name: str) -> Interconnect:
        """Safely retrieves an interconnect component by its name."""
        try:
            return self.device_interconnects[name]
        except KeyError:
            raise KeyError(
                f"Interconnect '{name}' not found on node type '{self.node_name}'. Available: {list(self.device_interconnects.keys())}"
            )

    def get_time_per_flop(self, dtype_size: int) -> Optional[float]:
        """
        Calculates compute time per byte based on FLOPs (example calculation).
        Result is in seconds per Byte.
        """
        compute_flops: Optional[float] = None  # Effective FLOPS
        match dtype_size:
            case 1:
                compute_flops = self.device_compute.fp8
            case 2:
                compute_flops = self.device_compute.bf16
            case 4:
                compute_flops = self.device_compute.fp32
            case _:
                raise ValueError(f"Unsupported dtype size: {dtype_size}.")

        if compute_flops is None or compute_flops <= 0:
            raise ValueError(
                f"Invalid compute performance for dtype size {dtype_size}: {compute_flops}."
            )

        return 1.0 / compute_flops

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NetworkNode):
            return NotImplemented
        return (
            self.node_name == other.node_name
            and self.device_compute == other.device_compute
            and self.device_memory_tiers == other.device_memory_tiers
            and self.device_interconnects == other.device_interconnects
        )

    def __hash__(self) -> int:
        interconnects_tuple = tuple(sorted(self.device_interconnects.items()))
        return hash(
            (
                self.node_name,
                self.device_compute,
                tuple(sorted(self.device_memory_tiers.items())),
                interconnects_tuple,
            )
        )


# Hardware registry
class HardwareRegistry:
    """Loads and stores definitions for Compute, Memory, Interconnect, and NetworkNode types."""

    def __init__(self):
        self.compute: Dict[str, Compute] = {}
        self.memory: Dict[str, Memory] = {}
        self.interconnect: Dict[str, Interconnect] = {}
        self.nodes: Dict[str, NetworkNode] = {}  # Stores resolved NetworkNode blueprints
        self._raw_node_configs: Dict[str, Dict[str, Any]] = {}

    def load_from_yaml(self, filepath: str):
        """Loads hardware definitions from a YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hardware library file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise TypeError(f"YAML file '{filepath}' must load as a dictionary.")
            self._load_components(data.get("components", {}))
            self._store_raw_nodes(data.get("nodes", {}))
            self._resolve_and_validate_nodes()
            print(f"Successfully loaded hardware library from '{filepath}'")
            print(
                f"  - Compute: {len(self.compute)}, Memory: {len(self.memory)}, "
                f"Interconnect: {len(self.interconnect)}, Nodes: {len(self.nodes)}"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{filepath}': {e}") from e
        except (ValidationError, TypeError, ValueError, KeyError) as e:
            raise ValueError(
                f"Error loading or validating hardware library '{filepath}': {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"Unexpected error loading hardware library '{filepath}': {e}") from e

    def _load_components(self, components_data: Dict[str, Dict[str, Any]]):
        """Loads Compute, Memory, and Interconnect components."""
        component_map = {
            "compute": (self.compute, Compute),
            "memory": (self.memory, Memory),
            "interconnect": (self.interconnect, Interconnect),
        }
        for comp_type, (registry, model_cls) in component_map.items():
            for name, config in components_data.get(comp_type, {}).items():
                if not isinstance(config, dict):
                    raise TypeError(
                        f"Config for {comp_type} component '{name}' must be a dictionary."
                    )
                try:
                    registry[name] = model_cls.model_validate(config)
                except ValidationError as e:
                    raise ValueError(
                        f"Validation failed for {comp_type} component '{name}':\n{e}"
                    ) from e

    def _store_raw_nodes(self, nodes_data: Dict[str, Dict[str, Any]]):
        """Stores the raw dictionary configuration for each node."""
        for name, config in nodes_data.items():
            if not isinstance(config, dict):
                raise TypeError(f"Config for node '{name}' must be a dictionary.")
            self._raw_node_configs[name] = config

    def _resolve_and_validate_nodes(self):
        """Resolves component references and validates NetworkNode models."""
        for node_name, raw_config in self._raw_node_configs.items():
            resolved_config = raw_config.copy()
            resolved_config.setdefault("node_name", node_name)
            # Resolve compute
            ref_compute = raw_config.get("device_compute")
            if not isinstance(ref_compute, str):
                raise ValueError(
                    f"Node '{node_name}' must reference 'device_compute' using a string name, got: {ref_compute}"
                )
            compute_obj = self.compute.get(ref_compute)
            if compute_obj is None:
                raise ValueError(
                    f"Node '{node_name}' references unknown compute component '{ref_compute}'. Available: {list(self.compute.keys())}"
                )
            resolved_config["device_compute"] = compute_obj

            # Resolve tiered memory (required)
            memory_tiers = raw_config.get("device_memory_tiers")
            if not isinstance(memory_tiers, dict) or not memory_tiers:
                raise ValueError(
                    f"Node '{node_name}' must have a non-empty 'device_memory_tiers' mapping."
                )
            resolved_tiers: Dict[str, Memory] = {}
            for tier_name, mem_ref in memory_tiers.items():
                if not isinstance(mem_ref, str):
                    raise ValueError(
                        f"Node '{node_name}' memory tier '{tier_name}' must reference memory by name (string), got: {mem_ref}"
                    )
                mem_obj = self.memory.get(mem_ref)
                if mem_obj is None:
                    raise ValueError(
                        f"Node '{node_name}' references unknown memory component '{mem_ref}' for tier '{tier_name}'. Available: {list(self.memory.keys())}"
                    )
                resolved_tiers[tier_name] = mem_obj
            resolved_config["device_memory_tiers"] = resolved_tiers

            # Resolve interconnects
            interconnect_refs = raw_config.get("device_interconnects")
            if not isinstance(interconnect_refs, dict):
                raise ValueError(
                    f"Node '{node_name}' must have an 'device_interconnects' dictionary mapping local names to component reference names."
                )
            resolved_interconnects: Dict[str, Interconnect] = {}
            for local_ic_name, ref_ic_name in interconnect_refs.items():
                if not isinstance(ref_ic_name, str):
                    raise ValueError(
                        f"Node '{node_name}' interconnect reference for '{local_ic_name}' must be a string name, got: {ref_ic_name}"
                    )
                interconnect_obj = self.interconnect.get(ref_ic_name)
                if interconnect_obj is None:
                    raise ValueError(
                        f"Node '{node_name}' references unknown interconnect component '{ref_ic_name}' for its '{local_ic_name}'. Available: {list(self.interconnect.keys())}"
                    )
                resolved_interconnects[local_ic_name] = interconnect_obj
            resolved_config["device_interconnects"] = resolved_interconnects
            try:
                self.nodes[node_name] = NetworkNode.model_validate(resolved_config)
            except ValidationError as e:
                raise ValueError(
                    f"Validation failed for node '{node_name}' after resolving references:\n{e}"
                ) from e

    def get_node(self, name: str) -> NetworkNode:
        """Retrieves a resolved NetworkNode blueprint by its type name."""
        try:
            return self.nodes[name]
        except KeyError:
            raise KeyError(
                f"NetworkNode type '{name}' not found in registry. Available: {list(self.nodes.keys())}"
            )
