"""Physical and logical topology specifications.

Pydantic models for datacenter physical topology (dimensions and interconnect
mapping) and logical topology (PP/TP/DP/EP mappings onto physical), including
computed properties (shape/total_nodes) and YAML loaders.
"""

import math
import os
from enum import Enum
from typing import Dict, List, Set, Tuple, Type, TypeVar, Union

import yaml

# Pydantic imports
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import override

from scalescope.hardware_spec import HardwareRegistry, Interconnect, NetworkNode


# Enums
class PhysicalTopology1DType(str, Enum):
    RING = "ring"
    FULLY_CONNECTED = "fully_connected"
    MESH = "mesh"
    TORUS = "torus"
    SWITCHED = "switched"


class LogicalDimensionTag(str, Enum):
    DP = "DP"
    TP = "TP"
    PP = "PP"
    MoE = "MoE"
    SP = "SP"

    @classmethod
    def new_dim(
        cls, logical_tag: str, maps_to_physical_dimension_name: str, size: int
    ) -> "LogicalDimensionMapping":
        return LogicalDimensionMapping(
            logical_tag=logical_tag,
            maps_to_physical_dimension_name=maps_to_physical_dimension_name,
            size=size,
        )


# Physical topology
class PhysicalDimension(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(..., description="Descriptive name for this physical dimension")
    interconnect_name: str = Field(..., description="Name of the Interconnect component used")
    topology_type: PhysicalTopology1DType = Field(..., description="Physical connection pattern")
    size: int = Field(..., description="Number of nodes along this dimension", gt=0)


T_PhysicalTopology = TypeVar("T_PhysicalTopology", bound="PhysicalTopology")


class PhysicalTopology(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(..., description="Name for this physical topology configuration")
    node_definition: NetworkNode = Field(..., description="The blueprint NetworkNode object used")
    dimensions: List[PhysicalDimension] = Field(
        ..., description="Ordered list of physical dimensions", min_length=1
    )

    @model_validator(mode="after")
    def check_interconnect_references(self) -> "PhysicalTopology":
        node_ic_keys = set(self.node_definition.device_interconnects.keys())
        for dim in self.dimensions:
            if dim.interconnect_name not in node_ic_keys:
                raise ValueError(
                    f"PhysDim '{dim.name}' refs invalid IC '{dim.interconnect_name}'. "
                    f"Available on node '{self.node_definition.node_name}': {list(node_ic_keys)}"
                )
        return self

    @computed_field
    @property
    def shape(self) -> List[int]:
        return [d.size for d in self.dimensions]

    @computed_field
    @property
    def total_nodes(self) -> int:
        return math.prod(self.shape) if self.shape else 0

    def get_dimension(self, dimension_name: str) -> PhysicalDimension:
        for dim in self.dimensions:
            if dim.name == dimension_name:
                return dim
        raise KeyError(
            f"Physical dimension '{dimension_name}' not found in topology '{self.name}'."
        )

    @classmethod
    def load_from_yaml(
        cls: Type[T_PhysicalTopology], filepath: str, hardware_registry: HardwareRegistry
    ) -> T_PhysicalTopology:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PhysicalTopology config file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)

            node_ref_name = config_dict.pop("node_definition_ref", None)
            if not isinstance(node_ref_name, str):
                raise ValueError(f"YAML '{filepath}' must contain 'node_definition_ref' (string).")

            node_obj = hardware_registry.get_node(node_ref_name)
            return cls(node_definition=node_obj, **config_dict)
        except Exception as e:
            raise ValueError(f"Error loading PhysicalTopology from '{filepath}': {e}") from e


# Logical topology definition
class LogicalDimensionMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    logical_tag: Union[LogicalDimensionTag, str] = Field(
        ..., description="Type/name of logical parallelism (DP, TP, etc.)"
    )
    maps_to_physical_dimension_name: str = Field(
        ..., description="Name of the PhysicalDimension used"
    )
    size: int = Field(..., description="Degree of this logical parallelism dimension", gt=0)


T_LogicalTopology = TypeVar("T_LogicalTopology", bound="LogicalTopology")


class LogicalTopology(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    physical_topology: PhysicalTopology
    dimensions: List[LogicalDimensionMapping] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_mapping(self) -> "LogicalTopology":
        phys_dims = {pd.name: pd for pd in self.physical_topology.dimensions}
        used_tags: Set[str] = set()
        for mapping in self.dimensions:
            tag_val = (
                mapping.logical_tag
                if isinstance(mapping.logical_tag, str)
                else mapping.logical_tag.value
            )
            if tag_val in used_tags:
                raise ValueError(f"LogicalTag '{tag_val}' is used multiple times in '{self.name}'.")
            used_tags.add(tag_val)
            if mapping.maps_to_physical_dimension_name not in phys_dims:
                raise ValueError(
                    f"Mapping '{tag_val}' maps to non-existent PhysDim '{mapping.maps_to_physical_dimension_name}'."
                )
            if mapping.size > phys_dims[mapping.maps_to_physical_dimension_name].size:
                raise ValueError(
                    f"Mapping '{tag_val}' size {mapping.size} exceeds its PhysDim size."
                )
        return self

    def get_mapping_for_tag(self, tag: Union[LogicalDimensionTag, str]) -> LogicalDimensionMapping:
        tag_val = tag.value if isinstance(tag, Enum) else tag
        for mapping in self.dimensions:
            map_tag_val = (
                mapping.logical_tag.value
                if isinstance(mapping.logical_tag, Enum)
                else mapping.logical_tag
            )
            if map_tag_val == tag_val:
                return mapping
        raise KeyError(f"Tag '{tag_val}' not found in LogicalTopology '{self.name}'.")

    def get_physical_dimension_for_tag(
        self, tag: Union[LogicalDimensionTag, str]
    ) -> PhysicalDimension:
        mapping = self.get_mapping_for_tag(tag)
        return self.physical_topology.get_dimension(mapping.maps_to_physical_dimension_name)

    def get_interconnect_for_tag(self, tag: Union[LogicalDimensionTag, str]) -> Interconnect:
        phys_dim = self.get_physical_dimension_for_tag(tag)
        return self.physical_topology.node_definition.get_interconnect(phys_dim.interconnect_name)

    def get_effective_bandwidth(
        self, tag_in_question: Union[LogicalDimensionTag, str], context: str = "concurrent_baseline"
    ) -> float:
        """Calculates effective bandwidth (Bps) considering sharing within this instance's dimensions."""
        self.get_mapping_for_tag(tag_in_question)  # Check existence
        phys_dim_for_tag = self.get_physical_dimension_for_tag(tag_in_question)
        interconnect = self.get_interconnect_for_tag(tag_in_question)
        base_ic_bandwidth_Bps = interconnect.bandwidth_Bps

        if context == "concurrent_baseline":
            num_sharers = sum(
                1
                for ldm in self.dimensions
                if self.get_physical_dimension_for_tag(ldm.logical_tag).interconnect_name
                == phys_dim_for_tag.interconnect_name
            )
            return base_ic_bandwidth_Bps / max(1, num_sharers)
        elif context == "max_exclusive":
            return base_ic_bandwidth_Bps
        else:
            raise ValueError(f"Unknown sharing context: {context}")

    def get_hockney_model_params(
        self,
        tag_in_view: Union[LogicalDimensionTag, str],
        float_size_bytes: float,
        context: str = "concurrent_baseline",
    ) -> Tuple[int, float, float, float]:
        mapping = self.get_mapping_for_tag(tag_in_view)
        interconnect = self.get_interconnect_for_tag(tag_in_view)
        node_def = self.physical_topology.node_definition

        gamma = (
            node_def.get_time_per_flop(float_size_bytes) / float_size_bytes if node_def else None
        )
        bandwidth_Bps = self.get_effective_bandwidth(tag_in_view, context=context)

        return mapping.size, interconnect.latency_sec, bandwidth_Bps, gamma or float("inf")

    @computed_field
    @property
    def shape(self) -> List[int]:
        """Returns the shape of the logical topology as a list of dimension sizes."""
        return [d.size for d in self.dimensions]

    @computed_field
    @property
    def num_chips(self) -> int:
        """Returns the total number of chips in the logical topology."""
        return math.prod(self.shape) if self.shape else 0

    def create_view(
        self: T_LogicalTopology,
        view_name: str,
        relevant_tags: List[Union[LogicalDimensionTag, str]],
        global_sharing_context: str = "concurrent_baseline",
    ) -> "LogicalTopologyView":
        """Creates a LogicalTopologyView focusing on relevant_tags."""
        mappings = [self.get_mapping_for_tag(tag) for tag in relevant_tags]
        return LogicalTopologyView(
            name=view_name,
            physical_topology=self.physical_topology,
            dimensions=mappings,
            original_topology_ref=self,
            global_sharing_context=global_sharing_context,
        )

    def create_view_from_prefix(
        self: T_LogicalTopology,
        view_name: str,
        prefix: str,
        global_sharing_context: str = "concurrent_baseline",
    ) -> "LogicalTopologyView":
        """
        Creates a view by finding all logical dimensions that share a common prefix.

        This is the primary interface for modeling hierarchical collectives where a single
        parallelism strategy (like 'EP') is split across multiple physical dimensions.

        Args:
            view_name: A name for the new view.
            prefix: The common prefix for the logical tags (e.g., "EP").
            global_sharing_context: The context for calculating bandwidth.

        Returns:
            A LogicalTopologyView containing all dimensions that match the prefix.
        """
        relevant_tags: List[str] = []
        # The convention is "PREFIX_SUFFIX"
        prefix_with_underscore = f"{prefix}_"

        for l_dim_map in self.dimensions:
            # Get the string value of the tag
            tag_val = (
                l_dim_map.logical_tag
                if isinstance(l_dim_map.logical_tag, str)
                else l_dim_map.logical_tag.value
            )

            # Check if the tag is an exact match or follows the prefix_suffix convention
            if tag_val == prefix or tag_val.startswith(prefix_with_underscore):
                relevant_tags.append(tag_val)

        if not relevant_tags:
            raise ValueError(
                f"No logical dimensions found with prefix '{prefix}' in topology '{self.name}'."
            )

        # The order of dimensions in the view can matter, so we sort them.
        # A common convention is to sort alphabetically, which might place 'EP_inter' before 'EP_intra'.
        relevant_tags.sort()

        # Call the original create_view method with the discovered tags.
        return self.create_view(
            view_name=view_name,
            relevant_tags=relevant_tags,
            global_sharing_context=global_sharing_context,
        )

    def create_virtual_view_from_tag(
        self: T_LogicalTopology,
        view_name: str,
        base_tag: Union[LogicalDimensionTag, str],
        virtual_size: int,
        global_sharing_context: str = "concurrent_baseline",
    ) -> "LogicalTopologyView":
        """
        Creates a 'virtual' view that reuses the physical interconnect of an existing tag
        but overrides the size without enforcing the physical-capacity validation.
        This is useful to model aggregated domains that communicate over a slower link.
        """
        # Reuse mapping properties but override size
        base_mapping = self.get_mapping_for_tag(base_tag)
        virtual_mapping = LogicalDimensionMapping(
            logical_tag=base_mapping.logical_tag,
            maps_to_physical_dimension_name=base_mapping.maps_to_physical_dimension_name,
            size=int(virtual_size),
        )
        return LogicalTopologyView(
            name=view_name,
            physical_topology=self.physical_topology,
            dimensions=[virtual_mapping],
            original_topology_ref=self,
            global_sharing_context=global_sharing_context,
            allow_virtual_resize=True,
        )

    # Note: derived views should be created via create_view / create_view_from_prefix.

    def get_tags_sorted_by_speed(self) -> List[str]:
        """
        Returns a list of all logical tags in the topology, sorted from fastest
        to slowest based on their underlying physical interconnect's performance.

        The sorting criteria are:
        1.  Bandwidth (Bps) in descending order (higher bandwidth is faster).
        2.  Latency (seconds) in ascending order as a tie-breaker (lower latency is faster).

        Returns:
            A list of logical tag strings, ordered by speed.
        """
        # Step 1: Create a list of tuples, where each tuple contains the
        # necessary information for sorting: (tag, bandwidth, latency).
        tag_metrics = []
        for l_dim_map in self.dimensions:
            # Get the string representation of the logical tag.
            tag_val = (
                l_dim_map.logical_tag
                if isinstance(l_dim_map.logical_tag, str)
                else l_dim_map.logical_tag.value
            )

            # Use the existing helper method to get the interconnect for this tag.
            interconnect = self.get_interconnect_for_tag(l_dim_map.logical_tag)

            # Store the metrics needed for sorting.
            tag_metrics.append((tag_val, interconnect.bandwidth_Bps, interconnect.latency_sec))

        # Step 2: Sort the list.
        # - The `key` lambda function returns a tuple `(bandwidth, -latency)`.
        # - We sort by bandwidth first (item[1]).
        # - For ties in bandwidth, we sort by latency. We use -latency (-item[2]) because
        #   the primary sort is `reverse=True`, so a larger value (-0.2e-6 is larger
        #   than -0.5e-6) will correctly come first, effectively sorting latency
        #   in ascending (faster) order.
        sorted_metrics = sorted(tag_metrics, key=lambda x: (x[1], -x[2]), reverse=True)

        # Step 3: Extract just the tag names from the sorted list and return.
        return [item[0] for item in sorted_metrics]

    def create_resized_view(
        self: T_LogicalTopology,
        view_name: str,
        new_sizes: Dict[Union[LogicalDimensionTag, str], int],
        global_sharing_context: str = "concurrent_baseline",
    ) -> "LogicalTopologyView":
        """
        Creates a new view by resizing specified logical dimensions.

        This method constructs a new LogicalTopologyView that includes all
        dimensions from the original topology. For dimensions whose tags are
        present in the `new_sizes` dictionary, their size is updated. All other
        dimensions remain unchanged.

        Args:
            view_name: A name for the new LogicalTopologyView.
            new_sizes: A dictionary mapping logical tags to their desired new size.
            global_sharing_context: The context for calculating bandwidth in the new view.

        Returns:
            A new LogicalTopologyView with the specified dimensions resized.

        Raises:
            ValueError: If a new size exceeds the underlying physical dimension's
                        size or if a tag in `new_sizes` is not in the topology.
        """
        new_mappings: List[LogicalDimensionMapping] = []

        # Normalize keys in the input dictionary to strings for consistent lookups
        normalized_new_sizes = {
            (k.value if isinstance(k, Enum) else k): v for k, v in new_sizes.items()
        }

        # Iterate over all original dimensions to build the new list of mappings
        for original_mapping in self.dimensions:
            tag_val = (
                original_mapping.logical_tag
                if isinstance(original_mapping.logical_tag, str)
                else original_mapping.logical_tag.value
            )

            if tag_val in normalized_new_sizes:
                # This dimension needs to be resized
                new_size = normalized_new_sizes[tag_val]

                # Validate the new size against its physical dimension
                phys_dim = self.get_physical_dimension_for_tag(tag_val)
                if new_size > phys_dim.size:
                    raise ValueError(
                        f"Cannot resize tag '{tag_val}' to size {new_size}. "
                        f"It exceeds the physical dimension '{phys_dim.name}' size of {phys_dim.size}."
                    )

                # Create a new mapping with the updated size using Pydantic's copy method
                resized_mapping = original_mapping.model_copy(update={"size": new_size})
                new_mappings.append(resized_mapping)
            else:
                # This dimension is unchanged, so add the original mapping
                new_mappings.append(original_mapping)

        # Create the final view with the full set of (potentially resized) mappings
        return LogicalTopologyView(
            name=view_name,
            physical_topology=self.physical_topology,
            dimensions=new_mappings,
            original_topology_ref=self,
            global_sharing_context=global_sharing_context,
        )

    @classmethod
    def load_from_yaml(
        cls: Type[T_LogicalTopology], filepath: str, physical_topology: PhysicalTopology
    ) -> T_LogicalTopology:
        # Implementation for loading LogicalTopology, similar to PhysicalTopology but simpler
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LogicalTopology config file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls(physical_topology=physical_topology, **config_dict)
        except Exception as e:
            raise ValueError(f"Error loading LogicalTopology from '{filepath}': {e}") from e


# Logical topology view (derived class)
class LogicalTopologyView(LogicalTopology):
    """A specialized view of a LogicalTopology for calculating performance of specific collectives."""

    original_topology_ref: LogicalTopology = Field(..., exclude=True, repr=False)
    global_sharing_context: str = Field(..., exclude=True, repr=False)
    # When True, skip enforcing mapping.size <= physical dimension size (virtual sizing).
    allow_virtual_resize: bool = Field(default=False, exclude=True, repr=False)

    @override
    def get_effective_bandwidth(
        self, tag_in_view: Union[LogicalDimensionTag, str], context: str = "concurrent_baseline"
    ) -> float:
        """Calculates bandwidth (Bps) for a tag, considering global and local sharing contexts."""
        self.get_mapping_for_tag(tag_in_view)  # Check existence in view
        phys_dim = self.get_physical_dimension_for_tag(tag_in_view)
        interconnect = self.get_interconnect_for_tag(tag_in_view)
        base_ic_bandwidth_Bps = interconnect.bandwidth_Bps

        # Step 1: Calculate the globally-aware bandwidth slice from the original topology
        globally_aware_slice_Bps: float
        if self.global_sharing_context == "concurrent_baseline":
            num_global_sharers = sum(
                1
                for ldm in self.original_topology_ref.dimensions
                if self.original_topology_ref.get_physical_dimension_for_tag(
                    ldm.logical_tag
                ).interconnect_name
                == phys_dim.interconnect_name
            )
            globally_aware_slice_Bps = base_ic_bandwidth_Bps / max(1, num_global_sharers)
        elif self.global_sharing_context == "max_exclusive":
            globally_aware_slice_Bps = base_ic_bandwidth_Bps
        else:
            raise ValueError(f"Unknown global_sharing_context: '{self.global_sharing_context}'")

        # Step 2: Apply further sharing based on context *within this view*
        if context == "concurrent_baseline":
            num_view_sharers = sum(
                1
                for ldm in self.dimensions
                if self.get_physical_dimension_for_tag(ldm.logical_tag).interconnect_name
                == phys_dim.interconnect_name
            )
            return globally_aware_slice_Bps / max(1, num_view_sharers)
        elif context == "max_exclusive":
            return globally_aware_slice_Bps
        else:
            raise ValueError(f"Unknown context_within_view: '{context}'")

    @model_validator(mode="after")
    def validate_mapping(self) -> "LogicalTopologyView":
        """
        Validate that mappings reference existing physical dimensions.
        If allow_virtual_resize is False, also enforce size <= physical capacity.
        """
        phys_dims = {pd.name: pd for pd in self.physical_topology.dimensions}
        used_tags: Set[str] = set()
        for mapping in self.dimensions:
            tag_val = (
                mapping.logical_tag
                if isinstance(mapping.logical_tag, str)
                else mapping.logical_tag.value
            )
            if tag_val in used_tags:
                raise ValueError(f"LogicalTag '{tag_val}' is used multiple times in '{self.name}'.")
            used_tags.add(tag_val)
            if mapping.maps_to_physical_dimension_name not in phys_dims:
                raise ValueError(
                    f"Mapping '{tag_val}' maps to non-existent PhysDim '{mapping.maps_to_physical_dimension_name}'."
                )
            if not self.allow_virtual_resize:
                if mapping.size > phys_dims[mapping.maps_to_physical_dimension_name].size:
                    raise ValueError(
                        f"Mapping '{tag_val}' size {mapping.size} exceeds its PhysDim size."
                    )
        return self
