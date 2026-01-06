from types import ModuleType

import pytest


def test_import():
    # Check if the module can be imported
    try:
        import scalescope
    except ImportError as e:
        pytest.fail(f"Failed to import scalescope: {e}")

    # Check if the module is of type ModuleType
    assert isinstance(scalescope, ModuleType), "scalescope is not a module"


def test_import_topology_spec_classes():
    # Check if classes/enums are available in topology_spec
    try:
        import scalescope.topology_spec as topo
    except ImportError as e:
        pytest.fail(f"Failed to import topology_spec: {e}")

    for name in ("PhysicalTopology1DType", "LogicalDimensionTag", "PhysicalTopology"):
        assert hasattr(topo, name), f"Missing '{name}' in topology_spec"


def test_import_hardware_spec_classes():
    # Check if classes are available in hardware_spec
    try:
        import scalescope.hardware_spec as hw
    except ImportError as e:
        pytest.fail(f"Failed to import hardware_spec: {e}")

    for name in ("Compute", "Interconnect", "Memory", "NetworkNode"):
        assert hasattr(hw, name), f"Missing '{name}' in hardware_spec"


def test_import_model_spec_classes():
    # Check if classes are available in model_spec
    try:
        import scalescope.model_spec as ms
    except ImportError as e:
        pytest.fail(f"Failed to import model_spec: {e}")

    assert hasattr(ms, "TransformerConfig"), "Missing 'TransformerConfig' in model_spec"


def test_import_train_spec_classes():
    # Check if classes/enums are available in workload_spec
    try:
        import scalescope.workload_spec as ws
    except ImportError as e:
        pytest.fail(f"Failed to import workload_spec: {e}")

    for name in (
        "ActivationStorageOptimizationStrategy",
        "DataParallelSpec",
        "PipelineParallelSpec",
        "TensorParallelSpec",
        "TrainSpec",
    ):
        assert hasattr(ws, name), f"Missing '{name}' in workload_spec"


def test_import_memory_model_classes():
    # Check if classes are available in memory_model
    try:
        import scalescope.memory_model as mm
    except ImportError as e:
        pytest.fail(f"Failed to import memory_model: {e}")

    for name in ("LayerMemoryTraffic", "MemoryAccess"):
        assert hasattr(mm, name), f"Missing '{name}' in memory_model"
