"""Contains helpers to convert sweep values, clone/patch specs, and other utilties
shared by the sweep runner and the CLI.
"""

import math
import operator
from enum import Enum
from functools import reduce
from typing import (
    Any,
    Dict,
    get_type_hints,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import numpy as np  # Using numpy for robust range generation, esp. with floats
from pydantic import BaseModel

from scalescope.workload_spec import (
    DataParallelVariant,
    PipelineParallelVariant,
    TensorParallelVariant,
)

# Allowed generator function registry


def power_of_two(min_val, max_val, dtype=int):
    """Generates powers of two within the range [min_val, max_val]."""
    if not (
        isinstance(min_val, int)
        and isinstance(max_val, int)
        and min_val >= 1
        and max_val >= min_val
    ):
        raise ValueError("power_of_two requires integer min_val >= 1 and max_val >= min_val")
    values = []
    current = 1
    while current < min_val:
        current *= 2
    while current <= max_val:
        values.append(current)
        current *= 2
    return [dtype(v) for v in values]  # Cast to requested type


def linear_space(start, stop, num, dtype=float):
    """Generates linearly spaced samples like np.linspace."""
    return np.linspace(start, stop, num, dtype=dtype).tolist()


def perfect_square_factors(min_val, max_val, dtype=int) -> List[List[int]]:
    """Generates a list of factor pairs [n, n] such that n*n
    is a perfect square within the inclusive range [min_val, max_val].

    Args:
        min_val: The minimum perfect square value to consider (inclusive).
        max_val: The maximum perfect square value to consider (inclusive).
        dtype: The desired data type for the factors 'n' (usually int).

    Returns:
        A list of lists, where each inner list is [n, n].
    """
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise ValueError("perfect_square_factors requires numeric min_val and max_val.")
    if min_val > max_val:
        raise ValueError(
            f"perfect_square_factors min_val ({min_val}) cannot be greater than max_val ({max_val})."
        )
    if min_val < 0:
        print(
            f"Warning: min_val ({min_val}) for perfect_square_factors is negative. Starting search from 0."
        )
        min_val = 0

    factor_pairs = []
    # Start checking from the integer part of the square root of min_val
    n = math.ceil(math.sqrt(min_val))

    while True:
        square = n * n
        if square > max_val:
            break  # Stop when the square exceeds the maximum value

        # Since we started at ceil(sqrt(min_val)), the first square is guaranteed
        # to be >= min_val. All subsequent squares will also be.
        try:
            factor = dtype(n)  # Ensure the factor itself is the correct type
            factor_pairs.append([factor, factor])
        except (ValueError, TypeError):
            raise TypeError(f"Could not convert factor {n} to requested dtype {dtype}")

        n += 1

    return factor_pairs


def _find_most_balanced_multiplicands(n):
    """
    Helper function to find the two most balanced multiplicands for a single number n.
    Returns a list [a, b] where a * b = n and a <= b, and abs(a-b) is minimized.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")

    if n == 1:
        return [1, 1]

    # Start searching from the integer part of the square root downwards
    # The first factor found will be the largest factor <= sqrt(n)
    # This ensures that factor1 and n // factor1 are as close as possible
    for i in range(int(math.sqrt(n)), 0, -1):  # Iterate from floor(sqrt(n)) down to 1
        if n % i == 0:
            factor1 = i
            factor2 = n // i
            # Since we iterate downwards from sqrt(n), factor1 <= sqrt(n)
            # and factor2 = n/factor1 >= n/sqrt(n) = sqrt(n).
            # So, factor1 will always be less than or equal to factor2.
            return [factor1, factor2]

    raise ValueError(f"No factors found for {n}. This should not happen for positive integers.")


def decompose_numbers_with_balanced_multiplicands(min_val, max_val, step):
    """
    For each number in the range [min_val, max_val] with a given step,
    returns its decomposition into two multiplicands that are as balanced as possible.

    Args:
        min_val (int): The minimum value of the range (inclusive).
        max_val (int): The maximum value of the range (inclusive).
        step (int): The step for the range. Must not be 0.

    Returns:
        dict: A dictionary where keys are the numbers in the range and
              values are lists of two multiplicands [a, b] such that a*b=number
              and abs(a-b) is minimized, with a <= b.
              Returns an error message as value for non-positive integers
              or if decomposition is not applicable.
    """
    if not all(isinstance(arg, int) for arg in [min_val, max_val, step]):
        raise TypeError("min_val, max_val, and step must be integers.")
    if step == 0:
        raise ValueError("Step cannot be zero.")
    if (step > 0 and min_val > max_val) or (step < 0 and min_val < max_val):
        # Range is empty or invalid direction
        return []

    results = []

    # Python's range(start, stop, step) does not include stop.
    # We need to adjust the 'stop' value to make max_val inclusive.
    # For simplicity, use Python's range and adjust the end condition.
    # If step > 0, range goes up to max_val + 1 (or slightly more to be safe with step)
    # If step < 0, range goes up to max_val - 1 (or slightly less to be safe with step)

    # A more robust way to handle range iteration:
    if step > 0:
        stop_val = max_val + 1
    else:  # step < 0
        stop_val = max_val - 1

    for num in range(min_val, stop_val, step):
        # The loop condition above already ensures num is within desired bounds
        # when combined with range behavior.
        # e.g. range(10, 0, -1) -> 10,9,...,1. stop_val = 0-1 = -1
        # e.g. range(0, 10, 1) -> 0,1,...,9. stop_val = 10+1 = 11
        if num > 0:
            results.append(_find_most_balanced_multiplicands(num))
    return results


# The actual registry mapping string names to function objects
ALLOWED_SWEEP_GENERATORS = {
    "power_of_two": power_of_two,
    "linear_space": linear_space,
    "perfect_square_factors": perfect_square_factors,
    "decompose_with_balanced_multiplicands": decompose_numbers_with_balanced_multiplicands,
}

# Type variable for Pydantic models used in configs
T_ConfigModel = TypeVar("T_ConfigModel", bound=BaseModel)


def _get_type_at_path(model_cls: Type[BaseModel], attr_path: str) -> Optional[Type]:
    """
    Helper to find the type hint of a potentially nested attribute within a model class.
    Handles dot notation for nested models.
    """
    current_type: Any = model_cls
    parts = attr_path.split(".")
    try:
        hints = {}
        for i, part in enumerate(parts):
            if not isinstance(current_type, type) or not issubclass(current_type, BaseModel):
                return None  # Cannot introspect non-BaseModel types mid-path

            hints = get_type_hints(current_type)
            field_type = hints.get(part)

            if field_type is None:
                # Check computed fields
                if hasattr(current_type, part) and isinstance(
                    getattr(current_type, part), property
                ):
                    prop = getattr(current_type, part)
                    if prop.fget:
                        try:
                            field_type = get_type_hints(prop.fget).get("return")
                        except TypeError:
                            field_type = None
                    else:
                        field_type = None
                if field_type is None:
                    return None  # Not found

            origin = getattr(field_type, "__origin__", None)
            args = getattr(field_type, "__args__", ())
            is_optional = origin is Union and type(None) in args
            effective_type = args[0] if is_optional and len(args) == 2 else field_type

            if i == len(parts) - 1:
                return effective_type
            else:
                current_type = effective_type
                if not isinstance(current_type, type) or not issubclass(current_type, BaseModel):
                    return None  # Stop if next step isn't a BaseModel
        return None
    except Exception as e:
        print(f"Debug: Error getting type hint for '{attr_path}': {e}")
        return None


def convert_sweep_values(
    sweep_dict_yaml: Dict[str, Union[List[Any], Dict[str, Any]]],
    config_object_map: Dict[str, BaseModel],
) -> Dict[str, List[Any]]:
    processed_sweep_dict: Dict[str, List[Any]] = {}
    if not isinstance(sweep_dict_yaml, dict):
        raise TypeError("Sweep configuration from YAML must be a dictionary.")

    for full_path, values_yaml in sweep_dict_yaml.items():
        path_parts = full_path.split(".", 1)
        if len(path_parts) < 2:
            raise ValueError(f"Invalid sweep path '{full_path}'.")
        object_key, nested_attr_path = path_parts

        base_object = config_object_map.get(object_key)
        if base_object is None:
            raise KeyError(f"Object key '{object_key}' not in config map.")
        if not isinstance(base_object, BaseModel):
            raise TypeError(f"Object '{object_key}' not BaseModel.")
        spec_cls = type(base_object)
        target_type = _get_type_at_path(spec_cls, nested_attr_path)

        generated_values: List[Any] = []

        # Determine value type: List, Range Dict, or Generator Dict
        if isinstance(values_yaml, list):
            # Handle list of values (Enum conversion included) ---
            raw_values_list = values_yaml
            if target_type and isinstance(target_type, type) and issubclass(target_type, Enum):
                for raw_value in raw_values_list:
                    if not isinstance(raw_value, str):
                        raise ValueError(
                            f"Expected string Enum for '{full_path}', got {type(raw_value)}"
                        )
                    try:
                        generated_values.append(target_type[raw_value])  # Try Name
                    except KeyError:
                        try:
                            generated_values.append(target_type(raw_value))  # Try Value
                        except ValueError:
                            raise ValueError(f"Invalid Enum value '{raw_value}' for '{full_path}'")
            else:
                generated_values = raw_values_list  # Pass through other types

        elif isinstance(values_yaml, dict):
            if "min" in values_yaml and "max" in values_yaml and "step" in values_yaml:
                # handle Range Dictionary
                if target_type and isinstance(target_type, type) and issubclass(target_type, Enum):
                    raise ValueError(f"Range sweep not supported for Enum '{full_path}'.")
                try:
                    min_val, max_val, step = (
                        values_yaml["min"],
                        values_yaml["max"],
                        values_yaml["step"],
                    )
                    if not all(isinstance(v, (int, float)) for v in [min_val, max_val, step]):
                        raise ValueError("Range values must be numeric.")
                    if step <= 0:
                        raise ValueError("Range step must be positive.")
                    if min_val > max_val:
                        raise ValueError("Range min > max.")

                    use_int = target_type is int
                    # Use numpy.arange for robust stepping
                    np_values = np.arange(min_val, max_val + step * 0.5, step)
                    # Filter potential values slightly over max due to float precision
                    np_values = np_values[np_values <= max_val + 1e-9]

                    if use_int:
                        if not np.all(np.isclose(np_values, np.round(np_values))):
                            print(f"Warning: Rounding range values for int field '{full_path}'.")
                        generated_values = np.round(np_values).astype(int).tolist()
                    else:
                        generated_values = np_values.astype(float).tolist()

                    if not generated_values:
                        print(f"Warning: Range spec for '{full_path}' generated empty list.")

                except Exception as e:
                    raise ValueError(f"Error processing range for '{full_path}': {e}")

            elif "generator" in values_yaml:
                # handle Generator Dictionary
                if target_type and isinstance(target_type, type) and issubclass(target_type, Enum):
                    raise ValueError(
                        f"Generator sweep is not supported for Enum type at path '{full_path}'. Provide a list of Enum names/values instead."
                    )

                generator_name = values_yaml["generator"]
                if not isinstance(generator_name, str):
                    raise ValueError(f"Function name for '{full_path}' must be a string.")

                generator_func = ALLOWED_SWEEP_GENERATORS.get(generator_name)
                if generator_func is None:
                    raise ValueError(
                        f"Unknown or disallowed function '{generator_name}' specified for '{full_path}'. "
                        f"Allowed generators: {list(ALLOWED_SWEEP_GENERATORS.keys())}"
                    )

                # Prepare arguments, excluding the 'generator' key itself
                generator_args = {k: v for k, v in values_yaml.items() if k != "generator"}

                # Attempt to pass target dtype if generator supports it
                if target_type in [int, float]:  # Add other supported types if needed
                    if "dtype" not in generator_args:  # Don't override if explicitly set
                        generator_args["dtype"] = target_type

                try:
                    print(
                        f"Debug: Calling generator '{generator_name}' for '{full_path}' with args: {generator_args}"
                    )
                    generated_values = generator_func(**generator_args)
                    if not isinstance(generated_values, list):
                        raise TypeError(
                            f"Generator function '{generator_name}' did not return a list."
                        )
                    print(f"Debug: Generated values: {generated_values}")
                except Exception as e:
                    raise ValueError(
                        f"Error executing generator '{generator_name}' for '{full_path}' with args {generator_args}: {e}"
                    ) from e

            else:
                # Dictionary didn't match known structures (range or generator)
                raise ValueError(
                    f"Invalid dictionary specification for sweep path '{full_path}'. "
                    "Expected list, range dict {{min, max, step}}, or generator dict {{generator, ...}}."
                )
        else:
            # Neither list nor dict
            raise TypeError(
                f"Sweep value for '{full_path}' must be list or dict, got {type(values_yaml)}."
            )

        # Handle TP spec
        if full_path == "tp_spec.variant" and isinstance(generated_values, str):
            try:
                converted_value = TensorParallelVariant(generated_values)
            except ValueError:
                raise ValueError(
                    f"Invalid TensorParallelVariant value '{generated_values}' in sweep for key '{full_path}'"
                )

        # Handle PP spec
        if full_path == "pp_spec.variant" and isinstance(generated_values, str):
            try:
                converted_value = PipelineParallelVariant(generated_values)
            except ValueError:
                raise ValueError(
                    f"Invalid PipelineParallelVariant value '{generated_values}' in sweep for key '{full_path}'"
                )

        # Handle DP spec
        if full_path == "dp_spec.variant" and isinstance(generated_values, str):
            try:
                converted_value = DataParallelVariant(generated_values)  # noqa F841
            except ValueError:
                raise ValueError(
                    f"Invalid DataParallelVariant value '{generated_values}' in sweep for key '{full_path}'"
                )

        # Store the final list of values
        processed_sweep_dict[full_path] = generated_values

    return processed_sweep_dict


def get_integer_factorizations(n: int, num_dimensions: int) -> List[List[int]]:
    """
    Finds all factorizations of n into num_dimensions positive integers.
    Order matters initially, but we can sort later if needed for balance.
    """
    if n == 0:
        return []
    if num_dimensions == 0:
        return (
            [[]] if n == 1 else []
        )  # Only way to get product 1 with 0 dims is an empty list (convention)
    if num_dimensions == 1:
        return [[n]] if n >= 1 else []

    results = []
    # Iterate for the first factor s_1. s_1 must be a divisor of n.
    # Max value for s_1 is n. Min is 1.
    for s1 in range(1, int(n**0.5) + 1):  # Iterate up to sqrt(n) for divisors
        if n % s1 == 0:
            divisors = [s1]
            if s1 * s1 != n:
                divisors.append(n // s1)  # Add the pair divisor

            for d in divisors:
                sub_factorizations = get_integer_factorizations(n // d, num_dimensions - 1)
                for sub_fact in sub_factorizations:
                    results.append([d] + sub_fact)
    return results


def get_nd_factorizations_recursive(target_product: int, num_dims: int) -> List[List[int]]:
    """
    Generates factorizations of target_product into num_dims positive integers.
    Example: target_product=4, num_dims=2 -> [[1,4], [2,2], [4,1]]
    """
    if num_dims == 1:
        if target_product >= 1:
            return [[target_product]]
        return []

    factors_list = []
    # Iterate s_1 from 1 up to target_product
    for s1 in range(1, target_product + 1):
        if target_product % s1 == 0:
            remaining_product = target_product // s1
            sub_factorizations = get_nd_factorizations_recursive(remaining_product, num_dims - 1)
            for sub_fact in sub_factorizations:
                factors_list.append([s1] + sub_fact)
    return factors_list


class ExpertTopologySlice:
    def __init__(self, expert_sub_mesh_shape: List[int], ep_degree: int, balance_score: float):
        self.expert_sub_mesh_shape = expert_sub_mesh_shape  # e.g., [4,4]
        self.ep_degree = ep_degree  # product of expert_sub_mesh_shape, e.g., 16
        self.balance_score = balance_score  # Lower is generally better

    def __repr__(self):
        return (
            f"ExpertSlice(shape={self.expert_sub_mesh_shape}, ep_degree={self.ep_degree}, "
            f"balance_score={self.balance_score:.2f})"
        )


def slice_mesh_for_experts(
    original_mesh_shape: List[int], num_experts: int
) -> Optional[ExpertTopologySlice]:
    """
    Slices an N-D mesh to accommodate a number of experts, aiming for balanced sub-meshes.
    Returns the shape of one expert's sub-mesh and its effective EP degree.
    Assumes all experts get an identically shaped sub-mesh.
    """
    if num_experts <= 0:
        raise ValueError("Number of experts must be positive.")
    if not original_mesh_shape:
        raise ValueError("Original mesh shape cannot be empty.")
    if any(d <= 0 for d in original_mesh_shape):
        raise ValueError("Original mesh dimensions must be positive.")

    num_mesh_dims = len(original_mesh_shape)

    # Step 2: Find factorizations of num_experts into num_mesh_dims integers [s_1, ..., s_N]
    # where s_i is the number of experts along dimension i of the original mesh.
    split_factorizations = get_nd_factorizations_recursive(num_experts, num_mesh_dims)

    valid_expert_slices: List[ExpertTopologySlice] = []

    for splits in split_factorizations:  # splits = [s_1, s_2, ..., s_N]
        is_valid_split = True
        expert_sub_mesh_shape = []
        for i in range(num_mesh_dims):
            original_dim_size = original_mesh_shape[i]
            num_splits_along_dim = splits[i]
            if original_dim_size % num_splits_along_dim != 0:
                is_valid_split = False
                break
            expert_sub_mesh_shape.append(original_dim_size // num_splits_along_dim)

        if is_valid_split:
            ep_degree = reduce(operator.mul, expert_sub_mesh_shape, 1)
            if ep_degree == 0:
                continue  # Should not happen if original_mesh_shape dims are positive

            # Calculate balance score (e.g., minimize variance or max/min ratio)
            # Using sum of squared differences from the geometric mean of sub_mesh_shape dimensions
            geom_mean = (
                ep_degree ** (1.0 / len(expert_sub_mesh_shape)) if expert_sub_mesh_shape else 0
            )
            balance_score = 0
            if geom_mean > 0:  # Avoid issues if ep_degree or len is 0/1 in weird cases
                for dim_size in expert_sub_mesh_shape:
                    balance_score += (dim_size - geom_mean) ** 2
            else:  # if only one dimension in expert_sub_mesh_shape, score is 0 (perfectly balanced for 1D)
                balance_score = 0 if len(expert_sub_mesh_shape) <= 1 else float("inf")

            valid_expert_slices.append(
                ExpertTopologySlice(expert_sub_mesh_shape, ep_degree, balance_score)
            )

    if not valid_expert_slices:
        # This can happen if num_experts cannot be factored in a way that divides the mesh dimensions.
        # e.g., mesh [7,1], num_experts=2. Cannot split 7 into 2 integer parts.
        # Fallback: try to assign total_chips / num_experts and then factor that?
        # For now, return None or raise error.
        print(
            f"Warning: Could not find a valid way to slice mesh {original_mesh_shape} "
            f"for {num_experts} experts such that original dimensions are evenly divisible by splits."
        )
        # Simplistic fallback: give each expert 1 chip from each dimension if possible,
        # and then the rest of chips are not used by other experts in a structured way.
        # This means ep_degree could be very small.
        # Or, if no perfect tiling, we might need a more complex assignment.
        # For this simplified function, we'll say no perfect tiling found.
        return None

    # Sort by balance_score (lower is better), then by largest ep_degree (prefer larger experts if balance is tied)
    valid_expert_slices.sort(key=lambda s: (s.balance_score, -s.ep_degree))

    return valid_expert_slices[0]  # Return the best one
