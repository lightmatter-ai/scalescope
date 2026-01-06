"""Sweep configuration utilities.

Generates concrete configuration dictionaries from high-level sweep specs.
"""

import itertools
import math
from typing import Any, Dict, Iterator, List, Tuple, TypeVar

# Pydantic imports needed for type hints and error handling
from pydantic import BaseModel, ValidationError

# Type variable for the spec class (e.g., TrainSpec)
T_ConfigModel = TypeVar("T_ConfigModel", bound=BaseModel)
# Type variable for generic nested models
T_Nested = TypeVar("T_Nested", bound=BaseModel)


def _set_nested_attr_pydantic(obj: T_Nested, attr_path: str, value: Any) -> T_Nested:
    """Recursively creates a new Pydantic model instance with a nested attribute updated.
    Uses model_copy() for immutability (Pydantic v2).

    Args:
        obj: The Pydantic model instance (must have frozen=True or be treated immutably).
        attr_path: Dot-separated path RELATIVE TO obj (e.g., "pp_spec.num_stages").
        value: The new value to set at the path.

    Returns:
        A new model instance with the update applied.

    Raises:
        AttributeError, TypeError, ValidationError: If path is invalid, object is not
                                                    a BaseModel mid-path, or update fails validation.
    """
    parts = attr_path.split(".", 1)
    current_attr = parts[0]
    if not hasattr(obj, current_attr):
        raise AttributeError(
            f"Object {type(obj).__name__} has no attr '{current_attr}' in path '{attr_path}'"
        )
    if len(parts) == 1:
        try:
            return obj.model_copy(update={current_attr: value})
        except ValidationError as e:
            raise ValidationError(
                f"Validation failed updating '{current_attr}' to '{value}': {e}"
            ) from e
        except Exception as e:
            raise TypeError(
                f"Failed to update '{current_attr}' on {type(obj).__name__}: {e}"
            ) from e
    else:
        nested_obj = getattr(obj, current_attr)
        if not isinstance(nested_obj, BaseModel):
            raise TypeError(f"Attr '{current_attr}' is not Pydantic model: {attr_path}")
        rest_of_path = parts[1]
        try:
            new_nested_obj = _set_nested_attr_pydantic(nested_obj, rest_of_path, value)
        except Exception as e:
            raise type(e)(
                f"Error setting nested '{rest_of_path}' within '{current_attr}': {e}"
            ) from e
        try:
            return obj.model_copy(update={current_attr: new_nested_obj})
        except ValidationError as e:
            raise ValidationError(
                f"Validation failed replacing nested '{current_attr}': {e}"
            ) from e
        except Exception as e:
            raise TypeError(f"Failed to replace nested '{current_attr}': {e}") from e


def generate_specs_from_sweep(
    base_specs: Dict[str, T_ConfigModel],  # Dict mapping name -> base spec instance
    sweep_config: Dict[str, List[Any]],  # Processed sweep config (Python types)
) -> Iterator[Tuple[Dict[str, Any], Dict[str, T_ConfigModel]]]:
    """
    Generates configuration object sets based on base objects and a sweep config.

    Args:
        base_specs: Dictionary mapping spec keys (e.g., 'train_spec', 'model_spec')
                      to their corresponding validated base Pydantic model instances.
        sweep_config: Dictionary mapping full sweep paths ('spec_key.attr.path')
                      to lists of *processed Python objects* (Enums, ints, etc.).

    Yields:
        Tuples of (sweep_params, current_configs_dict).
        - sweep_params: Dict representing the specific combination of sweep values used.
        - current_configs_dict: Dict mapping spec keys to the potentially modified
                                Pydantic model instances for this sweep iteration.
        Skips invalid combinations where applying sweep values causes validation errors.
    """
    if not sweep_config:
        print("No sweep config provided. Yielding only the base configuration set.")
        yield {}, base_specs.copy()  # Yield a copy of the base configs
        return

    sweep_full_paths = list(sweep_config.keys())
    value_lists = [sweep_config[path] for path in sweep_full_paths]

    # Input validation (already done in calling script, but basic check here too)
    if any(not isinstance(v, list) for v in value_lists):
        bad_keys = [k for k, v in sweep_config.items() if not isinstance(v, list)]
        print(
            f"Error in generate_configs_from_sweep: Sweep values must be lists. Found non-list for: {bad_keys}"
        )
        return iter([])
    if any(not v for v in value_lists):
        empty_params = [k for k, v in sweep_config.items() if not v]
        print(f"Warning: Sweep config has empty list(s) for: {empty_params}. No specs generated.")
        return iter([])

    try:
        total_combinations = math.prod(len(v) for v in value_lists)
    except Exception:
        print("Error: Could not calculate total sweep combinations.")
        return iter([])

    print(f"Starting sweep generation for {total_combinations} configuration combinations...")
    generated_count = 0
    skipped_count = 0

    for i, value_combination in enumerate(itertools.product(*value_lists)):
        # Map the full path to the specific value for this combination
        current_sweep_params: Dict[str, Any] = dict(zip(sweep_full_paths, value_combination))

        # Start with copies of the base config objects for this iteration
        # Use model_copy for Pydantic models to ensure independent copies
        current_specs: Dict[str, T_ConfigModel] = {
            key: obj.model_copy() for key, obj in base_specs.items()
        }

        try:
            # Apply all changes defined in this sweep combination
            for full_path, value in current_sweep_params.items():
                path_parts = full_path.split(".", 1)
                spec_key = path_parts[0]
                nested_attr_path = path_parts[1]

                # Get the *current* instance of the object to modify for this iteration
                target_object_instance = current_specs.get(spec_key)

                if target_object_instance is None:
                    # This should ideally be caught during sweep conversion, but double-check
                    raise KeyError(
                        f"Object key '{spec_key}' from sweep path '{full_path}' "
                        f"not found in base_configs during generation."
                    )

                # Apply the change using the nested attribute setter
                # It returns a *new* instance of the modified object
                modified_object_instance = _set_nested_attr_pydantic(
                    target_object_instance, nested_attr_path, value
                )

                # Update the dictionary holding the *current state* for this iteration
                current_specs[spec_key] = modified_object_instance

            # If all updates succeeded without validation error, yield the result
            generated_count += 1
            yield current_sweep_params, current_specs

        except (ValidationError, ValueError, AttributeError, TypeError, KeyError) as e:
            # Catch errors during object modification (e.g., validation fail, bad path)
            skipped_count += 1
            print(
                f"Skipping invalid combination {i+1}/{total_combinations}: {current_sweep_params}. Reason: {e}"
            )
            continue
        except Exception as e:  # Catch unexpected errors
            skipped_count += 1
            print(
                f"Skipping combo {i+1}/{total_combinations} due to unexpected error: {current_sweep_params}. Error: {e}"
            )
            continue

    print("\nSweep generation finished.")
    print(f"  Total combinations attempted: {total_combinations}")
    print(f"  Valid config sets generated: {generated_count}")
    print(f"  Invalid combinations skipped: {skipped_count}")
