"""New sweep system (DSL) for ScaleScope.

This module intentionally does NOT support the legacy sweep format.

Design:
- A sweep config contains:
  - `leaves`: named leaf definitions. Each leaf expands to a list[case] where each
    case is a mapping {full_path: value}. A leaf may set one or more paths, and its
    cases are aligned tuples.
  - `sweep`: an expression tree over leaves, using operators:
      - cartesian: cross product
      - zip: aligned by index (strict by default)
      - union: concatenate streams
    Each operator may optionally include:
      - where: list[str] verifier names
      - on_fail: "skip" (default) or "error"

Verifiers:
- Verifiers are allowlisted python callables that accept
  (case, base_specs, params) and return bool.
- YAML supports either plain string names or mapping entries with
  ``name`` and optional ``params``.

Note: This DSL produces sweep points (cases). It does not run the model; the caller does.
"""

import math
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
)

import yaml
from pydantic import BaseModel, ValidationError


def _model_fields_data(obj: BaseModel) -> dict[str, Any]:
    """Return a dict of *input* fields for `obj` without evaluating computed fields.

    We intentionally avoid `model_dump()` here because dumping can trigger evaluation of computed
    properties on nested models, which is both slower and can fail if those computed properties
    depend on optional fields that aren't present on all variants.
    """
    # Pydantic v2
    fields_v2 = getattr(type(obj), "model_fields", None)
    if fields_v2:
        return {k: getattr(obj, k) for k in fields_v2.keys()}
    # Pydantic v1 fallback (shouldn't normally happen in this repo)
    fields_v1 = getattr(type(obj), "__fields__", None) or {}
    return {k: getattr(obj, k) for k in fields_v1.keys()}


def _set_nested_attr_pydantic(obj: BaseModel, attr_path: str, value: Any) -> BaseModel:
    """Recursively creates a new Pydantic model instance with a nested attribute updated.

    This is the only shared helper we need from the legacy sweep system; we inline it here
    so the old sweep modules can be removed cleanly.
    """
    parts = attr_path.split(".", 1)
    current_attr = parts[0]
    if not hasattr(obj, current_attr):
        raise AttributeError(
            f"Object {type(obj).__name__} has no attr '{current_attr}' in path '{attr_path}'"
        )
    if len(parts) == 1:
        # IMPORTANT: `model_copy(update=...)` does not re-validate or coerce types in Pydantic v2.
        # For sweep YAML ergonomics, we want values like "SHARD_WEIGHTS_1D" to be coerced into
        # the appropriate Enum types, and nested dicts to be coerced into nested models.
        try:
            data = _model_fields_data(obj)
            data[current_attr] = value
            return type(obj).model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed updating '{current_attr}' to '{value}' on {type(obj).__name__}: {e}"
            ) from e
        except Exception as e:
            raise TypeError(
                f"Failed to update '{current_attr}' on {type(obj).__name__}: {e}"
            ) from e

    nested_obj = getattr(obj, current_attr)
    rest_of_path = parts[1]

    # Support indexing into lists/tuples using dot-path segments, e.g.:
    #   "dimensions.1.topology_type"
    # This is common for immutable Pydantic models that contain lists of nested models.
    if isinstance(nested_obj, (list, tuple)):
        idx_parts = rest_of_path.split(".", 1)
        idx_str = idx_parts[0]
        if not idx_str.isdigit():
            raise TypeError(
                f"Attr '{current_attr}' is a list/tuple, expected numeric index in path '{attr_path}', "
                f"got '{idx_str}'"
            )
        idx = int(idx_str)
        if idx < 0 or idx >= len(nested_obj):
            raise IndexError(
                f"Index {idx} out of range for '{current_attr}' (len={len(nested_obj)}) in path '{attr_path}'"
            )
        # If the path ends at the index, replace the element directly.
        if len(idx_parts) == 1:
            new_elem = value
        else:
            elem = nested_obj[idx]
            if not isinstance(elem, BaseModel):
                raise TypeError(
                    f"Element '{current_attr}[{idx}]' is not a Pydantic model, cannot set nested path "
                    f"'{idx_parts[1]}' in '{attr_path}'"
                )
            try:
                new_elem = _set_nested_attr_pydantic(elem, idx_parts[1], value)
            except Exception as e:
                raise type(e)(
                    f"Error setting nested '{idx_parts[1]}' within '{current_attr}[{idx}]': {e}"
                ) from e
        new_list = list(nested_obj)
        new_list[idx] = new_elem
        try:
            data = _model_fields_data(obj)
            data[current_attr] = new_list
            return type(obj).model_validate(data)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed replacing list '{current_attr}[{idx}]' on {type(obj).__name__}: {e}"
            ) from e
        except Exception as e:
            raise TypeError(f"Failed to replace list element '{current_attr}[{idx}]': {e}") from e

    if not isinstance(nested_obj, BaseModel):
        raise TypeError(f"Attr '{current_attr}' is not Pydantic model: {attr_path}")

    try:
        new_nested_obj = _set_nested_attr_pydantic(nested_obj, rest_of_path, value)
    except Exception as e:
        raise type(e)(f"Error setting nested '{rest_of_path}' within '{current_attr}': {e}") from e
    try:
        data = _model_fields_data(obj)
        data[current_attr] = new_nested_obj
        return type(obj).model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"Validation failed replacing nested '{current_attr}' on {type(obj).__name__}: {e}"
        ) from e
    except Exception as e:
        raise TypeError(f"Failed to replace nested '{current_attr}': {e}") from e


Case = Dict[str, Any]  # mapping full_path -> value
VerifierFn = Callable[[Case, Mapping[str, BaseModel], Mapping[str, Any] | None], bool]
Labels = Dict[str, Any]  # scalar-only values (enforced on leaf parse)


@dataclass(frozen=True)
class SweepPoint:
    case: Case
    labels: Labels


def _verifier_pydantic_valid(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """Verifier: returns True iff the case can be applied to the base_specs without validation error."""
    try:
        _apply_case_to_specs(case, base_specs)
        return True
    except Exception:
        return False


# ---- Generators (no numpy dependency; keep minimal and deterministic) ----


def _power_of_two(*, min_val: int, max_val: int) -> list[list[int]]:
    if min_val < 1 or max_val < min_val:
        raise SweepDSLError("power_of_two requires 1 <= min_val <= max_val")
    out: list[list[int]] = []
    v = 1
    while v < min_val:
        v *= 2
    while v <= max_val:
        out.append([v])
        v *= 2
    return out


def _perfect_square_factors(*, min_val: int, max_val: int) -> list[list[int]]:
    if min_val < 0 or max_val < min_val:
        raise SweepDSLError("perfect_square_factors requires 0 <= min_val <= max_val")
    out: list[list[int]] = []
    n = int(math.ceil(math.sqrt(min_val)))
    while True:
        sq = n * n
        if sq > max_val:
            break
        out.append([n, n])
        n += 1
    return out


def _decompose_with_balanced_multiplicands(
    *, min_val: int, max_val: int, step: int
) -> list[list[int]]:
    if step == 0:
        raise SweepDSLError("decompose_with_balanced_multiplicands requires step != 0")
    if (step > 0 and min_val > max_val) or (step < 0 and min_val < max_val):
        return []

    def best_pair(n: int) -> list[int]:
        if n <= 0:
            raise SweepDSLError("decompose_with_balanced_multiplicands requires positive integers")
        if n == 1:
            return [1, 1]
        root = int(math.sqrt(n))
        for a in range(root, 0, -1):
            if n % a == 0:
                return [a, n // a]
        raise SweepDSLError(f"no factors for {n}")

    out: list[list[int]] = []
    stop = max_val + (1 if step > 0 else -1)
    for n in range(min_val, stop, step):
        if n > 0:
            out.append(best_pair(n))
    return out


# Registry: generator name -> function(args)->list[list[Any]].
# Each emitted row must align with the leaf's `paths` length.
ALLOWED_SWEEP_GENERATORS: Dict[str, Callable[..., list[list[Any]]]] = {
    # 1D helpers
    "power_of_two": _power_of_two,
    # 2D split helpers
    "perfect_square_factors": _perfect_square_factors,
    "decompose_with_balanced_multiplicands": _decompose_with_balanced_multiplicands,
}


ALLOWED_SWEEP_VERIFIERS: Dict[str, VerifierFn] = {
    # Minimal default verifier; users can extend this registry later.
    "pydantic_valid": _verifier_pydantic_valid,
}


def _verifier_tp1d_attn_tp_leq_num_heads(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """Verifier: for TP1D attention, ensure the TP group does not exceed num_heads.

    Rationale: in 1D weight-stationary attention, sharding is fundamentally along attention heads.
    You cannot shard more ways than there are (query) heads without introducing duplication.
    """
    try:
        specs = _apply_case_to_specs(case, base_specs)
    except Exception:
        return False
    ws = specs.get("workload_spec")
    topo = specs.get("logical_topology")
    ms = specs.get("model_spec")
    if ws is None or topo is None or ms is None:
        return True

    # Only apply when attn variant is 1D.
    tp_attn_spec = getattr(ws, "tp_attn_spec", None)
    variant = getattr(tp_attn_spec, "variant", None)
    variant_val = getattr(variant, "name", None) or getattr(variant, "value", None) or str(variant)
    if "SHARD_WEIGHTS_1D" not in variant_val:
        return True

    try:
        tp_x = topo.get_mapping_for_tag("TP_Attn_X").size
        tp_y = topo.get_mapping_for_tag("TP_Attn_Y").size
    except Exception:
        return True

    tp = int(tp_x) * int(tp_y)
    num_heads = int(getattr(ms, "num_heads", 0) or 0)
    if num_heads <= 0:
        return True
    return tp <= num_heads


def _verifier_projection_groups_divides_proj_yz(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """
    Verifier: for TP2D decode sweeps, ensure `workload_spec.projection_groups` divides the
    resolved TP2D projection YZ size (product of split_2d.yz_tags sizes on the logical topology).

    If the case does not describe TP2D decode with projection split tags, this returns True.
    """
    try:
        specs = _apply_case_to_specs(case, base_specs)
    except Exception:
        return False
    ws = specs.get("workload_spec")
    topo = specs.get("logical_topology")
    if ws is None or topo is None:
        return True

    g = int(getattr(ws, "projection_groups", 1) or 1)
    if g <= 1:
        return True

    tp_proj_spec = getattr(ws, "tp_proj_spec", None)
    split_2d = getattr(tp_proj_spec, "split_2d", None) if tp_proj_spec is not None else None
    yz_tags = list(getattr(split_2d, "yz_tags", []) or [])
    if not yz_tags:
        return True

    tp_yz = 1
    for t in yz_tags:
        tag = t.value if hasattr(t, "value") else str(t)
        try:
            tp_yz *= topo.get_mapping_for_tag(tag).size
        except Exception:
            return False

    if tp_yz <= 0:
        return False
    return (tp_yz % g) == 0


def _verifier_decode_sharded_router_proj_y_eff_matches_moe_y(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """
    Verifier (decode + sharded_router):
    enforce projection-group effective Y (`Y_eff = TP_Proj_YZ / g`) to match MoE Y-shard size.

    Scope/intent:
    - Only active for decode workload specs with `decode_moe_router_sync_mode=sharded_router`.
    - Only enforced for 1xY MoE shapes (`TP_MoE_X == 1`) where expert input availability
      depends directly on Y-group compatibility.
    - If not applicable (non-decode, replicated router, missing 2D splits), returns True.
    """
    try:
        specs = _apply_case_to_specs(case, base_specs)
    except Exception:
        return False

    ws = specs.get("workload_spec")
    topo = specs.get("logical_topology")
    if ws is None or topo is None:
        return True

    # Decode-only guard: if field is absent, this is not an inference-decode case.
    if not hasattr(ws, "decode_moe_router_sync_mode"):
        return True

    router_mode = getattr(ws, "decode_moe_router_sync_mode", None)
    router_mode_val = (
        getattr(router_mode, "value", None)
        or getattr(router_mode, "name", None)
        or str(router_mode)
    ).lower()
    if "sharded_router" not in router_mode_val:
        return True

    # Resolve projection YZ size and g.
    tp_proj_spec = getattr(ws, "tp_proj_spec", None)
    proj_split = getattr(tp_proj_spec, "split_2d", None) if tp_proj_spec is not None else None
    proj_yz_tags = list(getattr(proj_split, "yz_tags", []) or [])
    if not proj_yz_tags:
        return True

    g = int(getattr(ws, "projection_groups", 1) or 1)
    if g <= 0:
        return False

    proj_yz = 1
    for t in proj_yz_tags:
        tag = t.value if hasattr(t, "value") else str(t)
        try:
            proj_yz *= int(topo.get_mapping_for_tag(tag).size)
        except Exception:
            return False
    if proj_yz <= 0 or (proj_yz % g) != 0:
        return False
    y_eff = proj_yz // g

    # Resolve MoE 2D X/YZ sizes.
    tp_moe_spec = getattr(ws, "tp_moe_spec", None)
    moe_split = getattr(tp_moe_spec, "split_2d", None) if tp_moe_spec is not None else None
    moe_x_tags = list(getattr(moe_split, "x_tags", []) or [])
    moe_yz_tags = list(getattr(moe_split, "yz_tags", []) or [])
    if not moe_x_tags or not moe_yz_tags:
        return True

    moe_x = 1
    for t in moe_x_tags:
        tag = t.value if hasattr(t, "value") else str(t)
        try:
            moe_x *= int(topo.get_mapping_for_tag(tag).size)
        except Exception:
            return False

    # Enforce only for 1xY cases.
    if moe_x != 1:
        return True

    moe_y = 1
    for t in moe_yz_tags:
        tag = t.value if hasattr(t, "value") else str(t)
        try:
            moe_y *= int(topo.get_mapping_for_tag(tag).size)
        except Exception:
            return False
    if moe_y <= 0:
        return False

    return y_eff == moe_y


def _verifier_decode_sharded_router_proj_y_eff_matches_moe_y_all_shapes(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """
    Verifier (decode + sharded_router):
    enforce projection-group effective Y (`Y_eff = TP_Proj_YZ / g`) to match MoE Y-shard size.

    Scope/intent:
    - Only active for decode workload specs with `decode_moe_router_sync_mode=sharded_router`.
    - Enforced for all MoE TP shapes in sharded-router decode path.
    - If not applicable (non-decode, replicated router, missing 2D splits), returns True.
    """
    try:
        specs = _apply_case_to_specs(case, base_specs)
    except Exception:
        return False

    ws = specs.get("workload_spec")
    topo = specs.get("logical_topology")
    if ws is None or topo is None:
        return True

    # Decode-only guard: if field is absent, this is not an inference-decode case.
    if not hasattr(ws, "decode_moe_router_sync_mode"):
        return True

    router_mode = getattr(ws, "decode_moe_router_sync_mode", None)
    router_mode_val = (
        getattr(router_mode, "value", None)
        or getattr(router_mode, "name", None)
        or str(router_mode)
    ).lower()
    if "sharded_router" not in router_mode_val:
        return True

    # Resolve projection YZ size and g.
    tp_proj_spec = getattr(ws, "tp_proj_spec", None)
    proj_split = getattr(tp_proj_spec, "split_2d", None) if tp_proj_spec is not None else None
    proj_yz_tags = list(getattr(proj_split, "yz_tags", []) or [])
    if not proj_yz_tags:
        return True

    g = int(getattr(ws, "projection_groups", 1) or 1)
    if g <= 0:
        return False

    proj_yz = 1
    for t in proj_yz_tags:
        tag = t.value if hasattr(t, "value") else str(t)
        try:
            proj_yz *= int(topo.get_mapping_for_tag(tag).size)
        except Exception:
            return False
    if proj_yz <= 0 or (proj_yz % g) != 0:
        return False
    y_eff = proj_yz // g

    # Resolve MoE 2D YZ size.
    tp_moe_spec = getattr(ws, "tp_moe_spec", None)
    moe_split = getattr(tp_moe_spec, "split_2d", None) if tp_moe_spec is not None else None
    moe_yz_tags = list(getattr(moe_split, "yz_tags", []) or [])
    if not moe_yz_tags:
        return True

    moe_y = 1
    for t in moe_yz_tags:
        tag = t.value if hasattr(t, "value") else str(t)
        try:
            moe_y *= int(topo.get_mapping_for_tag(tag).size)
        except Exception:
            return False
    if moe_y <= 0:
        return False

    return y_eff == moe_y


def _verifier_decode_fc_flexible_tp_moe_compat(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """
    Verifier (decode, FC-independent-rack style):
    - Attention TP budget is rack-local: TP_Attn_X * TP_Attn_Y == EP_SU * EP_SO
    - MoE TP must be axis-compatible and no larger than attention axes
    - Divisibility required for clean handoff/group mapping

    This is intentionally flexible (does not require attn == proj).
    """
    try:
        specs = _apply_case_to_specs(case, base_specs)
    except Exception:
        return False

    ws = specs.get("workload_spec")
    topo = specs.get("logical_topology")
    if ws is None or topo is None:
        return True

    # Decode-only guard.
    if not hasattr(ws, "decode_moe_router_sync_mode"):
        return True

    try:
        attn_x_map = topo.get_mapping_for_tag("TP_Attn_X")
        attn_y_map = topo.get_mapping_for_tag("TP_Attn_Y")
        moe_x_map = topo.get_mapping_for_tag("TP_MoE_X")
        moe_y_map = topo.get_mapping_for_tag("TP_MoE_Y")
    except Exception:
        return False

    attn_x = int(attn_x_map.size)
    attn_y = int(attn_y_map.size)
    moe_x = int(moe_x_map.size)
    moe_y = int(moe_y_map.size)
    if min(attn_x, attn_y, moe_x, moe_y) <= 0:
        return False

    # Rack-local attention budget is explicitly configured via verifier params.
    if not params or "attn_budget" not in params:
        return False
    attn_budget = params.get("attn_budget")
    if not isinstance(attn_budget, int) or attn_budget <= 0:
        return False
    expected_attn_budget = attn_budget
    if attn_x * attn_y != expected_attn_budget:
        return False

    # Keep MoE on corresponding physical axes.
    if moe_x_map.maps_to_physical_dimension_name != attn_x_map.maps_to_physical_dimension_name:
        return False
    if moe_y_map.maps_to_physical_dimension_name != attn_y_map.maps_to_physical_dimension_name:
        return False

    if moe_x > attn_x or moe_y > attn_y:
        return False
    if (attn_x % moe_x) != 0 or (attn_y % moe_y) != 0:
        return False

    return True


def _verifier_decode_nx1_router_compat(
    case: Case, base_specs: Mapping[str, BaseModel], params: Mapping[str, Any] | None = None
) -> bool:
    """
    Verifier (decode, Nx1 MoE TP):
    - If TP_MoE is Nx1 (TP_MoE_X > 1 and TP_MoE_Y == 1), allow:
      * replicated_router (preferred), or
      * sharded_router only when Y_eff == 1.
    - Non-Nx1 cases pass through unchanged.
    """
    try:
        specs = _apply_case_to_specs(case, base_specs)
    except Exception:
        return False

    ws = specs.get("workload_spec")
    topo = specs.get("logical_topology")
    if ws is None or topo is None:
        return True

    # Decode-only guard.
    if not hasattr(ws, "decode_moe_router_sync_mode"):
        return True

    try:
        moe_x = int(topo.get_mapping_for_tag("TP_MoE_X").size)
        moe_y = int(topo.get_mapping_for_tag("TP_MoE_Y").size)
        attn_y = int(topo.get_mapping_for_tag("TP_Attn_Y").size)
    except Exception:
        return False

    if min(moe_x, moe_y, attn_y) <= 0:
        return False

    # Only enforce for Nx1 shapes.
    if not (moe_x > 1 and moe_y == 1):
        return True

    router_mode = getattr(ws, "decode_moe_router_sync_mode", None)
    router_mode_val = (
        getattr(router_mode, "value", None)
        or getattr(router_mode, "name", None)
        or str(router_mode)
    ).lower()

    if "replicated_router" in router_mode_val:
        return True
    if "sharded_router" not in router_mode_val:
        return False

    if not hasattr(ws, "projection_groups"):
        return False
    g = getattr(ws, "projection_groups")
    if not isinstance(g, int) or g <= 0:
        return False
    if (attn_y % g) != 0:
        return False

    y_eff = attn_y // g
    return y_eff == 1


# Extend allowlist
ALLOWED_SWEEP_VERIFIERS["projection_groups_divides_proj_yz"] = (
    _verifier_projection_groups_divides_proj_yz
)
ALLOWED_SWEEP_VERIFIERS["decode_sharded_router_proj_y_eff_matches_moe_y"] = (
    _verifier_decode_sharded_router_proj_y_eff_matches_moe_y
)
ALLOWED_SWEEP_VERIFIERS["decode_sharded_router_proj_y_eff_matches_moe_y_all_shapes"] = (
    _verifier_decode_sharded_router_proj_y_eff_matches_moe_y_all_shapes
)
ALLOWED_SWEEP_VERIFIERS["decode_fc_flexible_tp_moe_compat"] = (
    _verifier_decode_fc_flexible_tp_moe_compat
)
ALLOWED_SWEEP_VERIFIERS["decode_nx1_router_compat"] = _verifier_decode_nx1_router_compat

ALLOWED_SWEEP_VERIFIERS["tp1d_attn_tp_leq_num_heads"] = _verifier_tp1d_attn_tp_leq_num_heads


class SweepDSLError(ValueError):
    pass


@dataclass(frozen=True)
class LeafSpec:
    name: str
    paths: list[str]
    cases: list[list[Any]] | None = None
    generator: dict[str, Any] | None = None
    labels: list[Labels] | None = None


@dataclass(frozen=True)
class WhereVerifierSpec:
    name: str
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class NodeSpec:
    kind: str  # "ref" | "cartesian" | "zip" | "union"
    ref: str | None = None
    children: list["NodeSpec"] | None = None
    strict: bool = True  # zip only (default strict)
    where: list[WhereVerifierSpec] | None = None
    on_fail: str = "skip"  # "skip" | "error"


@dataclass(frozen=True)
class ReportColumnSpec:
    name: str
    path: str


@dataclass(frozen=True)
class WherePredicateSpec:
    col: str
    op: str  # one of: == != < <= > >=
    value: Any


@dataclass(frozen=True)
class BestRuleSpec:
    name: str
    group_by: list[str]
    metric: str
    mode: str = "min"  # "min" | "max"
    where: list[WherePredicateSpec] | None = None


@dataclass(frozen=True)
class ReportSpec:
    columns: list[ReportColumnSpec]
    best: list[BestRuleSpec] | None = None


@dataclass(frozen=True)
class SweepConfig:
    leaves: dict[str, LeafSpec]
    sweep: NodeSpec
    report: ReportSpec | None = None

    @classmethod
    def load_from_yaml(cls, path: str) -> "SweepConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Empty YAML means "no sweep" (one default case).
        if data is None:
            return cls(
                leaves={
                    "__base": LeafSpec(
                        name="__base", paths=[], cases=[[]], generator=None, labels=[{}]
                    )
                },
                sweep=NodeSpec(kind="ref", ref="__base"),
                report=None,
            )
        if not isinstance(data, dict):
            raise SweepDSLError("Sweep config must be a mapping.")
        if not data:
            return cls(
                leaves={
                    "__base": LeafSpec(
                        name="__base", paths=[], cases=[[]], generator=None, labels=[{}]
                    )
                },
                sweep=NodeSpec(kind="ref", ref="__base"),
                report=None,
            )
        leaves_raw = data.get("leaves")
        if not isinstance(leaves_raw, dict) or not leaves_raw:
            raise SweepDSLError("Sweep config must include non-empty 'leaves' mapping.")
        leaves: dict[str, LeafSpec] = {}
        for name, spec in leaves_raw.items():
            if not isinstance(spec, dict):
                raise SweepDSLError(f"Leaf '{name}' must be a mapping.")
            paths = spec.get("paths")
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                raise SweepDSLError(f"Leaf '{name}': 'paths' must be a list[str].")
            cases = spec.get("cases")
            gen = spec.get("generator")
            labels = spec.get("labels")
            if (cases is None) == (gen is None):
                raise SweepDSLError(
                    f"Leaf '{name}' must set exactly one of 'cases' or 'generator'."
                )
            if cases is not None:
                if not isinstance(cases, list) or not cases:
                    raise SweepDSLError(f"Leaf '{name}': 'cases' must be a non-empty list.")
                if not all(isinstance(row, list) for row in cases):
                    raise SweepDSLError(f"Leaf '{name}': each case must be a list (aligned tuple).")
                if len(paths) == 0:
                    # Allow a single empty case as an identity leaf.
                    if not (len(cases) == 1 and cases[0] == []):
                        raise SweepDSLError(
                            f"Leaf '{name}': empty paths requires cases=[[]] (single identity case)."
                        )
                for row in cases:
                    if len(row) != len(paths):
                        raise SweepDSLError(
                            f"Leaf '{name}': case length {len(row)} != len(paths) {len(paths)}."
                        )

                if labels is not None:
                    if not isinstance(labels, list):
                        raise SweepDSLError(
                            f"Leaf '{name}': 'labels' must be a list when provided."
                        )
                    if len(labels) != len(cases):
                        raise SweepDSLError(
                            f"Leaf '{name}': len(labels)={len(labels)} must equal len(cases)={len(cases)}."
                        )
                    for i, lab in enumerate(labels):
                        if not isinstance(lab, dict):
                            raise SweepDSLError(
                                f"Leaf '{name}': labels[{i}] must be a mapping/dict."
                            )
                        for k, v in lab.items():
                            if not isinstance(k, str):
                                raise SweepDSLError(
                                    f"Leaf '{name}': label key must be str, got {type(k)}."
                                )
                            # scalar-only: str/int/float/bool/null
                            if isinstance(v, (list, dict)):
                                raise SweepDSLError(
                                    f"Leaf '{name}': label '{k}' must be a scalar (no lists/dicts)."
                                )
                    # Normalize: ensure every entry is a plain dict (not a custom mapping)
                    labels = [dict(l) for l in labels]  # noqa E741
                else:
                    labels = [{} for _ in cases]
            if gen is not None:
                if not isinstance(gen, dict) or "name" not in gen:
                    raise SweepDSLError(
                        f"Leaf '{name}': 'generator' must be mapping with key 'name'."
                    )
                if len(paths) == 0:
                    raise SweepDSLError(f"Leaf '{name}': generator not allowed with empty paths.")
                if labels is not None:
                    raise SweepDSLError(
                        f"Leaf '{name}': 'labels' is not supported with 'generator' (use explicit cases)."
                    )
            leaves[name] = LeafSpec(
                name=str(name), paths=list(paths), cases=cases, generator=gen, labels=labels
            )

        sweep_raw = data.get("sweep")
        if sweep_raw is None:
            raise SweepDSLError("Sweep config must include 'sweep' expression.")
        sweep = _parse_node(sweep_raw)

        report = _parse_report(data.get("report"))
        return cls(leaves=leaves, sweep=sweep, report=report)


def _parse_report(obj: Any) -> ReportSpec | None:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise SweepDSLError("report must be a mapping when provided.")
    cols_raw = obj.get("columns")
    if not isinstance(cols_raw, list) or not cols_raw:
        raise SweepDSLError("report.columns must be a non-empty list.")
    columns: list[ReportColumnSpec] = []
    seen_names: set[str] = set()
    for i, c in enumerate(cols_raw):
        if not isinstance(c, dict):
            raise SweepDSLError(
                f"report.columns[{i}] must be a mapping with keys 'name' and 'path'."
            )
        name = c.get("name")
        path = c.get("path")
        if not isinstance(name, str) or not name:
            raise SweepDSLError(f"report.columns[{i}].name must be a non-empty string.")
        if not isinstance(path, str) or not path:
            raise SweepDSLError(f"report.columns[{i}].path must be a non-empty string.")
        if name in seen_names:
            raise SweepDSLError(f"Duplicate report column name: {name!r}")
        seen_names.add(name)
        columns.append(ReportColumnSpec(name=name, path=path))

    best_raw = obj.get("best")
    if best_raw is None:
        return ReportSpec(columns=columns, best=None)
    if not isinstance(best_raw, list) or not best_raw:
        raise SweepDSLError("report.best must be a non-empty list when provided.")
    best: list[BestRuleSpec] = []
    for i, r in enumerate(best_raw):
        if not isinstance(r, dict):
            raise SweepDSLError(f"report.best[{i}] must be a mapping.")
        name = r.get("name")
        group_by = r.get("group_by")
        metric = r.get("metric")
        mode = r.get("mode", "min")
        where_raw = r.get("where")
        if not isinstance(name, str) or not name:
            raise SweepDSLError(f"report.best[{i}].name must be a non-empty string.")
        if (
            not isinstance(group_by, list)
            or not group_by
            or not all(isinstance(x, str) for x in group_by)
        ):
            raise SweepDSLError(f"report.best[{i}].group_by must be a non-empty list[str].")
        if not isinstance(metric, str) or not metric:
            raise SweepDSLError(f"report.best[{i}].metric must be a non-empty string.")
        if mode not in ("min", "max"):
            raise SweepDSLError(f"report.best[{i}].mode must be 'min' or 'max', got {mode!r}.")
        where: list[WherePredicateSpec] | None = None
        if where_raw is not None:
            if not isinstance(where_raw, list) or not all(isinstance(x, dict) for x in where_raw):
                raise SweepDSLError(f"report.best[{i}].where must be list[mapping] when provided.")
            where = []
            for j, p in enumerate(where_raw):
                col = p.get("col")
                op = p.get("op")
                value = p.get("value")
                if not isinstance(col, str) or not col:
                    raise SweepDSLError(
                        f"report.best[{i}].where[{j}].col must be a non-empty string."
                    )
                if op not in ("==", "!=", "<", "<=", ">", ">="):
                    raise SweepDSLError(
                        f"report.best[{i}].where[{j}].op must be one of == != < <= > >=, got {op!r}."
                    )
                where.append(WherePredicateSpec(col=col, op=op, value=value))
        best.append(
            BestRuleSpec(name=name, group_by=list(group_by), metric=metric, mode=mode, where=where)
        )
    return ReportSpec(columns=columns, best=best)


def _parse_node(obj: Any) -> NodeSpec:
    if isinstance(obj, dict):
        if "ref" in obj:
            if len(obj) != 1:
                raise SweepDSLError("ref node must only contain 'ref'.")
            r = obj["ref"]
            if not isinstance(r, str) or not r:
                raise SweepDSLError("ref must be a non-empty string.")
            return NodeSpec(kind="ref", ref=r)

        # operator nodes
        for kind in ("cartesian", "zip", "union"):
            if kind in obj:
                children_raw = obj[kind]
                if not isinstance(children_raw, list) or not children_raw:
                    raise SweepDSLError(f"{kind} node must contain non-empty list of children.")
                children = [_parse_node(x) for x in children_raw]
                strict = True
                if kind == "zip":
                    strict = bool(obj.get("strict", True))
                where_raw = obj.get("where")
                where: list[WhereVerifierSpec] | None = None
                if where_raw is not None:
                    if not isinstance(where_raw, list):
                        raise SweepDSLError(
                            f"{kind}.where must be list[str|mapping(name, params)]."
                        )
                    where = []
                    for i, entry in enumerate(where_raw):
                        if isinstance(entry, str):
                            if not entry:
                                raise SweepDSLError(
                                    f"{kind}.where[{i}] verifier name must be non-empty."
                                )
                            where.append(WhereVerifierSpec(name=entry, params=None))
                            continue
                        if not isinstance(entry, dict):
                            raise SweepDSLError(
                                f"{kind}.where[{i}] must be str or mapping with 'name'."
                            )
                        name = entry.get("name")
                        params = entry.get("params", None)
                        if not isinstance(name, str) or not name:
                            raise SweepDSLError(
                                f"{kind}.where[{i}].name must be a non-empty string."
                            )
                        if params is not None and not isinstance(params, dict):
                            raise SweepDSLError(f"{kind}.where[{i}].params must be a mapping.")
                        extra = set(entry.keys()) - {"name", "params"}
                        if extra:
                            raise SweepDSLError(
                                f"{kind}.where[{i}] has unknown keys: {sorted(extra)}"
                            )
                        where.append(WhereVerifierSpec(name=name, params=params))
                on_fail = obj.get("on_fail", "skip")
                if on_fail not in ("skip", "error"):
                    raise SweepDSLError(
                        f"{kind}.on_fail must be 'skip' or 'error', got {on_fail!r}."
                    )
                # Allow only these keys
                allowed = {kind, "where", "on_fail", "strict"}
                extra = set(obj.keys()) - allowed
                if extra:
                    raise SweepDSLError(f"{kind} node has unknown keys: {sorted(extra)}")
                return NodeSpec(
                    kind=kind, children=children, strict=strict, where=where, on_fail=on_fail
                )

    raise SweepDSLError(
        "Invalid sweep node; expected {ref: ...} or {cartesian|zip|union: [...]} mapping."
    )


def generate_cases(
    *, cfg: SweepConfig, base_specs: Mapping[str, BaseModel]
) -> Iterator[SweepPoint]:
    """Yield sweep points (case dict + merged labels) after applying verifier filters."""
    leaf_cache: dict[str, list[SweepPoint]] = {}

    def eval_leaf(name: str) -> list[SweepPoint]:
        if name in leaf_cache:
            return leaf_cache[name]
        if name not in cfg.leaves:
            raise SweepDSLError(f"Unknown leaf ref: {name!r}")
        spec = cfg.leaves[name]
        out: list[SweepPoint] = []
        if spec.cases is not None:
            labs = spec.labels if spec.labels is not None else [{} for _ in spec.cases]
            for row, lab in zip(spec.cases, labs):
                out.append(
                    SweepPoint(case={p: v for p, v in zip(spec.paths, row)}, labels=dict(lab))
                )
        else:
            gen = spec.generator or {}
            gen_name = gen.get("name")
            if gen_name not in ALLOWED_SWEEP_GENERATORS:
                raise SweepDSLError(
                    f"Leaf '{name}': unknown generator '{gen_name}'. Allowed: {sorted(ALLOWED_SWEEP_GENERATORS)}"
                )
            args = dict(gen.get("args") or {})
            values = ALLOWED_SWEEP_GENERATORS[gen_name](**args)
            if not isinstance(values, list) or not values:
                raise SweepDSLError(
                    f"Leaf '{name}': generator '{gen_name}' returned empty/non-list."
                )
            if not all(isinstance(row, list) for row in values):
                raise SweepDSLError(
                    f"Leaf '{name}': generator '{gen_name}' must return list[list]."
                )
            for row in values:
                if len(row) != len(spec.paths):
                    raise SweepDSLError(
                        f"Leaf '{name}': generator '{gen_name}' emitted row len {len(row)} != len(paths) {len(spec.paths)}."
                    )
                out.append(SweepPoint(case={p: v for p, v in zip(spec.paths, row)}, labels={}))
        leaf_cache[name] = out
        return out

    def verify_case(case: Case, *, where: list[WhereVerifierSpec] | None, on_fail: str) -> bool:
        if not where:
            return True
        for vspec in where:
            vname = vspec.name
            fn = ALLOWED_SWEEP_VERIFIERS.get(vname)
            if fn is None:
                raise SweepDSLError(
                    f"Unknown verifier '{vname}'. Allowed: {sorted(ALLOWED_SWEEP_VERIFIERS)}"
                )
            ok = bool(fn(case, base_specs, vspec.params))
            if not ok:
                if on_fail == "error":
                    raise SweepDSLError(f"Verifier '{vname}' failed for case: {case}")
                return False
        return True

    def merge_cases(a: Case, b: Case) -> Case:
        overlap = set(a).intersection(b)
        if overlap:
            raise SweepDSLError(
                f"Collision: paths assigned multiple times in one case: {sorted(overlap)}"
            )
        out = dict(a)
        out.update(b)
        return out

    def merge_labels(a: Labels, b: Labels) -> Labels:
        overlap = set(a).intersection(b)
        if overlap:
            raise SweepDSLError(
                f"Collision: label keys assigned multiple times in one sweep point: {sorted(overlap)}"
            )
        out = dict(a)
        out.update(b)
        return out

    def eval_node(node: NodeSpec) -> list[SweepPoint]:
        if node.kind == "ref":
            assert node.ref is not None
            return eval_leaf(node.ref)

        assert node.children is not None and node.kind in ("cartesian", "zip", "union")
        if node.kind == "union":
            out: list[SweepPoint] = []
            for c in node.children:
                out.extend(eval_node(c))
            return [x for x in out if verify_case(x.case, where=node.where, on_fail=node.on_fail)]

        if node.kind == "zip":
            child_lists = [eval_node(c) for c in node.children]
            if not child_lists:
                return []
            lens = [len(x) for x in child_lists]
            if node.strict and len(set(lens)) != 1:
                raise SweepDSLError(f"zip(strict=true) requires equal lengths, got {lens}")
            n = min(lens) if lens else 0
            out: list[SweepPoint] = []
            for i in range(n):
                merged_case: Case = {}
                merged_lab: Labels = {}
                for lst in child_lists:
                    merged_case = merge_cases(merged_case, lst[i].case)
                    merged_lab = merge_labels(merged_lab, lst[i].labels)
                if verify_case(merged_case, where=node.where, on_fail=node.on_fail):
                    out.append(SweepPoint(case=merged_case, labels=merged_lab))
            return out

        # cartesian
        child_lists = [eval_node(c) for c in node.children]
        out: list[SweepPoint] = []
        for combo in product(*child_lists):
            merged_case: Case = {}
            merged_lab: Labels = {}
            for part in combo:
                merged_case = merge_cases(merged_case, part.case)
                merged_lab = merge_labels(merged_lab, part.labels)
            if verify_case(merged_case, where=node.where, on_fail=node.on_fail):
                out.append(SweepPoint(case=merged_case, labels=merged_lab))
        return out

    for pt in eval_node(cfg.sweep):
        yield pt


def _apply_case_to_specs(case: Case, base_specs: Mapping[str, BaseModel]) -> Dict[str, BaseModel]:
    """Apply a case dict to base_specs and return a new specs dict (does not mutate base specs)."""
    current: Dict[str, BaseModel] = {k: v.model_copy() for k, v in base_specs.items()}
    for full_path, value in case.items():
        if not isinstance(full_path, str) or "." not in full_path:
            raise SweepDSLError(f"Invalid assignment path: {full_path!r}")
        spec_key, attr_path = full_path.split(".", 1)
        if spec_key not in current:
            raise SweepDSLError(f"Unknown spec key in path '{full_path}': {spec_key!r}")

        obj = current[spec_key]
        if spec_key == "logical_topology" and attr_path.startswith("tag_size."):
            tag = attr_path.split(".", 1)[1]
            new_size = int(value)
            dims = list(getattr(obj, "dimensions", []))
            found = False
            new_dims = []
            for d in dims:
                d_tag = (
                    d.logical_tag.value if hasattr(d.logical_tag, "value") else str(d.logical_tag)
                )
                if d_tag == tag:
                    found = True
                    new_dims.append(d.model_copy(update={"size": new_size}))
                else:
                    new_dims.append(d)
            if not found:
                raise SweepDSLError(f"LogicalTopology has no dimension with tag '{tag}'")
            current[spec_key] = obj.model_copy(update={"dimensions": new_dims})
        else:
            current[spec_key] = _set_nested_attr_pydantic(obj, attr_path, value)
    return current


def apply_case_to_specs(case: Case, *, base_specs: Mapping[str, BaseModel]) -> Dict[str, BaseModel]:
    """Public wrapper around _apply_case_to_specs."""
    return _apply_case_to_specs(case, base_specs)
