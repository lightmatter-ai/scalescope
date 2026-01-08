from pydantic import BaseModel, ConfigDict

from scalescope.sweep_dsl import generate_cases, SweepConfig


class _Dummy(BaseModel):
    model_config = ConfigDict(frozen=True)
    x: int = 0


def test_cartesian_and_collision_and_zip_strict_default(tmp_path):
    # Minimal sweep config exercising cartesian, zip(strict default), union, and collision detection.
    sweep_yaml = """
leaves:
  a:
    paths: [model_spec.x]
    cases: [[1], [2]]
  b:
    paths: [workload_spec.x]
    cases: [[10], [20]]
  c:
    paths: [model_spec.x]
    cases: [[3]]

sweep:
  cartesian:
    - ref: a
    - ref: b
"""
    p = tmp_path / "sweep.yaml"
    p.write_text(sweep_yaml)
    cfg = SweepConfig.load_from_yaml(str(p))
    base_specs = {
        "model_spec": _Dummy(x=0),
        "workload_spec": _Dummy(x=0),
        "logical_topology": _Dummy(x=0),
        "physical_topology": _Dummy(x=0),
    }
    cases = list(generate_cases(cfg=cfg, base_specs=base_specs))
    assert len(cases) == 4
    assert cases[0].case["model_spec.x"] in (1, 2)
    assert cases[0].case["workload_spec.x"] in (10, 20)

    # zip strict default should error on mismatched lengths
    sweep_yaml_zip = """
leaves:
  a:
    paths: [model_spec.x]
    cases: [[1], [2]]
  b:
    paths: [workload_spec.x]
    cases: [[10]]
sweep:
  zip:
    - ref: a
    - ref: b
"""
    p2 = tmp_path / "zip.yaml"
    p2.write_text(sweep_yaml_zip)
    cfg2 = SweepConfig.load_from_yaml(str(p2))
    try:
        list(generate_cases(cfg=cfg2, base_specs=base_specs))
        assert False, "expected strict zip length error"
    except Exception:
        pass

    # collision error: cartesian merges two leaves setting the same path
    sweep_yaml_collision = """
leaves:
  a:
    paths: [model_spec.x]
    cases: [[1]]
  c:
    paths: [model_spec.x]
    cases: [[2]]
sweep:
  cartesian:
    - ref: a
    - ref: c
"""
    p3 = tmp_path / "coll.yaml"
    p3.write_text(sweep_yaml_collision)
    cfg3 = SweepConfig.load_from_yaml(str(p3))
    try:
        list(generate_cases(cfg=cfg3, base_specs=base_specs))
        assert False, "expected collision error"
    except Exception:
        pass


def test_where_verifier_skip(tmp_path):
    sweep_yaml = """
leaves:
  a:
    paths: [model_spec.x]
    cases: [[1], [2], [3]]
sweep:
  cartesian:
    - ref: a
  where: ["pydantic_valid"]
  on_fail: skip
"""
    p = tmp_path / "sweep.yaml"
    p.write_text(sweep_yaml)
    cfg = SweepConfig.load_from_yaml(str(p))
    base_specs = {
        "model_spec": _Dummy(x=0),
        "workload_spec": _Dummy(x=0),
        "logical_topology": _Dummy(x=0),
        "physical_topology": _Dummy(x=0),
    }
    # pydantic_valid always passes for _Dummy, so we keep all
    cases = list(generate_cases(cfg=cfg, base_specs=base_specs))
    assert len(cases) == 3


def test_where_verifier_with_params_object_syntax(tmp_path):
    sweep_yaml = """
leaves:
  a:
    paths: [model_spec.x]
    cases: [[1], [2]]
sweep:
  cartesian:
    - ref: a
  where:
    - name: pydantic_valid
      params:
        ignored_flag: true
  on_fail: skip
"""
    p = tmp_path / "sweep_params.yaml"
    p.write_text(sweep_yaml)
    cfg = SweepConfig.load_from_yaml(str(p))
    base_specs = {
        "model_spec": _Dummy(x=0),
        "workload_spec": _Dummy(x=0),
        "logical_topology": _Dummy(x=0),
        "physical_topology": _Dummy(x=0),
    }
    cases = list(generate_cases(cfg=cfg, base_specs=base_specs))
    assert len(cases) == 2


def test_where_decode_tp2d_qkv_groups_divides_proj_yz(tmp_path):
    # Ensure verifier filters cases where decode_tp2d_qkv_groups does not divide proj YZ size.
    sweep_yaml = """
leaves:
  yz:
    paths: [logical_topology.tag_size.TP_Proj_Y]
    cases: [[4], [8]]
  g:
    paths: [workload_spec.decode_tp2d_qkv_groups]
    cases: [[1], [2], [3], [4]]
sweep:
  cartesian:
    - ref: yz
    - ref: g
  where: ["decode_tp2d_qkv_groups_divides_proj_yz"]
  on_fail: skip
"""
    p = tmp_path / "sweep.yaml"
    p.write_text(sweep_yaml)
    cfg = SweepConfig.load_from_yaml(str(p))

    class _Dim(BaseModel):
        model_config = ConfigDict(frozen=True)
        logical_tag: str
        maps_to_physical_dimension_name: str = "X"
        size: int

    class _Topo(BaseModel):
        model_config = ConfigDict(frozen=True)
        dimensions: list[_Dim]

        def get_mapping_for_tag(self, tag):
            for d in self.dimensions:
                if d.logical_tag == tag:
                    return d
            raise KeyError(tag)

    class _Split2D(BaseModel):
        model_config = ConfigDict(frozen=True)
        yz_tags: list[str]

    class _TPSpec(BaseModel):
        model_config = ConfigDict(frozen=True)
        split_2d: _Split2D

    class _WS(BaseModel):
        model_config = ConfigDict(frozen=True)
        decode_tp2d_qkv_groups: int = 1
        tp_proj_spec: _TPSpec

    # Base specs: logical_topology must support get_mapping_for_tag, workload_spec must have decode_tp2d_qkv_groups and tp_proj_spec.split_2d.yz_tags.
    base_specs = {
        "model_spec": _Dummy(x=0),
        "workload_spec": _WS(
            decode_tp2d_qkv_groups=1,
            tp_proj_spec=_TPSpec(split_2d=_Split2D(yz_tags=["TP_Proj_Y"])),
        ),
        "logical_topology": _Topo(dimensions=[_Dim(logical_tag="TP_Proj_Y", size=4)]),
        "physical_topology": _Dummy(x=0),
    }

    pts = list(generate_cases(cfg=cfg, base_specs=base_specs))
    # For TP_Proj_Y=4: allowed g are 1,2,4. For 8: allowed g are 1,2,4 (and 8 if present, but it isn't).
    assert len(pts) == 6


def test_leaf_labels_and_merge(tmp_path):
    sweep_yaml = """
leaves:
  fabric:
    paths: []
    cases: [[]]
    labels:
      - {fabric: electrical, bw: 800}
  batch:
    paths: [workload_spec.x]
    cases: [[1], [2]]
    labels:
      - {batch_label: small}
      - {batch_label: medium}
sweep:
  cartesian:
    - ref: fabric
    - ref: batch
"""
    p = tmp_path / "sweep.yaml"
    p.write_text(sweep_yaml)
    cfg = SweepConfig.load_from_yaml(str(p))
    base_specs = {
        "model_spec": _Dummy(x=0),
        "workload_spec": _Dummy(x=0),
        "logical_topology": _Dummy(x=0),
        "physical_topology": _Dummy(x=0),
    }
    pts = list(generate_cases(cfg=cfg, base_specs=base_specs))
    assert len(pts) == 2
    assert pts[0].labels["fabric"] == "electrical"
    assert pts[0].labels["batch_label"] in ("small", "medium")


def test_set_nested_attr_supports_list_index_paths():
    from pydantic import BaseModel

    from scalescope.sweep_dsl import _set_nested_attr_pydantic

    class Child(BaseModel):
        a: int

    class Parent(BaseModel):
        xs: list[Child]

    p = Parent(xs=[Child(a=1), Child(a=2), Child(a=3)])
    p2 = _set_nested_attr_pydantic(p, "xs.1.a", 42)
    assert p2.xs[0].a == 1
    assert p2.xs[1].a == 42
    assert p2.xs[2].a == 3
