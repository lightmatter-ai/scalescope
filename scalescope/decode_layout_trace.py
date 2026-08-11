"""Tensor layout tracing for the decode pipeline.

Records tensor partitioning at each stage boundary and formats
a verification table.  Does NOT infer communication — that is a
separate step.  This module is for shape correctness checking.

Usage
-----
    trace = build_decode_2dtp_trace(
        d_model=12288, d_ff=4096, n_heads=96, n_kv_heads=8, head_dim=128,
        n_experts=128, top_k=2, tp_x=32, tp_y=32, g=8, c=16,
        attn_domains_per_full_expert_set=1, full_expert_sets_per_attn_domain=1,
        attention_layers=attention_layer_classes(transformer_config),
        router_mode="sharded",
    )
    print(trace.format_table())
"""

import logging
import textwrap
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

from scalescope.model_spec import AttentionLayerClass
from scalescope.stats import CommStats

logger = logging.getLogger(__name__)

_COMM_STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "phase1_ag_x": "Attention input all-gather (X)",
    "phase1_ar_yz_q": "Q projection all-reduce (Y)",
    "phase1_ar_yz_qkv": "QKV projection all-reduce (Y)",
    "phase1_reduce_yz_kv": "KV reduce-to-cache-owner (Y)",
    "ln1_stats_ar_yz": "Pre-attention LayerNorm stats",
    "phase2_ar_y": "Decode attention output merge",
    "phase3_rs_x": "Output projection reduce-scatter (X)",
    "phase3_ag_yz": "Output projection all-gather (Y)",
    "moe_input_ag_proj_x": "MoE input all-gather (projection X)",
    "moe_input_ag_y": "MoE input Y-quotient all-gather",
    "moe_a1_ag_x": "MoE routed-input A1 all-gather (X)",
    "moe_1d_ar_tp": "MoE 1D tensor-parallel all-reduce",
    "moe_1d_ag_opt": "MoE 1D optimized input all-gather",
    "ln2_ag_yz": "Pre-MoE LayerNorm/router gather",
    "ln2_stats_ar_yz": "Pre-MoE LayerNorm stats",
    "moe_router_ar": "MoE router-logit all-reduce",
    "moe_c1_ar_yz": "MoE up-projection C1 all-reduce",
    "moe_c2_rs_x": "MoE down-projection C2 reduce-scatter",
    "combine_rs_exact_groups": "MoE-to-residual layout reduce-scatter",
    "combine_ar_projection_replicas": "MoE projection-replica synchronization",
    "combine_ag_cross_set": "MoE cross-set batch all-gather",
    "decode_dispatch_su": "MoE local-owner dispatch (already resident)",
    "decode_dispatch_so": "MoE dispatch across attention domains",
    "decode_combine_su": "MoE local-owner return (already resident)",
    "decode_combine_so": "MoE return across attention domains",
}

_LAYOUT_STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "layer_input": "Layer input",
    "phase1_x_gathered": "Attention input after X gather",
    "after_qkv_proj": "After QKV projection",
    "kv_cache_read": "KV-cache read",
    "attention_scores": "Attention scores",
    "attn_v_output": "Attention value output",
    "after_attention": "After attention",
    "after_output_proj": "After output projection",
    "moe_input": "MoE input in projection layout",
    "moe_input_reconciled": "MoE input after layout reconciliation",
    "moe_set_token_select": "MoE local expert-set token selection",
    "router_input": "MoE router input",
    "decode_dispatch": "MoE routed-token dispatch",
    "expert_input": "MoE expert input",
    "after_up_proj_c1": "After MoE up-projection C1",
    "after_down_proj_c2": "After MoE down-projection C2",
    "decode_combine_a2a": "MoE routed-output return",
    "after_combine": "After MoE expert combine",
    "after_combine_cross_set": "After cross-set MoE combine",
    "layer_output": "Layer output",
}


# ---------------------------------------------------------------------------
# Lightweight mesh / partition descriptors (intentionally decoupled from
# layout_model.py so the trace can be used without importing the full model).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshDim:
    name: str
    size: int


@dataclass(frozen=True)
class Mesh:
    dims: Tuple[MeshDim, ...]

    @property
    def dim_sizes(self) -> Dict[str, int]:
        return {d.name: d.size for d in self.dims}

    def size_of(self, name: str) -> int:
        for d in self.dims:
            if d.name == name:
                return d.size
        raise KeyError(f"Unknown mesh dim '{name}'. Known: {sorted(self.dim_sizes)}")


class P:
    """Partition descriptor for a single tensor axis."""

    __slots__ = ("kind", "mesh_dims")

    def __init__(self, kind: str, mesh_dims: Sequence[str] = ()):
        if kind not in ("sharded", "replicated", "partial"):
            raise ValueError(f"Unknown partition kind: {kind}")
        self.kind = kind
        self.mesh_dims = tuple(mesh_dims)

    def __repr__(self) -> str:
        if self.mesh_dims:
            return f"{self.kind}({','.join(self.mesh_dims)})"
        return self.kind

    def local_factor(self, mesh: Mesh) -> int:
        """Divisor for the global axis size to get the per-rank local size."""
        if self.kind == "sharded":
            f = 1
            for d in self.mesh_dims:
                f *= mesh.size_of(d)
            return f
        return 1  # replicated / partial → full size per rank


# ---------------------------------------------------------------------------
# Trace records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorRecord:
    """One tensor at one pipeline stage."""

    name: str
    global_shape: Dict[str, int]
    partitions: Dict[str, P]
    dtype_bytes: int = 2

    def per_rank_shape(self, mesh: Mesh) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for axis, size in self.global_shape.items():
            p = self.partitions.get(axis)
            if p is None:
                out[axis] = size
            else:
                f = p.local_factor(mesh)
                if size % f != 0:
                    raise ValueError(
                        f"{self.name}: axis {axis} (size {size}) not divisible "
                        f"by mesh factor {f} from {p}"
                    )
                out[axis] = size // f
        return out

    def per_rank_bytes(self, mesh: Mesh) -> int:
        shape = self.per_rank_shape(mesh)
        return reduce(mul, shape.values(), 1) * self.dtype_bytes


@dataclass(frozen=True)
class StageRecord:
    stage: str
    description: str
    tensor: TensorRecord
    note: str = ""
    flags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CommRecord:
    """One collective communication call."""

    stage: str
    collective: str  # AG, AR, RS, Reduce, AllReduce, TotalExchange
    topology: str  # e.g. "X(32)", "YZ(128)", "EP(128)"
    participants: int
    volume_bytes: float  # per-rank message size (input to Hockney)
    alpha_s: float  # startup latency (seconds, per layer)
    beta_s: float  # transmission time (seconds, per layer)
    total_s: float  # alpha + beta (seconds, per layer)
    phases: int = 1  # startup phase count
    note: str = ""


# ---------------------------------------------------------------------------
# The trace container
# ---------------------------------------------------------------------------


class NullLayoutTrace:
    """No-op trace that silently drops all ``record()`` calls.

    Use as a default so callers never need ``if trace is not None`` guards.
    """

    def record(self, *_args, **_kwargs) -> None:  # noqa: D401
        pass

    def record_comm(self, *_args, **_kwargs) -> None:  # noqa: D401
        pass


# Singleton — avoids creating a new object on every call.
NULL_LAYOUT_TRACE = NullLayoutTrace()


@dataclass
class LayoutTrace:
    mesh: Mesh
    records: List[StageRecord] = field(default_factory=list)
    comm_records: List[CommRecord] = field(default_factory=list)

    def record(
        self,
        stage: str,
        description: str,
        tensor: TensorRecord,
        *,
        note: str = "",
        flags: Sequence[str] = (),
    ) -> None:
        self.records.append(
            StageRecord(
                stage=stage,
                description=description,
                tensor=tensor,
                note=note,
                flags=tuple(flags),
            )
        )

    def record_comm(
        self,
        stage: str,
        collective: str,
        topology: str,
        participants: int,
        volume_bytes: float,
        stats: CommStats,
        *,
        note: str = "",
    ) -> None:
        """Record a single collective communication call."""
        self.comm_records.append(
            CommRecord(
                stage=stage,
                collective=collective,
                topology=topology,
                participants=participants,
                volume_bytes=volume_bytes,
                alpha_s=getattr(stats, "startup_latency", 0.0),
                beta_s=getattr(stats, "transmission_time", 0.0),
                total_s=(
                    getattr(stats, "startup_latency", 0.0)
                    + getattr(stats, "transmission_time", 0.0)
                ),
                phases=getattr(stats, "startup_phase_count", 1),
                note=note,
            )
        )

    # ---- query helpers ----

    def stages(self) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for r in self.records:
            if r.stage not in seen:
                seen.add(r.stage)
                out.append(r.stage)
        return out

    def tensors_at(self, stage: str) -> List[StageRecord]:
        return [r for r in self.records if r.stage == stage]

    def find(self, stage: str, tensor_name: str) -> StageRecord:
        for r in self.records:
            if r.stage == stage and r.tensor.name == tensor_name:
                return r
        raise KeyError(f"No tensor '{tensor_name}' at stage '{stage}'")

    # ---- shape verification ----

    def verify_layer_boundary(self) -> List[str]:
        """Check exact layer-input/output shape and partition identity."""
        issues: List[str] = []
        try:
            inp = self.find("layer_input", "x")
            out = self.find("layer_output", "x_out")
        except KeyError:
            return ["layer_input or layer_output not found in trace"]

        if inp.tensor.global_shape != out.tensor.global_shape:
            issues.append(
                f"MISMATCH: layer_input global={inp.tensor.global_shape} != "
                f"layer_output global={out.tensor.global_shape}"
            )

        for axis in sorted(set(inp.tensor.partitions) | set(out.tensor.partitions)):
            inp_partition = self._effective_partition(inp.tensor.partitions.get(axis), self.mesh)
            out_partition = self._effective_partition(out.tensor.partitions.get(axis), self.mesh)
            if inp_partition != out_partition:
                issues.append(
                    f"MISMATCH: axis '{axis}' layer_input={inp.tensor.partitions.get(axis)} != "
                    f"layer_output={out.tensor.partitions.get(axis)}"
                )

        inp_shape = inp.tensor.per_rank_shape(self.mesh)
        out_shape = out.tensor.per_rank_shape(self.mesh)
        if inp_shape != out_shape:
            issues.append(
                f"MISMATCH: layer_input per_rank={inp_shape} != layer_output per_rank={out_shape}"
            )
        return issues

    def assert_layer_boundary(self) -> None:
        """Fail when a trace does not close onto the next layer's input layout."""
        issues = self.verify_layer_boundary()
        if issues:
            raise ValueError("Decode layout trace does not close: " + "; ".join(issues))

    def collect_flags(self) -> List[Tuple[str, str, str]]:
        """Return all (stage, tensor, flag) triples for review."""
        return [(r.stage, r.tensor.name, f) for r in self.records for f in r.flags]

    # ---- comparison ----

    @staticmethod
    def _effective_partition(
        p: Optional["P"], mesh: "Mesh"
    ) -> Optional[Tuple[str, Tuple[str, ...]]]:
        """Return (kind, non-degenerate mesh_dims) or None.

        Mesh dimensions with size 1 are stripped from the partition descriptor
        because sharding along a size-1 axis is a no-op.  If all dims are
        degenerate the partition is treated as absent (None).
        """
        if p is None:
            return None
        effective_dims = tuple(d for d in p.mesh_dims if mesh.size_of(d) > 1)
        if p.kind == "sharded" and not effective_dims:
            return None
        return (p.kind, effective_dims)

    def compare_with(self, expected: "LayoutTrace") -> List[str]:
        """Compare this (actual) trace against an expected one.

        Focuses on partition structure (kind + mesh_dims) for each tensor axis.
        Mesh dimensions with size 1 are ignored during comparison so that
        ``sharded(X, Y_eff)`` with ``X=1`` matches ``sharded(Y_eff)``.
        Global shape is compared only for sharded axes so that batch-dimension
        mismatches (replicated/partial) don't produce false positives.
        """
        diffs: List[str] = [
            *(f"Actual trace boundary: {issue}" for issue in self.verify_layer_boundary()),
            *(f"Expected trace boundary: {issue}" for issue in expected.verify_layer_boundary()),
        ]

        exp_stages = expected.stages()
        act_stages = self.stages()

        for s in exp_stages:
            if s not in act_stages:
                diffs.append(f"Stage '{s}' in expected but missing from actual")
        for s in act_stages:
            if s not in exp_stages:
                diffs.append(f"Stage '{s}' in actual but not in expected")

        for stage in [s for s in exp_stages if s in act_stages]:
            exp_map = {r.tensor.name: r for r in expected.tensors_at(stage)}
            act_map = {r.tensor.name: r for r in self.tensors_at(stage)}

            for name in sorted(set(exp_map) | set(act_map)):
                if name not in act_map:
                    diffs.append(f"[{stage}/{name}] missing from actual")
                    continue
                if name not in exp_map:
                    diffs.append(f"[{stage}/{name}] in actual but not expected")
                    continue

                et = exp_map[name].tensor
                at = act_map[name].tensor

                all_axes = sorted(set(et.partitions) | set(at.partitions))
                for axis in all_axes:
                    ep_eff = self._effective_partition(et.partitions.get(axis), expected.mesh)
                    ap_eff = self._effective_partition(at.partitions.get(axis), self.mesh)
                    if ep_eff != ap_eff:
                        ep_raw = et.partitions.get(axis)
                        ap_raw = at.partitions.get(axis)
                        lhs = ap_raw if ap_raw is not None else "unpartitioned"
                        rhs = ep_raw if ep_raw is not None else "unpartitioned"
                        diffs.append(
                            f"[{stage}/{name}] axis '{axis}': actual={lhs} vs expected={rhs}"
                        )

                for axis in sorted(set(et.global_shape) | set(at.global_shape)):
                    ep = et.partitions.get(axis)
                    if ep is not None and ep.kind == "sharded":
                        es = et.global_shape.get(axis)
                        as_ = at.global_shape.get(axis)
                        if es != as_:
                            diffs.append(
                                f"[{stage}/{name}] axis '{axis}' global size: "
                                f"actual={as_} vs expected={es}"
                            )

        return diffs

    # ---- formatting ----

    # Display names: map internal tensor-axis labels to TP-oriented names.
    # X = head-parallel axis, Y = model-dim sharding axis.
    _AXIS_LABEL: ClassVar[Dict[str, str]] = {
        "B": "batch",
        "D": "d",
        "H": "h",
        "Dh": "dh",
        "Dff": "dff",
        "Dkv": "dkv",
        "Ctx": "ctx",
    }

    @classmethod
    def _display_axis(cls, ax: str) -> str:
        return cls._AXIS_LABEL.get(ax, ax)

    @classmethod
    def _fmt_shape(cls, shape: Dict[str, int]) -> str:
        return " ".join(f"{cls._display_axis(k)}={v}" for k, v in shape.items())

    @classmethod
    def _fmt_parts(cls, parts: Dict[str, "P"]) -> str:
        if not parts:
            return "—"
        abbrev = {"sharded": "shard", "replicated": "rep", "partial": "part"}
        frags: List[str] = []
        for ax, p in parts.items():
            dax = cls._display_axis(ax)
            kind = abbrev.get(p.kind, p.kind)
            if p.mesh_dims:
                frags.append(f"{dax}:{kind}({','.join(p.mesh_dims)})")
            else:
                frags.append(f"{dax}:{kind}")
        return "  ".join(frags)

    def format_table(self, *, compact: bool = False) -> str:
        lines: List[str] = []
        lines.append(f"Mesh: {self.mesh.dim_sizes}")
        lines.append(
            "Axes: batch=batch_size  d=d_model  h=n_heads(÷X)  "
            "dh=d_head  dff=d_ff  dkv=kv_dim  ctx=context_len"
        )
        lines.append("")

        col_w = (32, 26, 28, 36, 28, 72)
        column_separator = "  "
        hdr = column_separator.join(
            (
                f"{'Stage':<{col_w[0]}}",
                f"{'Tensor':<{col_w[1]}}",
                f"{'Global':<{col_w[2]}}",
                f"{'Sharding':<{col_w[3]}}",
                f"{'Per-Rank':<{col_w[4]}}",
                f"{'Note':<{col_w[5]}}",
            )
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))

        # Runtime records follow producer call order, not necessarily pipeline
        # order: TP records expert stages before comm_model later records the
        # cross-domain EP dispatch/return. Sort only for presentation so the
        # table reads router -> dispatch -> expert -> return -> combine.
        # Python's stable sort preserves insertion order within each stage.
        stage_order = {stage: index for index, stage in enumerate(_LAYOUT_STAGE_DISPLAY_NAMES)}
        prev_stage = ""
        for rec in sorted(
            self.records, key=lambda item: stage_order.get(item.stage, len(stage_order))
        ):
            if rec.stage != prev_stage:
                if rec.stage not in _LAYOUT_STAGE_DISPLAY_NAMES:
                    raise ValueError(f"Missing layout-table display name for stage: {rec.stage}")
                stage_label = _LAYOUT_STAGE_DISPLAY_NAMES[rec.stage]
            else:
                stage_label = ""
            prev_stage = rec.stage

            gs = self._fmt_shape(rec.tensor.global_shape)
            parts = self._fmt_parts(rec.tensor.partitions)
            try:
                pr = self._fmt_shape(rec.tensor.per_rank_shape(self.mesh))
            except (ValueError, KeyError) as e:
                pr = f"ERR: {e}"

            note = rec.note
            if rec.flags:
                note = "[FLAG] " + note

            cell_lines = (
                self._wrap_table_cell(stage_label, col_w[0]),
                self._wrap_table_cell(rec.tensor.name, col_w[1]),
                self._wrap_table_cell(gs, col_w[2]),
                self._wrap_table_cell(parts, col_w[3]),
                self._wrap_table_cell(pr, col_w[4]),
                self._wrap_table_cell(note, col_w[5]),
            )
            row_count = max(len(cell) for cell in cell_lines)
            for row_index in range(row_count):
                row_cells = tuple(
                    cell[row_index] if row_index < len(cell) else "" for cell in cell_lines
                )
                lines.append(
                    column_separator.join(
                        f"{row_cells[column_index]:<{col_w[column_index]}}"
                        for column_index in range(len(col_w))
                    )
                )

        issues = self.verify_layer_boundary()
        flags = self.collect_flags()
        if issues or flags:
            lines.append("")
            lines.append("=== Verification ===")
            for issue in issues:
                lines.append(f"  ISSUE: {issue}")
            for stage, tensor, flag in flags:
                lines.append(f"  FLAG [{stage}/{tensor}]: {flag}")

        if self.comm_records:
            lines.append("")
            lines.append(self.format_comm_table())

        return "\n".join(lines)

    # ---- collective communication table ----

    @staticmethod
    def _fmt_bytes(b: float) -> str:
        if b == 0:
            return "0"
        if b < 1024:
            return f"{b:.0f} B"
        if b < 1024**2:
            return f"{b / 1024:.2f} KiB"
        if b < 1024**3:
            return f"{b / 1024**2:.2f} MiB"
        return f"{b / 1024**3:.2f} GiB"

    @staticmethod
    def _fmt_us(s: float) -> str:
        if s == 0:
            return "0"
        return f"{s * 1e6:.2f} us"

    @staticmethod
    def _wrap_table_cell(value: str, width: int) -> List[str]:
        """Wrap a table cell, preserving explicit newline boundaries."""
        wrapped: List[str] = []
        logical_lines = value.splitlines()
        if not logical_lines:
            return [""]
        for logical_line in logical_lines:
            wrapped.extend(
                textwrap.wrap(
                    logical_line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return wrapped

    def format_comm_table(self) -> str:
        if not self.comm_records:
            return ""
        lines: List[str] = []
        lines.append("=== Collective Communications (per layer) ===")
        cw = (32, 8, 14, 6, 14, 12, 12, 12, 72)
        column_separator = "  "
        hdr = column_separator.join(
            (
                f"{'Stage':<{cw[0]}}",
                f"{'Coll':<{cw[1]}}",
                f"{'Topology':<{cw[2]}}",
                f"{'P':<{cw[3]}}",
                f"{'Volume/rank':<{cw[4]}}",
                f"{'Alpha':<{cw[5]}}",
                f"{'Beta':<{cw[6]}}",
                f"{'Total':<{cw[7]}}",
                f"{'Note':<{cw[8]}}",
            )
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))

        total_alpha = 0.0
        total_beta = 0.0
        total_time = 0.0
        nonzero_count = 0
        for cr in self.comm_records:
            total_alpha += cr.alpha_s
            total_beta += cr.beta_s
            total_time += cr.total_s
            is_zero = cr.total_s == 0
            if not is_zero:
                nonzero_count += 1
            tag = "  (noop)" if is_zero else ""
            if cr.stage not in _COMM_STAGE_DISPLAY_NAMES:
                raise ValueError(f"Missing communication-table display name for stage: {cr.stage}")
            display_stage = _COMM_STAGE_DISPLAY_NAMES[cr.stage]
            stage_lines = self._wrap_table_cell(display_stage, cw[0])
            note_lines = self._wrap_table_cell(f"{cr.note}{tag}", cw[8])
            row_count = max(len(stage_lines), len(note_lines))
            for row_index in range(row_count):
                stage_text = stage_lines[row_index] if row_index < len(stage_lines) else ""
                note_text = note_lines[row_index] if row_index < len(note_lines) else ""
                first_row = row_index == 0
                lines.append(
                    column_separator.join(
                        (
                            f"{stage_text:<{cw[0]}}",
                            f"{cr.collective if first_row else '':<{cw[1]}}",
                            f"{cr.topology if first_row else '':<{cw[2]}}",
                            f"{cr.participants if first_row else '':<{cw[3]}}",
                            f"{self._fmt_bytes(cr.volume_bytes) if first_row else '':<{cw[4]}}",
                            f"{self._fmt_us(cr.alpha_s) if first_row else '':<{cw[5]}}",
                            f"{self._fmt_us(cr.beta_s) if first_row else '':<{cw[6]}}",
                            f"{self._fmt_us(cr.total_s) if first_row else '':<{cw[7]}}",
                            f"{note_text:<{cw[8]}}",
                        )
                    )
                )

        lines.append("-" * len(hdr))
        label = f"TOTAL ({nonzero_count} active / {len(self.comm_records)} collectives)"
        summary = column_separator.join(
            (
                f"{label:<{cw[0]}}",
                f"{'':<{cw[1]}}",
                f"{'':<{cw[2]}}",
                f"{'':<{cw[3]}}",
                f"{'':<{cw[4]}}",
                f"{self._fmt_us(total_alpha):<{cw[5]}}",
                f"{self._fmt_us(total_beta):<{cw[6]}}",
                f"{self._fmt_us(total_time):<{cw[7]}}",
            )
        )
        lines.append(summary)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decode-specific mesh builder
# ---------------------------------------------------------------------------


def build_decode_2dtp_mesh(
    tp_x: int,
    tp_y: int,
    g: int,
    n_experts: int,
    c: int,
    moe_tp_x: int,
    moe_tp_yz: int,
    full_expert_sets_per_attn_domain: int,
) -> Mesh:
    y_eff = tp_y // g
    G = -(-n_experts // c)
    if g % full_expert_sets_per_attn_domain != 0:
        raise ValueError(
            f"projection groups ({g}) must be divisible by expert sets "
            f"({full_expert_sets_per_attn_domain})"
        )
    return Mesh(
        dims=(
            MeshDim("X", tp_x),
            MeshDim("Y", tp_y),
            MeshDim("Y_eff", y_eff),
            MeshDim("G", G),
            MeshDim("MoE_X", moe_tp_x),
            MeshDim("MoE_Y", moe_tp_yz),
            MeshDim("U", g),
            MeshDim("Set_Y", full_expert_sets_per_attn_domain),
            MeshDim("U_set", g // full_expert_sets_per_attn_domain),
        )
    )


# ---------------------------------------------------------------------------
# Decode pipeline stage builders
# ---------------------------------------------------------------------------


def _trace_attention_stages(
    trace: LayoutTrace,
    *,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    tp_x: int,
    tp_y: int,
    g: int,
    s: int,
    dtype_bytes: int,
    attention_layers: list[AttentionLayerClass],
    context_len: int = 1,
) -> None:
    # `d_head` is the per-head attention dimension straight from the model spec
    # (NOT d_model // n_heads — those are unequal under wide-Q layouts such as
    # MiniMax-M2.5, gpt-oss-120b, qwen3, where num_heads * head_dim != d_model).
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")
    y_eff = tp_y // g
    d_head = head_dim

    # Stage 0: layer_input — residual + LayerNorm output
    trace.record(
        "layer_input",
        "Residual + LayerNorm (start of layer)",
        TensorRecord(
            name="x",
            global_shape={"B": s, "D": d_model},
            partitions={
                "D": P("sharded", ("X", "Y_eff")),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=f"per_rank D = d/(X·Y_eff) = {d_model}/({tp_x}·{y_eff}) = {d_model // (tp_x * y_eff)}",
    )

    # Stage 1: after Phase 1 step 1 — X all-gather
    trace.record(
        "phase1_x_gathered",
        "After AG(X): full D/Y_eff slice per rank",
        TensorRecord(
            name="x_proj_in",
            global_shape={"B": s, "D": d_model},
            partitions={
                "D": P("sharded", ("Y_eff",)),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=f"comm: AG(X). D/Y_eff = {d_model // y_eff}",
    )

    # Stage 2: Q after projection + Y_eff all-reduce
    trace.record(
        "after_qkv_proj",
        "Q after proj + AR(Y_eff)",
        TensorRecord(
            name="Q",
            global_shape={"B": s, "H": n_heads, "Dh": d_head},
            partitions={
                "H": P("sharded", ("X",)),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=f"comm: AR(Y_eff). Q Y-replicated. heads/X = {n_heads // tp_x}",
    )

    # K, V after projection — reduce-to-owner on Y_eff.
    # For decode (s=1), the single token's K,V go to one Y-rank (the cache owner).
    # With GQA (n_kv_heads < tp_x), multiple X-ranks share one KV head — we model
    # the per-rank KV size as a flat dimension to avoid fractional head sharding.
    kv_heads_per_x = max(1, n_kv_heads // tp_x)
    kv_dim = kv_heads_per_x * d_head
    trace.record(
        "after_qkv_proj",
        "K,V after proj → reduce-to-owner(Y_eff) into KV-cache",
        TensorRecord(
            name="KV",
            global_shape={"B": s, "Dkv": kv_dim},
            partitions={},
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"comm: reduce-to-owner(Y_eff). per_rank Dkv = kv_heads_per_x × d_head "
            f"= {kv_heads_per_x}×{d_head} = {kv_dim}. "
            f"GQA: {n_kv_heads} KV-heads across X={tp_x} → "
            f"{kv_heads_per_x} per X-rank. "
            "KV-cache is Y-distributed."
        ),
    )

    # ---- Attention core stages (rank-local, no communication) ----

    heads_local = n_heads // tp_x
    # KV-cache read + attention scores, once per attention layer class.
    # Sliding-window layers see at most `window_size` cached tokens;
    # full-attention layers see the whole context.
    attn_classes = attention_layers
    for layer_class in attn_classes:
        ctx_cls = layer_class.kv_len(context_len)
        if len(attn_classes) == 1:
            name_suffix = ""
            class_note = ""
        else:
            name_suffix = f"[{layer_class.kind.value}]"
            class_note = f" [{layer_class.kind.value}: {layer_class.layer_fraction:.0%} of layers]"

        # KV-cache read: each X-rank reads its local KV heads × visible context
        trace.record(
            "kv_cache_read",
            "Read KV-cache for current context",
            TensorRecord(
                name=f"KV_cache{name_suffix}",
                global_shape={"Dkv": kv_dim, "Ctx": ctx_cls},
                partitions={},
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"rank-local. Each X-rank: {kv_heads_per_x} KV-heads × ctx={ctx_cls} × "
                f"d_head={d_head}. Total per rank = "
                f"{kv_heads_per_x * ctx_cls * d_head * dtype_bytes} B{class_note}"
            ),
        )

        # Attention scores: Q·K^T — [heads_local, s, ctx_cls] per rank
        trace.record(
            "attention_scores",
            "QK^T + online softmax (FlashDecoding)",
            TensorRecord(
                name=f"scores{name_suffix}",
                global_shape={"H": n_heads, "B": s, "Ctx": ctx_cls},
                partitions={"H": P("sharded", ("X",))},
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"rank-local compute. [{heads_local}, {s}, {ctx_cls}] per rank. "
                f"GQA: {heads_local} Q-heads share {kv_heads_per_x} KV-head(s) "
                f"(ratio {heads_local // max(1, kv_heads_per_x)}:1). "
                f"Online softmax: streaming, never fully materialized.{class_note}"
            ),
        )

    # Attention output before AR(Y): V-weighted sum per Y-shard
    trace.record(
        "attn_v_output",
        "Attention × V partial sums (per Y-shard)",
        TensorRecord(
            name="attn_partial",
            global_shape={"B": s, "H": n_heads, "Dh": d_head},
            partitions={
                "H": P("sharded", ("X",)),
                "B": P("partial", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"per Y-rank: partial sum over local KV-cache slice. "
            f"[{s}, {heads_local}, {d_head}] per rank. "
            f"Needs AR(Y) to merge partial sums + softmax corrections."
        ),
    )

    # Stage 3: after attention core + Phase 2 all-reduce on full Y
    trace.record(
        "after_attention",
        "Attention output after AR(Y) — online softmax merge",
        TensorRecord(
            name="attn_out",
            global_shape={"B": s, "D": d_model},
            partitions={
                "D": P("sharded", ("X",)),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=f"comm: AR(Y). Y-replicated, X-sharded. D/X = {d_model // tp_x}",
    )

    # Stage 4: after output projection — AG(Y_eff) [no-op for decode] + matmul + RS(X)
    trace.record(
        "after_output_proj",
        "Output proj: [AG(Y_eff) no-op] + matmul(W_O) + RS(X)",
        TensorRecord(
            name="proj_out",
            global_shape={"B": s, "D": d_model},
            partitions={
                "D": P("sharded", ("X", "Y_eff")),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"comm: RS(X). g={g} identical clones. " f"D/(X·Y_eff) = {d_model // (tp_x * y_eff)}"
        ),
    )


def _trace_moe_stages(
    trace: LayoutTrace,
    *,
    d_model: int,
    d_ff: int,
    n_experts: int,
    top_k: int,
    tp_x: int,
    tp_y: int,
    g: int,
    c: int,
    router_mode: str,
    s: int,
    dtype_bytes: int,
    attn_domains_per_full_expert_set: int,
    full_expert_sets_per_attn_domain: int,
    moe_tp_x: int,
    moe_tp_yz: int,
) -> None:
    y_eff = tp_y // g
    G = -(-n_experts // c)
    if full_expert_sets_per_attn_domain <= 0:
        raise ValueError(
            "full_expert_sets_per_attn_domain must be > 0, got "
            f"{full_expert_sets_per_attn_domain}"
        )
    if s % full_expert_sets_per_attn_domain != 0:
        raise ValueError(
            f"attention-domain tokens ({s}) must be divisible by expert sets "
            f"({full_expert_sets_per_attn_domain})"
        )
    set_tokens = s // full_expert_sets_per_attn_domain
    if g % full_expert_sets_per_attn_domain != 0:
        raise ValueError(
            f"projection groups ({g}) must be divisible by expert sets "
            f"({full_expert_sets_per_attn_domain})"
        )
    replicas_per_set = g // full_expert_sets_per_attn_domain

    _moe_d_mesh_dims: Tuple[str, ...] = tuple(
        d for d, size in (("MoE_X", moe_tp_x), ("MoE_Y", moe_tp_yz)) if size > 1
    )
    if tp_x % moe_tp_x != 0:
        raise ValueError(f"TP_Proj_X ({tp_x}) must be divisible by TP_MoE_X ({moe_tp_x})")
    if y_eff % moe_tp_yz != 0:
        raise ValueError(f"TP_Proj_Y_eff ({y_eff}) must be divisible by TP_MoE_Y ({moe_tp_yz})")
    moe_to_residual_layout_quotient_count = (tp_x // moe_tp_x) * (y_eff // moe_tp_yz)
    if attn_domains_per_full_expert_set > 1:
        if G % attn_domains_per_full_expert_set != 0:
            raise ValueError(
                f"Expert groups ({G}) must be divisible by feeder attention domains "
                f"({attn_domains_per_full_expert_set})"
            )
        local_group_count = G // attn_domains_per_full_expert_set
    else:
        local_group_count = G
    if local_group_count != moe_to_residual_layout_quotient_count * replicas_per_set:
        raise ValueError(
            f"Local expert groups ({local_group_count}) must equal MoE-to-residual "
            "layout quotients × per-set projection replicas "
            f"({moe_to_residual_layout_quotient_count} × {replicas_per_set})"
        )

    # Stage 5: incoming MoE input keeps the output-projection RS(X) layout.
    trace.record(
        "moe_input",
        "Residual add + LayerNorm (same layout as output_proj)",
        TensorRecord(
            name="x_moe",
            global_shape={"B": s, "D": d_model},
            partitions={
                "D": P("sharded", ("X", "Y_eff")),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=f"output-projection layout: D/(X·Y_eff) = {d_model // (tp_x * y_eff)}",
    )
    trace.record(
        "moe_set_token_select",
        "Local selection of the token slice assigned to this complete expert set",
        TensorRecord(
            name="x_moe_set",
            global_shape={"B": set_tokens, "D": d_model},
            partitions={
                "D": P("sharded", ("X", "Y_eff")),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"local zero-communication selection: B={s} -> "
            f"B_set={set_tokens} across {full_expert_sets_per_attn_domain} Y-aligned set(s)"
        ),
    )

    # The consumer completes RS+AG over projection X using all tokens, then
    # reconciles projection Y_eff with TP_MoE_Y.
    _reconciled_partitions = {"D": P("sharded", ("MoE_Y",))} if moe_tp_yz > 1 else {}
    if y_eff == moe_tp_yz:
        y_reconcile_note = "Y_eff already equals TP_MoE_Y; no Y collective"
        y_reconcile_flags: Tuple[str, ...] = ()
    elif y_eff > moe_tp_yz:
        y_reconcile_note = f"AG(Y quotient={y_eff // moe_tp_yz})"
        y_reconcile_flags = (
            "MODEL_LIMITATION: projection-Y quotient gathers assume aligned disjoint "
            "subgroups execute concurrently; cross-subgroup fabric contention is not modeled.",
        )
    else:
        raise ValueError(
            "Projection-to-MoE Y refinement is unsupported without explicit projection-replica "
            f"metadata: TP_Proj_Y_eff={y_eff}, TP_MoE_Y={moe_tp_yz}."
        )
    trace.record(
        "moe_input_reconciled",
        "After all-token AG(TP_Proj_X) and Y reconciliation",
        TensorRecord(
            name="x_moe_reconciled",
            global_shape={"B": set_tokens, "D": d_model},
            partitions={
                **_reconciled_partitions,
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"comm: AG(TP_Proj_X={tp_x}); {y_reconcile_note}. "
            f"D/TP_MoE_Y = {d_model // moe_tp_yz}"
        ),
        flags=y_reconcile_flags,
    )

    # Stage 6: router input — depends on router mode
    if router_mode == "replicated":
        trace.record(
            "router_input",
            "Replicated router: needs full D for W_gate matmul",
            TensorRecord(
                name="x_router",
                global_shape={"B": set_tokens, "D": d_model},
                partitions={
                    "D": P("replicated"),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"boundary AG(TP_Proj_X) plus AG(Y quotient), followed by "
                f"LN/router AG(TP_MoE_Y), provides full d_model={d_model}."
            ),
        )
    else:
        _router_d_partitions = {"D": P("sharded", ("MoE_Y",))} if moe_tp_yz > 1 else {}
        trace.record(
            "router_input",
            "Sharded router: AG(X) then partial logits + AR sync",
            TensorRecord(
                name="x_router",
                global_shape={"B": set_tokens, "D": d_model},
                partitions={
                    **_router_d_partitions,
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: AG(X) → D/TP_MoE_Y = {d_model // moe_tp_yz}. "
                "Router uses W_gate[D/Y_eff, E] → partial logits → Phase B AR."
            ),
        )

    # Cross-attention-domain dispatch A2A.
    # When one expert set spans multiple attention domains, the deterministic
    # handoff has already placed each token shard on aligned local expert
    # groups. Same-domain owners consume that copy without network traffic;
    # only remote-domain owners need A2Av. The logical trace cannot name the
    # physical cross-domain fabric: runtime uses EP_SU for a scale-up-resident
    # set and EP_SO for a scale-out-spanning set.
    if attn_domains_per_full_expert_set > 1:
        _dispatch_d_partitions = {"D": P("sharded", ("MoE_Y",))} if moe_tp_yz > 1 else {}
        trace.record(
            "decode_dispatch",
            "Remote-owner dispatch A2Av on the cross-domain fabric "
            "(attn_domains_per_full_expert_set > 1)",
            TensorRecord(
                name="x_dispatched",
                global_shape={"B": set_tokens, "D": d_model},
                partitions={
                    **_dispatch_d_partitions,
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: A2Av with explicit, evenly distributed expert-group "
                f"ownership across "
                f"attn_domains_per_full_expert_set="
                f"{attn_domains_per_full_expert_set} feeder domains. "
                "Local-owner routes require no network transfer because the "
                "exact handoff replicated each token shard across aligned local "
                "expert groups. Per routed token-expert pair per MoE-Y lane: "
                f"{d_model // moe_tp_yz} elements = "
                f"{(d_model // moe_tp_yz) * dtype_bytes} bytes. "
                "Same-domain network fraction=0; remote-owner network fraction="
                f"{attn_domains_per_full_expert_set - 1}/"
                f"{attn_domains_per_full_expert_set}."
            ),
        )

    # Stage 7: expert input
    _expert_d_partitions = {"D": P("sharded", ("MoE_Y",))} if moe_tp_yz > 1 else {}
    if attn_domains_per_full_expert_set > 1:
        expert_input_desc = (
            "Expert input (post-dispatch: tokens arrived from "
            f"{attn_domains_per_full_expert_set} feeder attention domains)"
        )
    else:
        expert_input_desc = (
            "Expert input (single attention domain per expert set: "
            "tokens already local, dispatch = 0)"
        )
    trace.record(
        "expert_input",
        expert_input_desc,
        TensorRecord(
            name="x_expert",
            global_shape={"B": set_tokens, "D": d_model},
            partitions={
                **_expert_d_partitions,
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"D/TP_MoE_Y = {d_model // moe_tp_yz}. "
            f"W_up: [D/Y, Dff/X] = [{d_model // moe_tp_yz}, {d_ff // moe_tp_x}]"
        ),
    )

    # Stage 8: after up-proj + C1
    trace.record(
        "after_up_proj_c1",
        "Up-proj matmul + C1 AR(TP_MoE_Y)",
        TensorRecord(
            name="h_expert",
            global_shape={"B": set_tokens, "Dff": d_ff},
            partitions={
                **({"Dff": P("sharded", ("MoE_X",))} if moe_tp_x > 1 else {}),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"comm: C1 AR(TP_MoE_Y={moe_tp_yz}); no-op iff TP_MoE_Y=1. "
            f"Dff/TP_MoE_X = {d_ff // moe_tp_x}."
        ),
    )

    # Stage 9: after activation + down-proj + C2
    trace.record(
        "after_down_proj_c2",
        "Activation + down-proj + C2 RS(TP_MoE_X)",
        TensorRecord(
            name="y_expert",
            global_shape={"B": set_tokens, "D": d_model},
            partitions={
                **({"D": P("sharded", _moe_d_mesh_dims)} if _moe_d_mesh_dims else {}),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"comm: C2 RS(TP_MoE_X={moe_tp_x}); no-op iff TP_MoE_X=1. "
            f"D/(TP_MoE_X·TP_MoE_Y) = {d_model // (moe_tp_x * moe_tp_yz)}."
        ),
    )

    # Cross-attention-domain return precedes local residual-layout restoration.
    if attn_domains_per_full_expert_set > 1:
        _returned_d_partitions = {"D": P("sharded", _moe_d_mesh_dims)} if _moe_d_mesh_dims else {}
        trace.record(
            "decode_combine_a2a",
            "Remote-owner return A2Av on the cross-domain fabric",
            TensorRecord(
                name="moe_out_returned",
                global_shape={"B": set_tokens, "D": d_model},
                partitions={
                    **_returned_d_partitions,
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: symmetric A2Av return across attn_domains_per_full_expert_set="
                f"{attn_domains_per_full_expert_set} feeder domains. "
                "Local-owner outputs are already resident; same-domain network "
                "fraction=0. Remote-owner network fraction="
                f"{attn_domains_per_full_expert_set - 1}/"
                f"{attn_domains_per_full_expert_set}."
            ),
        )

    # Stage 10: local residual-layout restoration after the return leg.
    combine_ops: List[str] = []
    if moe_to_residual_layout_quotient_count > 1:
        combine_ops.append(
            "exact ND RS over MoE-to-residual layout quotient="
            f"{moe_to_residual_layout_quotient_count} "
            f"(({tp_x}/{moe_tp_x})×({y_eff}/{moe_tp_yz}))"
        )
    if replicas_per_set > 1:
        combine_ops.append(f"AR over U_set={replicas_per_set} projection replicas per set")
    combine_note = "comm: " + "; ".join(combine_ops) if combine_ops else "ZERO — identity combine"
    _post_combine_d_dims = ("X", "Y_eff")

    trace.record(
        "after_combine",
        "MoE combine (aggregate expert outputs per token)",
        TensorRecord(
            name="moe_out",
            global_shape={"B": set_tokens, "D": d_model},
            partitions={
                **({"D": P("sharded", _post_combine_d_dims)} if _post_combine_d_dims else {}),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=combine_note + f". Output D/(X·Y_eff) = {d_model // (tp_x * y_eff)}.",
    )

    if full_expert_sets_per_attn_domain > 1:
        trace.record(
            "after_combine_cross_set",
            "Cross-set batch all-gather on Set_Y",
            TensorRecord(
                name="moe_out_full_batch",
                global_shape={"B": s, "D": d_model},
                partitions={
                    "D": P("sharded", _post_combine_d_dims),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: AG(Set_Y={full_expert_sets_per_attn_domain}) gathers disjoint "
                f"B_set={set_tokens} slices to B={s}; {replicas_per_set} aligned "
                "subgroup(s) execute concurrently."
            ),
            flags=(
                "MODEL_LIMITATION: concurrent cross-set AG subgroups assume no "
                "cross-subgroup fabric contention.",
            ),
        )

    # Stage 11: layer output — residual add → next layer
    trace.record(
        "layer_output",
        "Residual add → input to next layer",
        TensorRecord(
            name="x_out",
            global_shape={"B": s, "D": d_model},
            partitions={
                **({"D": P("sharded", _post_combine_d_dims)} if _post_combine_d_dims else {}),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(f"D/(X·Y_eff) = {d_model // (tp_x * y_eff)}. " "Must match layer_input."),
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_decode_2dtp_trace(
    *,
    d_model: int,
    d_ff: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_experts: int,
    top_k: int,
    tp_x: int,
    tp_y: int,
    g: int,
    c: int,
    attn_domains_per_full_expert_set: int,
    full_expert_sets_per_attn_domain: int,
    attention_layers: list[AttentionLayerClass],
    moe_tp_x: int,
    moe_tp_yz: int,
    batch_tokens: int,
    router_mode: str = "sharded",
    dtype_bytes: int = 2,
    context_len: int = 1,
) -> LayoutTrace:
    """Build the complete decode pipeline layout trace for one 2D TP layer.

    Parameters
    ----------
    d_model : model hidden dimension
    d_ff : MoE FFN hidden dimension
    n_heads : number of attention heads
    n_kv_heads : number of KV heads (GQA)
    head_dim : per-head attention dimension (independent of d_model under wide-Q;
        always pass `TransformerConfig.head_dim` verbatim)
    n_experts : total number of experts (E)
    top_k : experts selected per token
    tp_x : X-axis TP degree (head parallelism)
    tp_y : full Y-axis TP degree
    g : number of projection groups
    c : experts colocated per group
    attn_domains_per_full_expert_set : how many attention TP domains share one
        expert set. Equal to 1 means single-attention-domain (decode dispatch
        is free); > 1 emits the decode_dispatch / decode_combine_a2a stages.
    full_expert_sets_per_attn_domain : how many expert sets live inside one
        attention TP domain. Equal to 1 means one set per attention domain;
        > 1 keeps the cross-set AG path (the existing "S_attn > 1" regime).
        At most one of these two ratios is > 1 at a time.
    router_mode : "replicated" or "sharded"
    batch_tokens : number of decode tokens in the attention domain
    dtype_bytes : bytes per element (default 2 for bf16/fp16)
    context_len : KV-cache context length (prefill_seq_len)
    attention_layers : the model's attention layer classes (from
        `attention_layer_classes(transformer_config)`); KV-dependent stages
        are emitted once per class
    moe_tp_x : explicit MoE X-axis TP degree
    moe_tp_yz : explicit MoE YZ-axis TP degree
    """
    if tp_y % g != 0:
        raise ValueError(f"tp_y ({tp_y}) must be divisible by g ({g})")

    mesh = build_decode_2dtp_mesh(
        tp_x,
        tp_y,
        g,
        n_experts,
        c,
        moe_tp_x,
        moe_tp_yz,
        full_expert_sets_per_attn_domain,
    )
    trace = LayoutTrace(mesh=mesh)

    _trace_attention_stages(
        trace,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        tp_x=tp_x,
        tp_y=tp_y,
        g=g,
        s=batch_tokens,
        dtype_bytes=dtype_bytes,
        attention_layers=attention_layers,
        context_len=context_len,
    )
    _trace_moe_stages(
        trace,
        d_model=d_model,
        d_ff=d_ff,
        n_experts=n_experts,
        top_k=top_k,
        tp_x=tp_x,
        tp_y=tp_y,
        g=g,
        c=c,
        router_mode=router_mode,
        s=batch_tokens,
        dtype_bytes=dtype_bytes,
        attn_domains_per_full_expert_set=attn_domains_per_full_expert_set,
        full_expert_sets_per_attn_domain=full_expert_sets_per_attn_domain,
        moe_tp_x=moe_tp_x,
        moe_tp_yz=moe_tp_yz,
    )

    trace.assert_layer_boundary()
    return trace
