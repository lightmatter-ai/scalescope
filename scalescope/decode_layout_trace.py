"""Tensor layout tracing for the decode pipeline.

Records tensor partitioning at each stage boundary and formats
a verification table.  Does NOT infer communication — that is a
separate step.  This module is for shape correctness checking.

Usage
-----
    trace = build_decode_2dtp_trace(
        d_model=12288, d_ff=4096, n_heads=96, n_kv_heads=8, head_dim=128,
        n_experts=128, top_k=2, tp_x=32, tp_y=32, g=8, c=16,
        router_mode="sharded",
    )
    print(trace.format_table())
"""

import logging
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

from scalescope.stats import CommStats

logger = logging.getLogger(__name__)


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
        """Check that layer_input and layer_output have compatible layouts."""
        issues: List[str] = []
        try:
            inp = self.find("layer_input", "x")
            out = self.find("layer_output", "x_out")
        except KeyError:
            return ["layer_input or layer_output not found in trace"]

        inp_shape = inp.tensor.per_rank_shape(self.mesh)
        out_shape = out.tensor.per_rank_shape(self.mesh)
        if inp_shape != out_shape:
            issues.append(
                f"MISMATCH: layer_input per_rank={inp_shape} != "
                f"layer_output per_rank={out_shape}"
            )
        return issues

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
        diffs: List[str] = []

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

        col_w = (24, 14, 28, 36, 28)
        hdr = (
            f"{'Stage':<{col_w[0]}}"
            f"{'Tensor':<{col_w[1]}}"
            f"{'Global':<{col_w[2]}}"
            f"{'Sharding':<{col_w[3]}}"
            f"{'Per-Rank':<{col_w[4]}}"
            f"Note"
        )
        lines.append(hdr)
        lines.append("-" * max(len(hdr), 120))

        prev_stage = ""
        for rec in self.records:
            stage_label = rec.stage if rec.stage != prev_stage else ""
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

            line = (
                f"{stage_label:<{col_w[0]}}"
                f"{rec.tensor.name:<{col_w[1]}}"
                f"{gs:<{col_w[2]}}"
                f"{parts:<{col_w[3]}}"
                f"{pr:<{col_w[4]}}"
                f"{note}"
            )
            lines.append(line)

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

    def format_comm_table(self) -> str:
        if not self.comm_records:
            return ""
        lines: List[str] = []
        lines.append("=== Collective Communications (per layer) ===")
        cw = (28, 8, 14, 6, 14, 12, 12, 12)
        hdr = (
            f"{'Stage':<{cw[0]}}"
            f"{'Coll':<{cw[1]}}"
            f"{'Topology':<{cw[2]}}"
            f"{'P':<{cw[3]}}"
            f"{'Volume/rank':<{cw[4]}}"
            f"{'Alpha':<{cw[5]}}"
            f"{'Beta':<{cw[6]}}"
            f"{'Total':<{cw[7]}}"
            f"Note"
        )
        lines.append(hdr)
        lines.append("-" * max(len(hdr), 120))

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
            line = (
                f"{cr.stage:<{cw[0]}}"
                f"{cr.collective:<{cw[1]}}"
                f"{cr.topology:<{cw[2]}}"
                f"{cr.participants:<{cw[3]}}"
                f"{self._fmt_bytes(cr.volume_bytes):<{cw[4]}}"
                f"{self._fmt_us(cr.alpha_s):<{cw[5]}}"
                f"{self._fmt_us(cr.beta_s):<{cw[6]}}"
                f"{self._fmt_us(cr.total_s):<{cw[7]}}"
                f"{cr.note}{tag}"
            )
            lines.append(line)

        lines.append("-" * max(len(hdr), 120))
        label = f"TOTAL ({nonzero_count} active / {len(self.comm_records)} collectives)"
        summary = (
            f"{label:<{cw[0]}}"
            f"{'':<{cw[1]}}"
            f"{'':<{cw[2]}}"
            f"{'':<{cw[3]}}"
            f"{'':<{cw[4]}}"
            f"{self._fmt_us(total_alpha):<{cw[5]}}"
            f"{self._fmt_us(total_beta):<{cw[6]}}"
            f"{self._fmt_us(total_time):<{cw[7]}}"
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
) -> Mesh:
    y_eff = tp_y // g
    G = -(-n_experts // c)
    return Mesh(
        dims=(
            MeshDim("X", tp_x),
            MeshDim("Y", tp_y),
            MeshDim("Y_eff", y_eff),
            MeshDim("G", G),
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
    # KV-cache read: each X-rank reads its local KV heads × full context
    trace.record(
        "kv_cache_read",
        "Read KV-cache for current context",
        TensorRecord(
            name="KV_cache",
            global_shape={"Dkv": kv_dim, "Ctx": context_len},
            partitions={},
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"rank-local. Each X-rank: {kv_heads_per_x} KV-heads × ctx={context_len} × "
            f"d_head={d_head}. Total per rank = "
            f"{kv_heads_per_x * context_len * d_head * dtype_bytes} B"
        ),
    )

    # Attention scores: Q·K^T — [heads_local, s, context_len] per rank
    trace.record(
        "attention_scores",
        "QK^T + online softmax (FlashDecoding)",
        TensorRecord(
            name="scores",
            global_shape={"H": n_heads, "B": s, "Ctx": context_len},
            partitions={"H": P("sharded", ("X",))},
            dtype_bytes=dtype_bytes,
        ),
        note=(
            f"rank-local compute. [{heads_local}, {s}, {context_len}] per rank. "
            f"GQA: {heads_local} Q-heads share {kv_heads_per_x} KV-head(s) "
            f"(ratio {heads_local // max(1, kv_heads_per_x)}:1). "
            f"Online softmax: streaming, never fully materialized."
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
    moe_tp_x: int | None = None,
    moe_tp_yz: int | None = None,
) -> None:
    y_eff = tp_y // g
    G = -(-n_experts // c)

    # Detect 1×Y_eff replicated-router path (§11.3): MoE uses tp_x=1 but
    # attention uses tp_x>1. After AG(X)+AG(YZ) full D is available, C1 is
    # eliminated, and C2 becomes RS(YZ) instead of RS(X).
    _moe_tp_x = moe_tp_x if moe_tp_x is not None else tp_x
    _moe_tp_yz = moe_tp_yz if moe_tp_yz is not None else (tp_y // g)
    _moe_d_mesh_dims: Tuple[str, ...] = tuple(
        d for d, size in (("X", _moe_tp_x), ("Y_eff", _moe_tp_yz)) if size > 1
    )
    _replicated_1d_y = (
        _moe_tp_x == 1 and _moe_tp_yz > 1 and router_mode == "replicated" and tp_x > 1
    )
    _rs_handoff_to_attn_x = _moe_tp_x == 1 and tp_x > 1 and G == tp_x and _moe_tp_yz == y_eff
    _rs_handoff_to_attn_y_eff = _moe_tp_yz == 1 and _moe_tp_x == tp_x and y_eff > 1 and G == y_eff

    # Stage 5: MoE input (after residual add + LayerNorm — same layout as output_proj)
    trace.record(
        "moe_input",
        "Residual add + LayerNorm (same layout as output_proj)",
        TensorRecord(
            name="x_moe",
            global_shape={"B": s, "D": d_model},
            partitions={
                **({"D": P("sharded", _moe_d_mesh_dims)} if _moe_d_mesh_dims else {}),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=f"D/(moe_tp_x·moe_tp_yz) = {d_model // (_moe_tp_x * _moe_tp_yz)}",
    )

    # Stage 6: router input — depends on router mode
    if router_mode == "replicated":
        trace.record(
            "router_input",
            "Replicated router: needs full D for W_gate matmul",
            TensorRecord(
                name="x_router",
                global_shape={"B": s, "D": d_model},
                partitions={
                    "D": P("replicated"),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: AG(X) + AG(Y_eff) → full d_model={d_model}. "
                "See §3.1.1: is Y_eff AG actually needed?"
            ),
            flags=(
                (
                    "OPEN: replicated router with Y_eff>1 requires full d_model gather — "
                    "current code only models AG(X) (Phase A1)",
                    "OPEN: fused router+expert gather vs separate? See §9.5.8",
                )
                if y_eff > 1 and not _replicated_1d_y
                else ()
            ),
        )
    else:
        _router_d_partitions = {"D": P("sharded", ("Y_eff",))} if _moe_tp_yz > 1 else {}
        trace.record(
            "router_input",
            "Sharded router: AG(X) then partial logits + AR sync",
            TensorRecord(
                name="x_router",
                global_shape={"B": s, "D": d_model},
                partitions={
                    **_router_d_partitions,
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: AG(X) → D/Y_eff = {d_model // _moe_tp_yz}. "
                "Router uses W_gate[D/Y_eff, E] → partial logits → Phase B AR."
            ),
        )

    # Stage 7: expert input
    if _replicated_1d_y:
        # §11.3: After AG(attn_X) + AG(YZ), full D is available for expert matmul.
        trace.record(
            "expert_input",
            "Expert input after AG(X) + AG(YZ) — full D (§11.3)",
            TensorRecord(
                name="x_expert",
                global_shape={"B": s, "D": d_model},
                partitions={
                    "D": P("replicated"),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"full D={d_model} (AG(X={tp_x})+AG(YZ={_moe_tp_yz})). "
                f"W_up: [{d_model}, {d_ff // _moe_tp_yz}] (1×{_moe_tp_yz} TP)"
            ),
        )
    else:
        expert_input_flags: List[str] = []
        if router_mode == "replicated" and y_eff > 1:
            expert_input_flags.append(
                "OPEN: if router gathered full D, expert still uses D/Y_eff slice — "
                "extra bandwidth wasted? See §9.5.8"
            )
        _expert_d_partitions = {"D": P("sharded", ("Y_eff",))} if _moe_tp_yz > 1 else {}
        trace.record(
            "expert_input",
            "Expert input (D=1: tokens already local, dispatch = 0)",
            TensorRecord(
                name="x_expert",
                global_shape={"B": s, "D": d_model},
                partitions={
                    **_expert_d_partitions,
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"D/Y_eff = {d_model // _moe_tp_yz}. "
                f"W_up: [D/Y_eff, Dff/X] = [{d_model // _moe_tp_yz}, {d_ff // _moe_tp_x}]"
            ),
            flags=tuple(expert_input_flags),
        )

    # Stage 8: after up-proj + C1
    if _replicated_1d_y:
        # Full D input → complete matmul, no C1 needed.
        trace.record(
            "after_up_proj_c1",
            "Up-proj: complete (full D, no C1 — §11.3)",
            TensorRecord(
                name="h_expert",
                global_shape={"B": s, "Dff": d_ff},
                partitions={"B": P("replicated", ("Y",))},
                dtype_bytes=dtype_bytes,
            ),
            note=f"C1 SKIPPED. Dff={d_ff} (full). mlp_ratio = {d_ff / d_model:.3f}",
        )
    else:
        trace.record(
            "after_up_proj_c1",
            "Up-proj matmul + C1 AR(Y_eff)",
            TensorRecord(
                name="h_expert",
                global_shape={"B": s, "Dff": d_ff},
                partitions={
                    **({"Dff": P("sharded", ("X",))} if _moe_tp_x > 1 else {}),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: C1 AR(Y_eff). Dff/X = {d_ff // _moe_tp_x}. "
                f"mlp_ratio = d_ff/d_model = {d_ff}/{d_model} = {d_ff / d_model:.3f}"
            ),
        )

    # Stage 9: after activation + down-proj + C2
    if _replicated_1d_y:
        # C2 is RS(YZ) + local-select → D/(X·Y_eff) per rank.
        trace.record(
            "after_down_proj_c2",
            f"Down-proj + RS(Y={_moe_tp_yz}) + local-select (§11.3)",
            TensorRecord(
                name="y_expert",
                global_shape={"B": s, "D": d_model},
                partitions={
                    "D": P("sharded", ("X", "Y_eff")),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: RS(Y={_moe_tp_yz}) + local-select. "
                f"D/(X·Y_eff) = {d_model // (tp_x * y_eff)}. "
                f"W_down: [{d_ff // _moe_tp_yz}, {d_model}]"
            ),
        )
    else:
        trace.record(
            "after_down_proj_c2",
            "Activation + down-proj + C2 RS(X)",
            TensorRecord(
                name="y_expert",
                global_shape={"B": s, "D": d_model},
                partitions={
                    **({"D": P("sharded", _moe_d_mesh_dims)} if _moe_d_mesh_dims else {}),
                    "B": P("replicated", ("Y",)),
                },
                dtype_bytes=dtype_bytes,
            ),
            note=(
                f"comm: C2 RS(X). D/(X·Y_eff) = {d_model // (_moe_tp_x * _moe_tp_yz)}. "
                f"W_down: [Dff/X, D/Y_eff] = [{d_ff // _moe_tp_x}, {d_model // _moe_tp_yz}]"
            ),
        )

    # Stage 10: after combine
    if G == 1:
        combine_note = "ZERO — G=1, all experts local, combine is free local add before C2"
        combine_flags: Tuple[str, ...] = ()
    else:
        remote_frac = (G - 1) / G
        combine_note = (
            f"comm: per-token reduce across G={G} groups on Y. "
            f"≈{G-1}/{G} = {remote_frac:.3f} of tokens need cross-group reduce."
        )
        combine_flags = ()

    _post_combine_d_dims: Tuple[str, ...]
    if _rs_handoff_to_attn_x or _rs_handoff_to_attn_y_eff:
        _post_combine_d_dims = ("X", "Y_eff")
    else:
        _post_combine_d_dims = _moe_d_mesh_dims

    trace.record(
        "after_combine",
        "MoE combine (aggregate expert outputs per token)",
        TensorRecord(
            name="moe_out",
            global_shape={"B": s, "D": d_model},
            partitions={
                **({"D": P("sharded", _post_combine_d_dims)} if _post_combine_d_dims else {}),
                "B": P("replicated", ("Y",)),
            },
            dtype_bytes=dtype_bytes,
        ),
        note=(
            combine_note
            + (
                f" RS handoff to attn-X enabled: D/(X·Y_eff) = {d_model // (tp_x * y_eff)}."
                if _rs_handoff_to_attn_x
                else (
                    f" RS handoff to attn-Y_eff enabled: D/(X·Y_eff) = {d_model // (tp_x * y_eff)}."
                    if _rs_handoff_to_attn_y_eff
                    else ""
                )
            )
        ),
        flags=combine_flags,
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
        note=(
            f"D/({'X·Y_eff' if (_rs_handoff_to_attn_x or _rs_handoff_to_attn_y_eff) else 'moe_tp_x·moe_tp_yz'}) = "
            f"{(d_model // (tp_x * y_eff)) if (_rs_handoff_to_attn_x or _rs_handoff_to_attn_y_eff) else (d_model // (_moe_tp_x * _moe_tp_yz))}. "
            "Must match layer_input."
        ),
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
    router_mode: str = "sharded",
    s: int = 1,
    dtype_bytes: int = 2,
    context_len: int = 1,
    moe_tp_x: int | None = None,
    moe_tp_yz: int | None = None,
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
    router_mode : "replicated" or "sharded"
    s : sequence length (1 for decode)
    dtype_bytes : bytes per element (default 2 for bf16/fp16)
    context_len : KV-cache context length (prefill_seq_len)
    moe_tp_x : MoE X-axis TP degree (defaults to tp_x if None)
    moe_tp_yz : MoE YZ-axis TP degree (defaults to tp_y//g if None)
    """
    if tp_y % g != 0:
        raise ValueError(f"tp_y ({tp_y}) must be divisible by g ({g})")

    mesh = build_decode_2dtp_mesh(tp_x, tp_y, g, n_experts, c)
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
        s=s,
        dtype_bytes=dtype_bytes,
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
        s=s,
        dtype_bytes=dtype_bytes,
        moe_tp_x=moe_tp_x,
        moe_tp_yz=moe_tp_yz,
    )

    return trace
