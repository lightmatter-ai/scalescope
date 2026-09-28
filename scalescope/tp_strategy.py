"""Tensor-parallel (TP) communication strategies for attention and MoE.

Implements modeling for forward/backward TP collectives for 1D/2D weight sharding variants.
"""

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

from scalescope.decode_layout_trace import NULL_LAYOUT_TRACE, P, TensorRecord
from scalescope.ep_strategy import resolve_moe_token_scopes
from scalescope.model_spec import (
    attention_layer_classes,
    resolve_kv_heads_per_partition,
    TransformerConfig,
)
from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_all_reduce_on_nd_topology,
    hockney_model_reduce_on_nd_topology,
    hockney_model_reduce_scatter_on_nd_topology,
    hockney_model_total_exchange_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.topology_spec import LogicalTopology, PhysicalTopology1DType
from scalescope.workload_spec import (
    MoERouterSyncMode,
    SeqParallelAttnMode,
    TensorParallelVariant,
    TP2DProjAxisRole,
    TP2DSplitSpec,
)


def _tag_to_str(tag) -> str:
    return tag.value if hasattr(tag, "value") else str(tag)


@dataclass(frozen=True)
class Resolved2DSplit:
    x_dims: list[int]
    yz_dims: list[int]
    x_tags: list[str]
    yz_tags: list[str]
    tp_x: int
    tp_yz: int
    x_view: LogicalTopology
    yz_view: LogicalTopology


def _physical_axis_sizes(view: LogicalTopology) -> dict[str, int]:
    """Return the logical extent assigned to each physical dimension."""
    sizes: dict[str, int] = {}
    for dim in view.dimensions:
        physical_name = dim.maps_to_physical_dimension_name
        if physical_name in sizes:
            sizes[physical_name] *= dim.size
        else:
            sizes[physical_name] = dim.size
    return sizes


def _layout_quotient_view(
    *,
    numerator_view: LogicalTopology,
    denominator_view: LogicalTopology,
    view_name: str,
) -> LogicalTopology:
    """Build the exact physical-axis quotient ``numerator / denominator``.

    This is used only when every numerator extent is an integer multiple of
    the corresponding denominator extent.  Mixed coarsening/refinement across
    physical dimensions is rejected rather than approximated as a virtual
    one-dimensional collective.
    """
    numerator_sizes = _physical_axis_sizes(numerator_view)
    denominator_sizes = _physical_axis_sizes(denominator_view)
    numerator_physical_names = set(numerator_sizes)
    denominator_physical_names = set(denominator_sizes)
    if numerator_physical_names != denominator_physical_names:
        raise ValueError(
            f"{view_name}: numerator and denominator must map the same physical dimensions; "
            f"numerator={sorted(numerator_physical_names)}, "
            f"denominator={sorted(denominator_physical_names)}"
        )

    quotient_by_physical: dict[str, int] = {}
    for physical_name in numerator_physical_names:
        numerator_size = numerator_sizes[physical_name]
        denominator_size = denominator_sizes[physical_name]
        if numerator_size % denominator_size != 0:
            raise ValueError(
                f"{view_name}: layout extents are not divisible on physical dimension "
                f"{physical_name}: {numerator_size} / {denominator_size}"
            )
        quotient_by_physical[physical_name] = numerator_size // denominator_size

    new_sizes: dict[str, int] = {}
    remaining = dict(quotient_by_physical)
    for dim in numerator_view.dimensions:
        physical_name = dim.maps_to_physical_dimension_name
        factor = math.gcd(remaining[physical_name], dim.size)
        new_sizes[_tag_to_str(dim.logical_tag)] = factor
        remaining[physical_name] //= factor

    if any(size != 1 for size in remaining.values()):
        raise ValueError(f"{view_name}: quotient contains an unmapped physical extent: {remaining}")

    return numerator_view.create_resized_view(view_name=view_name, new_sizes=new_sizes)


def _decode_moe_input_layout_comm(
    *,
    proj_x_topology: LogicalTopology,
    proj_yz_topology: LogicalTopology,
    moe_yz_topology: LogicalTopology,
    tokens: int,
    d_model: int,
    float_size: float,
    layout_trace,
) -> CommStats:
    """Price the all-token output-projection -> MoE input layout handoff."""
    if tokens < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens}")
    if d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {d_model}")
    if float_size <= 0:
        raise ValueError(f"float_size must be > 0, got {float_size}")

    proj_y = proj_yz_topology.num_chips
    moe_y = moe_yz_topology.num_chips
    if proj_y <= 0 or moe_y <= 0:
        raise ValueError(f"Projection and MoE Y extents must be > 0, got {proj_y} and {moe_y}")
    if proj_y < moe_y:
        raise ValueError(
            "Projection-to-MoE Y refinement is unsupported without explicit projection-replica "
            f"metadata: TP_Proj_Y_eff={proj_y}, TP_MoE_Y={moe_y}. No scatter fallback is "
            "modeled; prove a local replica-to-shard selection or add an exact redistribution."
        )

    total = CommStats()

    # Output projection intentionally ends with RS(TP_Proj_X).  The MoE
    # consumer completes the split all-reduce with an all-token AG over that
    # same projection-X group.  This payload is independent of top-k and rho.
    x_after_ag_bytes = float_size * tokens * (d_model / proj_y)
    comm_x = hockney_model_all_gather_on_nd_topology(
        proj_x_topology, x_after_ag_bytes, float_size_bytes=float_size
    )
    total += comm_x
    layout_trace.record_comm(
        "moe_input_ag_proj_x",
        "AG",
        f"TP_Proj_X({proj_x_topology.num_chips})",
        proj_x_topology.num_chips,
        x_after_ag_bytes,
        comm_x,
        note="all-token consumer gather; completes output-projection RS+AG",
    )

    if proj_y == moe_y:
        return total

    if proj_y > moe_y:
        # The Y_p / Q_y quotient defines one representative gather subgroup.
        # There are Q_y aligned, disjoint subgroups, each with Y_p / Q_y ranks,
        # and the model assumes they execute concurrently.  We therefore charge
        # the critical-path time of one subgroup rather than summing Q_y
        # launches.  Explicit subgroup identities and contention when those
        # concurrent groups share physical fabric resources are not modeled.
        quotient_view = _layout_quotient_view(
            numerator_view=proj_yz_topology,
            denominator_view=moe_yz_topology,
            view_name="TP_Proj_Y_to_TP_MoE_Y_AG",
        )
        y_after_ag_bytes = float_size * tokens * (d_model / moe_y)
        comm_y = hockney_model_all_gather_on_nd_topology(
            quotient_view, y_after_ag_bytes, float_size_bytes=float_size
        )
        total += comm_y
        layout_trace.record_comm(
            "moe_input_ag_y",
            "AG",
            f"TP_Proj_Y/TP_MoE_Y({quotient_view.num_chips})",
            quotient_view.num_chips,
            y_after_ag_bytes,
            comm_y,
            note=(
                f"coarsen projection Y layout: {proj_y} -> {moe_y}; "
                f"{moe_y} disjoint subgroup(s) of size {proj_y // moe_y} run concurrently. "
                "MODEL_LIMITATION: cross-subgroup fabric contention is not modeled."
            ),
        )
        return total

    raise AssertionError("Unreachable projection-to-MoE Y reconciliation branch")


def resolve_tp2d_split(
    *, tp_view: LogicalTopology, split_2d: TP2DSplitSpec, view_name: str
) -> Resolved2DSplit:
    """Resolve a TP2DSplitSpec into indices/sizes and build a YZ-only view."""
    dim_tags = [_tag_to_str(d.logical_tag) for d in getattr(tp_view, "dimensions", [])]
    x_tags = list(split_2d.x_tags)
    yz_tags = list(split_2d.yz_tags)

    try:
        x_dims = [dim_tags.index(t) for t in x_tags]
    except ValueError as e:
        raise ValueError(f"{view_name}: x_tags contains unknown tag for this TP view: {e}") from e
    try:
        yz_dims = [dim_tags.index(t) for t in yz_tags]
    except ValueError as e:
        raise ValueError(f"{view_name}: yz_tags contains unknown tag for this TP view: {e}") from e

    shape = getattr(tp_view, "shape", [])
    tp_x = 1
    for idx in x_dims:
        tp_x *= shape[idx]
    tp_yz = 1
    for idx in yz_dims:
        tp_yz *= shape[idx]

    yz_view = tp_view.create_view(
        view_name=f"{view_name}_YZ_View",
        relevant_tags=yz_tags,
        global_sharing_context=getattr(tp_view, "global_sharing_context", "max_exclusive"),
    )
    x_view = tp_view.create_view(
        view_name=f"{view_name}_X_View",
        relevant_tags=x_tags,
        global_sharing_context=getattr(tp_view, "global_sharing_context", "max_exclusive"),
    )

    return Resolved2DSplit(
        x_dims=x_dims,
        yz_dims=yz_dims,
        x_tags=x_tags,
        yz_tags=yz_tags,
        tp_x=tp_x,
        tp_yz=tp_yz,
        x_view=x_view,
        yz_view=yz_view,
    )


# ---- TP-2D attention comm helpers (explicit stage separation) ----


@dataclass(frozen=True)
class _TP2DProjWeightBytes:
    """Full (model-level) attention projection weight sizes, in bytes.

    One field per projection GEMM: `w_qkv` for the fused QKV projection,
    `w_o` for the output projection. These are the *logical* matrix sizes
    with no TP sharding applied; callers divide by the relevant TP factors
    (e.g. `weights.w_qkv / tp_x_proj`) to get the per-rank shard or the
    panel size a collective actually moves.
    """

    w_qkv: float
    w_o: float


def _tp2d_proj_weight_bytes(
    transformer_config: TransformerConfig, float_size_bytes: float
) -> _TP2DProjWeightBytes:
    d_model = transformer_config.d_model
    head_dim = transformer_config.head_dim
    q_proj_out_dim = transformer_config.num_heads * head_dim
    kv_proj_out_dim = transformer_config.num_kv_heads * head_dim
    w_qkv = float_size_bytes * d_model * (q_proj_out_dim + 2 * kv_proj_out_dim)
    w_o = float_size_bytes * q_proj_out_dim * d_model
    return _TP2DProjWeightBytes(w_qkv=w_qkv, w_o=w_o)


def _tp2d_proj_activation_bytes(
    transformer_config: TransformerConfig,
    *,
    n_batch: int,
    n_seq: int,
    tp_x_proj: int,
    float_size_bytes: float,
) -> tuple[float, float]:
    """Full-sequence projection activation buffers for the local X shard.

    Used by the weight-stationary (`yz: shard_weights`) YZ legs, whose
    collectives move activations over the sequence dim:
    - QKV partials: [B, S, (q_out + 2*kv_out)/tp_x]
    - attention output: [B, S, q_out/tp_x]
    """
    head_dim = transformer_config.head_dim
    q_proj_out_dim = transformer_config.num_heads * head_dim
    kv_proj_out_dim = transformer_config.num_kv_heads * head_dim
    qkv_act = (
        float_size_bytes * n_batch * n_seq * (q_proj_out_dim + 2 * kv_proj_out_dim) / tp_x_proj
    )
    o_act = float_size_bytes * n_batch * n_seq * q_proj_out_dim / tp_x_proj
    return qkv_act, o_act


def _tp2d_attn_proj_comm_prefill_train_forward(
    *,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    proj_x_topology: LogicalTopology,
    proj_yz_topology: LogicalTopology,
    tp_x_proj: int,
    seq_shard: int,
    float_size_bytes: float,
) -> tuple[CommStats, CommStats]:
    """Forward projection-leg comm (phases 1 and 3) as a per-axis-role walk.

    Phase numbering refers to the three-stage TP-2D attention layer pass:
    phase 1 = QKV input projections, phase 2 = attention core, phase 3 =
    output projection. This helper prices only the projection legs (1 and 3);
    the attention core (2) — the ring/Ulysses/USP exchange — is priced by
    `_tp2d_attn_comm_phase2_prefill_train`, and the caller assembles all
    three into `AttnForwardCommPhases`.

    Starting layout (prefill/training): each rank holds a slice of the
    residual stream cut along two axes at once — the sequence dim across YZ
    (`seq_shard` ranks) and the feature dim (d_model) across X. Per rank:
    [B, S/seq_shard, d_model/X].

    X legs — identical in every role. X always shards weights (validated in
    `TP2DProjAxisRoles`), and the projection GEMMs contract over d_model, so
    each rank needs the full d_model — but only for its own S/seq_shard
    tokens (the GEMM is independent across tokens, so the sequence stays
    partitioned). Phase 1 all-gathers the token slab along X before the QKV
    GEMM ([B, S/seq_shard, d_model/X] -> [B, S/seq_shard, d_model]); phase 3
    reduce-scatters the W_O output along X (Megatron-SP).
    Both collectives carry the same payload: the [B, S/seq_shard, d_model]
    residual slab of the rank's local sequence shard. The attention-output
    activation (feature dim q_proj_out_dim) never crosses the wire on X —
    even under wide-Q layouts where q_proj_out_dim > d_model.

    YZ legs — decided by the role:
    - `shard_tokens`: weights are replicated across YZ. Nothing moves on YZ.
    - `circulate_weights`: weights are sharded across YZ at rest, so each
      layer all-gathers the local X-shard weight panel along YZ before the
      GEMM (FSDP/ZeRO-3 style). Volume depends on model size only — it is
      independent of sequence length.
    - `shard_weights` (weight-stationary, decode-style): weight panels stay
      pinned (sharded X x YZ) and *activations* move instead. Phase 1
      reduce-scatters the full-sequence QKV partial sums along YZ over the
      sequence dim, landing tokens in exactly the sequence-sharded layout
      phase 2 expects; phase 3 all-gathers the attention output along YZ
      before the W_O panel GEMM. Both volumes are proportional to S: this
      regime wins at short sequences (and decode) and loses to panel
      circulation at long context. One layout wrinkle: at rest the residual
      stream is feature-sharded on YZ ([B, S, d_model/YZ]) rather than
      sequence-sharded — but that slab has the same byte count, so the X
      legs above are unchanged.
    """
    d_model = transformer_config.d_model
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    roles = workload_spec.tp2d_proj_axis_roles
    if roles is None:
        raise ValueError(
            "TP-2D prefill/train projection comm requires tp2d_proj_axis_roles on the "
            "workload spec (TrainSpec / InferencePrefillSpec)."
        )

    # Full token slab (residual stream) for the local sequence shard.
    token_slab_bytes = float_size_bytes * n_batch * (n_seq / seq_shard) * d_model
    weights = _tp2d_proj_weight_bytes(transformer_config, float_size_bytes)
    qkv_act_bytes, o_act_bytes = _tp2d_proj_activation_bytes(
        transformer_config,
        n_batch=n_batch,
        n_seq=n_seq,
        tp_x_proj=tp_x_proj,
        float_size_bytes=float_size_bytes,
    )

    phase1 = hockney_model_all_gather_on_nd_topology(
        proj_x_topology, token_slab_bytes, float_size_bytes=float_size_bytes
    )
    phase3 = hockney_model_reduce_scatter_on_nd_topology(
        proj_x_topology, token_slab_bytes, float_size_bytes=float_size_bytes
    )

    match roles.yz:
        case TP2DProjAxisRole.SHARD_TOKENS:
            pass  # weights replicated on YZ: no weight legs
        case TP2DProjAxisRole.CIRCULATE_WEIGHTS:
            phase1 += hockney_model_all_gather_on_nd_topology(
                proj_yz_topology, weights.w_qkv / tp_x_proj, float_size_bytes=float_size_bytes
            )
            phase3 += hockney_model_all_gather_on_nd_topology(
                proj_yz_topology, weights.w_o / tp_x_proj, float_size_bytes=float_size_bytes
            )
        case TP2DProjAxisRole.SHARD_WEIGHTS:
            phase1 += hockney_model_reduce_scatter_on_nd_topology(
                proj_yz_topology, qkv_act_bytes, float_size_bytes=float_size_bytes
            )
            phase3 += hockney_model_all_gather_on_nd_topology(
                proj_yz_topology, o_act_bytes, float_size_bytes=float_size_bytes
            )
        case _:
            raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")

    return phase1, phase3


def _tp2d_attn_proj_comm_train_backward(
    *,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    proj_x_topology: LogicalTopology,
    proj_yz_topology: LogicalTopology,
    tp_x_proj: int,
    seq_shard: int,
    float_size_bytes: float,
) -> tuple[CommStats, CommStats]:
    """Backward projection-leg comm (phases 1 and 3), mirroring the forward walk.

    Same phase numbering and structure as
    `_tp2d_attn_proj_comm_prefill_train_forward`; the attention-core
    backward (phase 2) is priced separately. The general rule: every forward
    collective mirrors into its dual (all-gather <-> reduce-scatter) at
    equal volume, and the backward pass adds one genuinely new term — the
    weight-gradient (dW) reduction, whose form depends on how weights are
    laid out on YZ.

    X legs — identical in every role, mirroring forward at equal volume:
    the forward all-gather of the token slab becomes a reduce-scatter of
    the input gradients, and the forward reduce-scatter of the W_O output
    becomes an all-gather of the output gradients. Payload is the same
    [B, S/seq_shard, d_model] residual slab as forward.

    YZ legs — decided by the role (mirrored activation/panel legs first,
    then the dW reduction):
    - `shard_tokens`: no forward YZ comm, so nothing mirrors. dW: each rank
      computed a partial dW from its own sequence shard, so the full local
      (X-sharded) weight grad is all-reduced along YZ.
    - `circulate_weights`: the weight panels are re-gathered along YZ for
      the backward GEMMs (same all-gather as forward). dW: panel gradients
      are reduce-scattered back to their YZ owners. Together that is 2x the
      forward panel volume.
    - `shard_weights` (weight-stationary): the forward activation legs
      mirror at equal volume — the forward reduce-scatter of the QKV
      partials becomes an all-gather of dQKV (needed for the input-grad
      GEMM), and the forward all-gather of the attention output becomes a
      reduce-scatter of its gradient back to the sequence shard. dW: each
      rank's dW panel is a partial sum over its local sequence slice, so
      the panel-sized (X x YZ-sharded) weight grad is all-reduced along YZ.
    """
    d_model = transformer_config.d_model
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    roles = workload_spec.tp2d_proj_axis_roles
    if roles is None:
        raise ValueError(
            "TP-2D train backward projection comm requires tp2d_proj_axis_roles on the "
            "workload spec (TrainSpec)."
        )

    token_slab_bytes = float_size_bytes * n_batch * (n_seq / seq_shard) * d_model
    weights = _tp2d_proj_weight_bytes(transformer_config, float_size_bytes)
    qkv_act_bytes, o_act_bytes = _tp2d_proj_activation_bytes(
        transformer_config,
        n_batch=n_batch,
        n_seq=n_seq,
        tp_x_proj=tp_x_proj,
        float_size_bytes=float_size_bytes,
    )

    phase1 = hockney_model_reduce_scatter_on_nd_topology(
        proj_x_topology, token_slab_bytes, float_size_bytes=float_size_bytes
    )
    phase3 = hockney_model_all_gather_on_nd_topology(
        proj_x_topology, token_slab_bytes, float_size_bytes=float_size_bytes
    )

    match roles.yz:
        case TP2DProjAxisRole.SHARD_TOKENS:
            # No forward YZ comm to mirror. Both legs below are the new dW
            # term: each rank's weight grad is a partial sum over its own
            # sequence shard, all-reduced along YZ (full X-sharded dW).
            phase1 += hockney_model_all_reduce_on_nd_topology(
                proj_yz_topology, weights.w_qkv / tp_x_proj, float_size_bytes=float_size_bytes
            )
            phase3 += hockney_model_all_reduce_on_nd_topology(
                proj_yz_topology, weights.w_o / tp_x_proj, float_size_bytes=float_size_bytes
            )
        case TP2DProjAxisRole.CIRCULATE_WEIGHTS:
            # AG = re-gather the weight panel for the backward GEMMs (mirror
            # of the forward panel gather); RS = the new dW term — panel
            # gradients reduce-scattered back to their YZ owners.
            phase1 += hockney_model_all_gather_on_nd_topology(
                proj_yz_topology, weights.w_qkv / tp_x_proj, float_size_bytes=float_size_bytes
            ) + hockney_model_reduce_scatter_on_nd_topology(
                proj_yz_topology, weights.w_qkv / tp_x_proj, float_size_bytes=float_size_bytes
            )
            phase3 += hockney_model_all_gather_on_nd_topology(
                proj_yz_topology, weights.w_o / tp_x_proj, float_size_bytes=float_size_bytes
            ) + hockney_model_reduce_scatter_on_nd_topology(
                proj_yz_topology, weights.w_o / tp_x_proj, float_size_bytes=float_size_bytes
            )
        case TP2DProjAxisRole.SHARD_WEIGHTS:
            tp_yz_proj = proj_yz_topology.num_chips
            # Activation legs mirror forward (AG of dQKV / RS of the
            # attention-output grad); the all-reduce is the new dW term —
            # each rank's pinned panel grad (W/(X*YZ)) is a partial sum over
            # its local sequence slice, all-reduced along YZ.
            phase1 += hockney_model_all_gather_on_nd_topology(
                proj_yz_topology, qkv_act_bytes, float_size_bytes=float_size_bytes
            ) + hockney_model_all_reduce_on_nd_topology(
                proj_yz_topology,
                weights.w_qkv / (tp_x_proj * tp_yz_proj),
                float_size_bytes=float_size_bytes,
            )
            phase3 += hockney_model_reduce_scatter_on_nd_topology(
                proj_yz_topology, o_act_bytes, float_size_bytes=float_size_bytes
            ) + hockney_model_all_reduce_on_nd_topology(
                proj_yz_topology,
                weights.w_o / (tp_x_proj * tp_yz_proj),
                float_size_bytes=float_size_bytes,
            )
        case _:
            raise ValueError(f"Unsupported tp2d_proj_axis_roles.yz: {roles.yz}")

    return phase1, phase3


def _tp2d_attn_comm_phase3_output_proj(
    *,
    x_topology: LogicalTopology,
    yz_topology: LogicalTopology,
    input_volume_bytes: float,
    output_volume_bytes: float,
    tp_x: int,
    tp_yz: int,
    float_size_bytes: float,
    layout_trace=NULL_LAYOUT_TRACE,
) -> CommStats:
    """Phase 3: output projection comm for 2D TP (matches explicit algorithm).

    Assumed 2D sharding:
    - Input activation arrives from attention as [B, n_seq, E_in/(tp_x*tp_yz)] per GPU
      where E_in = num_heads * head_dim = q_proj_out_dim — the attention-output
      dim, distinct from d_model under "wide-Q" attention layouts (e.g.
      MiniMax-M2.5, gpt_oss_120b, qwen3 series).

    Procedure modeled:
    1) All-gather along YZ to reconstruct `d_h` (equivalently the full E_in/tp_x slice).
       Payload volume scales with E_in = q_proj_out_dim.
    2) Local matmul with W_O shard maps E_in → d_model (modeled in compute/memory, not here).
    3) Reduce-scatter along X to sum across head shards and keep output sharded as
       d_model/(tp_x*tp_yz). Payload volume scales with d_model (the matmul output).

    Caller must supply both `input_volume_bytes` (= B·n_seq·q_proj_out_dim·float_size)
    and `output_volume_bytes` (= B·n_seq·d_model·float_size) explicitly so the
    two distinct collectives are priced on their actual payloads. The two are
    equal whenever num_heads · head_dim == d_model (i.e. non-wide-Q), in which
    case both collectives price the same volume.
    """
    if x_topology.num_chips <= 1 and yz_topology.num_chips <= 1:
        return CommStats()

    comm = CommStats()

    # Step 1: all-gather along YZ — payload volume is the attention-output activation (q_proj_out_dim).
    yz_after_ag_bytes = input_volume_bytes / tp_x
    ag_yz_stats = hockney_model_all_gather_on_nd_topology(
        yz_topology, yz_after_ag_bytes, float_size_bytes=float_size_bytes
    )
    comm += ag_yz_stats
    layout_trace.record_comm(
        "phase3_ag_yz",
        "AG",
        f"YZ({yz_topology.num_chips})",
        yz_topology.num_chips,
        yz_after_ag_bytes,
        ag_yz_stats,
        note="output proj: gather d_h along YZ",
    )

    # Step 3: reduce-scatter along X — payload volume is the post-output-proj activation (d_model).
    x_before_rs_bytes = output_volume_bytes / tp_yz
    rs_x_stats = hockney_model_reduce_scatter_on_nd_topology(
        x_topology, x_before_rs_bytes, float_size_bytes=float_size_bytes
    )
    comm += rs_x_stats
    layout_trace.record_comm(
        "phase3_rs_x",
        "RS",
        f"X({x_topology.num_chips})",
        x_topology.num_chips,
        x_before_rs_bytes,
        rs_x_stats,
        note="output proj: scatter across X heads",
    )
    return comm


def _tp2d_attn_comm_phase2_decode_yz_reductions(
    *,
    yz_topology: LogicalTopology,
    transformer_config: TransformerConfig,
    workload_spec,
    tp_x: int,
    float_size_bytes: float,
    layout_trace=NULL_LAYOUT_TRACE,
) -> CommStats:
    """Phase 2 (decode only): online-softmax YZ collectives for sequence/KV sharding.

    This matches the "distributed FlashAttention / online softmax" procedure:
    1) Each YZ shard computes a partial numerator vector O_local and softmax statistics (m, ℓ)
       for its local KV-cache segment.
    2) YZ all-reduce combines (O_local, m, ℓ) across the sequence-sharded axis to produce the
       exact final output for this (batch, head) shard.

    Payload per rank (per query token):
      (head_dim + 2) scalars per (batch, head), i.e. O_local (head_dim) + (m, ℓ).

    Notes:
    - This models *sequence/context parallelism* on YZ (KV-cache is sharded on YZ).
    - The result is replicated across YZ ranks (no reduce-scatter on the output).
    """
    if yz_topology.num_chips <= 1:
        return CommStats()

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    num_heads = transformer_config.num_heads
    if num_heads % tp_x != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by TP-X size ({tp_x}); "
            f"query-head replication across X is not modeled."
        )
    heads_per_x = num_heads // tp_x
    head_dim = transformer_config.head_dim

    comm = CommStats()
    payload_bytes = float_size_bytes * n_batch * heads_per_x * (head_dim + 2)
    ar_y_stats = hockney_model_all_reduce_on_nd_topology(
        yz_topology, payload_bytes, float_size_bytes=float_size_bytes
    )
    comm += ar_y_stats
    layout_trace.record_comm(
        "phase2_ar_y",
        "AR",
        f"Y({yz_topology.num_chips})",
        yz_topology.num_chips,
        payload_bytes,
        ar_y_stats,
        note="FlashDecoding: merge O_local + (m,l) across Y",
    )
    return comm


def _tp2d_attn_comm_phase2_prefill_ring_kv_exchange(
    *,
    yz_topology: LogicalTopology,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    tp_x: int,
    float_size_bytes: float,
) -> CommStats:
    """Phase 2 (prefill): approximate RingAttention KV circulation on the YZ axis.

    Background / intent:
    - In TP-2D with sequence sharding over YZ ("context parallelism"), prefill attention can be
      implemented with a ring-style algorithm (e.g., RingAttention / ring FlashAttention).
    - Each YZ rank keeps its local K/V cache segment (roughly S/tp_yz tokens) and *streams*
      K/V blocks around the YZ ring. While blocks arrive, it updates online-softmax statistics
      and accumulates the partial output for its local query tokens.

    Why we use Option A (all-gather proxy):
    - A full ring pipeline model would explicitly represent (tp_yz-1) send/recv steps and
      potentially overlap compute with communication. That's extra machinery.
    - For an analytic latency/volume model, a reasonable first-order proxy is:
        "Ring KV circulation" ≈ "All-gather K and V along YZ"
      because both have the same essential property: *each rank receives all other ranks' K/V
      segments over the course of the operation*.

    What this proxy captures:
    - Correct per-rank *volume scaling* on YZ: ~((tp_yz-1)/tp_yz) of the full K/V buffer.
    - Correct dependence on sequence length S=n_seq (unlike decode, where YZ comm can be O(1) in S).

    What this proxy intentionally ignores:
    - Step-by-step pipelining details (how early output can be produced).
    - Any overlap between compute and comm inside the fused attention kernel.
    - The fact that a ring algorithm may not need to materialize "the full gathered KV" at once.

    Payload modeled:
    - We gather K and V for the full sequence (length `n_seq`) for the *local head shard* on X.
    - GQA-aware: K/V payload uses `num_kv_heads` (not `num_heads`).
    - Sliding-window layers only need the window preceding each query block,
      so their circulated KV is capped at `window_size` tokens; costs are
      blended per attention layer class.
    - Per-rank full-buffer size after the conceptual all-gather (per class):
        KV_full_bytes = 2 * B * kv_len(n_seq) * (H_kv/tp_x) * d_h * b
    """
    if yz_topology.num_chips <= 1:
        return CommStats()

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim

    kv_heads_per_x, _kv_rep = resolve_kv_heads_per_partition(num_kv_heads=num_kv_heads, tp_x=tp_x)
    comm = CommStats()
    for layer_class in attention_layer_classes(transformer_config):
        kv_full_bytes = float_size_bytes * (
            2 * n_batch * layer_class.kv_len(n_seq) * (kv_heads_per_x * head_dim)
        )
        comm += layer_class.layer_fraction * hockney_model_all_gather_on_nd_topology(
            yz_topology, kv_full_bytes, float_size_bytes=float_size_bytes
        )
    return comm


def _tp2d_attn_comm_phase2_prefill_ulysses_exchange(
    *,
    yz_topology: LogicalTopology,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    tp_x: int,
    float_size_bytes: float,
) -> CommStats:
    """Phase 2 (prefill/train): DeepSpeed-Ulysses all-to-all resharding on YZ.

    Schedule (arXiv:2309.14509):
    1) Q/K/V are produced sequence-sharded on YZ (S/P tokens per rank, all
       local heads). A total-exchange along YZ reshards them to head-sharded
       (all S tokens, local-heads/P heads per rank).
    2) Attention core runs fully locally (each rank owns whole heads over the
       full sequence) — no per-step communication, unlike the ring proxy.
    3) The attention output O is total-exchanged back from head-sharded to
       sequence-sharded for the output projection.

    Per-rank send-buffer sizes (the hockney total-exchange helper takes the
    local buffer and applies the (P-1)/P factor internally):
      a2a #1: B * (S/P) * (H/tp_x + 2 * H_kv/tp_x) * d_h * b   (Q + K + V)
      a2a #2: B * (S/P) * (H/tp_x) * d_h * b                    (O)

    Note the S/P in both buffers: per-rank a2a volume scales with the
    *local* token shard and shrinks as P grows. Ring instead delivers the
    full sequence's K/V to every rank — per-rank volume ~ B*S*2*(H_kv/tp_x)
    * d_h * b, roughly constant in P. The ratio ring/Ulysses works out to
    P*H_kv/(H + H_kv), so Ulysses wins once P > ~H/H_kv + 1 (e.g. P > 9
    for an 8:1 GQA model; P > 2 for MHA).

    Constraints (fail fast, not silently degraded):
    - Q heads must split across (tp_x, P): heads_per_x % P == 0.
    - KV heads per X shard must cover P ranks (kv_heads_per_x >= P);
      KV-head duplication under Ulysses is not modeled.

    SWA note: sliding-window layers still exchange their full Q/K/V slabs —
    the a2a reshards activations regardless of the attention mask — so no
    layer-class blending applies here.
    """
    if yz_topology.num_chips <= 1:
        return CommStats()

    p_seq = yz_topology.num_chips
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim

    heads_per_x = num_heads // tp_x
    if heads_per_x % p_seq != 0:
        raise ValueError(
            f"Ulysses sequence parallelism requires query heads per X shard "
            f"({heads_per_x}) to be divisible by the YZ degree ({p_seq})."
        )
    kv_heads_per_x, kv_rep = resolve_kv_heads_per_partition(num_kv_heads=num_kv_heads, tp_x=tp_x)
    if kv_rep > 1 or kv_heads_per_x < p_seq:
        raise NotImplementedError(
            f"Ulysses with KV-head duplication is not modeled: kv_heads_per_x="
            f"{kv_heads_per_x} (replication {kv_rep}) < YZ degree {p_seq}. "
            f"Use seq_parallel_attn_mode: ring for this layout."
        )

    qkv_slab_bytes = (
        float_size_bytes
        * n_batch
        * (n_seq / p_seq)
        * ((heads_per_x + 2 * kv_heads_per_x) * head_dim)
    )
    o_slab_bytes = float_size_bytes * n_batch * (n_seq / p_seq) * (heads_per_x * head_dim)

    return hockney_model_total_exchange_on_nd_topology(
        yz_topology, qkv_slab_bytes, float_size_bytes=float_size_bytes
    ) + hockney_model_total_exchange_on_nd_topology(
        yz_topology, o_slab_bytes, float_size_bytes=float_size_bytes
    )


def _validate_subgroup_collective_pricing(view: LogicalTopology, leg_desc: str) -> None:
    """Safeguard: reject collective pricing on subgroups of a physical ring.

    Views are priced with the 1D collective primitives selected by the
    *physical* dimension's topology type. That is exact for fully-connected
    and switched fabrics (any subset of an FC group is itself fully
    connected), but wrong for a strict subgroup of a physical RING dimension:
    the subgroup's members are strided/interleaved on the ring rather than
    neighbors, so a "ring" collective among them would actually hop across
    non-member ranks (and an FC-priced collective has no direct links at
    all). This model has no representation of that, so fail fast instead of
    silently mispricing. Applied to the phase-2 YZ view in all
    seq-parallel modes (ring, Ulysses, and both hybrid sub-views).

    A logical dimension that spans its physical ring completely
    (`logical size == physical size`) is contiguous by construction and
    passes.
    """
    for dim in view.dimensions:
        tag = _tag_to_str(dim.logical_tag)
        phys = view.get_physical_dimension_for_tag(tag)
        if phys.topology_type == PhysicalTopology1DType.RING and dim.size != phys.size:
            raise ValueError(
                f"Cannot price the {leg_desc} exchange on logical dimension "
                f"'{tag}' (size {dim.size}): it is a strict subgroup of physical "
                f"RING dimension '{phys.name}' (size {phys.size}). Subgroup "
                f"members of a ring are not neighbors, so neither ring nor "
                f"fully-connected collective pricing applies. Map this tag onto "
                f"a fully-connected/switched physical dimension, or size it to "
                f"span the full ring."
            )


def _tp2d_attn_comm_phase2_prefill_hybrid_exchange(
    *,
    yz_topology: LogicalTopology,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    tp_x: int,
    float_size_bytes: float,
    backward: bool,
) -> CommStats:
    """Phase 2 (prefill/train): USP-style hybrid — Ulysses x ring on YZ.

    The YZ axis factors as `P = P_u x P_r` via a tag partition
    (`seq_parallel_ulysses_tags` on the workload spec names the Ulysses
    subgroup tags; the remaining yz tags form the ring axis). Schedule
    (arXiv:2405.07719):

    1) Ulysses a2a inside each `P_u` subgroup converts part of the sequence
       sharding into head sharding: each rank ends with the subgroup's
       combined chunk (`S/P_r` tokens) for `heads_per_x/P_u` heads.
    2) Ring circulation across the `P_r` subgroups covers the remaining
       sequence extent for that head slice — the circulated KV shrinks by
       `P_u` relative to pure ring.
    3) The inverse a2a returns to fully sequence-sharded.

    Pricing reuses the pure-mode functions with shifted arguments:
    - Ulysses leg: priced on the Ulysses sub-view with the *hybrid* local
      token count (`n_seq/P_r` fed in, so slabs use `S/(P_u·P_r)` tokens).
      Its head-divisibility and KV-duplication constraints then bind only
      the `P_u` factor, as USP requires.
    - Ring leg: priced on the ring sub-view with effective head parallelism
      `tp_x·P_u` (the transient head sharding during the attention core),
      which shrinks the circulated KV heads per rank accordingly. SWA
      layer-class blending is inherited.

    Direction: one call prices one direction, selected by the `backward`
    flag — this is never a combined forward+backward total. The two
    directions share this function because backward runs the same two legs
    over the same views, only weighted differently: the ring leg costs 2x
    forward (K/V are re-circulated for the backward kernels, and dK/dV
    circulate back to their owners), while the Ulysses leg costs 1x (its
    two a2a's mirror at equal volume).
    """
    ulysses_tags = workload_spec.seq_parallel_ulysses_tags
    if ulysses_tags is None:
        raise ValueError(
            "seq_parallel_attn_mode=hybrid requires seq_parallel_ulysses_tags on "
            "the workload spec (TrainSpec / InferencePrefillSpec)."
        )
    dim_tags = [_tag_to_str(d.logical_tag) for d in yz_topology.dimensions]
    unknown = set(ulysses_tags) - set(dim_tags)
    if unknown:
        raise ValueError(
            f"seq_parallel_ulysses_tags {sorted(unknown)} not present in the "
            f"attention YZ view dimensions {dim_tags}."
        )
    ring_tags = [t for t in dim_tags if t not in set(ulysses_tags)]
    if not ring_tags:
        raise ValueError(
            "seq_parallel_ulysses_tags covers the entire YZ axis; use "
            "seq_parallel_attn_mode: ulysses instead."
        )

    sharing = getattr(yz_topology, "global_sharing_context", "max_exclusive")
    ulysses_view = yz_topology.create_view(
        view_name="TP_Attn_YZ_Ulysses_View",
        relevant_tags=list(ulysses_tags),
        global_sharing_context=sharing,
    )
    ring_view = yz_topology.create_view(
        view_name="TP_Attn_YZ_Ring_View",
        relevant_tags=ring_tags,
        global_sharing_context=sharing,
    )
    _validate_subgroup_collective_pricing(ulysses_view, "hybrid Ulysses (a2a)")
    _validate_subgroup_collective_pricing(ring_view, "hybrid ring (KV circulation)")

    p_u = ulysses_view.num_chips
    p_r = ring_view.num_chips

    ulysses_leg = _tp2d_attn_comm_phase2_prefill_ulysses_exchange(
        yz_topology=ulysses_view,
        transformer_config=transformer_config,
        workload_spec=workload_spec,
        # Local slabs hold S/(P_u * P_r) tokens; the pure-mode function divides
        # by its own group size (P_u), so pre-divide by P_r here.
        n_seq=n_seq / p_r,
        tp_x=tp_x,
        float_size_bytes=float_size_bytes,
    )
    ring_leg = _tp2d_attn_comm_phase2_prefill_ring_kv_exchange(
        yz_topology=ring_view,
        transformer_config=transformer_config,
        workload_spec=workload_spec,
        n_seq=n_seq,
        # Effective head parallelism during the attention core is tp_x * P_u.
        tp_x=tp_x * p_u,
        float_size_bytes=float_size_bytes,
    )
    ring_factor = 2 if backward else 1
    return ulysses_leg + ring_factor * ring_leg


def _tp2d_attn_comm_phase2_prefill_train(
    *,
    yz_topology: LogicalTopology,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    tp_x: int,
    float_size_bytes: float,
    backward: bool,
) -> CommStats:
    """Phase 2 (prefill/train): dispatch on the spec's seq-parallel mode.

    Backward weighting per mode: ring re-circulates KV and returns dK/dV
    (2x forward); Ulysses mirrors its two a2a's (1x forward); hybrid applies
    each factor to its own leg.
    """
    mode = workload_spec.seq_parallel_attn_mode
    if mode is None:
        raise ValueError(
            "TP-2D prefill/train attention comm requires seq_parallel_attn_mode on "
            "the workload spec (TrainSpec / InferencePrefillSpec)."
        )
    match mode:
        case SeqParallelAttnMode.RING:
            _validate_subgroup_collective_pricing(yz_topology, "ring (KV circulation)")
            fwd = _tp2d_attn_comm_phase2_prefill_ring_kv_exchange(
                yz_topology=yz_topology,
                transformer_config=transformer_config,
                workload_spec=workload_spec,
                n_seq=n_seq,
                tp_x=tp_x,
                float_size_bytes=float_size_bytes,
            )
            return 2 * fwd if backward else fwd
        case SeqParallelAttnMode.ULYSSES:
            _validate_subgroup_collective_pricing(yz_topology, "Ulysses (a2a)")
            return _tp2d_attn_comm_phase2_prefill_ulysses_exchange(
                yz_topology=yz_topology,
                transformer_config=transformer_config,
                workload_spec=workload_spec,
                n_seq=n_seq,
                tp_x=tp_x,
                float_size_bytes=float_size_bytes,
            )
        case SeqParallelAttnMode.HYBRID:
            return _tp2d_attn_comm_phase2_prefill_hybrid_exchange(
                yz_topology=yz_topology,
                transformer_config=transformer_config,
                workload_spec=workload_spec,
                n_seq=n_seq,
                tp_x=tp_x,
                float_size_bytes=float_size_bytes,
                backward=backward,
            )
        case _:
            raise ValueError(f"Unknown seq_parallel_attn_mode: {mode}")


def _tp2d_attn_comm_phase1_qkv_proj_decode(
    *,
    x_topology: LogicalTopology,
    yz_topology: LogicalTopology,
    transformer_config: TransformerConfig,
    workload_spec,
    tp_x: int,
    tp_yz: int,
    float_size_bytes: float,
    layout_trace=NULL_LAYOUT_TRACE,
) -> CommStats:
    """Phase 1 (decode): QKV projection comm for 2D TP with YZ sharding of E.

    Procedure modeled (decode token, n_seq=1):
    1) All-gather input activations along X to reconstruct the full E/tp_yz slice.
    2) All-reduce Q along YZ (Q is needed on all GPUs in the YZ column for attention).
    3) Reduce K and V along YZ to a single owner (KV-cache shard owner for this token).

    Notes:
    - Root selection for K/V (e.g., j = t mod tp_yz) does not change the collective cost in
      our Hockney model, so we use a generic reduce-to-root.
    - This function is GQA-aware: K/V payload uses `num_kv_heads`, while Q payload uses `num_heads`.
    """
    if x_topology.num_chips <= 1 and yz_topology.num_chips <= 1:
        return CommStats()

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim

    comm = CommStats()

    # Step 1: all-gather the layer input along X. The layer input is the
    # residual stream (dim = d_model), not the Q-output activation.
    x_after_ag_bytes = float_size_bytes * n_batch * (d_model / tp_yz)
    ag_x_stats = hockney_model_all_gather_on_nd_topology(
        x_topology, x_after_ag_bytes, float_size_bytes=float_size_bytes
    )
    comm += ag_x_stats
    layout_trace.record_comm(
        "phase1_ag_x",
        "AG",
        f"X({x_topology.num_chips})",
        x_topology.num_chips,
        x_after_ag_bytes,
        ag_x_stats,
        note="input gather: D/Y_eff → D/Y_eff per rank",
    )

    # Step 3: all-reduce Q along YZ (needed replicated within the YZ column).
    # Q's activation dim is num_heads * head_dim = q_proj_out_dim, NOT
    # d_model — they only coincide for non-wide-Q layouts.
    q_bytes = float_size_bytes * n_batch * (q_proj_out_dim / tp_x)
    kv_heads_per_x, _kv_rep = resolve_kv_heads_per_partition(num_kv_heads=num_kv_heads, tp_x=tp_x)
    kv_bytes = float_size_bytes * n_batch * (kv_heads_per_x * head_dim)

    try:
        kv_mode = workload_spec.decode_tp2d_kv_comm_mode
    except AttributeError as e:
        raise ValueError(
            "TP2D decode QKV comm requires InferenceDecodeSpec fields "
            "`decode_tp2d_kv_comm_mode`."
        ) from e

    match kv_mode:
        case "reduce_to_owner":
            # Q is needed replicated within the YZ column for attention.
            ar_q_stats = hockney_model_all_reduce_on_nd_topology(
                yz_topology, q_bytes, float_size_bytes=float_size_bytes
            )
            comm += ar_q_stats
            layout_trace.record_comm(
                "phase1_ar_yz_q",
                "AR",
                f"YZ({yz_topology.num_chips})",
                yz_topology.num_chips,
                q_bytes,
                ar_q_stats,
                note="Q replicate across YZ",
            )

            # K/V must end up on a single owner rank per YZ column (sharded KV-cache).
            # Always fuse K and V into a single reduce-to-owner to save latency.
            red_kv_stats = hockney_model_reduce_on_nd_topology(
                yz_topology, 2 * kv_bytes, float_size_bytes=float_size_bytes
            )
            comm += red_kv_stats
            layout_trace.record_comm(
                "phase1_reduce_yz_kv",
                "Reduce",
                f"YZ({yz_topology.num_chips})",
                yz_topology.num_chips,
                2 * kv_bytes,
                red_kv_stats,
                note="KV reduce-to-owner for KV-cache",
            )

        case "allreduce_and_drop":
            # Latency-optimized alternative:
            # Piggyback K/V on the existing YZ all-reduce by expanding its payload. This models
            # that (Q, K, V) are replicated across YZ after the collective; non-owner ranks then
            # discard K/V (drop is assumed free here).
            ar_qkv_stats = hockney_model_all_reduce_on_nd_topology(
                yz_topology, q_bytes + 2 * kv_bytes, float_size_bytes=float_size_bytes
            )
            comm += ar_qkv_stats
            layout_trace.record_comm(
                "phase1_ar_yz_qkv",
                "AR",
                f"YZ({yz_topology.num_chips})",
                yz_topology.num_chips,
                q_bytes + 2 * kv_bytes,
                ar_qkv_stats,
                note="Q+KV allreduce-and-drop",
            )

        case _:
            raise ValueError(f"Unknown decode_tp2d_kv_comm_mode: {kv_mode}")
    return comm


class TensorParallelStrategy(ABC):
    """Abstract base class for tensor parallel communication strategies."""

    @abstractmethod
    def forward_comm_training(
        self, transformer_config: TransformerConfig, moe_config, workload_spec, n_seq: int, **kwargs
    ) -> CommStats:
        """Forward pass communication for TRAINING stage."""
        pass

    @abstractmethod
    def forward_comm_prefill(
        self, transformer_config: TransformerConfig, moe_config, workload_spec, n_seq: int, **kwargs
    ) -> CommStats:
        """Forward pass communication for INFERENCE_PREFILL stage."""
        pass

    @abstractmethod
    def forward_comm_decode(
        self, transformer_config: TransformerConfig, moe_config, workload_spec, n_seq: int, **kwargs
    ) -> CommStats:
        """Forward pass communication for INFERENCE_DECODE stage."""
        pass

    @abstractmethod
    def backward_comm_training(
        self, transformer_config: TransformerConfig, moe_config, workload_spec, n_seq: int, **kwargs
    ) -> CommStats:
        """Backward pass communication for TRAINING stage."""
        pass


@dataclass(frozen=True)
class AttnForwardCommPhases:
    """Structured breakdown of attention forward communication into phases."""

    phase1_input_proj: CommStats
    phase2_attn_core: CommStats
    phase3_output_proj: CommStats

    @property
    def total(self) -> CommStats:
        return self.phase1_input_proj + self.phase2_attn_core + self.phase3_output_proj

    @property
    def projection_total(self) -> CommStats:
        return self.phase1_input_proj + self.phase3_output_proj

    @property
    def attn_core_total(self) -> CommStats:
        return self.phase2_attn_core


class TP1DAttnStrategy(TensorParallelStrategy):
    """Models 1D TP for the attention block."""

    def forward_comm_prefill_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        volume = float_size * n_batch * n_seq * d_model

        phase1 = CommStats()  # QKV projection is column-parallel (no AR)
        phase2 = CommStats()  # no explicit 1D comm modeled for core attention
        phase3 = hockney_model_all_reduce_on_nd_topology(
            proj_tp_topology, volume, float_size_bytes=float_size
        )
        return AttnForwardCommPhases(
            phase1_input_proj=phase1, phase2_attn_core=phase2, phase3_output_proj=phase3
        )

    def forward_comm_training_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        # Same structure as prefill for this strategy.
        return self.forward_comm_prefill_phases(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            proj_tp_topology=proj_tp_topology,
            **kwargs,
        )

    def forward_comm_decode_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        # Decode: same phase structure (AR is priced on output projection).
        return self.forward_comm_prefill_phases(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            proj_tp_topology=proj_tp_topology,
            **kwargs,
        )

    def _forward_comm_prefill_train(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        # Structured into phases (keep formulas unchanged for now):
        # 1) Input projection
        # 2) Attention computation
        # 3) Output projection
        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        volume = float_size * n_batch * n_seq * d_model

        # Phase 1 (projection domain): input projection
        #
        # For standard 1D TP attention (Megatron-style), the QKV projection is typically
        # column-parallel: each rank produces its local Q/K/V shards without an all-reduce.
        # The single "attention block" all-reduce is usually associated with the output
        # projection (row-parallel) partial sums.
        comm_phase1_input_proj = CommStats()
        comm_phase2_attn = CommStats()
        # Phase 3 (projection domain): output projection
        # Model a single all-reduce on the output activation volume.
        comm_phase3_output_proj = hockney_model_all_reduce_on_nd_topology(
            proj_tp_topology, volume, float_size_bytes=float_size
        )

        return comm_phase1_input_proj + comm_phase2_attn + comm_phase3_output_proj

    def forward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_prefill_train(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            attn_tp_topology=attn_tp_topology,
            proj_tp_topology=proj_tp_topology,
            **kwargs,
        )

    def forward_comm_prefill(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_prefill_train(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            attn_tp_topology=attn_tp_topology,
            proj_tp_topology=proj_tp_topology,
            **kwargs,
        )

    def forward_comm_decode(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        # Decode: allow separate projection TP vs attention TP.
        # For now, boundary transfers are placeholders (warn + zero cost).
        # Both TP domains are required (explicit, no None-default behavior).

        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        volume = float_size * n_batch * n_seq * d_model

        # Phase 1 (projection domain): input projection (no all-reduce in standard 1D TP)
        comm_phase1_input_proj = CommStats()

        # Phase 2 (attention domain): attention compute (no explicit 1D comm modeled here)
        comm_phase2_attn = CommStats()

        # Phase 3 (projection domain): output projection
        comm_phase3_output_proj = hockney_model_all_reduce_on_nd_topology(
            proj_tp_topology, volume, float_size_bytes=float_size
        )

        return comm_phase1_input_proj + comm_phase2_attn + comm_phase3_output_proj

    def backward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology = None,
        **kwargs,
    ) -> CommStats:
        # Backward gradient comm is same as forward activation comm for this strategy.
        fwd_comm = self.forward_comm_training(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            attn_tp_topology=attn_tp_topology,
            **kwargs,
        )
        # Structured into phases (keep formulas unchanged for now):
        # 1) Input projection grads (reuse forward_comm volume proxy)
        # 2) Attention backward (none modeled explicitly here)
        # 3) Layernorm backward comm
        comm_phase1 = fwd_comm
        comm_phase2 = CommStats()
        # Layernorm communication is part of the attention block's backward pass
        layernorm_comm_vol = 2 * 2 * workload_spec.float_size_bytes
        comm_phase3 = hockney_model_all_reduce_on_nd_topology(
            attn_tp_topology,
            layernorm_comm_vol,
            float_size_bytes=workload_spec.float_size_bytes,
        )
        return comm_phase1 + comm_phase2 + comm_phase3


class TP1DMoEStrategy(TensorParallelStrategy):
    """Models 1D TP for the MoE/FFN block."""

    def _forward_comm_all_modes(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        layout_trace = kwargs.pop("layout_trace", NULL_LAYOUT_TRACE)

        # Structured into phases:
        # 1) Expert-input replication comm (All-Reduce proxy)
        # 2) Expert exchange comm (All-Gather proxy)
        # 3) Output projection (none modeled explicitly here)
        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes

        comm_phase1 = CommStats(0, 0, 0, 0)
        comm_phase2 = CommStats(0, 0, 0, 0)
        comm_phase3 = CommStats(0, 0, 0, 0)
        num_shared = moe_config.num_shared_experts_per_dp_rank
        if num_shared > 0:
            raise NotImplementedError("Shared experts not implemented in 1D TP MoE strategy.")

        total_num_routed_experts = transformer_config.num_regular_experts
        if total_num_routed_experts > 0:
            tokens_per_set = (
                workload_spec.get_resolved_batches().batch_size_per_full_expert_set * n_seq
            )
            traffic_mult = moe_config.regular_expert_traffic_multiplier
            tokens_per_routed_expert = tokens_per_set / total_num_routed_experts
            # max volume that an expert could receive, scaled by traffic multiplier
            token_volume_per_routed_expert = (
                tokens_per_routed_expert * traffic_mult * d_model * float_size
            )
            # Scale comm by number of collocated experts per chip
            token_volume_per_routed_expert *= moe_config.num_colocated_experts
            # TP core collectives (AR/AG) are modeled on routed token payload processed
            # by expert TP shards (local + remote). Placement locality/remote_fraction is
            # modeled in EP A2A path, not in TP core collectives.
            routed_tp_volume = token_volume_per_routed_expert

            # Regular TP1D combine path: all-reduce across TP shards.
            # Payload uses routed_remote_volume (routed + non-local token payload proxy),
            # which we use consistently for TP-MoE communication scaling.
            ar_tp_stats = hockney_model_all_reduce_on_nd_topology(
                moe_tp_topology,
                routed_tp_volume,
                float_size_bytes=float_size,
            )
            comm_phase1 += ar_tp_stats
            layout_trace.record_comm(
                "moe_1d_ar_tp",
                "AR",
                f"MoE_TP({moe_tp_topology.num_chips})",
                moe_tp_topology.num_chips,
                routed_tp_volume,
                ar_tp_stats,
                note="1D TP MoE: expert AR across TP shards",
            )

            router_sync_mode = moe_config.router_sync_mode
            if (
                router_sync_mode == MoERouterSyncMode.SHARDED_ROUTER
                and total_num_routed_experts > 1
            ):
                if attn_tp_topology is None:
                    raise ValueError(
                        "TP1D MoE sharded_router requires attn_tp_topology for router synchronization."
                    )
                tokens_local = workload_spec.get_resolved_batches().batch_size_per_dp * n_seq
                router_bytes = float_size * tokens_local * total_num_routed_experts
                ar_router_stats = hockney_model_all_reduce_on_nd_topology(
                    attn_tp_topology, router_bytes, float_size_bytes=float_size
                )
                comm_phase1 += ar_router_stats
                layout_trace.record_comm(
                    "moe_router_ar",
                    "AR",
                    f"AttnTP({attn_tp_topology.num_chips})",
                    attn_tp_topology.num_chips,
                    router_bytes,
                    ar_router_stats,
                    note="1D sharded router: sync logits",
                )
            elif router_sync_mode == MoERouterSyncMode.REPLICATED_ROUTER:
                pass
            else:
                raise ValueError(f"Unsupported router_sync_mode: {router_sync_mode}")
            if bool(moe_config.tp_moe_a2a_replication_optimization_enabled):
                # A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize
                # Mixture-of-Experts Training
                # https://arxiv.org/pdf/2303.06318
                ag_opt_stats = hockney_model_all_gather_on_nd_topology(
                    moe_tp_topology, routed_tp_volume, float_size_bytes=float_size
                )
                comm_phase2 += ag_opt_stats
                layout_trace.record_comm(
                    "moe_1d_ag_opt",
                    "AG",
                    f"MoE_TP({moe_tp_topology.num_chips})",
                    moe_tp_topology.num_chips,
                    routed_tp_volume,
                    ag_opt_stats,
                    note="1D TP hybrid optimization AG",
                )

        return comm_phase1 + comm_phase2 + comm_phase3

    def forward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_all_modes(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            moe_tp_topology=moe_tp_topology,
            attn_tp_topology=attn_tp_topology,
            **kwargs,
        )

    def forward_comm_prefill(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_all_modes(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            moe_tp_topology=moe_tp_topology,
            attn_tp_topology=attn_tp_topology,
            **kwargs,
        )

    def forward_comm_decode(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_all_modes(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            moe_tp_topology=moe_tp_topology,
            attn_tp_topology=attn_tp_topology,
            **kwargs,
        )

    def backward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        **kwargs,
    ) -> CommStats:
        return self.forward_comm_training(
            transformer_config,
            moe_config,
            workload_spec,
            n_seq,
            moe_tp_topology=moe_tp_topology,
            **kwargs,
        )


class TP2DAttnStrategy(TensorParallelStrategy):
    """Models 2D TP for the attention block."""

    def forward_comm_training_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        """Prefill/training TP-2D attention comm with a coherent token layout.

        Phases are the three stages of one attention layer pass:
        phase 1 = QKV input projections, phase 2 = attention core
        (softmax(QK^T)V), phase 3 = output projection (W_O).

        The whole layer assumes one sequence-sharded layout: tokens live as
        [B, S/tp_yz, *] slabs on every rank throughout phases 1-3. Phase 1/3
        collectives are derived from the per-axis projection roles
        (`tp2d_proj_axis_roles`); phase 2 is the sequence-parallel attention
        core (`seq_parallel_attn_mode`: ring or Ulysses) on the attention YZ
        axis.
        """
        float_size = workload_spec.float_size_bytes

        # Attention TP sizes (sequence is sharded on the attention YZ axis).
        shape_attn = getattr(attn_tp_topology, "shape", [])
        tp_x_attn = 1
        for idx in x_dims:
            tp_x_attn *= shape_attn[idx]
        tp_yz_attn = 1
        for idx in yz_dims:
            tp_yz_attn *= shape_attn[idx]

        # Projection TP sizes
        shape_proj = getattr(proj_tp_topology, "shape", [])
        tp_x_proj = 1
        for idx in proj_x_dims:
            tp_x_proj *= shape_proj[idx]

        phase1, phase3 = _tp2d_attn_proj_comm_prefill_train_forward(
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            proj_x_topology=proj_x_topology,
            proj_yz_topology=proj_yz_topology,
            tp_x_proj=tp_x_proj,
            seq_shard=tp_yz_attn,
            float_size_bytes=float_size,
        )
        phase2 = _tp2d_attn_comm_phase2_prefill_train(
            yz_topology=yz_topology,
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            tp_x=tp_x_attn,
            float_size_bytes=float_size,
            backward=False,
        )
        return AttnForwardCommPhases(
            phase1_input_proj=phase1, phase2_attn_core=phase2, phase3_output_proj=phase3
        )

    def backward_comm_training_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        """Training backward comm: mirrored projection collectives + attention core.

        Phase 2 by mode:
        - ring: K/V are re-circulated for the backward kernels and dK/dV
          circulate back to their owners — 2x the forward ring volume.
        - Ulysses: backward mirrors the two forward all-to-alls (dO reshard
          in, dQ/dK/dV reshard out) at equal volume — 1x the forward cost.
        - hybrid: each leg gets its own factor (ring 2x, Ulysses 1x).
        """
        float_size = workload_spec.float_size_bytes

        shape_attn = getattr(attn_tp_topology, "shape", [])
        tp_x_attn = 1
        for idx in x_dims:
            tp_x_attn *= shape_attn[idx]
        tp_yz_attn = 1
        for idx in yz_dims:
            tp_yz_attn *= shape_attn[idx]

        shape_proj = getattr(proj_tp_topology, "shape", [])
        tp_x_proj = 1
        for idx in proj_x_dims:
            tp_x_proj *= shape_proj[idx]

        phase1, phase3 = _tp2d_attn_proj_comm_train_backward(
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            proj_x_topology=proj_x_topology,
            proj_yz_topology=proj_yz_topology,
            tp_x_proj=tp_x_proj,
            seq_shard=tp_yz_attn,
            float_size_bytes=float_size,
        )
        phase2 = _tp2d_attn_comm_phase2_prefill_train(
            yz_topology=yz_topology,
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            tp_x=tp_x_attn,
            float_size_bytes=float_size,
            backward=True,
        )
        return AttnForwardCommPhases(
            phase1_input_proj=phase1, phase2_attn_core=phase2, phase3_output_proj=phase3
        )

    def forward_comm_prefill_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        return self.forward_comm_training_phases(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            x_topology=x_topology,
            yz_topology=yz_topology,
            proj_tp_topology=proj_tp_topology,
            proj_x_dims=proj_x_dims,
            proj_yz_dims=proj_yz_dims,
            proj_x_topology=proj_x_topology,
            proj_yz_topology=proj_yz_topology,
            **kwargs,
        )

    def forward_comm_decode_phases(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> AttnForwardCommPhases:
        layout_trace = kwargs.pop("layout_trace", NULL_LAYOUT_TRACE)

        shape_attn = getattr(attn_tp_topology, "shape", [])
        tp_x_attn = 1
        for idx in x_dims:
            tp_x_attn *= shape_attn[idx]

        shape_proj = getattr(proj_tp_topology, "shape", [])
        tp_x_proj = 1
        for idx in proj_x_dims:
            tp_x_proj *= shape_proj[idx]
        tp_yz_proj = 1
        for idx in proj_yz_dims:
            tp_yz_proj *= shape_proj[idx]

        float_size = workload_spec.float_size_bytes
        d_model = transformer_config.d_model
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        n_heads = transformer_config.num_heads
        n_kv_heads = transformer_config.num_kv_heads
        d_head = transformer_config.head_dim
        s = n_batch * n_seq

        layout_trace.record(
            "layer_input",
            "Residual + LayerNorm",
            TensorRecord(
                "x",
                {"B": s, "D": d_model},
                {"D": P("sharded", ("X", "Y_eff")), "B": P("replicated", ("Y",))},
                dtype_bytes=float_size,
            ),
            note=f"actual: D/(X·Y_eff) = {d_model}/({tp_x_proj}·{tp_yz_proj})"
            f" = {d_model // (tp_x_proj * tp_yz_proj)}",
        )

        phase1 = _tp2d_attn_comm_phase1_qkv_proj_decode(
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            x_topology=proj_x_topology,
            yz_topology=proj_yz_topology,
            tp_x=tp_x_proj,
            tp_yz=tp_yz_proj,
            float_size_bytes=float_size,
            layout_trace=layout_trace,
        )

        # LN1 (pre-attention LayerNorm) — gather-then-normalize approach.
        # After AG(X) each rank has [B, D/Y_eff]. LN1 needs stats across full D,
        # so a tiny stats AR across Y_eff is required (no router to fuse with).
        ln1_stats_bytes = 2 * float_size * s
        comm_ln1 = hockney_model_all_reduce_on_nd_topology(
            proj_yz_topology, ln1_stats_bytes, float_size_bytes=float_size
        )
        phase1 += comm_ln1
        layout_trace.record_comm(
            "ln1_stats_ar_yz",
            "AR",
            f"YZ({proj_yz_topology.num_chips})",
            proj_yz_topology.num_chips,
            ln1_stats_bytes,
            comm_ln1,
            note=f"LN1 stats: mean+var across Y_eff={tp_yz_proj}",
        )

        layout_trace.record(
            "phase1_x_gathered",
            "After AG(X) + LN1",
            TensorRecord(
                "x_proj_in",
                {"B": s, "D": d_model},
                {"D": P("sharded", ("Y_eff",)), "B": P("replicated", ("Y",))},
                dtype_bytes=float_size,
            ),
            note=f"actual: AG(X) → D/Y_eff = {d_model // tp_yz_proj}",
        )

        phase2 = _tp2d_attn_comm_phase2_decode_yz_reductions(
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            yz_topology=yz_topology,
            tp_x=tp_x_attn,
            float_size_bytes=float_size,
            layout_trace=layout_trace,
        )

        layout_trace.record(
            "after_qkv_proj",
            "Q after proj + AR(Y_eff)",
            TensorRecord(
                "Q",
                {"B": s, "H": n_heads, "Dh": d_head},
                {"H": P("sharded", ("X",)), "B": P("replicated", ("Y",))},
                dtype_bytes=float_size,
            ),
            note=f"actual: heads/X = {n_heads // tp_x_attn}",
        )
        kv_per_x, _kv_rep = resolve_kv_heads_per_partition(num_kv_heads=n_kv_heads, tp_x=tp_x_attn)
        kv_dim = kv_per_x * d_head
        layout_trace.record(
            "after_qkv_proj",
            "K,V → reduce-to-owner(Y_eff)",
            TensorRecord("KV", {"B": s, "Dkv": kv_dim}, {}, dtype_bytes=float_size),
            note=f"actual: GQA {n_kv_heads} KV-heads, {kv_per_x} per X-rank",
        )

        context_len = getattr(workload_spec, "prefill_seq_len", 1)
        heads_local = n_heads // tp_x_attn
        # One kv_cache_read / attention_scores record per attention layer
        # class (full vs sliding window); mirrors the expected trace in
        # `decode_layout_trace._trace_attention_stages`.
        attn_classes = attention_layer_classes(transformer_config)
        for layer_class in attn_classes:
            ctx_cls = layer_class.kv_len(context_len)
            if len(attn_classes) == 1:
                name_suffix = ""
                class_note = ""
            else:
                name_suffix = f"[{layer_class.kind.value}]"
                class_note = (
                    f" [{layer_class.kind.value}: {layer_class.layer_fraction:.0%} of layers]"
                )
            layout_trace.record(
                "kv_cache_read",
                "Read KV-cache for current context",
                TensorRecord(
                    f"KV_cache{name_suffix}",
                    {"Dkv": kv_dim, "Ctx": ctx_cls},
                    {},
                    dtype_bytes=float_size,
                ),
                note=(
                    f"actual: {kv_per_x} KV-heads × ctx={ctx_cls} × d_head={d_head}. "
                    f"Per rank = {kv_per_x * ctx_cls * d_head * float_size} B{class_note}"
                ),
            )
            layout_trace.record(
                "attention_scores",
                "QK^T + online softmax (FlashDecoding)",
                TensorRecord(
                    f"scores{name_suffix}",
                    {"H": n_heads, "B": s, "Ctx": ctx_cls},
                    {"H": P("sharded", ("X",))},
                    dtype_bytes=float_size,
                ),
                note=(
                    f"actual: [{heads_local}, {s}, {ctx_cls}] per rank. "
                    f"GQA: {heads_local} Q-heads share {kv_per_x} KV-head(s) "
                    f"(ratio {heads_local // max(1, kv_per_x)}:1). "
                    f"Online softmax: streaming.{class_note}"
                ),
            )
        layout_trace.record(
            "attn_v_output",
            "Attention × V partial sums (per Y-shard)",
            TensorRecord(
                "attn_partial",
                {"B": s, "H": n_heads, "Dh": d_head},
                {"H": P("sharded", ("X",)), "B": P("partial", ("Y",))},
                dtype_bytes=float_size,
            ),
            note=(
                f"actual: [{s}, {heads_local}, {d_head}] per rank. "
                f"Partial over Y — needs AR(Y)."
            ),
        )

        layout_trace.record(
            "after_attention",
            "Attention output after AR(Y)",
            TensorRecord(
                "attn_out",
                {"B": s, "D": d_model},
                {"D": P("sharded", ("X",)), "B": P("replicated", ("Y",))},
                dtype_bytes=float_size,
            ),
            note=f"actual: AR(Y) → D/X = {d_model // tp_x_attn}",
        )

        volume_bytes = float_size * n_batch * n_seq * d_model

        # Phase 3 for decode: AG_YZ is redundant — after Phase 2 AR(Y), data is
        # already Y-replicated, so output proj matmul works on local [B, D/X]
        # with W_O shard [D/X, D/Y_eff].  Only RS_X is needed.
        # ag_yz_stats = hockney_model_all_gather_on_nd_topology(
        #     proj_yz_topology, volume_bytes / tp_x_proj, float_size_bytes=float_size
        # )
        x_before_rs_bytes = volume_bytes / tp_yz_proj
        rs_x_stats = hockney_model_reduce_scatter_on_nd_topology(
            proj_x_topology, x_before_rs_bytes, float_size_bytes=float_size
        )
        layout_trace.record_comm(
            "phase3_rs_x",
            "RS",
            f"X({proj_x_topology.num_chips})",
            proj_x_topology.num_chips,
            x_before_rs_bytes,
            rs_x_stats,
            note="output proj: scatter across X heads",
        )
        phase3 = rs_x_stats

        layout_trace.record(
            "after_output_proj",
            "Output proj + RS(X)",
            TensorRecord(
                "proj_out",
                {"B": s, "D": d_model},
                {"D": P("sharded", ("X", "Y_eff")), "B": P("replicated", ("Y",))},
                dtype_bytes=float_size,
            ),
            note=f"actual: RS(X) → D/(X·Y_eff) = {d_model // (tp_x_proj * tp_yz_proj)}",
        )

        return AttnForwardCommPhases(
            phase1_input_proj=phase1, phase2_attn_core=phase2, phase3_output_proj=phase3
        )

    def forward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        # Delegate to the phase-structured method (single source of truth).
        return self.forward_comm_training_phases(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            x_topology=x_topology,
            yz_topology=yz_topology,
            proj_tp_topology=proj_tp_topology,
            proj_x_dims=proj_x_dims,
            proj_yz_dims=proj_yz_dims,
            proj_x_topology=proj_x_topology,
            proj_yz_topology=proj_yz_topology,
            **kwargs,
        ).total

    def forward_comm_prefill(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        # Share with training (for now); keep decode separate.
        return self.forward_comm_training(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            x_topology=x_topology,
            yz_topology=yz_topology,
            proj_tp_topology=proj_tp_topology,
            proj_x_dims=proj_x_dims,
            proj_yz_dims=proj_yz_dims,
            proj_x_topology=proj_x_topology,
            proj_yz_topology=proj_yz_topology,
            **kwargs,
        )

    def forward_comm_decode(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        proj_tp_topology: LogicalTopology,
        proj_x_dims: list[int],
        proj_yz_dims: list[int],
        proj_x_topology: LogicalTopology,
        proj_yz_topology: LogicalTopology,
        **kwargs,
    ) -> CommStats:
        # All decode topology inputs are required (explicit, no None-default behavior).

        shape = getattr(attn_tp_topology, "shape", [])
        tp_x_attn = 1
        for idx in x_dims:
            tp_x_attn *= shape[idx]
        tp_yz_attn = 1
        for idx in yz_dims:
            tp_yz_attn *= shape[idx]

        tp_x_proj = 1
        for idx in proj_x_dims:
            tp_x_proj *= shape[idx]
        tp_yz_proj = 1
        for idx in proj_yz_dims:
            tp_yz_proj *= shape[idx]

        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp

        # NOTE: projection_groups is implemented as a *global* policy by creating derived
        # TP views with smaller effective YZ degrees. That way, comm/compute/memory/KV-cache are
        # all affected consistently. Therefore, the TP strategy itself should not special-case it
        # here; it should simply consume the provided (possibly-derived) topologies.
        yz_topology_eff = yz_topology
        tp_yz_proj_eff = tp_yz_proj
        proj_yz_topology_eff = proj_yz_topology

        # Phase 1 (projection domain): QKV projection comm
        comm1 = _tp2d_attn_comm_phase1_qkv_proj_decode(
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            x_topology=proj_x_topology,
            yz_topology=proj_yz_topology_eff,
            tp_x=tp_x_proj,
            tp_yz=tp_yz_proj_eff,
            float_size_bytes=float_size,
        )

        # Phase 2 (attention domain): decode-only YZ reductions
        comm2 = _tp2d_attn_comm_phase2_decode_yz_reductions(
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            yz_topology=yz_topology_eff,
            tp_x=tp_x_attn,
            float_size_bytes=float_size,
        )

        # Phase 3 (projection domain): output projection comm. Two distinct
        # volumes: the AG step ships the q_proj_out_dim-based attention output,
        # the RS step ships the d_model-based residual output. Equal for
        # non-wide-Q layouts; differ only when num_heads · head_dim ≠ d_model.
        num_heads = transformer_config.num_heads
        head_dim = transformer_config.head_dim
        q_proj_out_dim = num_heads * head_dim
        output_volume_bytes = float_size * n_batch * n_seq * d_model
        input_volume_bytes = float_size * n_batch * n_seq * q_proj_out_dim
        comm3 = _tp2d_attn_comm_phase3_output_proj(
            x_topology=proj_x_topology,
            yz_topology=proj_yz_topology_eff,
            input_volume_bytes=input_volume_bytes,
            output_volume_bytes=output_volume_bytes,
            tp_x=tp_x_proj,
            tp_yz=tp_yz_proj_eff,
            float_size_bytes=float_size,
        )

        return comm1 + comm2 + comm3

    def backward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        attn_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        **kwargs,
    ) -> CommStats:
        # Delegate to the phase-structured backward (mirrored projection
        # collectives + mode-dependent attention-core cost).
        return self.backward_comm_training_phases(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            **kwargs,
        ).total


class TP2DMoEStrategy(TensorParallelStrategy):
    """Models 2D TP for the MoE/FFN block."""

    def _forward_comm_all_modes(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        *,
        moe_tp_topology: LogicalTopology,
        x_dims: list[int],
        yz_dims: list[int],
        x_topology: LogicalTopology,
        yz_topology: LogicalTopology,
        attn_tp_topology: LogicalTopology,
        proj_x_topology: LogicalTopology | None,
        proj_yz_topology: LogicalTopology | None,
        expert_set_topology: LogicalTopology | None,
        is_decode: bool,
        **kwargs,
    ) -> CommStats:
        layout_trace = kwargs.pop("layout_trace", NULL_LAYOUT_TRACE)

        if moe_tp_topology is None:
            raise ValueError("TP2DMoEStrategy requires moe_tp_topology.")
        if not x_dims or not yz_dims:
            raise ValueError("TP2DMoEStrategy requires explicit x_dims and yz_dims.")
        if x_topology is None or yz_topology is None:
            raise ValueError("TP2DMoEStrategy requires x_topology and yz_topology.")
        if attn_tp_topology is None:
            raise ValueError("TP2DMoEStrategy requires attn_tp_topology.")
        # Decode always uses the deterministic projection-to-MoE layout
        # handoff. For D=1 it places the expert set's local tokens; for D>1 it
        # applies the same handoff independently to each feeder before the
        # remote-owner A2Av. Prefill/training retain the routed exchange path.
        exact_decode_handoff = is_decode
        single_domain_decode = is_decode and moe_config.attn_domains_per_full_expert_set == 1
        full_sets_per_attn_domain = moe_config.full_expert_sets_per_attn_domain
        if exact_decode_handoff:
            if full_sets_per_attn_domain is None:
                raise ValueError("TP2D MoE decode requires full_expert_sets_per_attn_domain.")
            if single_domain_decode and full_sets_per_attn_domain > 1:
                if expert_set_topology is None:
                    raise ValueError("Multiple Y-aligned expert sets require expert_set_topology.")
                if expert_set_topology.num_chips != full_sets_per_attn_domain:
                    raise ValueError(
                        f"expert_set_topology ({expert_set_topology.num_chips}) must equal "
                        f"full_expert_sets_per_attn_domain ({full_sets_per_attn_domain})"
                    )
        if exact_decode_handoff and (proj_x_topology is None or proj_yz_topology is None):
            raise ValueError(
                "TP2D MoE decode requires proj_x_topology and "
                "proj_yz_topology for the output-projection layout handoff."
            )

        d_model = transformer_config.d_model
        float_size = workload_spec.float_size_bytes
        d_ff = transformer_config.mlp_hidden_size

        shape = getattr(moe_tp_topology, "shape", [])
        tp_x = 1
        for idx in x_dims:
            tp_x *= shape[idx]
        tp_yz = 1
        for idx in yz_dims:
            tp_yz *= shape[idx]

        resolved_batches = workload_spec.get_resolved_batches()
        if exact_decode_handoff:
            assert full_sets_per_attn_domain is not None
            if resolved_batches.complete_expert_set_count != moe_config.num_complete_expert_sets:
                raise ValueError(
                    "Resolved expert-set count does not match expert placement: "
                    f"{resolved_batches.complete_expert_set_count} != "
                    f"{moe_config.num_complete_expert_sets}"
                )
            token_scopes = resolve_moe_token_scopes(
                resolved_batches=resolved_batches,
                tokens_per_sequence=n_seq,
                attn_domains_per_full_expert_set=(moe_config.attn_domains_per_full_expert_set),
                full_expert_sets_per_attn_domain=full_sets_per_attn_domain,
            )
            tokens_local = token_scopes.tokens_per_attention_domain
            tokens_per_set = token_scopes.tokens_per_full_expert_set
            tokens_per_feeder_set = token_scopes.tokens_per_feeder_set
        else:
            tokens_local = resolved_batches.batch_size_per_dp * n_seq
            tokens_per_set = resolved_batches.batch_size_per_full_expert_set * n_seq
            tokens_per_feeder_set = tokens_per_set
        # MoE-stage trace should reflect per-full-expert-set token ownership.
        # Decode traces one feeder→set edge; prefill/training retain full-set scope.
        s = tokens_per_feeder_set if exact_decode_handoff else tokens_per_set

        _moe_d_dims = []
        if tp_x > 1:
            _moe_d_dims.append("MoE_X")
        if tp_yz > 1:
            _moe_d_dims.append("MoE_Y")
        _moe_d_part = (
            {"D": P("sharded", tuple(_moe_d_dims)), "B": P("replicated", ("Y",))}
            if _moe_d_dims
            else {"B": P("replicated", ("Y",))}
        )
        _moe_d_local = d_model // (tp_x * tp_yz)
        groups_per_set = moe_config.num_colocated_groups_per_set
        if groups_per_set <= 0:
            raise ValueError(f"num_colocated_groups_per_set must be > 0, got {groups_per_set}")
        if attn_tp_topology is None:
            raise ValueError("TP2DMoEStrategy requires attn_tp_topology for combine tracing.")
        attn_tp_xpus = attn_tp_topology.num_chips
        if attn_tp_xpus <= 0:
            raise ValueError(f"attn_tp_topology.num_chips must be > 0, got {attn_tp_xpus}")

        if single_domain_decode:
            assert proj_x_topology is not None
            assert proj_yz_topology is not None
            target_x = proj_x_topology.num_chips
            target_y_eff = proj_yz_topology.num_chips
            if target_x % tp_x != 0 or target_y_eff % tp_yz != 0:
                raise ValueError(
                    "Exact single-domain combine requires projection layout to be a "
                    f"coarsening of MoE layout: projection={target_x}×{target_y_eff}, "
                    f"MoE={tp_x}×{tp_yz}"
                )
            moe_to_residual_layout_quotient_count = (target_x // tp_x) * (target_y_eff // tp_yz)
            assert full_sets_per_attn_domain is not None
            global_projection_replicas = attn_tp_xpus // (target_x * target_y_eff)
            if global_projection_replicas % full_sets_per_attn_domain != 0:
                raise ValueError(
                    f"Global projection replicas ({global_projection_replicas}) must be "
                    f"divisible by expert sets ({full_sets_per_attn_domain})"
                )
            projection_replicas = global_projection_replicas // full_sets_per_attn_domain
            if groups_per_set != moe_to_residual_layout_quotient_count * projection_replicas:
                raise ValueError(
                    f"Expert groups ({groups_per_set}) != MoE-to-residual layout "
                    "quotients × projection replicas "
                    f"({moe_to_residual_layout_quotient_count} × "
                    f"{projection_replicas})"
                )
            handoff_mode = (
                "exact RS(moe_to_residual_layout_quotient="
                f"{moe_to_residual_layout_quotient_count}) + "
                f"AR(projection_replicas={projection_replicas})"
            )
            _post_combine_d_part = {
                "D": P("sharded", ("X", "Y_eff")),
                "B": P("replicated", ("Y",)),
            }
            _post_combine_d_local = d_model // (target_x * target_y_eff)
            _post_combine_issue_flags: tuple[str, ...] = ()
        else:
            target_x = tp_x
            target_y_eff = tp_yz
            handoff_mode = "pass-specific MoE layout"
            _post_combine_d_part = _moe_d_part
            _post_combine_d_local = _moe_d_local
            _post_combine_issue_flags = ()
        num_complete_sets = moe_config.num_complete_expert_sets
        if num_complete_sets < 1:
            raise ValueError(f"num_complete_expert_sets must be > 0, got {num_complete_sets}")
        full_set_xpus = moe_config.full_expert_set_xpu_count
        if full_set_xpus <= 0:
            raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_set_xpus}")
        if attn_tp_xpus >= full_set_xpus:
            if attn_tp_xpus % full_set_xpus != 0:
                raise ValueError(
                    f"attn_tp_xpus ({attn_tp_xpus}) must be divisible by "
                    f"full_expert_set_xpu_count ({full_set_xpus})"
                )
            num_combine_sets = attn_tp_xpus // full_set_xpus
        else:
            num_combine_sets = 1

        if exact_decode_handoff:
            assert proj_x_topology is not None
            assert proj_yz_topology is not None
            proj_x = proj_x_topology.num_chips
            proj_y = proj_yz_topology.num_chips
            _proj_d_dims = []
            if proj_x > 1:
                _proj_d_dims.append("X")
            if proj_y > 1:
                _proj_d_dims.append("Y_eff")
            _proj_d_part = (
                {"D": P("sharded", tuple(_proj_d_dims)), "B": P("replicated", ("Y",))}
                if _proj_d_dims
                else {"B": P("replicated", ("Y",))}
            )
            layout_trace.record(
                "moe_input",
                "Residual add + LayerNorm (same layout as output_proj)",
                TensorRecord(
                    "x_moe",
                    {"B": tokens_local, "D": d_model},
                    _proj_d_part,
                    dtype_bytes=float_size,
                ),
                note=(
                    f"actual incoming projection layout: D/({proj_x}·{proj_y}) = "
                    f"{d_model // (proj_x * proj_y)}"
                ),
            )
            layout_trace.record(
                "moe_set_token_select",
                "Local selection of this complete expert set's token slice",
                TensorRecord(
                    "x_moe_set",
                    {"B": tokens_per_feeder_set, "D": d_model},
                    _proj_d_part,
                    dtype_bytes=float_size,
                ),
                note=(
                    f"actual: zero-communication token selection B={tokens_local} -> "
                    f"B_feeder_set={tokens_per_feeder_set} across "
                    f"{full_sets_per_attn_domain} local set(s)"
                ),
            )
        else:
            layout_trace.record(
                "moe_input",
                "MoE input in the existing pass-specific layout",
                TensorRecord("x_moe", {"B": s, "D": d_model}, _moe_d_part, dtype_bytes=float_size),
                note=(
                    f"actual: moe_tp_x={tp_x}, moe_tp_yz={tp_yz}, "
                    f"D/({tp_x}·{tp_yz}) = {_moe_d_local}"
                ),
            )

        # ---- Phase-structured comm model for 2D TP MoE (prefill/train) ----
        #
        # We keep two optimizations explicitly separate:
        # A) Expert-token transport shaping (orthogonal to router policy):
        #    1) All-gather input along X to reconstruct E/tp_yz for expert matmuls.
        # B) Router synchronization policy:
        #    2) Optional all-reduce of router scores on attention TP topology.
        #
        # Then follow TP math-collective phases:
        # 3) All-reduce expert up-projection partial sums along YZ (E is sharded on YZ).
        # 4) Reduce-scatter expert down-projection output along X to produce E/(tp_x*tp_yz).
        #
        # Notes / scope:
        # - Token exchange between experts (MoE all-to-all) is modeled separately in ep_strategy.
        # - Here we only model TP-specific collectives induced by sharding expert weights/activations.
        #
        total_comm = CommStats()
        if moe_config.num_shared_experts_per_dp_rank > 0:
            raise NotImplementedError("Shared experts not implemented in 2D TP MoE strategy.")

        # Phase A1. Decode consumes each feeder's output-projection layout
        # directly: always AG all feeder-set tokens over TP_Proj_X, then
        # reconcile TP_Proj_Y_eff with TP_MoE_Y. Prefill/training retain the
        # existing optional routed-token transport optimization.
        redistribution_optimization_enabled = bool(
            moe_config.tp_moe_a2a_replication_optimization_enabled
        )
        n_experts = transformer_config.num_regular_experts
        if n_experts > 0:
            # Build A1 payload from the same routed-token basis as TP1D/C-phases:
            # - traffic_mult captures selected-expert fanout + imbalance amplification,
            # - num_colocated_experts maps per-expert routed traffic to per-chip load,
            # - remote_fraction (locality across `num_colocated_groups_per_set` groups,
            #   under the no-spill invariant) is applied in the EP A2A path itself, not
            #   here; TP core collectives carry the per-rank routed volume.
            #
            # This keeps 1D/2D parity in the degenerate YZ=1 case by avoiding a mixed
            # payload basis (raw tokens for A1 vs routed tokens for C-phases).
            traffic_mult = moe_config.regular_expert_traffic_multiplier
            tokens_per_routed_expert = tokens_per_set / n_experts
            routed_tokens_per_chip = (
                tokens_per_routed_expert * traffic_mult * moe_config.num_colocated_experts
            )
        else:
            routed_tokens_per_chip = 0.0
        if exact_decode_handoff and n_experts > 0:
            assert proj_x_topology is not None
            assert proj_yz_topology is not None
            total_comm += _decode_moe_input_layout_comm(
                proj_x_topology=proj_x_topology,
                proj_yz_topology=proj_yz_topology,
                moe_yz_topology=yz_topology,
                tokens=tokens_per_feeder_set,
                d_model=d_model,
                float_size=float_size,
                layout_trace=layout_trace,
            )
            _reconciled_d_part = (
                {"D": P("sharded", ("MoE_Y",)), "B": P("replicated", ("Y",))}
                if tp_yz > 1
                else {"B": P("replicated", ("Y",))}
            )
            _reconciliation_flags = (
                (
                    "MODEL_LIMITATION: projection-Y quotient gathers assume aligned disjoint "
                    "subgroups execute concurrently; cross-subgroup fabric contention is not modeled.",
                )
                if proj_yz_topology.num_chips > tp_yz
                else ()
            )
            layout_trace.record(
                "moe_input_reconciled",
                "After AG(TP_Proj_X) and projection-Y to MoE-Y reconciliation",
                TensorRecord(
                    "x_moe_reconciled",
                    {"B": s, "D": d_model},
                    _reconciled_d_part,
                    dtype_bytes=float_size,
                ),
                note=f"actual: X replicated, D/tp_moe_y = {d_model // tp_yz}",
                flags=_reconciliation_flags,
            )
        elif redistribution_optimization_enabled:
            x_after_ag_bytes = float_size * routed_tokens_per_chip * (d_model / tp_yz)
            comm_transport = hockney_model_all_gather_on_nd_topology(
                x_topology, x_after_ag_bytes, float_size_bytes=float_size
            )
            total_comm += comm_transport
            layout_trace.record_comm(
                "moe_a1_ag_x",
                "AG",
                f"X({x_topology.num_chips})",
                x_topology.num_chips,
                x_after_ag_bytes,
                comm_transport,
                note="expert input: gather D/Y_eff along X",
            )

        # LN2 (pre-MoE LayerNorm) communication — "gather-then-normalize" approach.
        #
        # After AG(X) in Phase A1, each rank has [B, D/Y_eff]. LayerNorm requires
        # stats (mean, variance) across the full D, so communication across Y_eff
        # is needed when tp_yz > 1. When tp_yz == 1 the Hockney model returns zero
        # and the trace shows the collective as a noop.
        #
        # Three strategies depending on router mode and pass type:
        # - Decode + replicated router: AG(Y_eff) → full D locally. One collective
        #   that serves both LN (computed locally with full D) and the replicated
        #   router (needs full D for W_gate[D, E] matmul). Saves one alpha vs
        #   separate LN-stats-AR + router-gather. Crossover ≈ B=22 tokens; below
        #   that the saved alpha dominates the extra volume.
        # - Sharded router / prefill / train: LN stats AR(Y_eff). Tiny payload
        #   (2 scalars × B × float_size). Router uses D/Y_eff with logits AR (Phase B).
        # - Normalization-deferred sharded router: compute partial router logits from
        #   unnormalized activations, then pack those logits with the two LN statistics
        #   into one AR(Y_eff). This preserves the sharded router's payload while
        #   eliminating one collective startup.
        router_sync_mode = moe_config.router_sync_mode

        # LN2 + router comm (Phase B), branched by router mode:
        match router_sync_mode:
            case MoERouterSyncMode.REPLICATED_ROUTER:
                # Fused LN2 + replicated router: AG(Y_eff) gathers full D.
                # One collective serves both LN (local with full D) and the
                # replicated router (needs full D for W_gate[D, E]).
                ln2_ag_yz_bytes = float_size * tokens_local * d_model
                comm_ln2 = hockney_model_all_gather_on_nd_topology(
                    yz_topology, ln2_ag_yz_bytes, float_size_bytes=float_size
                )
                total_comm += comm_ln2
                layout_trace.record_comm(
                    "ln2_ag_yz",
                    "AG",
                    f"YZ({yz_topology.num_chips})",
                    yz_topology.num_chips,
                    ln2_ag_yz_bytes,
                    comm_ln2,
                    note=(
                        f"LN2 + replicated router fused: AG(Y_eff={tp_yz}) → "
                        f"full D={d_model} per rank"
                    ),
                )
            case MoERouterSyncMode.SHARDED_ROUTER:
                # LN2 stats AR: tiny payload (2 scalars × B × float_size).
                ln2_stats_bytes = 2 * float_size * tokens_local
                comm_ln2 = hockney_model_all_reduce_on_nd_topology(
                    yz_topology, ln2_stats_bytes, float_size_bytes=float_size
                )
                total_comm += comm_ln2
                layout_trace.record_comm(
                    "ln2_stats_ar_yz",
                    "AR",
                    f"YZ({yz_topology.num_chips})",
                    yz_topology.num_chips,
                    ln2_stats_bytes,
                    comm_ln2,
                    note=f"LN2 stats: mean+var across Y_eff={tp_yz}",
                )

                # Router logits AR: sync partial logits across Y_eff.
                # After AG(X) in Phase A1, D is sharded only along Y_eff;
                # X-ranks hold identical D/Y_eff slices and identical W_gate
                # shards, producing identical partial logits. Only Y_eff-ranks
                # contribute distinct partials. With <=1 routed experts,
                # logits synchronization is unnecessary.
                if n_experts > 1:
                    router_bytes = float_size * tokens_local * n_experts
                    comm_router_sync = hockney_model_all_reduce_on_nd_topology(
                        yz_topology, router_bytes, float_size_bytes=float_size
                    )
                    total_comm += comm_router_sync
                    layout_trace.record_comm(
                        "moe_router_ar",
                        "AR",
                        f"YZ({yz_topology.num_chips})",
                        yz_topology.num_chips,
                        router_bytes,
                        comm_router_sync,
                        note="sharded router: sync logits across Y_eff",
                    )
            case MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER:
                # Algebraically defer normalization until after reducing both
                # the stats and pre-normalized partial logits in one packed AR:
                #   [S, Q, P_1, ..., P_E] = AR_Y([S_r, Q_r, P_1,r, ..., P_E,r])
                # Same B*(E+2) element volume as the two ordinary sharded
                # buffers, but one fewer collective startup.
                if tp_yz <= 1:
                    raise ValueError("normalization_deferred_sharded_router requires TP_MoE_Y > 1")
                packed_router_bytes = float_size * tokens_local * (n_experts + 2)
                comm_packed_router_sync = hockney_model_all_reduce_on_nd_topology(
                    yz_topology,
                    packed_router_bytes,
                    float_size_bytes=float_size,
                )
                total_comm += comm_packed_router_sync
                layout_trace.record_comm(
                    "ln2_router_packed_ar_yz",
                    "AR",
                    f"YZ({yz_topology.num_chips})",
                    yz_topology.num_chips,
                    packed_router_bytes,
                    comm_packed_router_sync,
                    note=(
                        "normalization-deferred sharded router: packed "
                        "[sum, sumsq, partial logits]"
                    ),
                )
            case _:
                raise ValueError(f"Unsupported router_sync_mode: {router_sync_mode}")

        _d_yz = d_model // max(1, tp_yz)
        _dff_x = d_ff // max(1, tp_x)
        match router_sync_mode:
            case MoERouterSyncMode.REPLICATED_ROUTER:
                # Each rank needs full D for the local W_gate matmul.
                layout_trace.record(
                    "router_input",
                    "Replicated router: needs full D",
                    TensorRecord(
                        "x_router",
                        {"B": s, "D": d_model},
                        {"D": P("replicated"), "B": P("replicated", ("Y",))},
                        dtype_bytes=float_size,
                    ),
                    note=f"actual: replicated router, full d_model={d_model}",
                )
            case (
                MoERouterSyncMode.SHARDED_ROUTER
                | MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER
            ):
                # Same D/Y_eff activation layout. Deferred mode only changes
                # how logits are formed (packed AR), not x_router partitioning.
                _router_d_part = (
                    {"D": P("sharded", ("MoE_Y",)), "B": P("replicated", ("Y",))}
                    if tp_yz > 1
                    else {"B": P("replicated", ("Y",))}
                )
                layout_trace.record(
                    "router_input",
                    (
                        "Normalization-deferred sharded router: AG(X), "
                        "pre-normalized partial logits + packed AR"
                        if router_sync_mode
                        == MoERouterSyncMode.NORMALIZATION_DEFERRED_SHARDED_ROUTER
                        else "Sharded router: AG(X) then partial logits + AR"
                    ),
                    TensorRecord(
                        "x_router",
                        {"B": s, "D": d_model},
                        _router_d_part,
                        dtype_bytes=float_size,
                    ),
                    note=(
                        f"actual: {router_sync_mode.value}, " f"D/tp_yz = {_d_yz} (tp_yz={tp_yz})"
                    ),
                )
            case _:
                raise ValueError(f"Unsupported router_sync_mode: {router_sync_mode}")

        _expert_d_part = (
            {"D": P("sharded", ("MoE_Y",)), "B": P("replicated", ("Y",))}
            if tp_yz > 1
            else {"B": P("replicated", ("Y",))}
        )
        layout_trace.record(
            "expert_input",
            "Expert input after Phase A1",
            TensorRecord(
                "x_expert", {"B": s, "D": d_model}, _expert_d_part, dtype_bytes=float_size
            ),
            note=(
                f"actual: D/tp_yz = {_d_yz} (tp_yz={tp_yz}). "
                f"W_up: [{_d_yz}, {_dff_x}] (tp_x={tp_x})"
            ),
        )

        # Phase C1/C2: expert matmul partial-sum collectives (use tokens-per-routed-expert proxy).
        if n_experts > 0:
            # Effective tokens processed per chip for one routed expert, scaled by
            # traffic multiplier, colocation, and placement locality.
            # Keep this toggle-independent; optional optimization is modeled above as
            # an extra setup collective, while routed exchange bytes are handled by EP A2A.
            tokens_processed = routed_tokens_per_chip

            # Fused gate||up (one packed AR on [tokens, n_up * d_ff]):
            # one collective, payload scales by mlp_num_up_projections.
            n_up = transformer_config.mlp_num_up_projections
            up_out_bytes = n_up * float_size * tokens_processed * (d_ff / tp_x)
            comm_tp_up_reduce = hockney_model_all_reduce_on_nd_topology(
                yz_topology, up_out_bytes, float_size_bytes=float_size
            )
            total_comm += comm_tp_up_reduce
            layout_trace.record_comm(
                "moe_c1_ar_yz",
                "AR",
                f"YZ({yz_topology.num_chips})",
                yz_topology.num_chips,
                up_out_bytes,
                comm_tp_up_reduce,
                note="up-proj partial sums across YZ",
            )

            down_out_bytes = float_size * tokens_processed * (d_model / tp_yz)
            comm_tp_down_scatter = hockney_model_reduce_scatter_on_nd_topology(
                x_topology, down_out_bytes, float_size_bytes=float_size
            )
            total_comm += comm_tp_down_scatter
            layout_trace.record_comm(
                "moe_c2_rs_x",
                "RS",
                f"X({x_topology.num_chips})",
                x_topology.num_chips,
                down_out_bytes,
                comm_tp_down_scatter,
                note="down-proj scatter across X",
            )

        _dff_part = (
            {"Dff": P("sharded", ("MoE_X",)), "B": P("replicated", ("Y",))}
            if tp_x > 1
            else {"B": P("replicated", ("Y",))}
        )
        layout_trace.record(
            "after_up_proj_c1",
            "Up-proj + C1 AR(Y_eff)",
            TensorRecord("h_expert", {"B": s, "Dff": d_ff}, _dff_part, dtype_bytes=float_size),
            note=f"actual: C1 AR(TP_MoE_Y). Dff/tp_x = {d_ff // tp_x} (tp_x={tp_x})",
        )
        layout_trace.record(
            "after_down_proj_c2",
            "Down-proj + C2 RS(X)",
            TensorRecord("y_expert", {"B": s, "D": d_model}, _moe_d_part, dtype_bytes=float_size),
            note=f"actual: C2 RS(X). D_local = {_moe_d_local} (tp_x={tp_x}, tp_yz={tp_yz})",
        )
        if exact_decode_handoff and not single_domain_decode:
            # Cross-domain return A2Av and local residual-layout restoration
            # are priced and traced by ep_strategy after expert execution.
            return total_comm
        if num_combine_sets > 1:
            # After intra-set RS(G), rank-local D may hand off to next-layer X sharding.
            layout_trace.record(
                "after_combine",
                f"MoE combine RS(G) — per-set [{s}, D_local]",
                TensorRecord(
                    "moe_out", {"B": s, "D": d_model}, _post_combine_d_part, dtype_bytes=float_size
                ),
                note=(
                    f"actual: intra-set RS(G). B_set={s}, D_local={_post_combine_d_local} "
                    f"(moe_tp_x={tp_x}, moe_tp_yz={tp_yz}, target_X={target_x}, "
                    f"target_Y_eff={target_y_eff}, mode={handoff_mode})"
                ),
                flags=_post_combine_issue_flags,
            )
            # Cross-set AG(S) reconstructs the full batch.
            layout_trace.record(
                "after_combine_cross_set",
                f"Cross-set AG(S_attn={num_combine_sets}) → attn-domain batch",
                TensorRecord(
                    "moe_out_full_batch",
                    {"B": tokens_local, "D": d_model},
                    _post_combine_d_part,
                    dtype_bytes=float_size,
                ),
                note=(
                    f"actual: AG(S_attn={num_combine_sets}). "
                    f"[B_set={s}] × {num_combine_sets} → [B={tokens_local}], "
                    f"D_local={_post_combine_d_local}, mode={handoff_mode}"
                ),
                flags=_post_combine_issue_flags,
            )
        else:
            layout_trace.record(
                "after_combine",
                "Exact MoE output combine",
                TensorRecord(
                    "moe_out", {"B": s, "D": d_model}, _post_combine_d_part, dtype_bytes=float_size
                ),
                note=(
                    f"actual: {handoff_mode}. D_local = {_post_combine_d_local} "
                    f"(moe_tp_x={tp_x}, moe_tp_yz={tp_yz}, target_X={target_x}, "
                    f"target_Y_eff={target_y_eff}, mode={handoff_mode})"
                ),
                flags=_post_combine_issue_flags,
            )
        layout_trace.record(
            "layer_output",
            "Residual add → next layer",
            TensorRecord(
                "x_out",
                {"B": tokens_local, "D": d_model},
                _post_combine_d_part,
                dtype_bytes=float_size,
            ),
            note=(
                f"actual: D_local = {_post_combine_d_local} "
                f"(moe_tp_x={tp_x}, moe_tp_yz={tp_yz}, target_X={target_x}, "
                f"target_Y_eff={target_y_eff}, mode={handoff_mode})"
            ),
            flags=_post_combine_issue_flags,
        )

        return total_comm

    def forward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        x_topology: LogicalTopology | None = None,
        yz_topology: LogicalTopology | None = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_all_modes(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            moe_tp_topology=moe_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            x_topology=x_topology,
            yz_topology=yz_topology,
            attn_tp_topology=attn_tp_topology,
            is_decode=False,
            **kwargs,
        )

    def forward_comm_prefill(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        x_topology: LogicalTopology | None = None,
        yz_topology: LogicalTopology | None = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_all_modes(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            moe_tp_topology=moe_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            x_topology=x_topology,
            yz_topology=yz_topology,
            attn_tp_topology=attn_tp_topology,
            is_decode=False,
            **kwargs,
        )

    def forward_comm_decode(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        x_topology: LogicalTopology | None = None,
        yz_topology: LogicalTopology | None = None,
        attn_tp_topology: LogicalTopology | None = None,
        **kwargs,
    ) -> CommStats:
        return self._forward_comm_all_modes(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            moe_tp_topology=moe_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            x_topology=x_topology,
            yz_topology=yz_topology,
            attn_tp_topology=attn_tp_topology,
            is_decode=True,
            **kwargs,
        )

    def backward_comm_training(
        self,
        transformer_config: TransformerConfig,
        moe_config,
        workload_spec,
        n_seq: int,
        moe_tp_topology: LogicalTopology = None,
        x_dims: list[int] | None = None,
        yz_dims: list[int] | None = None,
        **kwargs,
    ) -> CommStats:
        return self.forward_comm_training(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            moe_tp_topology=moe_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            **kwargs,
        )


def get_tp_strategies(
    variant: TensorParallelVariant,
) -> tuple[TensorParallelStrategy, TensorParallelStrategy]:
    """Factory to create the TP strategies for Attn and MoE based on a variant."""
    match variant:
        case TensorParallelVariant.SHARD_WEIGHTS_1D:
            return TP1DAttnStrategy(), TP1DMoEStrategy()
        case TensorParallelVariant.SHARD_WEIGHTS_2D:
            return TP2DAttnStrategy(), TP2DMoEStrategy()
        case TensorParallelVariant.SHARD_ACTIVATIONS | TensorParallelVariant.NONE:
            raise NotImplementedError(f"TP variant {variant} is not implemented.")
        case _:
            raise ValueError(f"Unknown TP variant: {variant}")


def get_attn_tp_strategy(variant: TensorParallelVariant) -> TensorParallelStrategy:
    """Return the attention TP strategy for the given variant."""
    attn_strategy, _ = get_tp_strategies(variant)
    return attn_strategy


def get_moe_tp_strategy(variant: TensorParallelVariant) -> TensorParallelStrategy:
    """Return the MoE/FFN TP strategy for the given variant."""
    _, moe_strategy = get_tp_strategies(variant)
    return moe_strategy


def _tp2d_prod(tp_topology: LogicalTopology, dims: list[int]) -> int:
    shape = getattr(tp_topology, "shape", [])
    p = 1
    for idx in dims:
        p *= shape[idx]
    return p


def validate_1d_tp_invariants_attn(model_spec, tp_topology: LogicalTopology):
    """Validate 1D-TP invariants for attention.

    In 1D attention TP, the TP degree maps to head sharding, so query heads must
    be divisible by TP size. KV heads follow the same sharding/duplication policy
    used by the 2D-X validator.
    """
    tp = int(getattr(tp_topology, "num_chips", 1) or 1)
    if tp <= 0:
        raise ValueError(f"TP-1D size must be positive, got {tp}")

    num_heads = model_spec.num_heads
    num_kv_heads = getattr(model_spec, "num_kv_heads", num_heads)

    if num_heads % tp != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by TP-1D size ({tp})")

    # KV heads: allow either sharding (divisible) or duplication/replication when TP > num_kv_heads.
    if num_kv_heads % tp != 0 and tp % num_kv_heads != 0:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must divide TP-1D ({tp}) or be divisible by TP-1D "
            f"(got neither)."
        )
    if tp > num_kv_heads:
        rep = tp // num_kv_heads
        warnings.warn(
            f"KV-head duplication modeled (TP-1D): TP ({tp}) > num_kv_heads ({num_kv_heads}); "
            f"each KV head is replicated {rep}x across TP ranks.",
            stacklevel=2,
        )


def validate_2d_tp_invariants_attn(model_spec, tp_topology, x_dims: list[int], yz_dims: list[int]):
    """Validate 2D-TP invariants for *attention compute*.

    Semantics (current model):
    - X shards heads
    - YZ shards head_dim (and sequence/KV-cache in attention compute variants)

    Constraints we need for *core attention compute* in this codebase:
    - X shards heads  -> num_heads divisible by TP X size
    - X shards kv heads -> num_kv_heads divisible by TP X size

    Importantly, we do **not** require `d_model` to be divisible by TP YZ size here because
    we allow `TP_Attn_YZ` to represent *sequence/context parallelism* (sharding sequence/KV-cache),
    not sharding the model dimension.
    """
    tp_x = _tp2d_prod(tp_topology, x_dims) if x_dims else 1

    num_heads = model_spec.num_heads
    num_kv_heads = getattr(model_spec, "num_kv_heads", num_heads)

    if num_heads % tp_x != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by TP-X size ({tp_x})")
    # KV heads: allow either sharding (divisible) or duplication/replication when tp_x > num_kv_heads.
    if num_kv_heads % tp_x != 0 and tp_x % num_kv_heads != 0:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must divide TP-X ({tp_x}) or be divisible by TP-X "
            f"(got neither)."
        )
    if tp_x > num_kv_heads:
        rep = tp_x // num_kv_heads
        warnings.warn(
            f"KV-head duplication modeled: TP-X ({tp_x}) > num_kv_heads ({num_kv_heads}); "
            f"each KV head is replicated {rep}x across X ranks.",
            stacklevel=2,
        )


def validate_2d_tp_invariants_proj(model_spec, tp_topology, x_dims: list[int], yz_dims: list[int]):
    """Validate 2D-TP invariants for *projection TP* (QKV + output projection).

    Projections in our model shard the *model dimension* across X*YZ, so we require:
    - d_model divisible by TP total size (X*YZ)
    - num_heads divisible by TP X size
    - num_kv_heads divisible by TP X size

    Note: we intentionally do not enforce any sequence-length divisibility here.
    """
    tp_x = _tp2d_prod(tp_topology, x_dims) if x_dims else 1
    tp_yz = _tp2d_prod(tp_topology, yz_dims) if yz_dims else 1
    tp_total = tp_x * tp_yz

    d_model = model_spec.d_model
    num_heads = model_spec.num_heads
    num_kv_heads = getattr(model_spec, "num_kv_heads", num_heads)

    if d_model % tp_total != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by 2D TP total ({tp_total})")
    if num_heads % tp_x != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by TP-X size ({tp_x})")
    # KV heads: allow either sharding (divisible) or duplication/replication when tp_x > num_kv_heads.
    if num_kv_heads % tp_x != 0 and tp_x % num_kv_heads != 0:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must divide TP-X ({tp_x}) or be divisible by TP-X "
            f"(got neither)."
        )
    if tp_x > num_kv_heads:
        rep = tp_x // num_kv_heads
        warnings.warn(
            f"KV-head duplication modeled (projection TP): TP-X ({tp_x}) > num_kv_heads ({num_kv_heads}); "
            f"each KV head is replicated {rep}x across X ranks.",
            stacklevel=2,
        )


def validate_2d_tp_invariants_moe(model_spec, tp_topology, x_dims: list[int], yz_dims: list[int]):
    """Validate divisibility constraints for 2D TP on the MoE/FFN block.

    Unlike attention, MoE does not require (num_heads, num_kv_heads) divisibility by TP-X.
    We still require:
    - d_model divisible by TP total size (2D weight sharding)
    - d_ff divisible by TP-YZ size (FFN hidden split on YZ)
    """
    tp_x = _tp2d_prod(tp_topology, x_dims) if x_dims else 1
    tp_yz = _tp2d_prod(tp_topology, yz_dims) if yz_dims else 1
    tp_total = tp_x * tp_yz

    d_model = model_spec.d_model
    if d_model % tp_total != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by 2D TP total ({tp_total})")

    d_ff = getattr(model_spec, "mlp_hidden_size", None)
    if d_ff is not None and tp_yz > 0 and d_ff % tp_yz != 0:
        raise ValueError(
            f"FFN hidden size d_ff ({d_ff}) must be divisible by TP-YZ size ({tp_yz}) for 2D TP"
        )
