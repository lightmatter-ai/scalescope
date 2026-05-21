"""Tensor-parallel (TP) communication strategies for attention and MoE.

Implements modeling for forward/backward TP collectives for 1D/2D weight sharding variants.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

from scalescope.decode_layout_trace import NULL_LAYOUT_TRACE
from scalescope.model_spec import resolve_kv_heads_per_partition, TransformerConfig
from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_all_reduce_on_nd_topology,
    hockney_model_reduce_on_nd_topology,
    hockney_model_reduce_scatter_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.topology_spec import LogicalTopology
from scalescope.workload_spec import (
    MoERouterSyncMode,
    TensorParallelVariant,
    TP2DSplitSpec,
)


def _tag_to_str(tag) -> str:
    return tag.value if hasattr(tag, "value") else str(tag)


def _moe_remote_fraction(moe_config) -> float:
    """Smooth routed-token remote fraction across attention domains and colocated groups.

    Remote fraction = 1 - 1/(D*G), where:
    - D: attn_domains_per_full_expert_set
    - G: num_colocated_groups_per_set
    This is the same independence-based locality factor used by EP MoE A2A.
    """
    d = int(moe_config.attn_domains_per_full_expert_set)
    g = max(1, int(getattr(moe_config, "num_colocated_groups_per_set", 1) or 1))
    if d <= 0:
        raise ValueError(f"attn_domains_per_full_expert_set must be > 0, got {d}.")
    return max(0.0, 1.0 - (1.0 / float(d * g)))


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


def _tp2d_attn_comm_phase1_input_proj(
    *, tp_topology: LogicalTopology, volume_bytes: float, tp_x: int, float_size_bytes: float
) -> CommStats:
    """Phase 1: input projection comm proxy (keep formula unchanged)."""
    return hockney_model_all_reduce_on_nd_topology(
        tp_topology, volume_bytes / tp_x, float_size_bytes=float_size_bytes
    )


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
    equal whenever num_heads · head_dim == d_model (i.e. non-wide-Q), so this
    collapses to the legacy single-volume pricing in the MHA / non-wide-Q case.
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
    - Per-rank full-buffer size after the conceptual all-gather:
        KV_full_bytes = 2 * B * n_seq * (H_kv/tp_x) * d_h * b
    """
    if yz_topology.num_chips <= 1:
        return CommStats()

    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    num_kv_heads = transformer_config.num_kv_heads
    head_dim = transformer_config.head_dim

    kv_heads_per_x, _kv_rep = resolve_kv_heads_per_partition(num_kv_heads=num_kv_heads, tp_x=tp_x)
    kv_full_bytes = float_size_bytes * (2 * n_batch * n_seq * (kv_heads_per_x * head_dim))
    return hockney_model_all_gather_on_nd_topology(
        yz_topology, kv_full_bytes, float_size_bytes=float_size_bytes
    )


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


def _tp2d_attn_comm_prefill_train(
    *,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    attn_tp_topology: LogicalTopology,
    x_topology: LogicalTopology,
    yz_topology: LogicalTopology,
    tp_x: int,
    tp_yz: int,
) -> CommStats:
    """TP2D attention comm for training/prefill (no decode-only YZ reductions)."""
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim
    float_size = workload_spec.float_size_bytes
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    # Two distinct volumes for the output-projection collectives: AG_YZ
    # ships the attention-output activation (q_proj_out_dim), RS_X ships
    # the post-output-proj residual stream (d_model). Equal under non-wide-Q.
    output_volume_bytes = float_size * n_batch * n_seq * d_model
    input_volume_bytes = float_size * n_batch * n_seq * q_proj_out_dim
    comm1 = _tp2d_attn_comm_phase1_input_proj(
        tp_topology=attn_tp_topology,
        volume_bytes=output_volume_bytes,
        tp_x=tp_x,
        float_size_bytes=float_size,
    )
    comm2 = _tp2d_attn_comm_phase2_prefill_ring_kv_exchange(
        yz_topology=yz_topology,
        transformer_config=transformer_config,
        workload_spec=workload_spec,
        n_seq=n_seq,
        tp_x=tp_x,
        float_size_bytes=float_size,
    )
    comm3 = _tp2d_attn_comm_phase3_output_proj(
        x_topology=x_topology,
        yz_topology=yz_topology,
        input_volume_bytes=input_volume_bytes,
        output_volume_bytes=output_volume_bytes,
        tp_x=tp_x,
        tp_yz=tp_yz,
        float_size_bytes=float_size,
    )
    return comm1 + comm2 + comm3


def _tp2d_attn_comm_decode(
    *,
    transformer_config: TransformerConfig,
    workload_spec,
    n_seq: int,
    attn_tp_topology: LogicalTopology,
    x_topology: LogicalTopology,
    yz_topology: LogicalTopology,
    tp_x: int,
    tp_yz: int,
) -> CommStats:
    """TP2D attention comm for decode (includes decode-only YZ reductions)."""
    d_model = transformer_config.d_model
    num_heads = transformer_config.num_heads
    head_dim = transformer_config.head_dim
    q_proj_out_dim = num_heads * head_dim
    float_size = workload_spec.float_size_bytes
    n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
    output_volume_bytes = float_size * n_batch * n_seq * d_model
    input_volume_bytes = float_size * n_batch * n_seq * q_proj_out_dim
    comm1 = _tp2d_attn_comm_phase1_qkv_proj_decode(
        x_topology=x_topology,
        yz_topology=yz_topology,
        transformer_config=transformer_config,
        workload_spec=workload_spec,
        tp_x=tp_x,
        tp_yz=tp_yz,
        float_size_bytes=float_size,
    )
    comm2 = _tp2d_attn_comm_phase2_decode_yz_reductions(
        yz_topology=yz_topology,
        transformer_config=transformer_config,
        workload_spec=workload_spec,
        tp_x=tp_x,
        float_size_bytes=float_size,
    )
    comm3 = _tp2d_attn_comm_phase3_output_proj(
        x_topology=x_topology,
        yz_topology=yz_topology,
        input_volume_bytes=input_volume_bytes,
        output_volume_bytes=output_volume_bytes,
        tp_x=tp_x,
        tp_yz=tp_yz,
        float_size_bytes=float_size,
    )
    return comm1 + comm2 + comm3


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

    def _forward_comm_prefill_train(
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
        return self._forward_comm_prefill_train(
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
        return self._forward_comm_prefill_train(
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
        return self._forward_comm_prefill_train(
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
        d_model = transformer_config.d_model
        num_heads = transformer_config.num_heads
        head_dim = transformer_config.head_dim
        q_proj_out_dim = num_heads * head_dim
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        # `output_volume` is the residual-stream volume (d_model-based) used
        # by the input projection and the output-proj reduce-scatter.
        # `input_volume` is the attention-output activation volume
        # (q_proj_out_dim-based) shipped by the output-proj all-gather.
        # They coincide for non-wide-Q layouts.
        output_volume = float_size * n_batch * n_seq * d_model
        input_volume = float_size * n_batch * n_seq * q_proj_out_dim

        # Attention TP sizes
        shape_attn = getattr(attn_tp_topology, "shape", [])
        tp_x_attn = 1
        for idx in x_dims:
            tp_x_attn *= shape_attn[idx]

        # Projection TP sizes
        shape_proj = getattr(proj_tp_topology, "shape", [])
        tp_x_proj = 1
        for idx in proj_x_dims:
            tp_x_proj *= shape_proj[idx]
        tp_yz_proj = 1
        for idx in proj_yz_dims:
            tp_yz_proj *= shape_proj[idx]

        phase1 = _tp2d_attn_comm_phase1_input_proj(
            tp_topology=proj_tp_topology,
            volume_bytes=output_volume,
            tp_x=tp_x_proj,
            float_size_bytes=float_size,
        )
        phase2 = _tp2d_attn_comm_phase2_prefill_ring_kv_exchange(
            yz_topology=yz_topology,
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            tp_x=tp_x_attn,
            float_size_bytes=float_size,
        )
        phase3 = _tp2d_attn_comm_phase3_output_proj(
            x_topology=proj_x_topology,
            yz_topology=proj_yz_topology,
            input_volume_bytes=input_volume,
            output_volume_bytes=output_volume,
            tp_x=tp_x_proj,
            tp_yz=tp_yz_proj,
            float_size_bytes=float_size,
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

        from scalescope.decode_layout_trace import P, TensorRecord

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
        kv_per_x = max(1, n_kv_heads // tp_x_attn)
        kv_dim = kv_per_x * d_head
        layout_trace.record(
            "after_qkv_proj",
            "K,V → reduce-to-owner(Y_eff)",
            TensorRecord("KV", {"B": s, "Dkv": kv_dim}, {}, dtype_bytes=float_size),
            note=f"actual: GQA {n_kv_heads} KV-heads, {kv_per_x} per X-rank",
        )

        context_len = getattr(workload_spec, "prefill_seq_len", 1)
        heads_local = n_heads // tp_x_attn
        layout_trace.record(
            "kv_cache_read",
            "Read KV-cache for current context",
            TensorRecord(
                "KV_cache", {"Dkv": kv_dim, "Ctx": context_len}, {}, dtype_bytes=float_size
            ),
            note=(
                f"actual: {kv_per_x} KV-heads × ctx={context_len} × d_head={d_head}. "
                f"Per rank = {kv_per_x * context_len * d_head * float_size} B"
            ),
        )
        layout_trace.record(
            "attention_scores",
            "QK^T + online softmax (FlashDecoding)",
            TensorRecord(
                "scores",
                {"H": n_heads, "B": s, "Ctx": context_len},
                {"H": P("sharded", ("X",))},
                dtype_bytes=float_size,
            ),
            note=(
                f"actual: [{heads_local}, {s}, {context_len}] per rank. "
                f"GQA: {heads_local} Q-heads share {kv_per_x} KV-head(s) "
                f"(ratio {heads_local // max(1, kv_per_x)}:1). "
                f"Online softmax: streaming."
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
        # All topology inputs are required (explicit; no None-default behavior).
        d_model = transformer_config.d_model
        num_heads = transformer_config.num_heads
        head_dim = transformer_config.head_dim
        q_proj_out_dim = num_heads * head_dim
        float_size = workload_spec.float_size_bytes
        n_batch = workload_spec.get_resolved_batches().batch_size_per_dp
        output_volume = float_size * n_batch * n_seq * d_model
        input_volume = float_size * n_batch * n_seq * q_proj_out_dim

        # Compute TP sizes separately for attention vs projection domains.
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
        tp_yz_proj = 1
        for idx in proj_yz_dims:
            tp_yz_proj *= shape_proj[idx]

        # Phase 1 (projection domain): keep proxy formula unchanged, but price on projection TP.
        comm1 = _tp2d_attn_comm_phase1_input_proj(
            tp_topology=proj_tp_topology,
            volume_bytes=output_volume,
            tp_x=tp_x_proj,
            float_size_bytes=float_size,
        )

        # Phase 2 (attention domain): RingAttention proxy on attention YZ topology.
        comm2 = _tp2d_attn_comm_phase2_prefill_ring_kv_exchange(
            yz_topology=yz_topology,
            transformer_config=transformer_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            tp_x=tp_x_attn,
            float_size_bytes=float_size,
        )

        # Phase 3 (projection domain): output projection comm on projection views.
        comm3 = _tp2d_attn_comm_phase3_output_proj(
            x_topology=proj_x_topology,
            yz_topology=proj_yz_topology,
            input_volume_bytes=input_volume,
            output_volume_bytes=output_volume,
            tp_x=tp_x_proj,
            tp_yz=tp_yz_proj,
            float_size_bytes=float_size,
        )

        return comm1 + comm2 + comm3

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

        # NOTE: decode_tp2d_qkv_groups is implemented as a *global* policy by creating derived
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
        # For now, reuse training forward comm as a proxy (keeps formulas unchanged).
        return self.forward_comm_training(
            transformer_config=transformer_config,
            moe_config=moe_config,
            workload_spec=workload_spec,
            n_seq=n_seq,
            attn_tp_topology=attn_tp_topology,
            x_dims=x_dims,
            yz_dims=yz_dims,
            **kwargs,
        )


class TP2DMoEStrategy(TensorParallelStrategy):
    """Models 2D TP for the MoE/FFN block."""

    def _forward_comm_prefill_train(
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
        layout_trace = kwargs.pop("layout_trace", NULL_LAYOUT_TRACE)

        if moe_tp_topology is None:
            raise ValueError("TP2DMoEStrategy requires moe_tp_topology.")
        if not x_dims or not yz_dims:
            raise ValueError("TP2DMoEStrategy requires explicit x_dims and yz_dims.")
        if x_topology is None or yz_topology is None:
            raise ValueError("TP2DMoEStrategy requires x_topology and yz_topology.")

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
        n_batch = resolved_batches.batch_size_per_dp
        tokens_local = n_batch * n_seq
        tokens_per_set = resolved_batches.batch_size_per_full_expert_set * n_seq
        # MoE-stage trace should reflect per-full-expert-set token ownership.
        # For num_complete_sets>1 this avoids showing global batch on each set.
        s = tokens_per_set

        from scalescope.decode_layout_trace import P, TensorRecord

        _moe_d_dims = []
        if tp_x > 1:
            _moe_d_dims.append("X")
        if tp_yz > 1:
            _moe_d_dims.append("Y_eff")
        _moe_d_part = (
            {"D": P("sharded", tuple(_moe_d_dims)), "B": P("replicated", ("Y",))}
            if _moe_d_dims
            else {"B": P("replicated", ("Y",))}
        )
        _moe_d_local = d_model // max(1, tp_x * tp_yz)
        if hasattr(layout_trace, "mesh"):
            target_x = max(1, layout_trace.mesh.size_of("X"))
            target_y_eff = max(1, layout_trace.mesh.size_of("Y_eff"))
        else:
            # No-op trace path (prefill/train tests): fall back to local TP decomposition.
            target_x = tp_x
            target_y_eff = tp_yz
        groups_per_set = moe_config.num_colocated_groups_per_set
        if groups_per_set <= 0:
            raise ValueError(f"num_colocated_groups_per_set must be > 0, got {groups_per_set}")

        # Variant-aware combine handoff:
        # - 1xN (tp_x=1): RS(G) can hand off into next-layer X sharding when G matches target_X.
        # - Nx1 (tp_yz=1): RS(G) can hand off into next-layer Y_eff sharding when G matches target_Y_eff.
        # - NxM (both >1): not modeled yet; keep flagged.
        rs_handoff_1d_y = (
            tp_x == 1 and tp_yz == target_y_eff and target_x > 1 and groups_per_set == target_x
        )
        rs_handoff_1d_x = (
            tp_yz == 1 and tp_x == target_x and target_y_eff > 1 and groups_per_set == target_y_eff
        )
        rs_handoff_ok = rs_handoff_1d_y or rs_handoff_1d_x

        if rs_handoff_1d_y:
            handoff_mode = "1xN->X"
        elif rs_handoff_1d_x:
            handoff_mode = "Nx1->Y_eff"
        elif tp_x > 1 and tp_yz > 1:
            handoff_mode = "NxM-unsupported"
        else:
            handoff_mode = "1D-mismatch"

        if rs_handoff_ok:
            _post_combine_d_part = {
                "D": P("sharded", ("X", "Y_eff")),
                "B": P("replicated", ("Y",)),
            }
            _post_combine_d_local = d_model // max(1, target_x * target_y_eff)
            _post_combine_issue_flags: tuple[str, ...] = ()
        else:
            _post_combine_d_part = _moe_d_part
            _post_combine_d_local = _moe_d_local
            _post_combine_issue_flags = (
                "ISSUE: combine RS(G) handoff to next-layer X-sharded layout is not guaranteed "
                f"(moe_tp_x={tp_x}, moe_tp_yz={tp_yz}, groups_per_set={groups_per_set}, "
                f"target_X={target_x}, target_Y_eff={target_y_eff}, mode={handoff_mode}).",
            )
        num_complete_sets = moe_config.num_complete_expert_sets
        if num_complete_sets < 1:
            raise ValueError(f"num_complete_expert_sets must be > 0, got {num_complete_sets}")
        if attn_tp_topology is None:
            raise ValueError("TP2DMoEStrategy requires attn_tp_topology for combine-set tracing.")
        full_set_xpus = moe_config.full_expert_set_xpu_count
        if full_set_xpus <= 0:
            raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_set_xpus}")
        attn_tp_xpus = attn_tp_topology.num_chips
        if attn_tp_xpus <= 0:
            raise ValueError(f"attn_tp_topology.num_chips must be > 0, got {attn_tp_xpus}")
        if attn_tp_xpus >= full_set_xpus:
            if attn_tp_xpus % full_set_xpus != 0:
                raise ValueError(
                    f"attn_tp_xpus ({attn_tp_xpus}) must be divisible by "
                    f"full_expert_set_xpu_count ({full_set_xpus})"
                )
            num_combine_sets = attn_tp_xpus // full_set_xpus
        else:
            num_combine_sets = 1

        layout_trace.record(
            "moe_input",
            "Residual add + LayerNorm (same layout as output_proj)",
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

        # Phase A1 (transport shaping): all-gather X for expert input activations.
        # Each rank starts with [tokens, E/(tp_x*tp_yz)] and gathers to [tokens, E/tp_yz].
        #
        # Use routed-token basis (traffic multiplier + colocation + remote fraction) so
        # this phase is aligned with TP1D/C-phases semantics under expert imbalance.
        # TP core collectives are modeled on routed token payload processed by expert TP
        # shards (local + remote). Placement locality/remote_fraction is modeled in EP A2A.
        redistribution_optimization_enabled = bool(
            moe_config.tp_moe_a2a_replication_optimization_enabled
        )
        n_experts = transformer_config.num_regular_experts
        if n_experts > 0:
            # Build A1 payload from the same routed-token basis as TP1D/C-phases:
            # - traffic_mult captures selected-expert fanout + imbalance amplification,
            # - num_colocated_experts maps per-expert routed traffic to per-chip load,
            # - remote_fraction applies placement locality (D,G) consistently.
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
        # Optional TP2D redistribution optimization:
        # when enabled, model an additional X all-gather setup phase.
        # Base routed-token exchange volume itself is modeled by MoE A2A (ep_strategy)
        # and remains toggle-independent.
        if redistribution_optimization_enabled:
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
        # Two strategies depending on router mode and pass type:
        # - Decode + replicated router: AG(Y_eff) → full D locally. One collective
        #   that serves both LN (computed locally with full D) and the replicated
        #   router (needs full D for W_gate[D, E] matmul). Saves one alpha vs
        #   separate LN-stats-AR + router-gather. Crossover ≈ B=22 tokens; below
        #   that the saved alpha dominates the extra volume.
        # - Sharded router / prefill / train: LN stats AR(Y_eff). Tiny payload
        #   (2 scalars × B × float_size). Router uses D/Y_eff with logits AR (Phase B).
        router_sync_mode = moe_config.router_sync_mode

        # LN2 + router comm (Phase B), branched by router mode:
        if router_sync_mode == MoERouterSyncMode.REPLICATED_ROUTER:
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
        elif router_sync_mode == MoERouterSyncMode.SHARDED_ROUTER:
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
            # After AG(X) in Phase A1, D is sharded only along Y_eff; X-ranks hold
            # identical D/Y_eff slices and identical W_gate shards, producing identical
            # partial logits. Only Y_eff-ranks contribute distinct partials.
            # With <=1 routed experts, logits synchronization is unnecessary.
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
        else:
            raise ValueError(f"Unsupported router_sync_mode: {router_sync_mode}")

        _d_yz = d_model // max(1, tp_yz)
        _dff_x = d_ff // max(1, tp_x)
        if router_sync_mode == MoERouterSyncMode.REPLICATED_ROUTER:
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
        elif router_sync_mode == MoERouterSyncMode.SHARDED_ROUTER:
            _router_d_part = (
                {"D": P("sharded", ("Y_eff",)), "B": P("replicated", ("Y",))}
                if tp_yz > 1
                else {"B": P("replicated", ("Y",))}
            )
            layout_trace.record(
                "router_input",
                "Sharded router: AG(X) then partial logits + AR",
                TensorRecord(
                    "x_router", {"B": s, "D": d_model}, _router_d_part, dtype_bytes=float_size
                ),
                note=f"actual: sharded router, D/tp_yz = {_d_yz} (tp_yz={tp_yz})",
            )

        _expert_d_part = (
            {"D": P("sharded", ("Y_eff",)), "B": P("replicated", ("Y",))}
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
            {"Dff": P("sharded", ("X",)), "B": P("replicated", ("Y",))}
            if tp_x > 1
            else {"B": P("replicated", ("Y",))}
        )
        layout_trace.record(
            "after_up_proj_c1",
            "Up-proj + C1 AR(Y_eff)",
            TensorRecord("h_expert", {"B": s, "Dff": d_ff}, _dff_part, dtype_bytes=float_size),
            note=f"actual: C1 AR(Y_eff). Dff/tp_x = {d_ff // max(1, tp_x)} (tp_x={tp_x})",
        )
        layout_trace.record(
            "after_down_proj_c2",
            "Down-proj + C2 RS(X)",
            TensorRecord("y_expert", {"B": s, "D": d_model}, _moe_d_part, dtype_bytes=float_size),
            note=f"actual: C2 RS(X). D_local = {_moe_d_local} (tp_x={tp_x}, tp_yz={tp_yz})",
        )
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
                    "moe_out_full",
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
                "MoE combine RS(G)",
                TensorRecord(
                    "moe_out", {"B": s, "D": d_model}, _post_combine_d_part, dtype_bytes=float_size
                ),
                note=(
                    f"actual: combine RS(G). D_local = {_post_combine_d_local} "
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
        return self._forward_comm_prefill_train(
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
        return self._forward_comm_prefill_train(
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
        return self._forward_comm_prefill_train(
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
