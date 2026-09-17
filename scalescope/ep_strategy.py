"""Expert placement and per-XPU MoE layout inference.

Infers how distinct experts are placed across chips, taling into account
expert collocation and parallelization strategies.
"""

from dataclasses import dataclass
from math import ceil, gcd, isclose

from scalescope.decode_layout_trace import NULL_LAYOUT_TRACE, P, TensorRecord
from scalescope.model_spec import TransformerConfig
from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_all_reduce_on_nd_topology,
    hockney_model_reduce_scatter_on_nd_topology,
    hockney_model_send_recv_on_nd_topology,
    hockney_model_total_exchange_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.topology_spec import LogicalTopology, LogicalTopologyView
from scalescope.workload_spec import (
    DecodeCrossDomainEPMode,
    ExpertDomainPlacementKind,
    ExpertRoutingDistribution,
    MoEA2ACommMode,
    MoERouterSyncMode,
    MoESpecV2,
    ResolvedBatchSizes,
)

# Bytes per entry in the A2Av size vector: one int32 send or recv count
# per remote peer, exchanged before the variable-size payload so each
# destination can post its recv.
A2AV_SIZE_VECTOR_DTYPE_BYTES = 4


@dataclass
class ExpertXPUPlacement:
    # Colocation-diluted routing imbalance, excluding top-k fanout.
    effective_expert_imbalance: float
    num_shared_experts_per_dp_rank: int = 0
    num_regular_experts_per_dp_rank: int = 0
    num_all_experts_per_dp_rank: int = 0
    total_distinct_experts_per_set: int = 0
    regular_expert_traffic_multiplier: float = 0.0
    num_complete_expert_sets: int = 1
    # EP/placement-wide fields needed by comm/compute
    full_expert_set_xpu_count: int = 1
    num_colocated_experts: int = 1
    # Number of colocated expert groups in one full expert set.
    # Colocated group = the colocated expert bundle routed together and hosted by one TP group.
    num_colocated_groups_per_set: int = 1
    # Chips per colocated expert group (TP degree inside one group).
    xpus_per_colocated_group: int = 1
    ep_chips_per_pp_stage: int = 0
    # Size relationship between the attention TP domain and the expert
    # set's chip footprint. At most one of these is > 1 at any time:
    #   - attn_domains_per_full_expert_set > 1 ↔ expert set is larger than
    #     one attention domain → multiple attention domains feed into one
    #     expert set ("centralized pool", currently unsupported for decode).
    #   - full_expert_sets_per_attn_domain > 1 ↔ one attention domain
    #     hosts multiple full expert sets ("small experts per attention
    #     domain", supported).
    #   - both == 1 ↔ expert set fills exactly one attention domain.
    # Defaults are None so missing assignment by a constructor surfaces
    # as a TypeError at the first comparison, instead of silently being
    # treated as 1.
    attn_domains_per_full_expert_set: int | None = None
    full_expert_sets_per_attn_domain: int | None = None
    # Decode router shard factor (resolved by orchestration).
    router_shard_factor: int = 1
    # Router synchronization policy consumed by TP comm strategies.
    # - replicated_router: router logits available locally
    # - sharded_router: separate normalization and router-sync collectives
    # - normalization_deferred_sharded_router: one packed collective for both
    router_sync_mode: MoERouterSyncMode = MoERouterSyncMode.REPLICATED_ROUTER
    # TP-MoE redistribution optimization toggle:
    # - OFF: regular routed MoE A2A.
    # - ON: send tokens (or token shards) to an expert shard first, then run all-gather
    #   so that shard is available across the expert TP shards.
    tp_moe_a2a_replication_optimization_enabled: bool = True


@dataclass(frozen=True)
class ResolvedMoETokenScopes:
    """Validated batch and token ownership across attention domains and expert sets."""

    global_batch_size: int
    tokens_per_sequence: int
    attention_domain_count: int
    # Physical attention domains that carry at least one request/token. This
    # equals attention_domain_count in dense mode. In sparse wide-EP decode it
    # is smaller; inactive domains still retain their expert ownership.
    active_attention_domain_count: int
    complete_expert_set_count: int
    attn_domains_per_full_expert_set: int
    full_expert_sets_per_attn_domain: int
    # Number of attention-domain-to-expert-set feeder paths that carry tokens.
    # Feeder paths from inactive sparse attention domains are excluded.
    # This is separate from active_attention_domain_count because one active
    # domain can feed multiple expert sets. It closes the token accounting:
    # tokens_per_feeder_set * active_feeder_count == tokens_global.
    active_feeder_count: int
    batch_size_per_attention_domain: int
    batch_size_per_full_expert_set: int
    batch_size_per_feeder_set: int
    tokens_global: int
    tokens_per_attention_domain: int
    tokens_per_full_expert_set: int
    tokens_per_feeder_set: int


def resolve_moe_token_scopes(
    *,
    resolved_batches: ResolvedBatchSizes,
    tokens_per_sequence: int,
    attn_domains_per_full_expert_set: int,
    full_expert_sets_per_attn_domain: int,
) -> ResolvedMoETokenScopes:
    """Extend canonical workload batch sizes with exact MoE token scopes."""

    values = {
        "tokens_per_sequence": tokens_per_sequence,
        "attn_domains_per_full_expert_set": attn_domains_per_full_expert_set,
        "full_expert_sets_per_attn_domain": full_expert_sets_per_attn_domain,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    global_batch_size = resolved_batches.global_batch_size
    attention_domain_count = resolved_batches.attention_domain_count
    active_attention_domain_count = resolved_batches.active_attention_domain_count
    complete_expert_set_count = resolved_batches.complete_expert_set_count
    batch_size_per_attention_domain = resolved_batches.batch_size_per_dp
    batch_size_per_full_expert_set = resolved_batches.batch_size_per_full_expert_set
    resolved_values = {
        "global_batch_size": global_batch_size,
        "attention_domain_count": attention_domain_count,
        "active_attention_domain_count": active_attention_domain_count,
        "complete_expert_set_count": complete_expert_set_count,
        "batch_size_per_dp": batch_size_per_attention_domain,
        "batch_size_per_full_expert_set": batch_size_per_full_expert_set,
    }
    for name, value in resolved_values.items():
        if value <= 0:
            raise ValueError(f"resolved_batches.{name} must be > 0, got {value}")
    if active_attention_domain_count > attention_domain_count:
        raise ValueError(
            "resolved_batches.active_attention_domain_count must not exceed "
            f"attention_domain_count: {active_attention_domain_count} > "
            f"{attention_domain_count}"
        )
    if batch_size_per_attention_domain * active_attention_domain_count != global_batch_size:
        raise ValueError(
            "Resolved active attention-domain batch does not reconstruct the global batch: "
            f"{batch_size_per_attention_domain} * {active_attention_domain_count} "
            f"!= {global_batch_size}"
        )
    if batch_size_per_full_expert_set * complete_expert_set_count != global_batch_size:
        raise ValueError(
            "Resolved full-expert-set batch does not reconstruct the global batch: "
            f"{batch_size_per_full_expert_set} * {complete_expert_set_count} "
            f"!= {global_batch_size}"
        )

    left_edges = attention_domain_count * full_expert_sets_per_attn_domain
    right_edges = complete_expert_set_count * attn_domains_per_full_expert_set
    if left_edges != right_edges:
        raise ValueError(
            "Attention-domain/expert-set incidence must close exactly: "
            f"{attention_domain_count} * {full_expert_sets_per_attn_domain} "
            f"!= {complete_expert_set_count} * {attn_domains_per_full_expert_set}"
        )
    # Sparse attention domains occur in small-batch wide-EP decode when there
    # are fewer active sequences than physical attention domains. One request
    # is assigned to each active domain, while the remaining domains perform
    # no attention work but still retain their expert ownership.
    sparse_attention_domains = active_attention_domain_count < attention_domain_count
    if sparse_attention_domains:
        if complete_expert_set_count != 1:
            raise ValueError(
                "Sparse attention-domain token scopes require exactly one complete "
                f"expert set, got {complete_expert_set_count}"
            )
        if full_expert_sets_per_attn_domain != 1:
            raise ValueError(
                "Sparse attention-domain token scopes require "
                "full_expert_sets_per_attn_domain=1, got "
                f"{full_expert_sets_per_attn_domain}"
            )
        if attn_domains_per_full_expert_set != attention_domain_count:
            raise ValueError(
                "Sparse attention-domain token scopes require one wide-EP set spanning "
                "every physical attention domain: "
                f"{attn_domains_per_full_expert_set} != {attention_domain_count}"
            )
        if batch_size_per_attention_domain != 1:
            raise ValueError(
                "Sparse attention-domain token scopes require one request per active "
                f"feeder, got {batch_size_per_attention_domain}"
            )
        if active_attention_domain_count != global_batch_size:
            raise ValueError(
                "Sparse attention-domain token scopes require one active feeder per "
                f"request: {active_attention_domain_count} != {global_batch_size}"
            )
        batch_size_per_feeder_set = 1
        active_feeder_count = active_attention_domain_count
    else:
        if batch_size_per_attention_domain % full_expert_sets_per_attn_domain != 0:
            raise ValueError(
                f"batch_size_per_attention_domain ({batch_size_per_attention_domain}) must be "
                "divisible by full_expert_sets_per_attn_domain "
                f"({full_expert_sets_per_attn_domain})"
            )
        if batch_size_per_full_expert_set % attn_domains_per_full_expert_set != 0:
            raise ValueError(
                f"batch_size_per_full_expert_set ({batch_size_per_full_expert_set}) must be "
                "divisible by attn_domains_per_full_expert_set "
                f"({attn_domains_per_full_expert_set})"
            )

        # A feeder-set scope is one attention-domain -> expert-set edge. Derive
        # its batch size independently from each endpoint of that edge.
        batch_size_per_feeder_set_via_attention_domain = (
            batch_size_per_attention_domain // full_expert_sets_per_attn_domain
        )
        batch_size_per_feeder_set_via_expert_set = (
            batch_size_per_full_expert_set // attn_domains_per_full_expert_set
        )
        if (
            batch_size_per_feeder_set_via_attention_domain
            != batch_size_per_feeder_set_via_expert_set
        ):
            raise AssertionError(
                "Feeder-set batch derivations disagree: "
                "via attention domain="
                f"{batch_size_per_feeder_set_via_attention_domain}, "
                f"via expert set={batch_size_per_feeder_set_via_expert_set}"
            )
        batch_size_per_feeder_set = batch_size_per_feeder_set_via_attention_domain
        active_feeder_count = active_attention_domain_count * full_expert_sets_per_attn_domain

    tokens_global = global_batch_size * tokens_per_sequence
    tokens_per_attention_domain = batch_size_per_attention_domain * tokens_per_sequence
    tokens_per_full_expert_set = batch_size_per_full_expert_set * tokens_per_sequence
    tokens_per_feeder_set = batch_size_per_feeder_set * tokens_per_sequence
    if tokens_per_attention_domain * active_attention_domain_count != tokens_global:
        raise AssertionError("Attention-domain token scopes do not reconstruct global tokens")
    if tokens_per_full_expert_set * complete_expert_set_count != tokens_global:
        raise AssertionError("Expert-set token scopes do not reconstruct global tokens")
    if tokens_per_feeder_set * active_feeder_count != tokens_global:
        raise AssertionError("Feeder-set token scopes do not reconstruct global tokens")

    return ResolvedMoETokenScopes(
        global_batch_size=global_batch_size,
        tokens_per_sequence=tokens_per_sequence,
        attention_domain_count=attention_domain_count,
        active_attention_domain_count=active_attention_domain_count,
        complete_expert_set_count=complete_expert_set_count,
        attn_domains_per_full_expert_set=attn_domains_per_full_expert_set,
        full_expert_sets_per_attn_domain=full_expert_sets_per_attn_domain,
        active_feeder_count=active_feeder_count,
        batch_size_per_attention_domain=batch_size_per_attention_domain,
        batch_size_per_full_expert_set=batch_size_per_full_expert_set,
        batch_size_per_feeder_set=batch_size_per_feeder_set,
        tokens_global=tokens_global,
        tokens_per_attention_domain=tokens_per_attention_domain,
        tokens_per_full_expert_set=tokens_per_full_expert_set,
        tokens_per_feeder_set=tokens_per_feeder_set,
    )


def _logical_tag_string(tag) -> str:
    """Return the stable string identifier for an enum-backed or plain logical tag."""
    return tag.value if hasattr(tag, "value") else str(tag)


def _physical_axis_sizes(view: LogicalTopology) -> dict[str, int]:
    """Return each physical axis's aggregate logical extent in ``view``.

    Multiple logical dimensions may form a Cartesian factorization of one
    physical domain. Their sizes are multiplied so layout comparisons operate
    on physical rank ownership rather than individual logical tag names.
    """
    sizes: dict[str, int] = {}
    for dimension in view.dimensions:
        physical_name = dimension.maps_to_physical_dimension_name
        if physical_name in sizes:
            sizes[physical_name] *= dimension.size
        else:
            sizes[physical_name] = dimension.size
    return sizes


def _merge_logical_views(
    *,
    first_view: LogicalTopology,
    second_view: LogicalTopology,
    view_name: str,
) -> LogicalTopologyView:
    """Merge two disjoint logical views over the same physical topology.

    The merged view preserves both dimension lists and uses concurrent
    bandwidth sharing. Duplicate logical tags are rejected because they would
    make participant extents and subsequent quotient calculations ambiguous.
    """
    if first_view.physical_topology is not second_view.physical_topology:
        raise ValueError(f"{view_name}: views must reference the same physical topology object")

    dimensions = [*first_view.dimensions, *second_view.dimensions]
    tags = [_logical_tag_string(dimension.logical_tag) for dimension in dimensions]
    if len(tags) != len(set(tags)):
        raise ValueError(f"{view_name}: duplicate logical tags are not allowed: {tags}")

    original_topology_ref = getattr(first_view, "original_topology_ref", first_view)

    return LogicalTopologyView(
        name=view_name,
        physical_topology=first_view.physical_topology,
        dimensions=dimensions,
        original_topology_ref=original_topology_ref,
        global_sharing_context="concurrent_baseline",
    )


def _exact_layout_quotient_view(
    *,
    numerator_view: LogicalTopology,
    denominator_view: LogicalTopology,
    view_name: str,
) -> LogicalTopologyView:
    """Build the exact participant quotient from two axis-aligned layouts.

    Numerator and denominator must cover the same physical axes, and every
    numerator extent must be divisible by its denominator extent. The physical
    quotient is then factored back through the numerator's logical dimensions
    in their existing order, producing a resized view suitable for pricing the
    collective that reconciles the two layouts. Unsupported refinements or
    residual unmapped factors fail explicitly.
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
                f"{view_name}: non-divisible physical extent on {physical_name}: "
                f"{numerator_size} / {denominator_size}"
            )
        quotient_by_physical[physical_name] = numerator_size // denominator_size

    new_sizes: dict[str, int] = {}
    remaining = dict(quotient_by_physical)
    for dimension in numerator_view.dimensions:
        physical_name = dimension.maps_to_physical_dimension_name
        factor = gcd(remaining[physical_name], dimension.size)
        new_sizes[_logical_tag_string(dimension.logical_tag)] = factor
        remaining[physical_name] //= factor

    if any(size != 1 for size in remaining.values()):
        raise ValueError(f"{view_name}: quotient has unmapped physical extents: {remaining}")

    return numerator_view.create_resized_view(view_name=view_name, new_sizes=new_sizes)


def _resolve_attn_expert_size_ratio(
    *, full_expert_set_xpu_count: int, attn_tp_size: int
) -> tuple[int, int]:
    """Resolve the size-ratio pair between the expert set and one attention TP domain.

    Returns ``(attn_domains_per_full_expert_set, full_expert_sets_per_attn_domain)``.

    The two returned values capture how the expert set's chip footprint
    relates to one attention TP domain's chip footprint. They are
    mutually exclusive (at most one is > 1):
      - ``attn_domains_per_full_expert_set > 1`` ↔ the expert set spans
        multiple attention TP domains (centralized pool).
      - ``full_expert_sets_per_attn_domain > 1`` ↔ a single attention TP
        domain hosts multiple full expert sets.

    Invariants enforced:
      - Both values are positive integers.
      - The larger of ``full_expert_set_xpu_count`` and ``attn_tp_size``
        must be an exact multiple of the smaller; otherwise we raise
        ``ValueError`` rather than silently rounding (per the
        explicit-only rule).
    """
    if full_expert_set_xpu_count <= 0:
        raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_expert_set_xpu_count}")
    if attn_tp_size <= 0:
        raise ValueError(f"attn_tp_size must be > 0, got {attn_tp_size}")

    if full_expert_set_xpu_count == attn_tp_size:
        return 1, 1
    if full_expert_set_xpu_count > attn_tp_size:
        if full_expert_set_xpu_count % attn_tp_size != 0:
            raise ValueError(
                f"full_expert_set_xpu_count ({full_expert_set_xpu_count}) must be a "
                f"multiple of attn_tp_size ({attn_tp_size}) when the expert set is "
                f"larger than the attention TP domain."
            )
        return full_expert_set_xpu_count // attn_tp_size, 1
    if attn_tp_size % full_expert_set_xpu_count != 0:
        raise ValueError(
            f"attn_tp_size ({attn_tp_size}) must be a multiple of "
            f"full_expert_set_xpu_count ({full_expert_set_xpu_count}) when the "
            f"attention TP domain is larger than the expert set."
        )
    return 1, attn_tp_size // full_expert_set_xpu_count


@dataclass(frozen=True)
class ExpertActivationCardinality:
    """Per-step expected active-expert cardinality under top-k routing.

    Computed under the sole ``bottleneck_group_collector`` model (worst-loaded
    colocated group = the ``c`` hottest experts of the global Zipf(rho)).

    Fields:
    - e_support: full expert-set size ``E`` (``num_regular_experts``). The
      distribution has full support.
    - e_active: expected count of distinct experts touched on the worst-loaded
      colocated group this step. For assignment share ``p_k``, top-k distinct
      routing gives per-token selection probability ``top_k * p_k`` and active
      probability ``1 - (1 - top_k * p_k)^tokens_per_set``.
    - active_expert_fraction: ``e_active / num_colocated_experts`` (i.e.
      ``e_active / c``) in ``[0, 1]``. The multiplier applied to resident-weight
      memory-traffic terms under ``RESIDENT_ACTIVE_SUBSET``; dividing by ``c``
      (the chip's resident set, not the full expert set) exposes the per-shard
      weight read.
    """

    e_support: float
    e_active: float
    active_expert_fraction: float


def validate_expert_imbalance_realizability(
    *,
    expert_imbalance: float,
    num_regular_experts: int,
    num_selected_experts: int,
) -> None:
    """Fail fast if ``expert_imbalance`` is outside the top-k realizable range.

    This bound is purely combinatorial: it follows from top-k routing alone and
    is independent of the routing-load distribution (Zipf or otherwise). The
    distribution shape only matters later, when we turn ``rho`` into an active-
    expert count; here we just reject a ``rho`` that no top-k assignment can
    produce.

    Definitions. With ``E = num_regular_experts``, ``k = num_selected_experts``,
    and ``T`` tokens each routing to ``k`` *distinct* experts, let ``p_i`` be
    expert ``i``'s **share** of all routing assignments (so ``sum_i p_i = 1``).
    The mean share is ``1/E``, and ``rho = p_max / (1/E) = p_max * E`` is the
    max-to-mean ratio.

    Upper bound. The total number of assignments is ``T * k``. Because a token's
    ``k`` experts are distinct, any single expert receives at most one slot per
    token, i.e. at most ``T`` assignments, so its share obeys
    ``p_i <= T / (T * k) = 1/k``. Substituting ``p_max <= 1/k`` into
    ``rho = p_max * E`` gives ``rho <= E / k = num_regular_experts /
    num_selected_experts``.

    Lower bound. The max share is always at least the mean, so ``rho >= 1``
    (``rho == 1`` is perfectly balanced routing).

    Realizable range: ``rho in [1, num_regular_experts / num_selected_experts]``.
    Example: ``E=8, k=2`` gives ``[1, 4]`` — even if every token picked the same
    two experts, each would take ``T`` of the ``2T`` slots (share ``1/2``,
    ``rho = 4``); you cannot do worse because a token cannot spend both slots on
    one expert.

    Dense models are represented as a single routed expert
    (``num_regular_experts == num_selected_experts == 1``) and ARE checked: the
    range collapses to ``rho == 1`` (a single expert cannot be imbalanced). The
    check is skipped only for attention-only models with no FFN/routing
    (``num_regular_experts == 0`` or ``num_selected_experts == 0``).
    """
    if num_regular_experts <= 0 or num_selected_experts <= 0:
        return
    upper = num_regular_experts / num_selected_experts
    if not (1.0 <= expert_imbalance <= upper):
        raise ValueError(
            f"expert_imbalance={expert_imbalance} is outside the top-k realizable range "
            f"[1, num_regular_experts/num_selected_experts] = [1, {upper}] for "
            f"E={num_regular_experts}, top_k={num_selected_experts}. Top-k routing with "
            f"distinct experts per token bounds rho <= E/top_k."
        )


_ZIPF_EXPONENT_CACHE: dict[tuple[int, float], float] = {}


def _solve_zipf_exponent(*, num_regular_experts: int, expert_imbalance: float) -> float:
    """Solve for the Zipf exponent ``s`` such that ``E / H_{E,s} = rho``.

    H_{E,s} = sum_{k=1..E} k^{-s} is the generalized harmonic number, and the
    Zipf distribution sets ``p_k = k^{-s} / H_{E,s}``. The spec ties
    ``rho = p_max * E = p_1 * E = E / H_{E,s}``, so given (E, rho) we recover
    ``s`` by a monotone 1-D root find:

      f(s) := H_{E,s} - E/rho      (strictly decreasing in s on s > 0;
                                    f(0) = E - E/rho > 0 for rho > 1,
                                    f(+inf) -> 1 - E/rho < 0)

    At ``rho == 1`` the target H = E is hit only at ``s = 0`` (uniform);
    we short-circuit to avoid log(1)/0 issues.

    Returns the non-negative exponent ``s``.
    """
    if num_regular_experts <= 0:
        raise ValueError(f"num_regular_experts must be > 0, got {num_regular_experts}")
    if expert_imbalance < 1.0:
        raise ValueError(f"expert_imbalance must be >= 1, got {expert_imbalance}")
    cache_key = (int(num_regular_experts), float(expert_imbalance))
    cached = _ZIPF_EXPONENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if num_regular_experts == 1:
        _ZIPF_EXPONENT_CACHE[cache_key] = 0.0
        return 0.0
    target = num_regular_experts / expert_imbalance
    if abs(target - num_regular_experts) <= 1e-12:
        _ZIPF_EXPONENT_CACHE[cache_key] = 0.0
        return 0.0

    def h(s: float) -> float:
        if s == 0.0:
            return float(num_regular_experts)
        total = 0.0
        for k in range(1, num_regular_experts + 1):
            total += k ** (-s)
        return total

    lo, hi = 0.0, 1.0
    while h(hi) > target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError(
                f"Zipf exponent search failed to bracket rho={expert_imbalance}, "
                f"E={num_regular_experts} (hi={hi}, H(hi)={h(hi)}, target={target})."
            )
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    s = 0.5 * (lo + hi)
    _ZIPF_EXPONENT_CACHE[cache_key] = s
    return s


def _zipf_active_experts_the_hottest_plus_random(
    *,
    num_regular_experts: int,
    num_selected_experts: int,
    expert_imbalance: float,
    num_colocated_experts: int,
    tokens_per_set: float,
) -> float:
    """Expected distinct active experts on the straggler under *random* placement.

    Under load-agnostic (random/hash) placement the worst-loaded chip holds the
    single hottest expert (#1, placement-invariant — it is hot wherever it lands)
    plus ``c - 1`` experts sampled from the *tail* of the global Zipf, rather than
    the ``c`` globally hottest. Here ``p_k`` is expert ``k``'s share of all
    token-expert assignments. Because top-k routing selects distinct experts, a
    token selects expert ``k`` with marginal probability
    ``num_selected_experts * p_k``. With
    ``a_k = 1 - (1 - num_selected_experts * p_k)^tokens_per_set``, the expected
    distinct active count is

        E_active_chip = a_1 + (c - 1) * mean_{k=2..E} a_k

    i.e. one hottest expert plus ``c - 1`` average tail experts. The fillers use
    the cold-tail mean ``mean a_{2..E}`` (not the next-hottest ``a_2..a_c``).

    ``E_active`` is left unfloored: this is an *expectation* over random placement
    and routing, so a value below 1.0 is the honest answer. At small token count
    even the hottest expert can be missed, and the tail fillers are smaller still,
    so the expected distinct-active count is genuinely fractional (the straggler
    reads less than one full expert's weights per step on average). Flooring at
    1.0 would force a full per-expert HBM read every step in that regime,
    over-counting weight traffic and inflating decode step time at small batch /
    low top-k. Leaving it unfloored keeps the correct small-token asymptote while
    still rising to the full Zipf active count as the token count grows.

    Corners:
      c = 1     : a_1
      c = E     : a_1 + (E-1)·mean_{k≥2} a_k = sum_{k=1..E} a_k (full Zipf count)
      rho = 1   : all a_k equal ⇒ plain coupon collector (c·a)
    """
    if tokens_per_set <= 0:
        return 0.0
    s = _solve_zipf_exponent(
        num_regular_experts=num_regular_experts, expert_imbalance=expert_imbalance
    )
    if s == 0.0:
        # Uniform routing (rho == 1): every expert has assignment share 1/E and
        # per-token selection probability top_k/E. The hottest expert and the
        # cold-tail average therefore coincide. No floor here (or below): this
        # is an expectation, so a value < 1 at tiny token count is intentional.
        per_expert_active = (
            1.0 - (1.0 - num_selected_experts / num_regular_experts) ** tokens_per_set
        )
        return num_colocated_experts * per_expert_active
    # Skewed routing (rho > 1, s > 0). Zipf normalizer over all E experts:
    # H_{E,s} = sum_{k=1..E} k^{-s}, giving per-expert probability p_k = k^{-s}/H.
    h_es = 0.0
    for k in range(1, num_regular_experts + 1):
        h_es += k ** (-s)
    # The hottest expert (k = 1, p_1 = 1/H_{E,s}) is placement-invariant — it is
    # hot wherever it lands, so it always sits on the straggler. Its per-token
    # selection probability is top_k * p_1.
    p_1 = (1.0 ** (-s)) / h_es
    hottest_selection_probability = num_selected_experts * p_1
    if not 0.0 <= hottest_selection_probability <= 1.0:
        raise ValueError(
            "Hottest-expert selection probability must be in [0, 1], got "
            f"{hottest_selection_probability} for E={num_regular_experts}, "
            f"top_k={num_selected_experts}, rho={expert_imbalance}"
        )
    a_1 = 1.0 - (1.0 - hottest_selection_probability) ** tokens_per_set
    if num_colocated_experts == 1 or num_regular_experts == 1:
        # Only the hottest expert is hosted (c = 1) or it is the sole expert.
        return a_1
    # The other c - 1 experts on the straggler are sampled at random from the cold
    # tail (k = 2..E), so each contributes the tail's *average* activation rather
    # than the next-hottest experts. Compute that mean activation over k = 2..E.
    tail_sum = 0.0
    for k in range(2, num_regular_experts + 1):
        p_k = (k ** (-s)) / h_es
        selection_probability = num_selected_experts * p_k
        if not 0.0 <= selection_probability <= 1.0:
            raise ValueError(
                "Expert selection probability must be in [0, 1], got "
                f"{selection_probability} for expert_rank={k}, "
                f"E={num_regular_experts}, top_k={num_selected_experts}, "
                f"rho={expert_imbalance}"
            )
        tail_sum += 1.0 - (1.0 - selection_probability) ** tokens_per_set
    tail_mean = tail_sum / (num_regular_experts - 1)
    # Expected distinct-active count = the always-present hottest expert plus the
    # c - 1 random tail fillers at their average activation. Left unfloored on
    # purpose (expectation, not a worst-case bound).
    return a_1 + (num_colocated_experts - 1) * tail_mean


def compute_expert_activation_cardinality(
    *,
    tokens_per_set: float,
    num_selected_experts: int,
    num_regular_experts: int,
    num_colocated_experts: int,
    expert_imbalance: float,
    routing_distribution: ExpertRoutingDistribution,
) -> ExpertActivationCardinality:
    """Compute the active-expert quantity used for the per-chip weight-read
    multiplier under ``RESIDENT_ACTIVE_SUBSET``.

      bottleneck_group_collector_random (expected straggler under random
      placement; holds hottest #1 + (c-1) random tail experts, unfloored):
          p_k       = k^{-s} / H_{E,s},  s solved from E / H_{E,s} = rho
          q_k       = top_k * p_k
          a_k       = 1 - (1 - q_k)^tokens_per_set
          E_active  = a_1 + (c - 1) * mean_{k=2..E} a_k
          frac      = E_active / c
      An expectation (so unfloored); corner-exact at c=1, c=E, and rho=1.
      See `_zipf_active_experts_the_hottest_plus_random`.

    Why occupancy. ``p_k`` is expert ``k``'s share of all routing assignments.
    Each token selects ``num_selected_experts`` distinct experts, so its
    marginal probability of selecting expert ``k`` is ``q_k = top_k * p_k``.
    Across ``tokens_per_set`` independent tokens, expert ``k`` is missed with
    probability ``(1 - q_k)^tokens_per_set`` and therefore hit at least once
    with probability ``1 - (1 - q_k)^tokens_per_set``. Summing that hit
    probability over the ``c`` hosted experts gives the expected number of
    distinct weight shards the worst chip must read this step. The skew sets
    how fast experts saturate: under a steep Zipf only the few hot experts are
    commonly selected, while under uniform routing enough tokens eventually
    touch all ``c``.

      Step time is set by the worst-loaded chip, so this multiplier is the
      physically-grounded choice. The memory model multiplies resident weight
      by ``frac``; since ``frac = E_active / c`` this equals
      ``E_active * (w / TP_MoE)``, exposing the per-shard read so 2D TP_MoE
      (finer sharding) shrinks an outlier's bottleneck. At c=1 it returns
      ``a_1`` (the hottest expert's hit probability, < 1 at small m). At c=E it
      becomes the full Zipf active count (within-chip skew preserved).

    ``rho = expert_imbalance`` is the **raw** (un-damped) per-expert
    max/mean routing-load ratio at single-expert placement (SC26 §II-D). The
    bottleneck path uses raw ``rho`` because the straggler model captures
    worst-group concentration directly (no ``rho_eff`` damping).

    Returned fields:
    - ``e_support``: effective support size of the distribution (``E``).
    - ``e_active``: the worst-chip distinct-active-expert count.
    - ``active_expert_fraction``: the multiplier to apply to resident-weight
      memory traffic — ``e_active`` divided by ``c`` (the bottleneck
      "denominator" is the chip's resident set, not the full expert set).

    Defense-in-depth: validates ``expert_imbalance`` against the top-k
    realizability bound before computing.
    """
    if num_regular_experts <= 0:
        raise ValueError(
            f"num_regular_experts must be > 0 to define active-expert cardinality, "
            f"got {num_regular_experts}"
        )
    if num_selected_experts <= 0:
        raise ValueError(f"num_selected_experts must be > 0, got {num_selected_experts}")
    if num_colocated_experts <= 0:
        raise ValueError(f"num_colocated_experts must be > 0, got {num_colocated_experts}")
    if num_colocated_experts > num_regular_experts:
        raise ValueError(
            f"num_colocated_experts ({num_colocated_experts}) cannot exceed "
            f"num_regular_experts ({num_regular_experts})"
        )
    if tokens_per_set < 0:
        raise ValueError(f"tokens_per_set must be >= 0, got {tokens_per_set}")
    validate_expert_imbalance_realizability(
        expert_imbalance=expert_imbalance,
        num_regular_experts=num_regular_experts,
        num_selected_experts=num_selected_experts,
    )

    # Distinct active experts on the straggler chip. Dividing by c makes the
    # memory model's `resident * frac` equal `e_active * (w/TP_MoE)`, exposing the
    # per-shard weight read so finer sharding (2D TP_MoE) shrinks the outlier
    # bottleneck. Under random placement the straggler holds the hottest expert
    # (#1) plus c-1 cold-tail-average fillers (expectation, unfloored).
    match routing_distribution:
        case ExpertRoutingDistribution.BOTTLENECK_GROUP_COLLECTOR_RANDOM:
            active_experts_estimator = _zipf_active_experts_the_hottest_plus_random
        case _:
            raise ValueError(f"Unsupported routing_distribution: {routing_distribution}")

    e_support = float(num_regular_experts)
    if tokens_per_set <= 0:
        e_active = 0.0
        per_chip_frac = 0.0
    else:
        e_active = active_experts_estimator(
            num_regular_experts=num_regular_experts,
            num_selected_experts=num_selected_experts,
            expert_imbalance=expert_imbalance,
            num_colocated_experts=num_colocated_experts,
            tokens_per_set=tokens_per_set,
        )
        per_chip_frac = e_active / num_colocated_experts
    active_expert_fraction = per_chip_frac

    return ExpertActivationCardinality(
        e_support=e_support,
        e_active=e_active,
        active_expert_fraction=active_expert_fraction,
    )


def dilute_expert_imbalance(
    *, expert_imbalance: float, num_regular_experts: int, num_colocated_experts: int
) -> float:
    """Damp expert imbalance as colocation increases (dilute / single-outlier law).

    The one straggler law used on the token path. Under a Zipf load the worst
    chip is driven by one dominant expert diluted by ``c - 1`` cold fillers, so
    the surviving imbalance attenuates as ``1/c``:

        effective_imbalance = 1 + (rho - 1) * A(c),   A(c) = (E - c) / (c * (E - 1))

    where E is total regular experts and c is colocated experts per XPU (clamped
    to ``[1, E]``). This is analytically exact for the spike+uniform
    (dominant-outlier) load and is the correct exponent for Zipf routing.

    Properties:
    - c = 1  -> A = 1 -> effective imbalance = expert_imbalance
    - c = E  -> A = 0 -> effective imbalance = 1
    - monotonic and continuous for 1 <= c <= E
    """
    if num_regular_experts <= 1:
        return 1.0

    c = min(max(1, int(num_colocated_experts)), int(num_regular_experts))
    e = int(num_regular_experts)
    base = max(0.0, float(expert_imbalance) - 1.0)
    a = max(0.0, (e - c) / (c * (e - 1)))
    return 1.0 + base * a


def infer_expert_placement_per_xpu(
    *,
    model_spec: TransformerConfig,
    moe_v2: MoESpecV2,
    attn_tp_size: int,
    xpus_per_colocated_group: int,
    ep_chips_per_pp_stage: int,
    router_sync_mode: MoERouterSyncMode,
    router_shard_factor: int,
    tp_moe_a2a_replication_optimization_enabled: bool,
) -> ExpertXPUPlacement:
    """Infer per-XPU expert placement within a DP group.

    Compute how many routed and shared experts are hosted per DP rank,
    how many DP ranks are required to hold a complete expert set, and how many complete
    expert sets fit into the EP domain of a single PP stage. It also derives helper
    quantities used by communication and compute models (e.g., traffic multipliers).
    We also define explicit expert-group metadata:
    - expert group: the colocated expert bundle served by one TP expert group
    - num_colocated_groups_per_set: number of such groups in one full expert set
    - xpus_per_colocated_group: chips per colocated expert group

    NB: shared experts are not yet supported

    Args:
    - model_spec: Transformer configuration. Uses:
        - num_regular_experts
        - num_shared_experts
        - num_selected_experts (for traffic multiplier)
    - moe_v2: MoE configuration (MoESpecV2). Uses:
        - num_colocated_experts: number of experts colocated per XPU (>= 1)
        - shared_expert_copies: replication factor for shared experts (optional)
        - expert_imbalance: traffic multiplier (>= 1.0)
        - EP_SU (logical tag size) defines the high-bandwidth expert group size.
    - attn_tp_size: TP size of the attention path (chips per DP rank for TP_Attn).
      Used to divide a complete expert set across DP ranks.
    - xpus_per_colocated_group: expert tensor-parallel degree (chips per colocated group).
    - ep_chips_per_pp_stage: total EP chips available per PP stage (DP_Attn * TP_Attn).
    - router_sync_mode: router synchronization policy for TP comm strategies.
    - router_shard_factor: number of ranks across which the router weight is sharded.
      For replicated router: 1. For a sharded router mode: Y_eff (2D TP), or
      attn_tp_size for the ordinary 1D sharded router.
    - tp_moe_a2a_replication_optimization_enabled: enables TP-MoE expert-input redistribution optimization.

    Returns
    - ExpertXPUPlacement
    """

    if router_shard_factor < 1:
        raise ValueError(f"Invalid router_shard_factor: {router_shard_factor}")

    num_shared = model_spec.num_shared_experts
    normalized_xpus_per_colocated_group = max(1, int(xpus_per_colocated_group))
    num_colocated_experts = int(moe_v2.num_colocated_experts)

    num_regular = model_spec.num_regular_experts
    total_distinct_experts = num_regular + num_shared * (moe_v2.shared_expert_copies or 1)
    if total_distinct_experts == 0:
        # No MoE experts in the model: treat expert placement as a no-op so sweeps can focus on
        # attention/projection behavior without needing EP divisibility constraints.
        return ExpertXPUPlacement(
            effective_expert_imbalance=1.0,
            num_shared_experts_per_dp_rank=0,
            num_regular_experts_per_dp_rank=0,
            num_all_experts_per_dp_rank=0,
            total_distinct_experts_per_set=0,
            regular_expert_traffic_multiplier=0.0,
            num_complete_expert_sets=1,
            full_expert_set_xpu_count=1,
            num_colocated_experts=num_colocated_experts,
            num_colocated_groups_per_set=0,
            xpus_per_colocated_group=normalized_xpus_per_colocated_group,
            ep_chips_per_pp_stage=int(ep_chips_per_pp_stage),
            attn_domains_per_full_expert_set=1,
            full_expert_sets_per_attn_domain=1,
            router_shard_factor=router_shard_factor,
            router_sync_mode=router_sync_mode,
            tp_moe_a2a_replication_optimization_enabled=tp_moe_a2a_replication_optimization_enabled,
        )
    # Colocation reduces the number of XPUs needed to host a full set of distinct experts,
    # regardless of whether experts are TP-sharded. Account for both expert TP degree and colocation.
    full_expert_set_xpu_count = ceil(
        (total_distinct_experts * normalized_xpus_per_colocated_group) / num_colocated_experts
    )
    num_colocated_groups_per_set = ceil(total_distinct_experts / num_colocated_experts)
    # No-spill invariant: each colocated expert group must live entirely
    # inside a single attention TP domain. We rely on this in the locality
    # model used by `calculate_moe_comm_hierarchical_total_exchange` (remote
    # fraction = 1 - 1/G under "groups don't cross attention domains").
    # Concretely: attn_tp_size must be a positive integer multiple of
    # xpus_per_colocated_group. This rejects the (rare) case where a group's
    # q chips span multiple attention TP domains. Cross-domain expert sets
    # (attn_domains_per_full_expert_set > 1) are still supported as long as
    # groups themselves stay within one attention domain.
    if attn_tp_size % normalized_xpus_per_colocated_group != 0:
        raise ValueError(
            f"xpus_per_colocated_group ({normalized_xpus_per_colocated_group}) "
            f"must divide attn_tp_size ({attn_tp_size}) so that no colocated "
            "expert group spills across attention TP domains. Multi-attention-"
            "domain expert sets are supported, but a single group is required "
            "to live in one attention domain."
        )
    attn_domains_per_full_expert_set, full_expert_sets_per_attn_domain = (
        _resolve_attn_expert_size_ratio(
            full_expert_set_xpu_count=full_expert_set_xpu_count,
            attn_tp_size=attn_tp_size,
        )
    )
    num_routed_experts_per_dp_rank = num_regular // attn_domains_per_full_expert_set
    num_shared_experts_per_dp_rank = (
        num_shared * (moe_v2.shared_expert_copies or 1) // attn_domains_per_full_expert_set
    )
    num_of_all_experts_per_dp_rank = num_routed_experts_per_dp_rank + num_shared_experts_per_dp_rank

    # Determine how many complete expert sets fit into the available EP domain per PP stage
    if ep_chips_per_pp_stage % full_expert_set_xpu_count != 0:
        raise ValueError(
            f"EP chips ({ep_chips_per_pp_stage}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_complete_expert_sets = ep_chips_per_pp_stage // full_expert_set_xpu_count

    # Regular expert traffic multiplier with colocation-aware damping. The token
    # path always uses the dilute (1/c, Zipf-exact) straggler law, independent of
    # the routing-distribution selector: the dominant-outlier attenuation is the
    # better-justified exponent for skewed routing.
    effective_imbalance = dilute_expert_imbalance(
        expert_imbalance=moe_v2.expert_imbalance,
        num_regular_experts=num_regular,
        num_colocated_experts=num_colocated_experts,
    )
    regular_expert_traffic_multiplier = effective_imbalance * model_spec.num_selected_experts

    return ExpertXPUPlacement(
        effective_expert_imbalance=effective_imbalance,
        num_shared_experts_per_dp_rank=num_shared_experts_per_dp_rank,
        num_regular_experts_per_dp_rank=num_routed_experts_per_dp_rank,
        num_all_experts_per_dp_rank=num_of_all_experts_per_dp_rank,
        total_distinct_experts_per_set=int(total_distinct_experts),
        regular_expert_traffic_multiplier=regular_expert_traffic_multiplier,
        num_complete_expert_sets=int(num_complete_expert_sets),
        # EP/placement-wide fields
        full_expert_set_xpu_count=int(full_expert_set_xpu_count),
        num_colocated_experts=num_colocated_experts,
        num_colocated_groups_per_set=int(num_colocated_groups_per_set),
        xpus_per_colocated_group=normalized_xpus_per_colocated_group,
        ep_chips_per_pp_stage=int(ep_chips_per_pp_stage),
        attn_domains_per_full_expert_set=attn_domains_per_full_expert_set,
        full_expert_sets_per_attn_domain=full_expert_sets_per_attn_domain,
        router_shard_factor=router_shard_factor,
        router_sync_mode=router_sync_mode,
        tp_moe_a2a_replication_optimization_enabled=tp_moe_a2a_replication_optimization_enabled,
    )


def calculate_moe_comm_first_total_exchange(
    ep_topology_view,
    transformer_config,
    moe_config,
    token_scopes: ResolvedMoETokenScopes,
    float_size,
    tp_moe_y_shards: int,
    workload_spec=None,
    layout_trace=NULL_LAYOUT_TRACE,
    a2a_leg: str = "dispatch",
):
    return calculate_moe_comm_hierarchical_total_exchange(
        ep_topology_view,
        transformer_config,
        moe_config,
        token_scopes,
        float_size,
        workload_spec=workload_spec,
        tp_moe_y_shards=tp_moe_y_shards,
        layout_trace=layout_trace,
        a2a_leg=a2a_leg,
    )


def calculate_moe_comm_second_total_exchange(
    ep_topology_view,
    transformer_config,
    moe_config,
    token_scopes: ResolvedMoETokenScopes,
    float_size,
    tp_moe_y_shards: int,
    workload_spec=None,
    layout_trace=NULL_LAYOUT_TRACE,
):
    # Return payload matches dispatch (same routed tokens, transpose).
    # Size-vector A2A is dispatch-only; combine reuses those counts.
    return calculate_moe_comm_first_total_exchange(
        ep_topology_view,
        transformer_config,
        moe_config,
        token_scopes,
        float_size,
        workload_spec=workload_spec,
        tp_moe_y_shards=tp_moe_y_shards,
        layout_trace=layout_trace,
        a2a_leg="combine",
    )


def _price_dispatch_size_vector_or_zero(
    *,
    a2a_leg: str,
    topology: LogicalTopology,
    participant_count: int,
    layout_trace,
    stage: str,
    fabric: str,
    note: str,
) -> CommStats:
    """Charge the A2Av size vector on dispatch only.

    Combine is the transpose of dispatch: both sides already know the
    per-peer lengths from the outbound send / inbound recv counts, so
    the return leg does not post a second size collective.
    """
    if a2a_leg != "dispatch":
        return CommStats()
    size_stats = price_a2av_size_vector_exchange(
        topology=topology,
        participant_count=participant_count,
    )
    layout_trace.record_comm(
        stage,
        "A2A",
        fabric,
        participant_count,
        (participant_count - 1) * A2AV_SIZE_VECTOR_DTYPE_BYTES,
        size_stats,
        note=note,
    )
    return size_stats


def price_a2av_size_vector_exchange(
    *,
    topology: LogicalTopology,
    participant_count: int,
) -> CommStats:
    """Price the A2Av size-vector collective (per-peer send/recv counts).

    Variable-size dispatch cannot post the payload recv until every destination
    knows its inbound length. That is one integer of
    ``A2AV_SIZE_VECTOR_DTYPE_BYTES`` to each of the other
    ``participant_count - 1`` ranks on ``topology``. Payload activations are
    priced separately. ``participant_count`` must be > 1; D=1 decode never
    calls this. Combine must not call this: return recvs reuse dispatch counts.
    """
    if participant_count <= 1:
        raise ValueError(
            "A2Av size-vector exchange requires participant_count > 1, " f"got {participant_count}"
        )
    remote_peers = participant_count - 1
    bytes_per_sender = remote_peers * A2AV_SIZE_VECTOR_DTYPE_BYTES
    return hockney_model_send_recv_on_nd_topology(
        topology=topology,
        bytes_per_sender=bytes_per_sender,
    )


def calculate_moe_decode_cross_domain_exchange(
    *,
    ep_topology_view: LogicalTopology,
    transformer_config: TransformerConfig,
    moe_config,
    workload_spec,
    token_scopes: ResolvedMoETokenScopes,
    float_size: int,
    tp_moe_y_shards: int,
    layout_trace=NULL_LAYOUT_TRACE,
    a2a_leg: str,
) -> CommStats:
    """Price decode dispatch or return A2Av on EP_SO.

    Expert groups are owned evenly across attention domains. Without building
    a rank-level traffic matrix, the model starts from the balanced routed load
    per expert-group-sized endpoint and scales it by the colocation-diluted
    expert imbalance. This gives the busiest receiving endpoint directly.

    The exact per-feeder handoff prepositions each ``D / TP_MoE_Y`` token shard
    at every aligned local expert group. Therefore, the ``1/D`` local-owner
    fraction needs no EP_SU dispatch or return. Only the remote ``(D-1)/D``
    fraction is charged to EP_SO. Dispatch EP_SO is two Hockney phases: an
    int32 size-vector all-to-all among the ``D`` feeders, then the payload
    A2Av. Destinations cannot post the payload recv without the counts.
    Combine payload is the transpose of dispatch, so return recvs reuse those
    counts; no second size collective. After the return, the exact
    reduce-scatter and replica all-reduce restore the residual layout.

    This path assumes direct per-XPU EP_SO endpoints with no local
    ingress/egress staging. Topologies with centralized scale-out endpoints
    require a separate staging model.
    """

    if transformer_config.num_regular_experts == 0:
        return CommStats()
    if workload_spec.decode_cross_domain_ep_mode != DecodeCrossDomainEPMode.EXPLICIT_OWNER_A2AV:
        raise ValueError(
            "Cross-domain decode requires " "decode_cross_domain_ep_mode=explicit_owner_a2av"
        )
    ownership = workload_spec.moe_spec_v2.expert_domain_placement
    if ownership is None:
        raise ValueError("Cross-domain decode requires expert_domain_placement")
    if ownership.kind != ExpertDomainPlacementKind.BALANCED_EXPERT_GROUPS:
        raise ValueError(
            "Cross-domain decode currently supports only "
            "expert_domain_placement.kind=balanced_expert_groups"
        )
    if ownership.domain_tag != "EP_SO":
        raise ValueError("Cross-domain decode requires expert_domain_placement.domain_tag=EP_SO")
    if a2a_leg not in {"dispatch", "combine"}:
        raise ValueError(f"a2a_leg must be 'dispatch' or 'combine', got {a2a_leg}")
    tokens_per_full_expert_set = token_scopes.tokens_per_full_expert_set
    tokens_per_feeder_set = token_scopes.tokens_per_feeder_set
    feeders_per_full_expert_set = token_scopes.attn_domains_per_full_expert_set
    active_feeder_count = token_scopes.active_feeder_count
    if feeders_per_full_expert_set != moe_config.attn_domains_per_full_expert_set:
        raise ValueError(
            "Token-scope feeder count does not match expert placement: "
            f"{feeders_per_full_expert_set} != "
            f"{moe_config.attn_domains_per_full_expert_set}"
        )
    if tokens_per_feeder_set * active_feeder_count != tokens_per_full_expert_set:
        raise ValueError(
            "Active feeder-set token scopes do not reconstruct full-expert-set tokens: "
            f"{tokens_per_feeder_set} * {active_feeder_count} "
            f"!= {tokens_per_full_expert_set}"
        )
    if feeders_per_full_expert_set <= 1:
        raise ValueError(
            "feeders_per_full_expert_set must be > 1, got " f"{feeders_per_full_expert_set}"
        )
    if tp_moe_y_shards <= 0:
        raise ValueError(f"tp_moe_y_shards must be > 0, got {tp_moe_y_shards}")

    groups_per_set = moe_config.num_colocated_groups_per_set
    if groups_per_set <= 0:
        raise ValueError(f"num_colocated_groups_per_set must be > 0, got {groups_per_set}")
    if groups_per_set % feeders_per_full_expert_set != 0:
        raise ValueError(
            f"num_colocated_groups_per_set ({groups_per_set}) must be divisible by "
            f"feeders_per_full_expert_set ({feeders_per_full_expert_set})"
        )
    xpus_per_group = moe_config.xpus_per_colocated_group
    if xpus_per_group <= 0:
        raise ValueError(f"xpus_per_colocated_group must be > 0, got {xpus_per_group}")
    if xpus_per_group % tp_moe_y_shards != 0:
        raise ValueError(
            f"xpus_per_colocated_group ({xpus_per_group}) must be divisible by "
            f"tp_moe_y_shards ({tp_moe_y_shards})"
        )

    so_mapping = ep_topology_view.get_mapping_for_tag("EP_SO")
    if so_mapping.size != feeders_per_full_expert_set:
        raise ValueError(
            f"EP_SO size ({so_mapping.size}) must equal feeders_per_full_expert_set "
            f"({feeders_per_full_expert_set})"
        )
    su_view = ep_topology_view.create_view(
        view_name=f"Decode_{a2a_leg}_EP_SU",
        relevant_tags=["EP_SU"],
        global_sharing_context="max_exclusive",
    )
    so_view = ep_topology_view.create_view(
        view_name=f"Decode_{a2a_leg}_EP_SO",
        relevant_tags=["EP_SO"],
        global_sharing_context="max_exclusive",
    )

    selected_experts_per_token = transformer_config.num_selected_experts
    if selected_experts_per_token <= 0:
        raise ValueError(
            "num_selected_experts must be > 0 for cross-domain MoE exchange, "
            f"got {selected_experts_per_token}"
        )
    effective_imbalance = moe_config.effective_expert_imbalance
    if effective_imbalance < 1.0:
        raise ValueError("effective_expert_imbalance must be >= 1, got " f"{effective_imbalance}")
    expected_routed_multiplier = effective_imbalance * selected_experts_per_token
    if not isclose(
        moe_config.regular_expert_traffic_multiplier,
        expected_routed_multiplier,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "regular_expert_traffic_multiplier must equal "
            "effective_expert_imbalance * num_selected_experts: "
            f"{moe_config.regular_expert_traffic_multiplier} != "
            f"{effective_imbalance} * {selected_experts_per_token}"
        )
    balanced_pairs_per_endpoint = (
        tokens_per_full_expert_set * selected_experts_per_token / groups_per_set
    )
    busiest_receiver_pairs = balanced_pairs_per_endpoint * effective_imbalance
    groups_per_feeder_domain = groups_per_set // feeders_per_full_expert_set
    balanced_sender_pairs = (
        tokens_per_feeder_set * selected_experts_per_token / groups_per_feeder_domain
    )
    bytes_per_token = transformer_config.d_model * float_size / tp_moe_y_shards

    # Each feeder domain owns 1/D of the expert groups in the shared set
    # (D = feeders_per_full_expert_set). Routes to that owner fraction are
    # already local; all remaining token-expert pairs leave over EP_SO.
    local_owner_fraction = 1.0 / feeders_per_full_expert_set
    sender_remote_fraction = 1.0 - local_owner_fraction

    # Sparse wide-EP keeps expert owners on every physical feeder domain even
    # when some domains have no input sequences and therefore send no tokens.
    inactive_feeder_count = feeders_per_full_expert_set - active_feeder_count
    if inactive_feeder_count > 0:
        # An expert group on an inactive attention domain receives every routed
        # pair over EP_SO. Since ownership does not pin the hottest group to an
        # active domain, use this conservative busiest-receiver endpoint.
        receiver_remote_fraction = 1.0
    else:
        # Every receiver domain is also an active sender, so its locally
        # originated 1/D owner share does not consume EP_SO bandwidth.
        receiver_remote_fraction = sender_remote_fraction

    # Sender pressure starts from balanced routed pairs on one feeder endpoint;
    # receiver pressure starts from the imbalance-scaled hottest expert group.
    # Apply the side-specific remote fraction before comparing endpoints.
    busiest_remote_sender_pairs = balanced_sender_pairs * sender_remote_fraction
    busiest_remote_receiver_pairs = busiest_receiver_pairs * receiver_remote_fraction

    # A2Av completes at the slower endpoint. Price the larger of the busiest
    # sender's outbound load and the busiest receiver's inbound load.
    bottleneck_remote_pairs = max(
        busiest_remote_sender_pairs,
        busiest_remote_receiver_pairs,
    )
    # Local-owner routes are already resident after the deterministic handoff.
    # Keep the EP_SU trace record as an explicit no-op so the table documents
    # why the balanced 1/D owner fraction does not become network traffic.
    su_bytes_per_xpu = 0.0
    so_bytes_per_xpu = bottleneck_remote_pairs * bytes_per_token

    su_stats = CommStats()
    payload_stats = hockney_model_send_recv_on_nd_topology(
        topology=so_view,
        bytes_per_sender=so_bytes_per_xpu,
    )
    stage_prefix = "decode_dispatch" if a2a_leg == "dispatch" else "decode_combine"
    layout_trace.record_comm(
        f"{stage_prefix}_su",
        "NO-OP",
        f"EP_SU({su_view.num_chips})",
        su_view.num_chips,
        su_bytes_per_xpu,
        su_stats,
        note=(
            f"{a2a_leg}: local-owner fraction=1/{feeders_per_full_expert_set}, but "
            "the exact per-feeder handoff already placed each token shard on "
            "every aligned local expert group; EP_SU network bytes=0"
        ),
    )
    size_stats = _price_dispatch_size_vector_or_zero(
        a2a_leg=a2a_leg,
        topology=so_view,
        participant_count=feeders_per_full_expert_set,
        layout_trace=layout_trace,
        stage=f"{stage_prefix}_so_size",
        fabric=f"EP_SO({so_view.num_chips})",
        note=(
            f"{a2a_leg}: A2Av size vector; "
            f"{feeders_per_full_expert_set - 1} remote peers × "
            f"{A2AV_SIZE_VECTOR_DTYPE_BYTES} B int32 counts; "
            "required before the payload recv can be posted"
        ),
    )
    so_stats = size_stats + payload_stats
    layout_trace.record_comm(
        f"{stage_prefix}_so",
        "A2Av",
        f"EP_SO({so_view.num_chips})",
        so_view.num_chips,
        so_bytes_per_xpu,
        payload_stats,
        note=(
            f"{a2a_leg}: busiest endpoint; remote_fraction="
            f"{feeders_per_full_expert_set - 1}/{feeders_per_full_expert_set} "
            "per active sender; "
            f"active_feeders={active_feeder_count}; "
            f"inactive_feeders={inactive_feeder_count}; "
            f"balanced_pairs/endpoint={balanced_pairs_per_endpoint:g}; "
            f"effective_imbalance={effective_imbalance:g}; "
            f"busiest_receiver_pairs={busiest_receiver_pairs:g}; "
            f"balanced_sender_pairs={balanced_sender_pairs:g}; "
            f"bottleneck_remote_pairs={bottleneck_remote_pairs:g}; "
            f"bytes/token={bytes_per_token:g}"
        ),
    )
    if a2a_leg == "dispatch":
        tensor_stage = "decode_dispatch"
        tensor_name = "x_dispatched"
        d_partitions = (
            {"D": P("sharded", ("MoE_Y",)), "B": P("replicated", ("Y",))}
            if tp_moe_y_shards > 1
            else {"B": P("replicated", ("Y",))}
        )
    else:
        tensor_stage = "decode_combine_a2a"
        tensor_name = "moe_out_returned"
        moe_dims = tuple(
            name
            for name, size in (
                ("MoE_X", moe_config.xpus_per_colocated_group // tp_moe_y_shards),
                ("MoE_Y", tp_moe_y_shards),
            )
            if size > 1
        )
        d_partitions = (
            {"D": P("sharded", moe_dims), "B": P("replicated", ("Y",))}
            if moe_dims
            else {"B": P("replicated", ("Y",))}
        )
    layout_trace.record(
        tensor_stage,
        f"Remote-owner {a2a_leg} A2Av on EP_SO; local-owner data already resident",
        TensorRecord(
            tensor_name,
            {"B": tokens_per_feeder_set, "D": transformer_config.d_model},
            d_partitions,
            dtype_bytes=float_size,
        ),
        note=(
            "actual: explicit, evenly distributed expert-group ownership with "
            "busiest-endpoint timing; "
            f"local_owner_fraction={local_owner_fraction:g} (already local), "
            f"active_feeders={active_feeder_count}/"
            f"{feeders_per_full_expert_set}; "
            f"sender_SO_network_fraction={sender_remote_fraction:g}; "
            f"feeder-equivalent B={tokens_per_feeder_set}"
        ),
    )
    if not isclose(
        so_bytes_per_xpu,
        bottleneck_remote_pairs * bytes_per_token,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise AssertionError("EP_SO exchange bytes do not conserve remote routed payload")
    return so_stats


def calculate_moe_combine_decode(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    token_scopes: ResolvedMoETokenScopes,
    attn_tp_xpus: int,
    float_size: int,
    tp_moe_x_shards: int,
    tp_moe_y_shards: int,
    num_local_colocated_groups: int,
    record_layout_output: bool,
    projection_groups: int,
    attn_x_topology: LogicalTopology,
    attn_yz_topology: LogicalTopology,
    proj_x_topology: LogicalTopology,
    proj_yz_topology: LogicalTopology,
    moe_x_topology: LogicalTopology,
    moe_yz_topology: LogicalTopology,
    expert_set_topology: LogicalTopology | None,
    layout_trace=NULL_LAYOUT_TRACE,
) -> CommStats:
    """Restore decode-time expert outputs to one attention-domain residual layout.

    For a single-domain expert set this follows local expert execution directly.
    For a cross-domain set it follows the A2Av return leg and combines only the
    expert groups owned by this feeder domain. ``num_local_colocated_groups`` is
    therefore explicit and must describe one feeder→set edge.

    The function returns **per-layer** communication statistics. The caller
    multiplies the result by ``num_layers_per_pp_stage``, matching the TP
    strategy convention in ``comm_model.py``.

    Definitions used below:

    - ``B_edge`` is the canonical token count for one attention-domain →
      expert-set edge. It is supplied by ``token_scopes.tokens_per_feeder_set``.
    - ``D_expert = d_model / (TP_MoE_X * TP_MoE_Y)`` is one expert rank's
      output feature shard.
    - ``D_residual = d_model / (TP_Proj_X * TP_Proj_Y_eff)`` is the residual
      stream feature shard required on one rank by the next layer.
    - ``U = TP_Attn_Y / TP_Proj_Y_eff`` is the global projection-replica
      count. With ``S`` expert sets in one attention domain, each set owns
      ``U_set = U / S`` aligned replicas. Thus ``U == S`` means
      ``U_set == 1`` and no replica all-reduce is needed.

    Each colocated expert group first sums the weighted outputs of its local
    top-k experts into a dense ``[B_edge, D_expert]`` buffer. Entries for tokens
    not routed to that group are zero. Top-k and routing imbalance therefore
    affect expert work but do not scale the inter-group combine payload.

    The combine repairs three distinct kinds of ownership in order: feature
    shards, projection replicas, and token slices.

    1. **MoE-to-residual layout-quotient reduce-scatter.** Build the physical-axis quotient
       ``(TP_Proj_X × TP_Proj_Y_eff) / (TP_MoE_X × TP_MoE_Y)`` and run RS on
       that quotient. The division is axis-by-axis, so this may be an ND
       collective. Within each aligned layout-quotient subgroup, RS both sums
       partial expert contributions and narrows ``D_expert`` to the
       ``D_residual`` shard expected by the next layer. Its output is therefore
       ready for the replica AR below. The RS input volume per rank is
       ``B_edge * D_expert * float_size``. It is a no-op when the quotient has
       one participant.
    2. **Projection-replica all-reduce.** If ``U_set > 1``, run AR across
       aligned replicas inside the same expert set—not across expert sets.
       Members own the same ``B_edge`` tokens and ``D_residual`` shard but hold
       disjoint expert-group contributions. AR sums those contributions and leaves
       ``[B_edge, D_residual]`` replicated across the per-set replica axis.
       The payload is ``B_edge * D_residual * float_size``.
    3. **Cross-set batch all-gather.** If ``S > 1``, each Y-aligned set owns a
       disjoint ``B_edge`` token slice. These slices must be concatenated, not
       summed, so AG across ``expert_set_topology`` reconstructs
       ``[B_attention_domain, D_residual]`` for the next layer.
       Matching replica subgroups execute concurrently; contention between
       those subgroups is not yet modeled.

    All quotients are derived from exact physical-axis ownership. Mismatched
    axes, non-divisible extents, inconsistent set counts, or non-integral
    replica partitioning fail explicitly rather than using virtual participant
    approximations.
    """

    if transformer_config.num_regular_experts == 0:
        return CommStats()

    group_count = num_local_colocated_groups
    num_attention_domain_tokens = token_scopes.tokens_per_attention_domain
    tokens_per_feeder_set = token_scopes.tokens_per_feeder_set
    full_set_xpus = moe_config.full_expert_set_xpu_count
    xpus_per_group = moe_config.xpus_per_colocated_group
    if group_count <= 0:
        raise ValueError(f"num_local_colocated_groups must be > 0, got {group_count}")
    if full_set_xpus <= 0:
        raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_set_xpus}")
    if xpus_per_group <= 0:
        raise ValueError(f"xpus_per_colocated_group must be > 0, got {xpus_per_group}")
    if num_attention_domain_tokens <= 0:
        raise ValueError(
            "num_attention_domain_tokens must be > 0, " f"got {num_attention_domain_tokens}"
        )
    if tp_moe_x_shards <= 0 or tp_moe_y_shards <= 0:
        raise ValueError(
            f"MoE TP extents must be > 0, got X={tp_moe_x_shards}, Y={tp_moe_y_shards}"
        )
    if projection_groups <= 0:
        raise ValueError(f"projection_groups must be > 0, got {projection_groups}")
    if xpus_per_group != tp_moe_x_shards * tp_moe_y_shards:
        raise ValueError(
            f"xpus_per_colocated_group ({xpus_per_group}) must equal "
            f"TP_MoE_X × TP_MoE_Y ({tp_moe_x_shards} × {tp_moe_y_shards})"
        )
    local_set_xpus = min(full_set_xpus, attn_tp_xpus)
    if group_count * xpus_per_group != local_set_xpus:
        raise ValueError(
            f"group_count × xpus_per_group = {group_count} × {xpus_per_group} "
            f"!= local expert-set footprint ({local_set_xpus})"
        )
    if attn_tp_xpus != attn_x_topology.num_chips * attn_yz_topology.num_chips:
        raise ValueError(
            f"attn_tp_xpus ({attn_tp_xpus}) does not match resolved attention views "
            f"({attn_x_topology.num_chips} × {attn_yz_topology.num_chips})"
        )
    if proj_x_topology.num_chips != attn_x_topology.num_chips:
        raise ValueError(
            f"TP_Proj_X ({proj_x_topology.num_chips}) must equal "
            f"TP_Attn_X ({attn_x_topology.num_chips}) for residual-layout closure"
        )
    if moe_x_topology.num_chips != tp_moe_x_shards:
        raise ValueError(f"Resolved TP_MoE_X ({moe_x_topology.num_chips}) != {tp_moe_x_shards}")
    if moe_yz_topology.num_chips != tp_moe_y_shards:
        raise ValueError(f"Resolved TP_MoE_Y ({moe_yz_topology.num_chips}) != {tp_moe_y_shards}")
    if attn_yz_topology.num_chips % proj_yz_topology.num_chips != 0:
        raise ValueError(
            f"TP_Attn_Y ({attn_yz_topology.num_chips}) must be divisible by "
            f"TP_Proj_Y_eff ({proj_yz_topology.num_chips})"
        )

    resolved_projection_groups = attn_yz_topology.num_chips // proj_yz_topology.num_chips
    if resolved_projection_groups != projection_groups:
        raise ValueError(
            f"projection_groups ({projection_groups}) != resolved attention/projection Y "
            f"quotient ({resolved_projection_groups})"
        )

    if full_set_xpus > attn_tp_xpus:
        num_combine_sets = 1
    else:
        if attn_tp_xpus % full_set_xpus != 0:
            raise ValueError(
                f"attn_tp_xpus ({attn_tp_xpus}) must be divisible by "
                f"full_expert_set_xpu_count ({full_set_xpus})"
            )
        num_combine_sets = attn_tp_xpus // full_set_xpus
    if num_combine_sets != moe_config.full_expert_sets_per_attn_domain:
        raise ValueError(
            f"Resolved combine sets ({num_combine_sets}) != "
            "full_expert_sets_per_attn_domain "
            f"({moe_config.full_expert_sets_per_attn_domain})"
        )
    if num_combine_sets != token_scopes.full_expert_sets_per_attn_domain:
        raise ValueError(
            f"Resolved combine sets ({num_combine_sets}) != token-scope "
            "full_expert_sets_per_attn_domain "
            f"({token_scopes.full_expert_sets_per_attn_domain})"
        )
    if resolved_projection_groups % num_combine_sets != 0:
        raise ValueError(
            f"Projection replicas ({resolved_projection_groups}) must be divisible by "
            f"expert sets ({num_combine_sets})"
        )
    replicas_per_set = resolved_projection_groups // num_combine_sets
    if num_combine_sets > 1:
        if expert_set_topology is None:
            raise ValueError("Multiple expert sets require explicit expert_set_topology")
        if expert_set_topology.num_chips != num_combine_sets:
            raise ValueError(
                f"expert_set_topology ({expert_set_topology.num_chips}) != "
                f"combine sets ({num_combine_sets})"
            )

    projection_layout = _merge_logical_views(
        first_view=proj_x_topology,
        second_view=proj_yz_topology,
        view_name="Projection_Residual_Layout",
    )
    moe_layout = _merge_logical_views(
        first_view=moe_x_topology,
        second_view=moe_yz_topology,
        view_name="MoE_Output_Layout",
    )
    moe_to_residual_layout_quotient_view = _exact_layout_quotient_view(
        numerator_view=projection_layout,
        denominator_view=moe_layout,
        view_name="MoE_To_Residual_Layout_Quotient",
    )
    global_replica_view = _exact_layout_quotient_view(
        numerator_view=attn_yz_topology,
        denominator_view=proj_yz_topology,
        view_name="MoE_Combine_Global_Projection_Replicas",
    )
    moe_to_residual_layout_quotient_count = moe_to_residual_layout_quotient_view.num_chips
    if global_replica_view.num_chips != projection_groups:
        raise ValueError(
            f"Replica quotient ({global_replica_view.num_chips}) != "
            f"projection_groups ({projection_groups})"
        )
    if num_combine_sets == 1:
        replica_view = global_replica_view
    else:
        if len(global_replica_view.dimensions) != 1:
            raise ValueError(
                "Y-aligned multiple-set combine requires one projection-replica "
                f"dimension; got {len(global_replica_view.dimensions)}"
            )
        replica_tag = _logical_tag_string(global_replica_view.dimensions[0].logical_tag)
        replica_view = global_replica_view.create_resized_view(
            view_name="MoE_Combine_Per_Set_Projection_Replicas",
            new_sizes={replica_tag: replicas_per_set},
        )
    if group_count != moe_to_residual_layout_quotient_count * replicas_per_set:
        raise ValueError(
            f"Expert groups per set ({group_count}) must equal MoE-to-residual "
            "layout quotients × per-set projection replicas "
            f"({moe_to_residual_layout_quotient_count} × {replicas_per_set})"
        )
    if tokens_per_feeder_set * num_combine_sets != num_attention_domain_tokens:
        raise ValueError(
            "Canonical feeder-set token scopes do not close over the attention "
            f"domain: {tokens_per_feeder_set} × "
            f"{num_combine_sets} != {num_attention_domain_tokens}"
        )
    d_model = transformer_config.d_model
    expert_output_shard = d_model / (tp_moe_x_shards * tp_moe_y_shards)
    residual_output_shard = d_model / (proj_x_topology.num_chips * proj_yz_topology.num_chips)
    rs_input_bytes = tokens_per_feeder_set * expert_output_shard * float_size
    if moe_to_residual_layout_quotient_count == 1:
        per_layer_rs = CommStats()
    else:
        per_layer_rs = hockney_model_reduce_scatter_on_nd_topology(
            view=moe_to_residual_layout_quotient_view,
            P_total_bytes_before_rs=rs_input_bytes,
            float_size_bytes=float_size,
        )
    layout_quotient_shape = " × ".join(
        f"{_logical_tag_string(dimension.logical_tag)}={dimension.size}"
        for dimension in moe_to_residual_layout_quotient_view.dimensions
    )
    layout_trace.record_comm(
        "combine_rs_exact_groups",
        "RS",
        layout_quotient_shape,
        moe_to_residual_layout_quotient_count,
        rs_input_bytes,
        per_layer_rs,
        note=(
            "Dense expert-output combine over the exact MoE-to-residual "
            "layout quotient. "
            f"B_feeder_set={tokens_per_feeder_set}, D_expert={expert_output_shard:g}, "
            f"D_residual={residual_output_shard:g}."
        ),
    )

    per_layer = per_layer_rs
    if replicas_per_set > 1:
        replica_payload_bytes = tokens_per_feeder_set * residual_output_shard * float_size
        per_layer_replica_ar = hockney_model_all_reduce_on_nd_topology(
            view=replica_view,
            P_total_bytes=replica_payload_bytes,
            float_size_bytes=float_size,
        )
        replica_shape = " × ".join(
            f"{_logical_tag_string(dimension.logical_tag)}={dimension.size}"
            for dimension in replica_view.dimensions
        )
        layout_trace.record_comm(
            "combine_ar_projection_replicas",
            "AR",
            replica_shape,
            replicas_per_set,
            replica_payload_bytes,
            per_layer_replica_ar,
            note=(
                "Sum disjoint expert contributions and replicate the final residual shard "
                f"across projection replicas. B_feeder_set={tokens_per_feeder_set}, "
                f"D_residual={residual_output_shard:g}."
            ),
        )
        per_layer = per_layer + per_layer_replica_ar

    if num_combine_sets > 1:
        assert expert_set_topology is not None
        tokens_per_attn_domain = tokens_per_feeder_set * num_combine_sets
        if tokens_per_attn_domain != num_attention_domain_tokens:
            raise AssertionError(
                f"Cross-set gather output tokens ({tokens_per_attn_domain}) != "
                f"attention-domain tokens ({num_attention_domain_tokens})"
            )
        cross_set_output_bytes = tokens_per_attn_domain * residual_output_shard * float_size
        per_layer_cross_set_ag = hockney_model_all_gather_on_nd_topology(
            view=expert_set_topology,
            P_total_bytes_after_ag=cross_set_output_bytes,
            float_size_bytes=float_size,
        )
        set_shape = " × ".join(
            f"{_logical_tag_string(dimension.logical_tag)}={dimension.size}"
            for dimension in expert_set_topology.dimensions
        )
        layout_trace.record_comm(
            "combine_ag_cross_set",
            "AG",
            set_shape,
            num_combine_sets,
            cross_set_output_bytes,
            per_layer_cross_set_ag,
            note=(
                f"Gather disjoint B/{num_combine_sets} token slices after per-set combine: "
                f"B_feeder_set={tokens_per_feeder_set} -> "
                f"B_domain={tokens_per_attn_domain}. "
                f"{replicas_per_set} aligned subgroup(s) run concurrently. "
                "MODEL_LIMITATION: cross-subgroup fabric contention is not modeled."
            ),
        )
        per_layer = per_layer + per_layer_cross_set_ag

    if record_layout_output:
        output_partitions = {
            "D": P("sharded", ("X", "Y_eff")),
            "B": P("replicated", ("Y",)),
        }
        output_tensor = TensorRecord(
            "moe_out",
            {"B": num_attention_domain_tokens, "D": transformer_config.d_model},
            output_partitions,
            dtype_bytes=float_size,
        )
        layout_trace.record(
            "after_combine",
            "Local residual-layout restoration after cross-domain return",
            output_tensor,
            note=(
                "actual: exact RS(moe_to_residual_layout_quotient="
                f"{moe_to_residual_layout_quotient_count}) + "
                f"AR(projection_replicas={replicas_per_set}); "
                f"D_local={residual_output_shard:g}"
            ),
        )
        layout_trace.record(
            "layer_output",
            "Residual add → next layer",
            TensorRecord(
                "x_out",
                {"B": num_attention_domain_tokens, "D": transformer_config.d_model},
                output_partitions,
                dtype_bytes=float_size,
            ),
            note=f"actual: restored incoming projection layout; D_local={residual_output_shard:g}",
        )

    return per_layer


def calculate_moe_comm_hierarchical_total_exchange(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    token_scopes: ResolvedMoETokenScopes,
    float_size: int,
    tp_moe_y_shards: int,
    workload_spec=None,
    layout_trace=NULL_LAYOUT_TRACE,
    a2a_leg: str = "dispatch",
):
    """Per-layer MoE A2A exchange cost (hierarchical or fused).

    Returns **per-layer** comm stats; the caller is responsible for
    multiplying by ``num_layers_per_pp_stage``.

    ``a2a_leg`` labels the layout-trace stage emitted by this call:
    ``"dispatch"`` (token routing to experts) or ``"combine"`` (return leg).
    Per-set token volume comes from the canonical ``token_scopes`` rather than
    being reconstructed from topology dimensions. On dispatch, each fabric that
    carries a routing-dependent payload also pays an int32 size-vector A2A
    among that fabric's participants before the payload exchange. Combine is
    the transpose of dispatch, so return recvs reuse those counts.
    """
    if transformer_config.num_regular_experts == 0:
        return CommStats()

    d_model = transformer_config.d_model

    try:
        full_expert_set_xpu_count = int(moe_config.full_expert_set_xpu_count)
        xpus_per_colocated_group = int(moe_config.xpus_per_colocated_group)
        num_colocated_groups_per_set = int(moe_config.num_colocated_groups_per_set)
        attn_domains_per_full_expert_set = int(moe_config.attn_domains_per_full_expert_set)
    except AttributeError as e:
        raise ValueError(
            "moe_config must be produced by infer_expert_placement_per_xpu and include "
            "`full_expert_set_xpu_count`, `xpus_per_colocated_group`, "
            "`num_colocated_groups_per_set`, `attn_domains_per_full_expert_set`."
        ) from e
    if (
        full_expert_set_xpu_count <= 0
        or xpus_per_colocated_group <= 0
        or num_colocated_groups_per_set <= 0
        or attn_domains_per_full_expert_set <= 0
    ):
        raise ValueError(
            "Invalid expert placement fields: expected full_expert_set_xpu_count>0, "
            "xpus_per_colocated_group>0, num_colocated_groups_per_set>0, "
            "attn_domains_per_full_expert_set>0."
        )
    routed_expert_traffic_multiplier = float(
        getattr(moe_config, "regular_expert_traffic_multiplier")
    )
    # Use EP chips per PP stage (DP_Attn * TP_Attn) if provided by caller via merged placement
    try:
        ep_total_xpus = int(moe_config.ep_chips_per_pp_stage)
    except AttributeError as e:
        raise ValueError(
            "moe_config must include `ep_chips_per_pp_stage` (produced by infer_expert_placement_per_xpu)."
        ) from e
    if ep_total_xpus <= 0:
        raise ValueError(f"Invalid ep_chips_per_pp_stage: {ep_total_xpus}. Expected > 0.")
    if ep_total_xpus % full_expert_set_xpu_count != 0:
        raise ValueError(
            f"EP chips ({ep_total_xpus}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_full_sets = ep_total_xpus // full_expert_set_xpu_count
    if num_full_sets != token_scopes.complete_expert_set_count:
        raise ValueError(
            f"Topology-derived expert-set count ({num_full_sets}) != token-scope "
            f"complete_expert_set_count ({token_scopes.complete_expert_set_count})"
        )
    tokens_per_set = token_scopes.tokens_per_full_expert_set
    # Smooth locality model (independence approximation):
    #   routing_nonlocal_fraction = 1 - 1 / (attn_domains_per_full_expert_set
    #                                        * num_colocated_groups_per_set)
    #
    # The two factors index distinct "routing buckets" inside one expert set:
    #   - attn_domains_per_full_expert_set = number of feeder attention TP
    #     domains that share one expert set,
    #   - num_colocated_groups_per_set    = number of destination colocated
    #     expert groups inside that set.
    #
    # NOTE — this is intentionally coarse and is a known approximation:
    # under the no-spill placement invariant (each colocated group lives in
    # exactly one attention TP domain), the *correct* total-volume fraction
    # is 1 - 1/num_colocated_groups_per_set, and the
    # attn_domains_per_full_expert_set factor is only needed when separating
    # the within-attention-domain remote share from the cross-attention-
    # domain remote share (which only matters for cross-rack fabric
    # accounting). For the scale-up-resident wide-EP path the two formulas
    # pick the same physical fabric — only the volume scales differ. The
    # product form is kept pending the refinement below.
    #
    # TODO: refine into separate within-attention-domain and cross-attention-
    # domain fractions so scale-out-spanning wide-EP sets attribute the right
    # volume to EP_SU vs. EP_SO.
    # Current product form double-counts locality when both factors are > 1
    # and under-counts the SO leg when only the cross-domain share is remote.
    routing_buckets_per_set = attn_domains_per_full_expert_set * num_colocated_groups_per_set
    routing_nonlocal_fraction = max(0.0, 1.0 - (1.0 / float(routing_buckets_per_set)))
    # Model routed exchange at expert-group granularity, then map to per-XPU payload
    # by dividing across TP ranks inside the group.
    tokens_per_group = (
        tokens_per_set
        * routed_expert_traffic_multiplier
        * routing_nonlocal_fraction
        / num_colocated_groups_per_set
    )
    # EP A2A toggle semantics:
    # - optimization ON: send one grouped copy over EP exchange, then redistribute inside TP group.
    # - optimization OFF: each TP rank carries its own routed copy over EP exchange.
    # This makes EP A2A volume larger in OFF mode, as intended.
    optimization_enabled = bool(getattr(workload_spec, "tp_moe_a2a_replication_optimization", True))
    a2a_replication_factor = 1.0 if optimization_enabled else float(xpus_per_colocated_group)
    tokens_per_xpu = tokens_per_group / xpus_per_colocated_group * a2a_replication_factor

    # Resolve this once so hierarchical and fused paths are based on the same EP
    # interpretation. EP_SU is the scale-up XPU-domain capacity for experts.
    try:
        ep_su_domain_size_xpus = ep_topology_view.get_mapping_for_tag("EP_SU").size
    except Exception as e:
        raise ValueError(
            "EP topology must include tag `EP_SU` to resolve the scale-up " "expert domain size."
        ) from e
    if ep_su_domain_size_xpus <= 0:
        raise ValueError(f"Invalid EP_SU size: {ep_su_domain_size_xpus}. Expected > 0.")

    moe_a2a_comm_mode = workload_spec.moe_a2a_comm_mode
    # Shard MoE A2A activation payload by TP_MoE_Y.
    # For 1D MoE TP this value is 1, so behavior is unchanged.
    if tp_moe_y_shards <= 0:
        raise ValueError(f"tp_moe_y_shards must be > 0, got {tp_moe_y_shards}")
    bytes_per_token = (d_model * float_size) / tp_moe_y_shards

    # Prefill and single-attention-domain decode also pass through here; the
    # records are still emitted (NULL_LAYOUT_TRACE is a no-op for callers that
    # don't pass a real trace).
    trace_stage = "decode_dispatch_a2a" if a2a_leg == "dispatch" else "decode_combine_a2a"

    if moe_a2a_comm_mode == MoEA2ACommMode.FUSED_ND:
        # fused_nd: one total-exchange on the full EP topology, not two
        # staged SU/SO collectives. "Fused" means flatten the hierarchical
        # split; it is not kernel fusion. The N-D Hockney walk prices hops
        # on every mesh dimension. Experimental; current sweeps use
        # hierarchical_split.
        payload_bytes_per_xpu = tokens_per_xpu * bytes_per_token
        payload_stats = hockney_model_total_exchange_on_nd_topology(
            ep_topology_view,
            P_total_send_bytes_per_node=payload_bytes_per_xpu,
            float_size_bytes=float_size,
        )
        per_layer = payload_stats
        if payload_bytes_per_xpu > 0 and ep_total_xpus > 1:
            size_stats = _price_dispatch_size_vector_or_zero(
                a2a_leg=a2a_leg,
                topology=ep_topology_view,
                participant_count=ep_total_xpus,
                layout_trace=layout_trace,
                stage=f"{trace_stage}_fused_size",
                fabric="EP_FUSED",
                note=(
                    f"{a2a_leg}: A2Av size vector; "
                    f"{ep_total_xpus - 1} remote peers × "
                    f"{A2AV_SIZE_VECTOR_DTYPE_BYTES} B int32 counts"
                ),
            )
            per_layer = size_stats + payload_stats
        layout_trace.record_comm(
            f"{trace_stage}_fused",
            "A2A",
            "EP_FUSED",
            ep_total_xpus,
            payload_bytes_per_xpu,
            payload_stats,
            note=(
                f"{a2a_leg} A2A fused over EP topology; "
                f"tokens_per_xpu={tokens_per_xpu:.3g}, bytes/token={bytes_per_token:.3g}, "
                f"routing_nonlocal_fraction={routing_nonlocal_fraction:.3f} "
                f"(routing_buckets_per_set={routing_buckets_per_set} = "
                f"attn_domains_per_full_expert_set={attn_domains_per_full_expert_set} * "
                f"num_colocated_groups_per_set={num_colocated_groups_per_set})"
            ),
        )
    elif moe_a2a_comm_mode == MoEA2ACommMode.HIERARCHICAL_SPLIT:
        # Hierarchical split: scale-up (EP_SU) carries the in-domain share,
        # scale-out (second-fastest EP tag, typically EP_SO) carries the
        # cross-rack share when the expert set spills out of one SU domain.
        # Clamp SU size to the set footprint so frac_local <= 1.
        if ep_su_domain_size_xpus > full_expert_set_xpu_count:
            ep_su_domain_size_xpus = full_expert_set_xpu_count
        su_fabric_fraction = min(1.0, ep_su_domain_size_xpus / max(1, full_expert_set_xpu_count))
        so_fabric_fraction = max(0.0, 1.0 - su_fabric_fraction)

        su_fabric_bytes_per_xpu = tokens_per_xpu * su_fabric_fraction * bytes_per_token
        so_fabric_bytes_per_xpu = tokens_per_xpu * so_fabric_fraction * bytes_per_token

        su_fabric_stats_per_layer = CommStats()
        so_fabric_stats_per_layer = CommStats()
        topology_view_speed_sorted_tags = ep_topology_view.get_tags_sorted_by_speed()

        if ep_su_domain_size_xpus > 1 and su_fabric_bytes_per_xpu > 0:
            su_fabric_fastest_tag = topology_view_speed_sorted_tags[0]
            su_fabric_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
                view_name="EP_SU_Fabric_Virtual",
                base_tag=su_fabric_fastest_tag,
                virtual_size=ep_su_domain_size_xpus,
                global_sharing_context="max_exclusive",
            )
            su_payload_stats = hockney_model_total_exchange_on_nd_topology(
                su_fabric_view,
                P_total_send_bytes_per_node=su_fabric_bytes_per_xpu,
                float_size_bytes=float_size,
            )
            su_size_stats = _price_dispatch_size_vector_or_zero(
                a2a_leg=a2a_leg,
                topology=su_fabric_view,
                participant_count=ep_su_domain_size_xpus,
                layout_trace=layout_trace,
                stage=f"{trace_stage}_intra_size",
                fabric=f"{su_fabric_fastest_tag}({ep_su_domain_size_xpus})",
                note=(
                    f"{a2a_leg}: A2Av size vector on EP_SU; "
                    f"{ep_su_domain_size_xpus - 1} remote peers × "
                    f"{A2AV_SIZE_VECTOR_DTYPE_BYTES} B int32 counts"
                ),
            )
            su_fabric_stats_per_layer = su_size_stats + su_payload_stats
            layout_trace.record_comm(
                f"{trace_stage}_intra",
                "A2A",
                f"{su_fabric_fastest_tag}({ep_su_domain_size_xpus})",
                ep_su_domain_size_xpus,
                su_fabric_bytes_per_xpu,
                su_payload_stats,
                note=(
                    f"{a2a_leg} A2A on EP_SU (scale-up, intra-rack); "
                    f"su_fabric_fraction={su_fabric_fraction:.3f}, "
                    f"bytes/token={bytes_per_token:.3g}"
                ),
            )

        so_fabric_participant_xpus = max(0, full_expert_set_xpu_count - ep_su_domain_size_xpus)
        if so_fabric_participant_xpus > 1 and so_fabric_bytes_per_xpu > 0:
            tag_index = 1 if len(topology_view_speed_sorted_tags) > 1 else 0
            so_fabric_tag = topology_view_speed_sorted_tags[tag_index]
            so_fabric_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
                view_name="EP_SO_Fabric_Virtual",
                base_tag=so_fabric_tag,
                virtual_size=so_fabric_participant_xpus,
                global_sharing_context="max_exclusive",
            )
            so_payload_stats = hockney_model_total_exchange_on_nd_topology(
                so_fabric_view,
                P_total_send_bytes_per_node=so_fabric_bytes_per_xpu,
                float_size_bytes=float_size,
            )
            so_size_stats = _price_dispatch_size_vector_or_zero(
                a2a_leg=a2a_leg,
                topology=so_fabric_view,
                participant_count=so_fabric_participant_xpus,
                layout_trace=layout_trace,
                stage=f"{trace_stage}_inter_size",
                fabric=f"{so_fabric_tag}({so_fabric_participant_xpus})",
                note=(
                    f"{a2a_leg}: A2Av size vector on second-fastest EP tag; "
                    f"{so_fabric_participant_xpus - 1} remote peers × "
                    f"{A2AV_SIZE_VECTOR_DTYPE_BYTES} B int32 counts"
                ),
            )
            so_fabric_stats_per_layer = so_size_stats + so_payload_stats
            layout_trace.record_comm(
                f"{trace_stage}_inter",
                "A2A",
                f"{so_fabric_tag}({so_fabric_participant_xpus})",
                so_fabric_participant_xpus,
                so_fabric_bytes_per_xpu,
                so_payload_stats,
                note=(
                    f"{a2a_leg} A2A on second-fastest EP tag (scale-out, "
                    f"typically EP_SO; cross-rack); "
                    f"so_fabric_fraction={so_fabric_fraction:.3f}, "
                    f"bytes/token={bytes_per_token:.3g}"
                ),
            )

        per_layer = su_fabric_stats_per_layer + so_fabric_stats_per_layer
    else:
        raise ValueError(f"Unsupported moe_a2a_comm_mode: {moe_a2a_comm_mode}")
    return per_layer
