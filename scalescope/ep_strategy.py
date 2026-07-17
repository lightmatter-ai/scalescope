"""Expert placement and per-XPU MoE layout inference.

Infers how distinct experts are placed across chips, taling into account
expert collocation and parallelization strategies.
"""

from dataclasses import dataclass
from math import ceil

from scalescope.decode_layout_trace import NULL_LAYOUT_TRACE
from scalescope.model_spec import TransformerConfig
from scalescope.network_topology_model import (
    hockney_model_all_gather_on_nd_topology,
    hockney_model_reduce_scatter_on_nd_topology,
    hockney_model_total_exchange_on_nd_topology,
)
from scalescope.stats import CommStats
from scalescope.workload_spec import (
    ExpertRoutingDistribution,
    MoEA2ACommMode,
    MoERouterSyncMode,
    MoESpecV2,
)


@dataclass
class ExpertXPUPlacement:
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
    # - sharded_router: explicit router-sync collective required
    router_sync_mode: MoERouterSyncMode = MoERouterSyncMode.REPLICATED_ROUTER
    # TP-MoE redistribution optimization toggle:
    # - OFF: regular routed MoE A2A.
    # - ON: send tokens (or token shards) to an expert shard first, then run all-gather
    #   so that shard is available across the expert TP shards.
    tp_moe_a2a_replication_optimization_enabled: bool = True


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
      colocated group this step (the ``c`` hottest experts; collector's-problem
      expectation ``max(1, sum_{k=1..c} (1 - (1 - p_k)^m))``).
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
    expert_imbalance: float,
    num_colocated_experts: int,
    m: float,
) -> float:
    """Expected distinct active experts on the straggler under *random* placement.

    Under load-agnostic (random/hash) placement the worst-loaded chip holds the
    single hottest expert (#1, placement-invariant — it is hot wherever it lands)
    plus ``c - 1`` experts sampled from the *tail* of the global Zipf, rather than
    the ``c`` globally hottest. With ``a_k = 1 - (1 - p_k)^m`` the expected
    distinct active count is

        E_active_chip = a_1 + (c - 1) * mean_{k=2..E} a_k

    i.e. one hottest expert plus ``c - 1`` average tail experts. The fillers use
    the cold-tail mean ``mean a_{2..E}`` (not the next-hottest ``a_2..a_c``).

    ``E_active`` is left unfloored: this is an *expectation* over random placement
    and routing, so a value below 1.0 is the honest answer. At small ``m`` even
    the hottest expert is hit with probability ``a_1 = 1 - (1 - p_1)^m ≈ m·p_1 <
    1``, and the tail fillers are smaller still, so the expected distinct-active
    count is genuinely fractional (the straggler reads less than one full expert's
    weights per step on average). Flooring at 1.0 would force a full per-expert
    HBM read every step in that regime, over-counting weight traffic and inflating
    decode step time at small batch / low top-k. Leaving it unfloored keeps the
    correct small-``m`` asymptote (linear in ``m``) while still rising to the full
    Zipf active count as ``m`` grows.

    Corners:
      c = 1     : a_1
      c = E     : a_1 + (E-1)·mean_{k≥2} a_k = sum_{k=1..E} a_k (full Zipf count)
      rho = 1   : all a_k equal ⇒ plain coupon collector (c·a)
    """
    if m <= 0:
        return 0.0
    s = _solve_zipf_exponent(
        num_regular_experts=num_regular_experts, expert_imbalance=expert_imbalance
    )
    if s == 0.0:
        # Uniform routing (rho == 1): every expert has probability p = 1/E, so the
        # hottest expert and the cold-tail average coincide and the hottest-plus-
        # random split collapses to the plain coupon collector (c experts, each
        # active w.p. 1 - (1 - 1/E)^m). No floor here (or below): this is an
        # expectation, so a value < 1 at tiny m is intentional.
        per_expert_active = 1.0 - (1.0 - 1.0 / num_regular_experts) ** m
        return num_colocated_experts * per_expert_active
    # Skewed routing (rho > 1, s > 0). Zipf normalizer over all E experts:
    # H_{E,s} = sum_{k=1..E} k^{-s}, giving per-expert probability p_k = k^{-s}/H.
    h_es = 0.0
    for k in range(1, num_regular_experts + 1):
        h_es += k ** (-s)
    # The hottest expert (k = 1, p_1 = 1/H_{E,s}) is placement-invariant — it is
    # hot wherever it lands, so it always sits on the straggler. a_1 is its
    # probability of being hit by >= 1 of the m draws.
    a_1 = 1.0 - (1.0 - (1.0 ** (-s)) / h_es) ** m
    if num_colocated_experts == 1 or num_regular_experts == 1:
        # Only the hottest expert is hosted (c = 1) or it is the sole expert.
        return a_1
    # The other c - 1 experts on the straggler are sampled at random from the cold
    # tail (k = 2..E), so each contributes the tail's *average* activation rather
    # than the next-hottest experts. Compute that mean activation over k = 2..E.
    tail_sum = 0.0
    for k in range(2, num_regular_experts + 1):
        p_k = (k ** (-s)) / h_es
        tail_sum += 1.0 - (1.0 - p_k) ** m
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
          a_k       = 1 - (1 - p_k)^m
          E_active  = a_1 + (c - 1) * mean_{k=2..E} a_k
          frac      = E_active / c
      An expectation (so unfloored); corner-exact at c=1, c=E, and rho=1.
      See `_zipf_active_experts_the_hottest_plus_random`.

    Why coupon collector. ``m = tokens_per_set * num_selected_experts`` is the
    number of routing "draws" (token->expert assignments) that hit one full
    expert set this step. The quantity we need is how many *distinct* experts
    those ``m`` draws touch — which is exactly the coupon-collector (occupancy)
    problem, generalized to *unequal* coupon probabilities. Expert ``k`` is
    missed by one draw with probability ``1 - p_k``, missed by all ``m``
    independent draws with probability ``(1 - p_k)^m``, and therefore hit at
    least once with probability ``1 - (1 - p_k)^m``. Summing that hit
    probability over the ``c`` hosted experts gives the expected number of
    distinct experts collected, i.e. how many distinct weight shards the worst
    chip must read this step. The skew sets how fast coupons saturate: under a
    steep Zipf only the few hot coupons are ever collected (the chip reads ~1
    shard no matter how large ``m`` gets), while under uniform routing enough
    draws eventually collect all ``c`` (the textbook coupon-collector limit).

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

    m = tokens_per_set * num_selected_experts

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
    if m <= 0:
        e_active = 0.0
        per_chip_frac = 0.0
    else:
        e_active = active_experts_estimator(
            num_regular_experts=num_regular_experts,
            expert_imbalance=expert_imbalance,
            num_colocated_experts=num_colocated_experts,
            m=m,
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
      For replicated router: 1. For sharded router: Y_eff (2D TP) or attn_tp_size (1D TP).
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
    num_global_batch_tokens,
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
        num_global_batch_tokens,
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
    num_global_batch_tokens,
    float_size,
    tp_moe_y_shards: int,
    workload_spec=None,
    layout_trace=NULL_LAYOUT_TRACE,
):
    # assume for now that we get the same stats when we are returning tokens after they are processed by teh experts
    return calculate_moe_comm_first_total_exchange(
        ep_topology_view,
        transformer_config,
        moe_config,
        num_global_batch_tokens,
        float_size,
        workload_spec=workload_spec,
        tp_moe_y_shards=tp_moe_y_shards,
        layout_trace=layout_trace,
        a2a_leg="combine",
    )


def calculate_moe_combine_decode(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    num_global_batch_tokens: int,
    attn_tp_xpus: int,
    float_size: int,
    tp_moe_x_shards: int,
    tp_moe_y_shards: int,
    projection_groups: int,
    layout_trace=NULL_LAYOUT_TRACE,
) -> CommStats:
    """Single-attention-domain decode combine: per-layer reduce-scatter
    across the colocated expert groups within one expert set.

    Used only when ``attn_domains_per_full_expert_set == 1`` (each expert set
    is contained in exactly one attention TP domain, so dispatch is free and
    only the combine needs to be modeled). Multi-attention-domain decode goes
    through the symmetric A2A path in ``comm_model.py`` instead.

    Returns **per-layer** comm stats; the caller is responsible for
    multiplying by ``num_layers_per_pp_stage`` (consistent with the TP
    strategy pattern in ``comm_model.py``).

    The combine is a reduce-scatter **along G** (matching ranks across groups).
    There are tp_x × Y_eff ranks per group; each (X, Y_eff) position has one
    independent reduce-scatter with G participants.

    Each rank holds a dense output buffer of shape [B, D_local] where
    D_local = d_model / xpus_per_group.  After expert processing, entries
    for tokens not routed to this group's experts are zero.

    1. **Reduce** — tokens whose top_k experts span multiple groups have
       partial weighted sums in each contributing group; RS reduces them.
    2. **Scatter/Handoff** — RS also shards outputs across participants,
       which can align with next-layer X sharding when group↔X mapping matches.

    RS input volume per rank = B_set × D_local × float_size  (the full dense buffer
    for one expert set, where B_set = B_global / num_complete_expert_sets).
    No traffic_mult scaling: the top_k fan-out is resolved locally inside
    each group (the group sums its local expert outputs per token before
    the inter-group combine).  The RS operates on the dense buffer
    regardless of routing sparsity.

    When G=1 the reduce-scatter is a no-op (zero cost).

    **Cross-set All-Gather** (when combine-domain set count S_attn > 1):
    After the intra-set RS, each rank has [B_set, D_local].  An AG(S)
    across matching ranks in different expert sets reconstructs the full
    [B_attn_domain, D_local] needed by the next layer in the current
    attention token domain.  When S_attn=1 this step is skipped.
    """

    if transformer_config.num_regular_experts == 0:
        return CommStats()

    G = moe_config.num_colocated_groups_per_set

    ep_chips = moe_config.ep_chips_per_pp_stage
    full_set_xpus = moe_config.full_expert_set_xpu_count
    xpus_per_group = moe_config.xpus_per_colocated_group

    if full_set_xpus <= 0:
        raise ValueError(f"full_expert_set_xpu_count must be > 0, got {full_set_xpus}")
    if xpus_per_group <= 0:
        raise ValueError(f"xpus_per_colocated_group must be > 0, got {xpus_per_group}")
    if ep_chips <= 0:
        raise ValueError(f"ep_chips_per_pp_stage must be > 0, got {ep_chips}")
    if attn_tp_xpus <= 0:
        raise ValueError(f"attn_tp_xpus must be > 0, got {attn_tp_xpus}")
    if ep_chips % full_set_xpus != 0:
        raise ValueError(
            f"ep_chips_per_pp_stage ({ep_chips}) must be divisible by "
            f"full_expert_set_xpu_count ({full_set_xpus})"
        )
    if G * xpus_per_group != full_set_xpus:
        raise ValueError(
            f"Group decomposition inconsistent: G ({G}) × xpus_per_group ({xpus_per_group}) "
            f"= {G * xpus_per_group}, expected full_expert_set_xpu_count = {full_set_xpus}"
        )
    if tp_moe_x_shards <= 0:
        raise ValueError(f"tp_moe_x_shards must be > 0, got {tp_moe_x_shards}")
    if projection_groups <= 0:
        raise ValueError(f"projection_groups must be > 0, got {projection_groups}")
    if xpus_per_group % tp_moe_y_shards != 0:
        raise ValueError(
            f"xpus_per_colocated_group ({xpus_per_group}) must be divisible by "
            f"tp_moe_y_shards ({tp_moe_y_shards})"
        )
    if xpus_per_group % tp_moe_x_shards != 0:
        raise ValueError(
            f"xpus_per_colocated_group ({xpus_per_group}) must be divisible by "
            f"tp_moe_x_shards ({tp_moe_x_shards})"
        )

    d_model = transformer_config.d_model

    # D_local: each rank's shard of d_model within the group (sharded by X × Y_eff).
    d_local_bytes = (d_model / xpus_per_group) * float_size

    # Dense RS input buffer: every rank materializes [B, D_local] — zeros for
    # tokens not routed to this group's experts.
    num_processing_sets = moe_config.num_complete_expert_sets
    if num_processing_sets <= 0:
        raise ValueError(f"num_complete_expert_sets must be > 0, got {num_processing_sets}")
    if attn_tp_xpus >= full_set_xpus:
        if attn_tp_xpus % full_set_xpus != 0:
            raise ValueError(
                f"attn_tp_xpus ({attn_tp_xpus}) must be divisible by "
                f"full_expert_set_xpu_count ({full_set_xpus})"
            )
        num_combine_sets = attn_tp_xpus // full_set_xpus
    else:
        # Attention domain smaller than one full expert set: no cross-set combine in this domain.
        num_combine_sets = 1
    if num_combine_sets <= 0:
        raise ValueError(
            f"combine-domain set count must be > 0, got {num_combine_sets} "
            f"(attn_tp_xpus={attn_tp_xpus}, full_set_xpus={full_set_xpus})"
        )
    tokens_per_set = num_global_batch_tokens / num_processing_sets

    combine_bytes_per_rank = tokens_per_set * d_local_bytes

    # Variant-aware axis assignment:
    # - 1xN: RS(G) and cross-set AG are modeled on TP_Attn_X.
    # - Nx1: RS(G) and cross-set AG are modeled on TP_Attn_Y.
    # - NxM: keep explicit mixed-mode marker (not fully modeled).
    if tp_moe_x_shards == 1:
        rs_handoff_mode = "1xN->X"
        preferred_base_tag = "TP_Attn_X"
    elif tp_moe_y_shards == 1:
        rs_handoff_mode = "Nx1->Y_eff"
        preferred_base_tag = "TP_Attn_Y"
    else:
        rs_handoff_mode = "NxM-unsupported"
        preferred_base_tag = "TP_Attn_X"

    # EP view contains only EP_* tags; for axis selection we need TP_Attn_* tags.
    # Use the EP view's original topology reference to access the full tag namespace.
    topo_ref = ep_topology_view.original_topology_ref
    axis_note = ""
    combine_base_tag = preferred_base_tag
    try:
        _ = topo_ref.get_mapping_for_tag(combine_base_tag)
    except Exception:
        ep_tags = ep_topology_view.get_tags_sorted_by_speed()
        if not ep_tags:
            raise ValueError("EP topology view has no tags to map combine axis.")
        combine_base_tag = ep_tags[0]
        axis_note = (
            f" NOTE: {preferred_base_tag} not defined in logical topology; "
            f"using EP axis {combine_base_tag} for combine-domain modeling."
        )
    cross_set_base_tag = combine_base_tag
    # The reduce-scatter runs across G groups on a selected attention-domain axis.
    # Create a virtual view with G participants so the Hockney model uses
    # the correct alpha count and bandwidth sharing.
    # TODO(moe-combine): model explicit subgroup membership (set_id, local_shard_id)
    # for RS/AG participants. Current path models axis + virtual participant count,
    # but not exact rank pairing/placement within the axis.
    combine_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
        view_name="Combine_G",
        base_tag=combine_base_tag,
        virtual_size=G,
        global_sharing_context="max_exclusive",
    )

    per_layer_rs = hockney_model_reduce_scatter_on_nd_topology(
        view=combine_view,
        P_total_bytes_before_rs=combine_bytes_per_rank,
        float_size_bytes=float_size,
    )

    rs_issue_note = ""
    if rs_handoff_mode == "NxM-unsupported":
        rs_issue_note = (
            " ISSUE: combine RS(G) handoff for mixed MoE TP (NxM) is not modeled yet "
            f"(tp_moe_x_shards={tp_moe_x_shards}, tp_moe_y_shards={tp_moe_y_shards})."
        )

    layout_trace.record_comm(
        "combine_rs_g",
        "RS",
        f"{combine_base_tag}({G})",
        G,
        combine_bytes_per_rank,
        per_layer_rs,
        note=(
            f"single-attention-domain dense combine/handoff: RS across G={G} colocated expert groups, "
            f"buffer=[B_set={tokens_per_set:.0f}, D_local={d_model/xpus_per_group:.0f}]"
            f", S_processing={num_processing_sets}, S_attn={num_combine_sets}"
            f", mode={rs_handoff_mode}, rs_axis={combine_base_tag}"
            f"{axis_note}"
            f"{rs_issue_note}"
        ),
    )

    per_layer = per_layer_rs
    per_layer_cross_set = CommStats()
    per_layer_combine_u = CommStats()

    # Cross-set All-Gather: when combine-domain set count > 1, each set
    # processed only tokens_per_set = B_global / S tokens.  After the
    # intra-set RS(G), every rank has its set's [B/S, D_local] shard.  To
    # reconstruct the full [B_attn_domain, D_local] for the next layer we need
    # an AG(S) across matching ranks in different expert sets.
    #
    # AG input per rank  = tokens_per_set × D_local  (one set's slice)
    # AG output per rank = B_attn_domain × D_local   (current attn-domain batch)
    if num_combine_sets > 1:
        tokens_per_attn_domain = tokens_per_set * num_combine_sets
        cross_set_ag_output_bytes = tokens_per_attn_domain * d_local_bytes

        cross_set_domain_note = ""
        cross_set_phys = topo_ref.get_mapping_for_tag(
            cross_set_base_tag
        ).maps_to_physical_dimension_name
        combine_phys = topo_ref.get_mapping_for_tag(
            combine_base_tag
        ).maps_to_physical_dimension_name
        try:
            attn_x_phys = topo_ref.get_mapping_for_tag("TP_Attn_X").maps_to_physical_dimension_name
            attn_y_phys = topo_ref.get_mapping_for_tag("TP_Attn_Y").maps_to_physical_dimension_name
            if cross_set_phys not in (attn_x_phys, attn_y_phys):
                cross_set_domain_note = (
                    " ISSUE: cross-set AG base domain does not map to attention TP axes "
                    f"(base_tag={cross_set_base_tag}, phys={cross_set_phys}, "
                    f"attn_x_phys={attn_x_phys}, attn_y_phys={attn_y_phys})."
                )
            else:
                # Same-axis RS and cross-set AG can be valid: they may use different
                # subgroups/virtual participant sets of the same physical dimension.
                cross_set_domain_note = (
                    f" verified_domain: rs_axis={combine_base_tag}({combine_phys}), "
                    f"cross_set_axis={cross_set_base_tag}({cross_set_phys})."
                )
        except Exception:
            cross_set_domain_note = (
                f" NOTE: TP_Attn_X/Y tags unavailable; rs_axis={combine_base_tag}({combine_phys}), "
                f"cross_set_axis={cross_set_base_tag}({cross_set_phys})."
            )

        cross_set_view = ep_topology_view.original_topology_ref.create_virtual_view_from_tag(
            view_name="Combine_CrossSet",
            base_tag=cross_set_base_tag,
            virtual_size=num_combine_sets,
            global_sharing_context="max_exclusive",
        )

        per_layer_cross_set = hockney_model_all_gather_on_nd_topology(
            view=cross_set_view,
            P_total_bytes_after_ag=cross_set_ag_output_bytes,
            float_size_bytes=float_size,
        )

        layout_trace.record_comm(
            "combine_ag_cross_set",
            "AG",
            f"{cross_set_base_tag}({num_combine_sets})",
            num_combine_sets,
            cross_set_ag_output_bytes,
            per_layer_cross_set,
            note=(
                f"Cross-set AG: S_attn={num_combine_sets} sets, each contributes "
                f"[B/S={tokens_per_set:.0f}, D_local={d_model/xpus_per_group:.0f}] "
                f"→ attn-domain [B={tokens_per_attn_domain:.0f}, D_local={d_model/xpus_per_group:.0f}]"
                f", ag_axis={cross_set_base_tag}"
                f"{cross_set_domain_note}"
            ),
        )

    else:
        tokens_per_attn_domain = tokens_per_set

    # projection_groups>1 means projection-domain work is split into
    # multiple independent YZ subgroups. After combine RS(G) (and optional cross-set
    # AG), we need an extra AG over that group axis so every rank in the active
    # attention shard sees the full post-combine token domain.
    if projection_groups > 1:
        # projection_groups (`g`) is introduced by shrinking projection-domain
        # YZ tags in run_scalescope. This reconciliation AG therefore stays on the
        # attention Y axis regardless of whether RS(G)/cross-set AG used X or Y.
        # It models restoring full visibility across projection groups in the YZ domain.
        u_base_tag = "TP_Attn_Y"
        try:
            _ = topo_ref.get_mapping_for_tag(u_base_tag)
        except Exception as e:
            raise ValueError(
                f"Missing required logical tag '{u_base_tag}' for projection-group combine AG(U) "
                f"(g={projection_groups}, combine_axis={combine_base_tag})."
            ) from e

        combine_u_ag_output_bytes = tokens_per_attn_domain * d_local_bytes
        combine_u_view = topo_ref.create_virtual_view_from_tag(
            view_name="Combine_U",
            base_tag=u_base_tag,
            virtual_size=projection_groups,
            global_sharing_context="max_exclusive",
        )
        per_layer_combine_u = hockney_model_all_gather_on_nd_topology(
            view=combine_u_view,
            P_total_bytes_after_ag=combine_u_ag_output_bytes,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            "combine_ag_u",
            "AG",
            f"{u_base_tag}({projection_groups})",
            projection_groups,
            combine_u_ag_output_bytes,
            per_layer_combine_u,
            note=(
                f"Projection-group combine AG(U): g={projection_groups}, "
                f"[B={tokens_per_attn_domain:.0f}, D_local={d_model/xpus_per_group:.0f}] "
                "to restore full combine-domain visibility for next-layer handoff."
            ),
        )
    # Keep post-RS gather stages serialized for now.
    per_layer = per_layer + per_layer_cross_set + per_layer_combine_u

    # Conservative NxM handoff remap:
    # For mixed MoE TP (tp_moe_x_shards>1 and tp_moe_y_shards>1), RS(G) does not
    # guarantee alignment with next-layer attention sharding. Model an explicit
    # remap as AG+RS on TP_Attn_X over the full post-combine buffer to avoid
    # optimistic winners while subgroup-exact mapping is not implemented.
    if tp_moe_x_shards > 1 and tp_moe_y_shards > 1:
        remap_base_tag = "TP_Attn_X"
        try:
            remap_size = topo_ref.get_mapping_for_tag(remap_base_tag).size
        except Exception as e:
            raise ValueError(
                f"Missing required logical tag '{remap_base_tag}' for NxM combine remap "
                f"(tp_moe_x_shards={tp_moe_x_shards}, tp_moe_y_shards={tp_moe_y_shards})."
            ) from e
        remap_output_bytes = tokens_per_attn_domain * d_local_bytes
        remap_view = topo_ref.create_virtual_view_from_tag(
            view_name="Combine_Remap_X",
            base_tag=remap_base_tag,
            virtual_size=remap_size,
            global_sharing_context="concurrent_baseline",
        )
        per_layer_remap_ag = hockney_model_all_gather_on_nd_topology(
            view=remap_view,
            P_total_bytes_after_ag=remap_output_bytes,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            "combine_remap_ag_x",
            "AG",
            f"{remap_base_tag}({remap_size})",
            remap_size,
            remap_output_bytes,
            per_layer_remap_ag,
            note=(
                f"Conservative NxM handoff remap (step 1/2): AG on {remap_base_tag}. "
                f"buffer=[B={tokens_per_attn_domain:.0f}, D_local={d_model/xpus_per_group:.0f}]"
            ),
        )
        per_layer_remap_rs = hockney_model_reduce_scatter_on_nd_topology(
            view=remap_view,
            P_total_bytes_before_rs=remap_output_bytes,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            "combine_remap_rs_x",
            "RS",
            f"{remap_base_tag}({remap_size})",
            remap_size,
            remap_output_bytes,
            per_layer_remap_rs,
            note=(
                f"Conservative NxM handoff remap (step 2/2): RS on {remap_base_tag}. "
                "Used until subgroup-exact remap is modeled."
            ),
        )
        per_layer = per_layer + per_layer_remap_ag + per_layer_remap_rs

    return per_layer


def calculate_moe_comm_hierarchical_total_exchange(
    ep_topology_view,
    transformer_config: TransformerConfig,
    moe_config,
    num_global_batch_tokens: int,
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
        or num_colocated_groups_per_set < 0
        or attn_domains_per_full_expert_set <= 0
    ):
        raise ValueError(
            "Invalid expert placement fields: expected full_expert_set_xpu_count>0, "
            "xpus_per_colocated_group>0, num_colocated_groups_per_set>=0, "
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
    if ep_total_xpus % max(1, full_expert_set_xpu_count) != 0:
        raise ValueError(
            f"EP chips ({ep_total_xpus}) must be divisible by full_expert_set_xpu_count ({full_expert_set_xpu_count})"
        )
    num_full_sets = ep_total_xpus // max(1, full_expert_set_xpu_count)

    tokens_per_set = num_global_batch_tokens / max(1, num_full_sets)
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
    # accounting). For the intra-rack centralized-pool path the two formulas
    # pick the same physical fabric — only the volume scales differ. We
    # keep the current product form for now so historical baselines are
    # preserved.
    #
    # TODO: refine into separate within-attention-domain and cross-attention-
    # domain fractions so cross-rack centralized pools (full_expert_set spans
    # scale-out fabric) attribute the right volume to EP_SU vs. EP_SO.
    # Current product form double-counts locality when both factors are > 1
    # and under-counts the SO leg when only the cross-domain share is remote.
    routing_buckets_per_set = max(
        1, attn_domains_per_full_expert_set * max(1, num_colocated_groups_per_set)
    )
    routing_nonlocal_fraction = max(0.0, 1.0 - (1.0 / float(routing_buckets_per_set)))
    # Model routed exchange at expert-group granularity, then map to per-XPU payload
    # by dividing across TP ranks inside the group.
    tokens_per_group = (
        tokens_per_set
        * routed_expert_traffic_multiplier
        * routing_nonlocal_fraction
        / max(1, num_colocated_groups_per_set)
    )
    # EP A2A toggle semantics:
    # - optimization ON: send one grouped copy over EP exchange, then redistribute inside TP group.
    # - optimization OFF: each TP rank carries its own routed copy over EP exchange.
    # This makes EP A2A volume larger in OFF mode, as intended.
    optimization_enabled = bool(getattr(workload_spec, "tp_moe_a2a_replication_optimization", True))
    a2a_replication_factor = (
        1.0 if optimization_enabled else float(max(1, xpus_per_colocated_group))
    )
    tokens_per_xpu = tokens_per_group / max(1, xpus_per_colocated_group) * a2a_replication_factor

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
    tp_moe_y_shards = max(1, int(tp_moe_y_shards))
    bytes_per_token = (d_model * float_size) / tp_moe_y_shards

    # Stage name in the layout trace.  Mirrors the TensorRecord stages
    # emitted by `decode_layout_trace._trace_moe_stages`:
    #   - decode_dispatch  (a2a_leg="dispatch")
    #   - decode_combine_a2a (a2a_leg="combine")
    # Prefill and single-attention-domain decode also pass through here; the
    # records are still emitted (NULL_LAYOUT_TRACE is a no-op for callers that
    # don't pass a real trace).
    trace_stage = "decode_dispatch" if a2a_leg == "dispatch" else "decode_combine_a2a"

    if moe_a2a_comm_mode == MoEA2ACommMode.FUSED_ND:
        # Experimental path: model MoE exchange as a single fused N-D total-exchange on EP topology.
        payload_bytes_per_xpu = tokens_per_xpu * bytes_per_token
        per_layer = hockney_model_total_exchange_on_nd_topology(
            ep_topology_view,
            P_total_send_bytes_per_node=payload_bytes_per_xpu,
            float_size_bytes=float_size,
        )
        layout_trace.record_comm(
            f"{trace_stage}_fused",
            "A2A",
            "EP_FUSED",
            ep_total_xpus,
            payload_bytes_per_xpu,
            per_layer,
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
            su_fabric_stats_per_layer = hockney_model_total_exchange_on_nd_topology(
                su_fabric_view,
                P_total_send_bytes_per_node=su_fabric_bytes_per_xpu,
                float_size_bytes=float_size,
            )
            layout_trace.record_comm(
                f"{trace_stage}_intra",
                "A2A",
                f"{su_fabric_fastest_tag}({ep_su_domain_size_xpus})",
                ep_su_domain_size_xpus,
                su_fabric_bytes_per_xpu,
                su_fabric_stats_per_layer,
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
            so_fabric_stats_per_layer = hockney_model_total_exchange_on_nd_topology(
                so_fabric_view,
                P_total_send_bytes_per_node=so_fabric_bytes_per_xpu,
                float_size_bytes=float_size,
            )
            layout_trace.record_comm(
                f"{trace_stage}_inter",
                "A2A",
                f"{so_fabric_tag}({so_fabric_participant_xpus})",
                so_fabric_participant_xpus,
                so_fabric_bytes_per_xpu,
                so_fabric_stats_per_layer,
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
