"""Avalanche dynamics on a flow network, and measurement of the branching ratio.

One avalanche: block a seed segment, re-solve the pressure field, let the
segments whose transit time grew block with the probability the kernel assigns
them, and iterate until nothing more blocks.

Why the blocking probability is *conditional*
---------------------------------------------
The kernel assigns every perfused segment a standing blocking probability
``p_base = p_block(T_base)`` in the undisturbed network, and that number is not
small. Sampling it directly on every round would occlude the whole network
immediately, with or without any redistribution -- the "quiescent" state would
not be quiescent, and no branching ratio would be measurable.

What an avalanche actually is, is the network's *response to redistribution*: a
segment blocks now because losing a neighbour lengthened its transit time,
given that it survived at its baseline hazard. That is the conditional
probability

    p_cond = (p_block(T_now) - p_block(T_base)) / (1 - p_block(T_base))

which is zero whenever the transit time has not changed, so the undisturbed
network is genuinely quiescent and every block in an avalanche is attributable
to redistribution. The slow baseline occlusion this construction factors out is
the *driving*, and it belongs to a separate, slower process; it is not part of
the avalanche.

Two sizes are recorded, and they differ
---------------------------------------
``size``
    Segments actively occluded.
``deperfused``
    Segments that lost perfusion, whether by occluding themselves or by being
    stranded behind something that did. This is the larger number, and it is
    the one with a physical referent -- a stranded subtree is just as
    unperfused as an occluded segment. Use ``deperfused`` as the avalanche size
    unless there is a specific reason not to.

The branching ratio is always measured, never assumed, with two independent
estimators (mean first-generation offspring, and ``R = 1 - 1/<S>`` from the
mean total progeny of a subcritical branching process). They agree only to the
extent the mean-field branching picture applies; a disagreement is a result
about the network, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernel import BlockingKernel
from .network import FlowNetwork, FlowState

__all__ = [
    "Avalanche",
    "BranchingRatio",
    "avalanche_ensemble",
    "branching_ratio",
    "critical_shift",
    "expected_branching_ratio",
    "generation_profile",
    "run_avalanche",
    "shift_for_ratio",
    "sweep_control",
]


@dataclass(frozen=True)
class Avalanche:
    """Outcome of a single avalanche.

    ``terminated`` records *why* the cascade stopped, and it decides whether the
    recorded size is usable:

    ``"quiescent"``
        A round produced no new blocks. The avalanche ended on its own, so its
        size is a genuine sample from the size distribution.
    ``"exhausted"``
        No perfused, unblocked segment was left to block. The avalanche ate the
        network, so its size is censored at the system size and says more about
        ``n_edges`` than about the dynamics.
    ``"max_rounds"``
        Hit the generation cap while still spreading.

    Only ``"quiescent"`` avalanches may be fed to :mod:`cascadence.scaling`.
    Mixing the other two into a size distribution manufactures a spurious
    cutoff at the system size and biases any fitted exponent.
    """

    seed_edge: int
    size: int
    deperfused: int
    generations: tuple[int, ...]
    terminated: str
    throughput_loss: float

    @property
    def censored(self) -> bool:
        """Whether the size is limited by the system rather than the dynamics."""
        return self.terminated != "quiescent"

    @property
    def rounds(self) -> int:
        return len(self.generations)

    @property
    def offspring(self) -> int:
        """Segments blocked in the first generation, i.e. the seed's offspring."""
        return self.generations[1] if len(self.generations) > 1 else 0


def run_avalanche(
    network: FlowNetwork,
    kernel: BlockingKernel,
    rng: np.random.Generator,
    *,
    seed_edge: int | None = None,
    baseline: FlowState | None = None,
    baseline_p: np.ndarray | None = None,
    max_rounds: int = 200,
) -> Avalanche:
    """Run one avalanche from a single seeded block.

    Parameters
    ----------
    seed_edge
        Which edge to occlude first. Defaults to a uniformly random perfused
        edge. Seeding non-uniformly (say, on the most vulnerable segment)
        changes the measured branching ratio, so the choice is part of the
        protocol and must be reported.
    baseline, baseline_p
        Pre-computed undisturbed state and its blocking probabilities. Pass
        these when running many avalanches on the same network -- they are what
        the conditional probability is referenced to and recomputing them per
        realisation is both wasteful and a source of inconsistency.
    max_rounds
        Cap on cascade generations. Hitting it sets ``runaway``, which means the
        configuration is supercritical and the avalanche has no finite size; do
        not fold such realisations into a size distribution.
    """
    if baseline is None:
        baseline = network.solve()
    if baseline_p is None:
        baseline_p = kernel.p_block(baseline.transit_time)

    baseline_throughput = network.total_flow(baseline)
    perfused_edges = np.flatnonzero(baseline.perfused)
    if perfused_edges.size == 0:
        raise ValueError("baseline network has no perfused edges")

    if seed_edge is None:
        seed_edge = int(rng.choice(perfused_edges))
    elif not baseline.perfused[seed_edge]:
        raise ValueError(f"seed edge {seed_edge} is not perfused at baseline")

    blocked = np.zeros(network.n_edges, dtype=bool)
    blocked[seed_edge] = True
    generations = [1]
    terminated = "max_rounds"

    for _ in range(max_rounds):
        state = network.solve(blocked)
        candidates = state.perfused & ~blocked
        if not candidates.any():
            # Nothing left that could block: the avalanche consumed the network
            # and its size is censored at the system size, not self-limiting.
            terminated = "exhausted"
            break

        p_now = kernel.p_block(state.transit_time[candidates])
        p_was = baseline_p[candidates]
        # Probability of blocking now given survival at the baseline hazard.
        p_cond = np.clip(
            (p_now - p_was) / np.clip(1.0 - p_was, 1e-12, None), 0.0, 1.0
        )
        newly = rng.random(p_cond.size) < p_cond
        if not newly.any():
            terminated = "quiescent"
            break

        blocked[np.flatnonzero(candidates)[newly]] = True
        generations.append(int(np.count_nonzero(newly)))

    final = network.solve(blocked)
    deperfused = int(np.count_nonzero(baseline.perfused & ~final.perfused))
    final_throughput = network.total_flow(final)
    loss = (
        1.0 - final_throughput / baseline_throughput
        if baseline_throughput > 0.0
        else 0.0
    )

    return Avalanche(
        seed_edge=int(seed_edge),
        size=int(np.count_nonzero(blocked)),
        deperfused=deperfused,
        generations=tuple(generations),
        terminated=terminated,
        throughput_loss=float(loss),
    )


def avalanche_ensemble(
    network: FlowNetwork,
    kernel: BlockingKernel,
    rng: np.random.Generator,
    *,
    n_realizations: int = 500,
    max_rounds: int = 200,
) -> list[Avalanche]:
    """Run many independent avalanches against one shared baseline."""
    baseline = network.solve()
    baseline_p = kernel.p_block(baseline.transit_time)
    return [
        run_avalanche(
            network,
            kernel,
            rng,
            baseline=baseline,
            baseline_p=baseline_p,
            max_rounds=max_rounds,
        )
        for _ in range(n_realizations)
    ]


@dataclass(frozen=True)
class BranchingRatio:
    """Two independent estimates of the branching ratio, plus diagnostics."""

    from_offspring: float
    from_mean_size: float
    offspring_sem: float
    mean_size: float
    mean_deperfused: float
    censored_fraction: float
    n_realizations: int

    @property
    def subcritical(self) -> bool:
        """Whether the configuration produced finite, uncensored avalanches.

        With any censored realisation the mean total progeny is truncated by the
        system size, so ``from_mean_size`` is biased toward 1 from below and
        cannot be read as a branching ratio at all.
        """
        return self.censored_fraction == 0.0 and self.from_offspring < 1.0

    @property
    def consistent(self) -> bool:
        """Whether the estimators agree within two standard errors of the first.

        Disagreement means the mean-field branching approximation does not hold
        on this network -- typically because redistribution is correlated across
        generations rather than tree-like. That is a finding to report, not an
        error to suppress.
        """
        if self.censored_fraction > 0.0:
            return False
        tol = max(2.0 * self.offspring_sem, 0.02)
        return abs(self.from_offspring - self.from_mean_size) <= tol


def branching_ratio(
    network: FlowNetwork,
    kernel: BlockingKernel,
    rng: np.random.Generator,
    *,
    n_realizations: int = 500,
    max_rounds: int = 200,
) -> BranchingRatio:
    """Measure the branching ratio two independent ways.

    ``from_offspring``
        Mean number of segments blocked in the first generation. This is the
        branching ratio by definition, and needs no assumption about the
        cascade's shape.
    ``from_mean_size``
        ``1 - 1/<S>``, inverting the mean total progeny of a subcritical
        Galton-Watson process. Valid only if the cascade really is branching,
        which is precisely what comparing it against the first estimator tests.

    Both are meaningless if any realisation was censored by the system size;
    ``censored_fraction`` reports that, and neither
    :attr:`BranchingRatio.consistent` nor :attr:`BranchingRatio.subcritical`
    will certify such a sweep point.
    """
    ensemble = avalanche_ensemble(
        network, kernel, rng, n_realizations=n_realizations, max_rounds=max_rounds
    )
    offspring = np.array([av.offspring for av in ensemble], dtype=float)
    sizes = np.array([av.size for av in ensemble], dtype=float)
    deperfused = np.array([av.deperfused for av in ensemble], dtype=float)
    censored = np.array([av.censored for av in ensemble], dtype=bool)

    mean_size = float(sizes.mean())
    return BranchingRatio(
        from_offspring=float(offspring.mean()),
        from_mean_size=float(1.0 - 1.0 / mean_size) if mean_size > 0.0 else np.nan,
        offspring_sem=float(offspring.std(ddof=1) / np.sqrt(offspring.size)),
        mean_size=mean_size,
        mean_deperfused=float(deperfused.mean()),
        censored_fraction=float(censored.mean()),
        n_realizations=len(ensemble),
    )


def expected_branching_ratio(
    network: FlowNetwork,
    kernel: BlockingKernel,
    *,
    seeds: np.ndarray | None = None,
    n_seeds: int | None = None,
    rng: np.random.Generator | None = None,
    baseline: FlowState | None = None,
    baseline_p: np.ndarray | None = None,
) -> float:
    """Exact expected first-generation offspring, without Monte Carlo.

    The branching ratio is the mean number of segments a single block sets off.
    Given the seed, that is a sum of independent Bernoulli trials, so its
    expectation is available in closed form:

        E[offspring | seed e] = sum over candidates of p_cond

    which needs one pressure solve per seed and carries no sampling noise at
    all. :func:`branching_ratio` estimates the same quantity by simulating
    whole cascades; this is the estimator to use when only ``R`` is wanted --
    it is both cheaper and exact, which is what makes bisecting for ``R = 1``
    in :func:`critical_shift` practical.

    Averaging over every perfused seed (the default) makes the result a
    deterministic function of the network and kernel. Pass ``n_seeds`` to
    subsample on large networks, in which case ``rng`` is required.

    When comparing across control settings -- a sweep, or the bisection in
    :func:`critical_shift` -- draw the subsample **once** and pass it as
    ``seeds``. Re-drawing per evaluation leaves each call unbiased but makes
    the sequence of calls a different function each time, which silently
    defeats any procedure that assumes it is comparing like with like.
    """
    if baseline is None:
        baseline = network.solve()
    if baseline_p is None:
        baseline_p = kernel.p_block(baseline.transit_time)

    perfused = np.flatnonzero(baseline.perfused)
    if perfused.size == 0:
        raise ValueError("baseline network has no perfused edges")

    if seeds is not None:
        seeds = np.atleast_1d(np.asarray(seeds, dtype=np.int64))
        if not baseline.perfused[seeds].all():
            raise ValueError("every seed must be perfused at baseline")
    elif n_seeds is not None and n_seeds < perfused.size:
        if rng is None:
            raise ValueError("rng is required when subsampling seeds")
        seeds = rng.choice(perfused, size=n_seeds, replace=False)
    else:
        seeds = perfused

    blocked = np.zeros(network.n_edges, dtype=bool)
    total = 0.0
    for seed in seeds:
        blocked[:] = False
        blocked[seed] = True
        state = network.solve(blocked)
        candidates = state.perfused & ~blocked
        if not candidates.any():
            continue
        p_now = kernel.p_block(state.transit_time[candidates])
        p_was = baseline_p[candidates]
        total += float(
            np.clip((p_now - p_was) / np.clip(1.0 - p_was, 1e-12, None), 0.0, 1.0).sum()
        )
    return total / len(seeds)


def generation_profile(ensemble: list[Avalanche]) -> dict[str, np.ndarray]:
    """Branching ratio resolved generation by generation.

    ``R_k`` is the total number of segments blocked in generation ``k+1``
    divided by the total blocked in generation ``k``, pooled over the ensemble.

    This is the diagnostic that explains why tuning the branching ratio to 1
    does not make this system critical. On a flow network with global
    redistribution the ratio is **not a single number**: measured at the
    setting where ``R_1 = 1``, the second generation branches at roughly half
    that rate and the ratio then plateaus well below 1. Successive generations
    re-attack a neighbourhood the previous one already depleted of its
    susceptible segments, so the cascade is not a tree and no single rate
    describes it.

    Three observations follow, and they are consistent with each other:

    * ``R_1 = 1`` is not a critical condition, so avalanches stay small there.
    * ``<S> = 1/(1-R)`` is invalid, since it assumes one rate for all
      generations -- which is why the two estimators in
      :func:`branching_ratio` disagree near criticality.
    * Reaching ``R_k = 1`` for ``k >= 2`` would need ``R_1`` around 2, and by
      then avalanches span the system outright.

    Returns a dict whose three arrays are aligned and equal in length:
    ``generation`` (the index ``k``, starting at 1), ``ratio`` (``R_k``) and
    ``total`` (segments blocked in generation ``k``, the numerator of ``R_k``).
    Ratios are reported only while the denominator is large enough to mean
    anything, so the profile ends where the statistics run out rather than
    trailing off into noise.
    """
    if not ensemble:
        raise ValueError("empty ensemble")

    depth = max(len(av.generations) for av in ensemble)
    totals = np.zeros(depth, dtype=float)
    for avalanche in ensemble:
        for index, count in enumerate(avalanche.generations):
            totals[index] += count

    usable = totals >= 20.0
    last = int(np.argmin(usable)) if not usable.all() else depth
    last = max(last, 1)

    return {
        "generation": np.arange(1, last, dtype=np.int64),
        "ratio": totals[1:last] / totals[: last - 1],
        "total": totals[1:last],
    }


def shift_for_ratio(
    network: FlowNetwork,
    kernel,
    target: float,
    *,
    bracket: tuple[float, float] = (0.0, 20.0),
    tol: float = 0.02,
    max_iter: int = 40,
    n_seeds: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Locate the control shift at which the branching ratio equals ``target``.

    Comparing avalanche statistics across system sizes requires matching on the
    branching ratio, not on the raw control shift: the same shift produces
    different ratios on different lattices, so a shift-matched comparison
    silently compares different physical states and any trend in ``L`` it shows
    is an artefact of that mismatch.

    See :func:`critical_shift` for the ``target = 1`` case and for the details
    of the bisection.
    """
    if target <= 0.0:
        raise ValueError("target branching ratio must be positive")
    return _bisect_ratio(
        network, kernel, target, bracket, tol, max_iter, n_seeds, rng
    )


def critical_shift(
    network: FlowNetwork,
    kernel,
    *,
    bracket: tuple[float, float] = (0.0, 20.0),
    tol: float = 0.02,
    max_iter: int = 40,
    n_seeds: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Locate the control shift at which the branching ratio crosses 1.

    Bisects :func:`expected_branching_ratio`, which is monotonically decreasing
    in the shift (raising the median delay time can only make propagation
    harder). The seed set is drawn **once** up front and held fixed across
    every evaluation, so the bisection descends a single deterministic
    function; re-drawing it per step would make each step probe a slightly
    different function and the bracket would collapse onto the wrong point.

    Raises ``ValueError`` if the bracket does not straddle ``R = 1``; widen it
    rather than assuming a crossing exists.
    """
    return _bisect_ratio(network, kernel, 1.0, bracket, tol, max_iter, n_seeds, rng)


def _bisect_ratio(
    network: FlowNetwork,
    kernel,
    target: float,
    bracket: tuple[float, float],
    tol: float,
    max_iter: int,
    n_seeds: int | None,
    rng: np.random.Generator | None,
) -> float:
    if not hasattr(kernel, "shifted"):
        raise TypeError("kernel must implement .shifted(delta)")

    baseline = network.solve()
    perfused = np.flatnonzero(baseline.perfused)
    if n_seeds is not None and n_seeds < perfused.size:
        if rng is None:
            raise ValueError("rng is required when subsampling seeds")
        seeds = rng.choice(perfused, size=n_seeds, replace=False)
    else:
        seeds = perfused

    lo, hi = float(bracket[0]), float(bracket[1])

    def ratio(shift: float) -> float:
        shifted = kernel.shifted(shift)
        return expected_branching_ratio(
            network,
            shifted,
            seeds=seeds,
            baseline=baseline,
            baseline_p=shifted.p_block(baseline.transit_time),
        )

    r_lo, r_hi = ratio(lo), ratio(hi)
    if r_lo < target or r_hi > target:
        raise ValueError(
            f"bracket [{lo}, {hi}] does not straddle R={target} "
            f"(R={r_lo:.3f} and R={r_hi:.3f}); widen it"
        )

    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sweep_control(
    network: FlowNetwork,
    kernel,
    shifts: np.ndarray,
    rng: np.random.Generator,
    *,
    n_realizations: int = 300,
    max_rounds: int = 200,
) -> dict[str, np.ndarray]:
    """Measure the branching ratio across a control-parameter sweep.

    ``kernel`` must expose ``shifted(delta)`` (see
    :class:`cascadence.kernel.RaceKernel`). Each ``shift`` translates the median
    log delay time, so the sweep traces out where ``R`` crosses 1 -- the
    quantity the whole model hinges on, and one that has to be located
    numerically rather than assumed.

    Returns arrays keyed ``shift``, ``r_offspring``, ``r_mean_size``,
    ``mean_size``, ``mean_deperfused``, ``censored_fraction``.
    """
    if not hasattr(kernel, "shifted"):
        raise TypeError("kernel must implement .shifted(delta) to be swept")

    rows = []
    for shift in np.asarray(shifts, dtype=float):
        measured = branching_ratio(
            network,
            kernel.shifted(float(shift)),
            rng,
            n_realizations=n_realizations,
            max_rounds=max_rounds,
        )
        rows.append(
            (
                shift,
                measured.from_offspring,
                measured.from_mean_size,
                measured.mean_size,
                measured.mean_deperfused,
                measured.censored_fraction,
            )
        )
    arr = np.asarray(rows, dtype=float)
    return {
        "shift": arr[:, 0],
        "r_offspring": arr[:, 1],
        "r_mean_size": arr[:, 2],
        "mean_size": arr[:, 3],
        "mean_deperfused": arr[:, 4],
        "censored_fraction": arr[:, 5],
    }
