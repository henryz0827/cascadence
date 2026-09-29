"""Cascade dynamics: the conditional construction, censoring, branching ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cascadence.cascade import (
    avalanche_ensemble,
    branching_ratio,
    critical_shift,
    expected_branching_ratio,
    generation_profile,
    run_avalanche,
    shift_for_ratio,
)
from cascadence.kernel import RaceKernel
from cascadence.network import lattice_network


@dataclass(frozen=True)
class ConstantKernel:
    """Blocking probability independent of transit time."""

    value: float

    def p_block(self, transit_time: np.ndarray) -> np.ndarray:
        return np.full(np.shape(transit_time), self.value, dtype=float)

    def shifted(self, delta: float) -> ConstantKernel:
        return ConstantKernel(value=self.value)


def _bed(seed: int = 42, shape: tuple[int, ...] = (16, 12)):
    return lattice_network(shape, np.random.default_rng(seed))


def test_transit_time_independent_kernel_never_propagates():
    """The decisive test of the conditional construction.

    With ``p_block`` independent of ``T``, the current and baseline
    probabilities coincide, so the conditional probability is identically zero
    and no segment may block beyond the seed -- however large the standing
    blocking probability is. If this fails, the model is sampling the baseline
    hazard instead of the response to redistribution, and every avalanche it
    reports is an artefact.
    """
    net = _bed()
    rng = np.random.default_rng(0)
    for value in (0.01, 0.5, 0.99):
        for _ in range(25):
            avalanche = run_avalanche(net, ConstantKernel(value), rng)
            assert avalanche.size == 1
            assert avalanche.generations == (1,)
            assert avalanche.terminated == "quiescent"
            assert not avalanche.censored


def test_unreachable_kernel_gives_singleton_avalanches():
    net = _bed()
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    # Median delay time far above any transit time in the network.
    kernel = RaceKernel(
        log_tau_median=float(np.log(finite.max()) + 40.0), log_tau_sd=1.0
    )
    ensemble = avalanche_ensemble(
        net, kernel, np.random.default_rng(1), n_realizations=40
    )
    assert all(av.size == 1 for av in ensemble)
    assert all(av.terminated == "quiescent" for av in ensemble)


def test_permissive_kernel_is_flagged_as_censored():
    """A system-spanning cascade must be reported, not silently recorded.

    It terminates by running out of segments, not by self-limiting, so its size
    measures ``n_edges`` rather than the dynamics. ``terminated`` has to say so,
    otherwise these realisations poison any fitted size distribution.
    """
    net = _bed()
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel(log_tau_median=float(np.log(np.median(finite))), log_tau_sd=2.0)
    ensemble = avalanche_ensemble(
        net, kernel, np.random.default_rng(2), n_realizations=60
    )
    censored = [av for av in ensemble if av.censored]
    assert censored, "expected system-spanning avalanches at this control setting"
    assert all(av.terminated == "exhausted" for av in censored)
    assert max(av.deperfused for av in ensemble) > 0.5 * baseline.n_perfused


def test_deperfused_never_below_blocked_size():
    """Stranded segments count as unperfused, so deperfused >= size always."""
    net = _bed()
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel(
        log_tau_median=float(np.log(np.median(finite)) + 4.0), log_tau_sd=2.0
    )
    ensemble = avalanche_ensemble(
        net, kernel, np.random.default_rng(3), n_realizations=120
    )
    assert all(av.deperfused >= av.size for av in ensemble)
    assert all(0.0 <= av.throughput_loss <= 1.0 for av in ensemble)


def test_branching_ratio_decreases_with_control_shift():
    """Raising the delay time monotonically suppresses propagation."""
    net = _bed()
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel(log_tau_median=float(np.log(np.median(finite))), log_tau_sd=2.0)

    measured = [
        branching_ratio(
            net, kernel.shifted(shift), np.random.default_rng(7), n_realizations=120
        ).from_offspring
        for shift in (4.0, 6.0, 8.0, 10.0)
    ]
    assert all(a > b for a, b in zip(measured, measured[1:], strict=False)), measured
    assert measured[0] > measured[-1]


def test_deep_subcritical_estimators_agree():
    """Far below criticality the mean-field branching relation does hold."""
    net = _bed()
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel(
        log_tau_median=float(np.log(np.median(finite)) + 9.0), log_tau_sd=2.0
    )
    measured = branching_ratio(
        net, kernel, np.random.default_rng(11), n_realizations=400
    )
    assert measured.censored_fraction == 0.0
    assert measured.subcritical
    assert measured.consistent, measured
    assert measured.from_offspring == pytest.approx(measured.from_mean_size, abs=0.05)


def test_mean_field_relation_fails_near_criticality():
    """Documented finding, pinned as a test so a refactor cannot hide it.

    Near the critical point the two branching-ratio estimators disagree by tens
    of percent: global pressure redistribution correlates offspring across
    generations, so the cascade is not a tree and ``<S> = 1/(1-R)`` does not
    hold. The practical consequence is that a branching ratio must not be
    inferred from a mean avalanche size in this regime.
    """
    net = _bed(shape=(24, 20))
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    kernel = RaceKernel(
        log_tau_median=float(np.log(np.median(finite)) + 6.0), log_tau_sd=2.0
    )
    measured = branching_ratio(
        net, kernel, np.random.default_rng(13), n_realizations=300
    )
    assert measured.censored_fraction == 0.0
    assert not measured.consistent, measured
    assert measured.from_offspring > measured.from_mean_size + 0.1, measured


def test_seed_must_be_perfused():
    net = _bed()
    baseline = net.solve()
    dead = np.flatnonzero(~baseline.perfused)
    kernel = RaceKernel(log_tau_median=0.0, log_tau_sd=1.0)
    if dead.size:
        with pytest.raises(ValueError, match="not perfused"):
            run_avalanche(
                net, kernel, np.random.default_rng(0), seed_edge=int(dead[0])
            )


def test_explicit_seed_is_honoured():
    net = _bed()
    baseline = net.solve()
    edge = int(np.flatnonzero(baseline.perfused)[0])
    kernel = RaceKernel(
        log_tau_median=float(np.log(baseline.transit_time[baseline.perfused].max()) + 40),
        log_tau_sd=1.0,
    )
    avalanche = run_avalanche(
        net, kernel, np.random.default_rng(0), seed_edge=edge
    )
    assert avalanche.seed_edge == edge
    assert avalanche.size == 1


# ---------------------------------------------------------------------------
# the closed-form branching ratio and the bisections built on it
# ---------------------------------------------------------------------------


def _centred_kernel(net, offset: float = 0.0) -> RaceKernel:
    baseline = net.solve()
    finite = baseline.transit_time[baseline.perfused]
    return RaceKernel.from_polymerization(
        tau_ref=float(np.median(finite)), exponent_n=20.0, conc_log_sd=0.1
    ).shifted(offset)


def test_closed_form_matches_monte_carlo():
    """The exact estimator must agree with simulating whole cascades."""
    net = _bed()
    kernel = _centred_kernel(net, 7.0)
    exact = expected_branching_ratio(net, kernel)
    sampled = branching_ratio(net, kernel, np.random.default_rng(1), n_realizations=600)
    assert abs(exact - sampled.from_offspring) < 3.0 * sampled.offspring_sem


def test_closed_form_is_deterministic_for_a_fixed_seed_set():
    """Regression: a re-drawn seed subsample makes each call a different function.

    ``expected_branching_ratio`` consumes ``rng`` when it subsamples, so calling
    it repeatedly with one generator evaluates a slightly different function
    each time. Every procedure that compares calls -- a sweep, or the bisection
    in ``critical_shift`` -- must therefore pin the seed set once and pass it
    in. Both behaviours are asserted here so neither can drift.
    """
    net = _bed()
    kernel = _centred_kernel(net, 6.0)
    baseline = net.solve()
    seeds = np.flatnonzero(baseline.perfused)[:120]

    fixed = [expected_branching_ratio(net, kernel, seeds=seeds) for _ in range(3)]
    assert fixed[0] == pytest.approx(fixed[1]) == pytest.approx(fixed[2])

    shared = np.random.default_rng(0)
    redrawn = [
        expected_branching_ratio(net, kernel, n_seeds=120, rng=shared) for _ in range(3)
    ]
    assert len({round(value, 12) for value in redrawn}) > 1

    with pytest.raises(ValueError, match="rng is required"):
        expected_branching_ratio(net, kernel, n_seeds=10)


def test_critical_shift_lands_on_unit_branching_ratio():
    """End-to-end check: verify the located shift against the full seed set."""
    net = _bed()
    kernel = _centred_kernel(net)
    shift = critical_shift(net, kernel, tol=0.01)
    assert expected_branching_ratio(net, kernel.shifted(shift)) == pytest.approx(
        1.0, abs=0.02
    )


@pytest.mark.parametrize("target", [0.5, 1.5, 2.0])
def test_shift_for_ratio_hits_its_target(target):
    net = _bed()
    kernel = _centred_kernel(net)
    shift = shift_for_ratio(net, kernel, target, tol=0.01)
    assert expected_branching_ratio(net, kernel.shifted(shift)) == pytest.approx(
        target, rel=0.03
    )


def test_bisection_rejects_a_bracket_that_does_not_straddle():
    net = _bed()
    kernel = _centred_kernel(net)
    with pytest.raises(ValueError, match="does not straddle"):
        critical_shift(net, kernel, bracket=(12.0, 20.0))
    with pytest.raises(ValueError, match="target branching ratio must be positive"):
        shift_for_ratio(net, kernel, 0.0)


def test_branching_ratio_collapses_after_the_first_generation():
    """The structural reason R = 1 is not a critical condition here.

    Measured at the setting where the first-generation ratio is 1, the second
    generation branches at roughly half that rate and the ratio then plateaus
    well below 1: successive generations re-attack a neighbourhood the previous
    one already stripped of its susceptible segments, so no single rate
    describes the cascade. Pinned as a test because three other results depend
    on it -- the small avalanches at R = 1, the failure of <S> = 1/(1-R), and
    the absence of any scale-free regime.
    """
    net = _bed(shape=(20, 20))
    kernel = _centred_kernel(net)
    shift = critical_shift(net, kernel, n_seeds=400, rng=np.random.default_rng(7))
    ensemble = avalanche_ensemble(
        net, kernel.shifted(shift), np.random.default_rng(5), n_realizations=1500
    )
    profile = generation_profile(ensemble)

    assert profile["ratio"].size >= 4
    # The three arrays describe the same generations and must stay aligned.
    assert profile["generation"].shape == profile["ratio"].shape
    assert profile["generation"].shape == profile["total"].shape
    assert profile["generation"][0] == 1

    first = profile["ratio"][0]
    assert first == pytest.approx(1.0, abs=0.12), profile["ratio"]
    # The drop after the first generation is the finding, not noise.
    assert profile["ratio"][1] < 0.8 * first, profile["ratio"]
    # ...and what follows stays below 1 rather than recovering.
    assert profile["ratio"][1:5].max() < 0.85, profile["ratio"]

    with pytest.raises(ValueError, match="empty ensemble"):
        generation_profile([])


def test_seeds_must_be_perfused():
    net = _bed()
    baseline = net.solve()
    dead = np.flatnonzero(~baseline.perfused)
    if dead.size:
        with pytest.raises(ValueError, match="every seed must be perfused"):
            expected_branching_ratio(net, _centred_kernel(net), seeds=dead[:1])
