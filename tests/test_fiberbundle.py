"""Validation against the ELS fiber bundle's analytically known exponents.

These are the load-bearing tests of the repository. If they fail, every
exponent ``cascadence.scaling`` reports anywhere else is suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascadence.fiberbundle import avalanche_sizes, critical_force, uniform_bundle
from cascadence.scaling import exponent_profile, fit_powerlaw


def _brute_force_avalanches(thresholds: np.ndarray, n_steps: int = 200_000):
    """Reference implementation: ramp the external force and watch it break.

    Deliberately written a completely different way from the record-maxima
    algorithm under test -- it simulates the actual loading protocol, finding
    the fixed point of the broken set at each force level.
    """
    x = np.sort(thresholds)
    n = x.size
    f_c = float(np.max((n - np.arange(n)) * x))
    sizes = []
    n_broken = 0
    for force in np.linspace(0.0, f_c * (1.0 + 1e-7), n_steps):
        k = n_broken
        while k < n and force / (n - k) >= x[k]:
            k += 1
        if k > n_broken:
            sizes.append(k - n_broken)
            n_broken = k
    return np.asarray(sizes)


@pytest.mark.parametrize("n_fibers", [30, 60, 120])
def test_extraction_matches_loading_simulation(n_fibers):
    """Record maxima of (N-k)x_k reproduce an explicit force-ramp simulation."""
    rng = np.random.default_rng(7)
    thresholds = uniform_bundle(n_fibers, rng)
    reference = _brute_force_avalanches(thresholds)
    extracted = avalanche_sizes(thresholds, include_final=True)
    np.testing.assert_array_equal(extracted, reference)
    assert extracted.sum() == n_fibers


def test_final_avalanche_is_dropped_by_default():
    rng = np.random.default_rng(1)
    thresholds = uniform_bundle(500, rng)
    with_final = avalanche_sizes(thresholds, include_final=True)
    without = avalanche_sizes(thresholds)
    assert without.size == with_final.size - 1
    np.testing.assert_array_equal(without, with_final[:-1])
    # The dropped one is the catastrophic failure: O(N), far above the rest.
    assert with_final[-1] > 10 * np.median(with_final[:-1])


def test_critical_force_matches_load_curve_peak():
    rng = np.random.default_rng(2)
    thresholds = uniform_bundle(1000, rng)
    x = np.sort(thresholds)
    n = x.size
    assert critical_force(thresholds) == pytest.approx(
        float(np.max((n - np.arange(n)) * x))
    )


def test_full_history_exponent_converges_to_five_halves():
    """The asymptotic burst exponent is 5/2 (Hemmer-Hansen).

    Checked as a *convergence* rather than a single fit: alpha(xmin) must
    approach 5/2 from above and plateau there. Fitting from xmin=1 gives ~2.72
    because the 5/2 law is asymptotic, which is exactly why
    ``exponent_profile`` exists and why a lone fitted number is not a
    measurement.
    """
    rng = np.random.default_rng(11)
    sizes = np.concatenate(
        [avalanche_sizes(uniform_bundle(300_000, rng)) for _ in range(12)]
    )

    profile = exponent_profile(sizes, min_tail=250)
    assert profile["alpha"].size >= 12

    # Fitting from xmin=1 overestimates the exponent (~2.71), because the 5/2
    # law is asymptotic and the head of the distribution is not a power law.
    assert profile["alpha"][0] > 2.6

    # The deep end of the profile settles onto 5/2. Averaged over the last few
    # cutoffs, because individually they sit on a few hundred points each and
    # fluctuate by a few percent.
    deep = profile["alpha"][-5:]
    assert abs(deep.mean() - 2.5) < 0.07, profile

    # Convergence must be a descent toward 5/2, not a wander.
    assert profile["alpha"][-1] < profile["alpha"][0] - 0.1
    assert profile["alpha"][:4].mean() > deep.mean()


def test_near_critical_window_crosses_over_to_three_halves():
    """Restricting to the near-failure window drives the small-size slope to 3/2.

    The crossover is the point: 5/2 and 3/2 are the *same* mechanism sampled
    over different windows, not two mechanisms. Any claim that a measured
    exponent discriminates between mechanisms has to control for this.
    """
    rng = np.random.default_rng(3)
    windows = [0.80, 0.90, 0.95, 0.99, 0.999]
    pooled = {w: [] for w in windows}
    for _ in range(30):
        thresholds = uniform_bundle(200_000, rng)
        for window in windows:
            pooled[window].append(avalanche_sizes(thresholds, force_window=window))

    alphas = [fit_powerlaw(np.concatenate(pooled[w]), xmin=1).alpha for w in windows]

    # Monotone decrease as the window tightens onto the critical point.
    assert all(a > b for a, b in zip(alphas, alphas[1:], strict=False)), alphas
    # Wide window sits near the full-history value, tight window near 3/2.
    assert alphas[0] > 1.9, alphas
    assert abs(alphas[-1] - 1.5) < 0.12, alphas


def test_rejects_degenerate_input():
    with pytest.raises(ValueError, match="strictly positive"):
        avalanche_sizes(np.array([0.5, 0.0, 0.2]))
    with pytest.raises(ValueError, match="force_window"):
        avalanche_sizes(np.array([0.5, 0.3, 0.2]), force_window=1.5)
    assert avalanche_sizes(np.array([])).size == 0
