"""Estimator checks on synthetic data with known parameters."""

from __future__ import annotations

import numpy as np
import pytest

from cascadence.scaling import (
    bootstrap_alpha,
    exponent_profile,
    fit_lognormal,
    fit_powerlaw,
    fit_powerlaw_cutoff,
    lr_test_cutoff,
    vuong_test,
)


def _sample_discrete_powerlaw(
    alpha: float,
    xmin: int,
    size: int,
    rng: np.random.Generator,
    x_max: int = 200_000,
) -> np.ndarray:
    """Exact inverse-CDF sampling from a truncated discrete power law."""
    support = np.arange(xmin, x_max + 1, dtype=np.int64)
    weights = support.astype(float) ** (-alpha)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return support[np.searchsorted(cdf, rng.random(size))]


def _sample_discrete_cutoff(
    alpha: float,
    rate: float,
    xmin: int,
    size: int,
    rng: np.random.Generator,
    x_max: int = 200_000,
) -> np.ndarray:
    support = np.arange(xmin, x_max + 1, dtype=np.int64)
    log_w = -alpha * np.log(support) - rate * support
    weights = np.exp(log_w - log_w.max())
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return support[np.searchsorted(cdf, rng.random(size))]


@pytest.mark.parametrize("alpha", [1.8, 2.5, 3.2])
def test_recovers_known_exponent(alpha):
    rng = np.random.default_rng(int(alpha * 1000))
    data = _sample_discrete_powerlaw(alpha, 1, 60_000, rng)
    fit = fit_powerlaw(data, xmin=1)
    assert fit.alpha == pytest.approx(alpha, abs=0.03)
    assert fit.n_tail == data.size


def test_recovers_exponent_with_selected_xmin():
    """KS-selected xmin must not bias the exponent on a genuine power law."""
    rng = np.random.default_rng(99)
    data = _sample_discrete_powerlaw(2.5, 4, 60_000, rng)
    fit = fit_powerlaw(data)
    assert fit.xmin >= 4
    assert fit.alpha == pytest.approx(2.5, abs=0.06)
    assert fit.ks < 0.02


def test_exponent_profile_is_flat_for_a_true_power_law():
    """A real asymptotic exponent shows up as a plateau in alpha(xmin)."""
    rng = np.random.default_rng(5)
    data = _sample_discrete_powerlaw(2.5, 1, 200_000, rng)
    profile = exponent_profile(data, min_tail=300)
    assert profile["alpha"].size >= 6
    assert np.all(np.abs(profile["alpha"] - 2.5) < 0.1), profile["alpha"]
    # Flat means small spread, not merely close on average.
    assert profile["alpha"].std() < 0.04


def test_bootstrap_interval_covers_the_truth():
    rng = np.random.default_rng(17)
    data = _sample_discrete_powerlaw(2.5, 1, 20_000, rng)
    lo, hi, reps = bootstrap_alpha(data, n_boot=120, rng=rng, xmin=1)
    assert lo < 2.5 < hi
    assert hi - lo < 0.15
    assert reps.size >= 100


def test_cutoff_test_finds_no_cutoff_in_a_pure_power_law():
    rng = np.random.default_rng(23)
    data = _sample_discrete_powerlaw(2.2, 1, 60_000, rng)
    lr, p, fit = lr_test_cutoff(data, xmin=1)
    assert p > 0.05, (lr, p, fit)
    assert lr < 4.0


def test_cutoff_test_detects_a_real_cutoff():
    rng = np.random.default_rng(29)
    rate = 1.0 / 40.0
    data = _sample_discrete_cutoff(1.6, rate, 1, 60_000, rng)
    lr, p, fit = lr_test_cutoff(data, xmin=1)
    assert p < 1e-6, (lr, p, fit)
    assert fit.xi == pytest.approx(1.0 / rate, rel=0.35)
    assert not fit.support_capped


def test_cutoff_fit_reduces_to_power_law_when_rate_is_zero():
    rng = np.random.default_rng(31)
    data = _sample_discrete_powerlaw(2.4, 1, 40_000, rng)
    pure = fit_powerlaw(data, xmin=1)
    cut = fit_powerlaw_cutoff(data, xmin=1)
    # The nested model can never fit worse than the null.
    assert cut.loglik >= pure.loglik - 1e-6
    assert cut.alpha == pytest.approx(pure.alpha, abs=0.25)


def test_vuong_prefers_power_law_on_power_law_data():
    rng = np.random.default_rng(37)
    data = _sample_discrete_powerlaw(2.5, 1, 60_000, rng)
    statistic, p = vuong_test(data, xmin=1)
    assert statistic > 0.0
    assert p < 0.05


def test_lognormal_fit_is_well_behaved():
    rng = np.random.default_rng(41)
    values = np.rint(np.exp(rng.normal(2.0, 1.0, size=40_000))).astype(np.int64)
    data = values[values >= 1]
    fit = fit_lognormal(data, xmin=1)
    assert np.isfinite(fit.loglik)
    assert 1.0 < fit.mu < 3.0
    assert 0.5 < fit.sigma < 1.6


def test_input_validation():
    with pytest.raises(ValueError, match="integer-valued"):
        fit_powerlaw(np.array([1.0, 2.5, 3.0]))
    with pytest.raises(ValueError, match=">= 1"):
        fit_powerlaw(np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="non-finite"):
        fit_powerlaw(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="no data"):
        fit_powerlaw(np.array([]))
    with pytest.raises(ValueError, match="need >="):
        fit_powerlaw(np.array([1, 2, 3, 4, 5]), xmin=1)
