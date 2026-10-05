"""Consistency bounds and the mixed-Poisson null."""

from __future__ import annotations

import numpy as np
import pytest

from voc.summaries import (
    PUBLISHED,
    MixedPoisson,
    check_consistency,
    excess_over_heterogeneity,
)


def test_published_figures_are_jointly_impossible():
    """The finding this module exists for, pinned so it cannot quietly change.

    Pooling Shah 2019's count summaries with Platt 1991's tail requires patients
    with 1-3 episodes to supply more episodes than three each would allow. No
    distribution of the annual count satisfies it, so the two sources describe
    different populations and cannot be combined -- which is why summary
    statistics cannot settle the mechanism question.
    """
    result = check_consistency(
        p_zero=PUBLISHED["p_zero"].value,
        mean_per_year=PUBLISHED["mean_per_year"].value,
        p_above=PUBLISHED["p_high_utilizer"].value,
        tail_episode_share=PUBLISHED["tail_episode_share"].value,
    )
    assert not result.feasible
    assert result.middle_band_needs > result.middle_band_max
    assert any("at most" in reason for reason in result.reasons)
    assert "INFEASIBLE" in result.report()


def test_a_self_consistent_set_passes():
    """Summaries drawn from one real distribution must be reported feasible.

    Guards against a bound so tight it rejects everything, which would make the
    infeasibility finding meaningless.
    """
    # 60% zero, 30% with one episode, 10% with five.
    mean = 0.3 * 1 + 0.1 * 5
    tail_share = (0.1 * 5) / mean
    result = check_consistency(
        p_zero=0.6, mean_per_year=mean, p_above=0.1, tail_episode_share=tail_share
    )
    assert result.feasible, result.report()
    assert "FEASIBLE" in result.report()


def test_bounds_catch_a_tail_that_is_too_light():
    """A tail group too small to supply the episodes attributed to it."""
    # 5% of patients have >3 episodes, so they supply at least 4 each = 0.20,
    # but they are credited with only 0.02 * 1.0 of a mean of 1.0.
    result = check_consistency(
        p_zero=0.5, mean_per_year=1.0, p_above=0.05, tail_episode_share=0.02
    )
    assert not result.feasible
    assert any("at least 4" in reason for reason in result.reasons)


def test_bounds_catch_a_middle_band_credited_with_too_little():
    # 40% of patients have 1-3 episodes, so they supply at least 0.40, but are
    # credited with 0.1 of a mean of 1.0.
    result = check_consistency(
        p_zero=0.5, mean_per_year=1.0, p_above=0.1, tail_episode_share=0.9
    )
    assert not result.feasible
    assert any("at least one" in reason for reason in result.reasons)


def test_consistency_input_validation():
    base = dict(p_zero=0.5, mean_per_year=1.0, p_above=0.1, tail_episode_share=0.3)
    with pytest.raises(ValueError, match="p_zero must be a probability"):
        check_consistency(**{**base, "p_zero": 1.5})
    with pytest.raises(ValueError, match="exceeds 1"):
        check_consistency(**{**base, "p_zero": 0.95, "p_above": 0.3})
    with pytest.raises(ValueError, match="threshold must be at least 1"):
        check_consistency(**base, threshold=0)


def test_mixed_poisson_reproduces_the_summaries_it_was_fitted_to():
    model = MixedPoisson.from_summaries(p_zero=0.745, mean_per_year=1.422)
    assert model.shape * model.scale == pytest.approx(1.422, rel=1e-6)
    assert (1.0 + model.scale) ** (-model.shape) == pytest.approx(0.745, rel=1e-6)
    # Extreme overdispersion: a Poisson with this mean would leave only 24% at zero.
    assert model.shape < 0.5
    assert np.exp(-1.422) < 0.745


def test_mixed_poisson_rejects_underdispersed_input():
    """P(0) below the Poisson value cannot come from any Gamma mixing.

    Mixing only ever adds dispersion, so this is unreachable rather than merely
    hard to fit, and the error says so instead of returning a bad fit.
    """
    with pytest.raises(ValueError, match="no Gamma mixing"):
        MixedPoisson.from_summaries(p_zero=0.1, mean_per_year=1.422)


def test_short_interval_prediction_is_monotone_and_bounded():
    model = MixedPoisson.from_summaries(p_zero=0.745, mean_per_year=1.422)
    values = [model.p_next_within(d) for d in (1, 3, 7, 14, 30, 365)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a < b for a, b in zip(values[:-1], values[1:], strict=True))


def test_fitted_mixing_fails_the_independent_tail_check():
    """Over-identification: the fit must also reproduce a summary it never saw.

    It does not. Fitted to Shah's zero-share and mean, the Gamma predicts around
    11% of patients above three episodes per year supplying over 80% of
    episodes, against Platt's published 5% and 33%. Either the mixing is badly
    misspecified or the cohorts differ -- and with only summaries there is no
    way to tell which, which is the point.
    """
    model = MixedPoisson.from_summaries(p_zero=0.745, mean_per_year=1.422)
    assert model.p_above(3) > 2 * PUBLISHED["p_high_utilizer"].value
    assert model.tail_episode_share(3) > 2 * PUBLISHED["tail_episode_share"].value


def test_excess_ratio_crosses_one_across_horizons():
    """The signature that the inputs are incommensurable, not that the mechanism is odd.

    Adding self-excitation to a heterogeneity null can only push observed rates
    *above* predicted, at every horizon. A ratio falling through 1 as the window
    widens cannot be produced that way, so it indicts the inputs.
    """
    model = MixedPoisson.from_summaries(p_zero=0.745, mean_per_year=1.422)
    rows = excess_over_heterogeneity(
        model,
        {
            3.0: PUBLISHED["revisit_3d"],
            7.0: PUBLISHED["revisit_7d"],
            14.0: PUBLISHED["revisit_14d"],
            30.0: PUBLISHED["readmit_30d"],
        },
    )
    ratios = [ratio for *_, ratio in rows]
    assert ratios[0] > 1.0
    assert ratios[-1] < 1.0
    assert all(a > b for a, b in zip(ratios[:-1], ratios[1:], strict=True))


def test_every_published_figure_carries_its_provenance():
    """A number without its event definition is not usable here.

    The whole failure mode this module documents is arithmetic performed across
    incompatible definitions, so the definition travels with the value.
    """
    for key, estimate in PUBLISHED.items():
        assert estimate.source, key
        assert estimate.cohort, key
        assert estimate.definition, key
        assert str(estimate).startswith(estimate.quantity)

    # The two that most obviously cannot be compared say so in their definitions.
    assert "ENCOUNTER-level" in PUBLISHED["revisit_3d"].definition
    assert "INPATIENT" in PUBLISHED["readmit_30d"].definition
