"""Can published summary statistics answer the question without buying data?

The cheapest possible study: take the numbers already in the literature -- the
share of patients with no crisis in a year, the mean rate, the heavy tail, the
short-interval revisit rates -- and ask whether they discriminate between the
two mechanisms that produce clustered recurrent events:

**heterogeneity**
    patients differ in a fixed rate; events within a patient are Poisson. Short
    intervals are over-represented only because a randomly chosen *event* comes
    disproportionately from a high-rate patient.
**self-excitation**
    an event raises the near-term risk of the next. Short intervals are
    over-represented beyond what heterogeneity alone explains.

The answer is no, and the reason is worth the module: the published numbers are
**not mutually consistent**, so they cannot be pooled, and no amount of
modelling fixes that. :func:`check_consistency` establishes it without assuming
any distribution at all -- not a mixing family, not a parametric form, nothing.

Why that is the useful finding
------------------------------
It converts "should I spend the money?" from a judgement into a derivation. The
obstacle is not statistical power, which more published numbers would fix. It
is that each number comes from a different cohort under a different event
definition, and the disagreements between them are as large as the effect being
tested. The only repair is one cohort with one definition, measured on the same
patients -- which is exactly what individual-level data is, and exactly what
cannot be reconstructed from summaries.

Every estimate here therefore carries its source, cohort and event definition,
and comparisons that cross definitions are flagged rather than silently made.

The figures in ``PUBLISHED`` were gathered from secondary sources and have NOT
been checked against the primary publications. Verify each at its source before
any of this reaches a protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = [
    "PUBLISHED",
    "ConsistencyCheck",
    "MixedPoisson",
    "PublishedEstimate",
    "check_consistency",
    "excess_over_heterogeneity",
]


@dataclass(frozen=True)
class PublishedEstimate:
    """A number from the literature, with the provenance that decides its use.

    ``definition`` is not documentation. Two estimates with different event
    definitions are not comparable, and the whole difficulty here is that
    combining them looks like analysis while being arithmetic on incommensurable
    quantities.
    """

    value: float
    quantity: str
    source: str
    cohort: str
    definition: str

    def __str__(self) -> str:
        return f"{self.quantity}={self.value:g} [{self.source}; {self.cohort}]"


# Gathered from secondary sources; each needs checking at the primary
# publication. Kept together so the definitional mismatches are visible at a
# glance rather than buried at the point of use.
PUBLISHED = {
    "p_zero": PublishedEstimate(
        value=0.745,
        quantity="P(no VOC in a year)",
        source="Shah 2019, J Health Econ Outcomes Res",
        cohort="claims, 20,909 patients, all ages (adults 0.691)",
        definition="VOC episode per the study's claims algorithm",
    ),
    "mean_per_year": PublishedEstimate(
        value=1.4220,
        quantity="mean VOC per patient-year",
        source="Shah 2019, J Health Econ Outcomes Res",
        cohort="claims, adults (142.20 per 100 person-years)",
        definition="VOC episode per the study's claims algorithm",
    ),
    "p_high_utilizer": PublishedEstimate(
        value=0.05,
        quantity="P(more than 3 episodes per year)",
        source="Platt 1991, NEJM",
        cohort="CSSCD, prospective, all ages, 1979-1988",
        definition="clinically ascertained painful episode",
    ),
    "tail_episode_share": PublishedEstimate(
        value=0.33,
        quantity="share of all episodes from the >3/yr group",
        source="Platt 1991, NEJM",
        cohort="CSSCD, prospective, all ages, 1979-1988",
        definition="clinically ascertained painful episode",
    ),
    "revisit_3d": PublishedEstimate(
        value=0.17,
        quantity="P(ED revisit within 3 days)",
        source="Walsh 2023, Am J Hematol",
        cohort="40 US EDs, 13,847 index encounters",
        definition="ENCOUNTER-level revisit, NOT collapsed into episodes",
    ),
    "revisit_7d": PublishedEstimate(
        value=0.24,
        quantity="P(ED revisit within 7 days)",
        source="Walsh 2023, Am J Hematol",
        cohort="40 US EDs, 13,847 index encounters",
        definition="ENCOUNTER-level revisit, NOT collapsed into episodes",
    ),
    "revisit_14d": PublishedEstimate(
        value=0.31,
        quantity="P(ED revisit within 14 days)",
        source="Walsh 2023, Am J Hematol",
        cohort="40 US EDs, 13,847 index encounters",
        definition="ENCOUNTER-level revisit, NOT collapsed into episodes",
    ),
    "readmit_30d": PublishedEstimate(
        value=0.269,
        quantity="P(inpatient readmission within 30 days)",
        source="NRD 2016, Sci Rep 2020",
        cohort="67,887 index sickle-cell-crisis discharges",
        definition="INPATIENT readmission after an INPATIENT discharge only",
    ),
}


# ---------------------------------------------------------------------------
# the assumption-free consistency bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsistencyCheck:
    """Whether a set of count summaries is jointly achievable by any distribution."""

    middle_band_needs: float
    middle_band_max: float
    middle_band_min: float
    tail_needs: float
    tail_min: float
    feasible: bool
    reasons: tuple[str, ...]

    def report(self) -> str:
        lines = [
            "assumption-free consistency of the count summaries",
            "  (no mixing family, no parametric form -- bounds any distribution obeys)",
            "",
            f"  episodes required from the 1..m band : {self.middle_band_needs:.4f}",
            f"  most that band could contribute      : {self.middle_band_max:.4f}",
            f"  least that band could contribute     : {self.middle_band_min:.4f}",
            f"  episodes required from the >m tail   : {self.tail_needs:.4f}",
            f"  least the tail could contribute      : {self.tail_min:.4f}",
            "",
        ]
        if self.feasible:
            lines.append("  FEASIBLE -- the summaries can describe one population")
        else:
            lines.append(
                "  INFEASIBLE -- no distribution of the annual count satisfies these"
            )
            lines.extend(f"    {reason}" for reason in self.reasons)
            lines.append("")
            lines.append(
                "  The numbers therefore describe different populations under "
                "different\n  event definitions, and cannot be pooled. No model "
                "choice repairs this."
            )
        return "\n".join(lines)


def check_consistency(
    *,
    p_zero: float,
    mean_per_year: float,
    p_above: float,
    tail_episode_share: float,
    threshold: int = 3,
) -> ConsistencyCheck:
    """Bound-check four count summaries against every possible distribution.

    Let ``N`` be a patient's episode count in a year and ``m = threshold``. Any
    distribution whatsoever obeys:

    * patients with ``1 <= N <= m`` contribute at least ``P(1<=N<=m)`` episodes
      and at most ``m * P(1<=N<=m)``;
    * patients with ``N > m`` contribute at least ``(m+1) * P(N>m)``.

    The published figures fix how many episodes each group must contribute --
    ``(1 - tail_episode_share) * mean`` and ``tail_episode_share * mean``. If a
    requirement falls outside its bound, the figures are jointly impossible, and
    that conclusion rests on nothing but counting.
    """
    if not 0.0 <= p_zero <= 1.0:
        raise ValueError("p_zero must be a probability")
    if not 0.0 <= p_above <= 1.0:
        raise ValueError("p_above must be a probability")
    if not 0.0 <= tail_episode_share <= 1.0:
        raise ValueError("tail_episode_share must be a probability")
    if mean_per_year < 0.0:
        raise ValueError("mean_per_year must be non-negative")
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    if p_zero + p_above > 1.0:
        raise ValueError("p_zero + p_above exceeds 1")

    p_middle = 1.0 - p_zero - p_above
    middle_needs = (1.0 - tail_episode_share) * mean_per_year
    middle_max = threshold * p_middle
    middle_min = p_middle
    tail_needs = tail_episode_share * mean_per_year
    tail_min = (threshold + 1) * p_above

    reasons = []
    if middle_needs > middle_max + 1e-12:
        reasons.append(
            f"patients with 1..{threshold} episodes must supply "
            f"{middle_needs:.4f} episodes but can supply at most {middle_max:.4f} "
            f"({threshold} each across {p_middle:.1%} of patients)"
        )
    if middle_needs < middle_min - 1e-12:
        reasons.append(
            f"patients with 1..{threshold} episodes must supply only "
            f"{middle_needs:.4f} episodes but there are {p_middle:.1%} of them, "
            "each with at least one"
        )
    if tail_needs < tail_min - 1e-12:
        reasons.append(
            f"patients with more than {threshold} episodes must supply "
            f"{tail_needs:.4f} episodes but there are {p_above:.1%} of them, "
            f"each with at least {threshold + 1}"
        )

    return ConsistencyCheck(
        middle_band_needs=middle_needs,
        middle_band_max=middle_max,
        middle_band_min=middle_min,
        tail_needs=tail_needs,
        tail_min=tail_min,
        feasible=not reasons,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# pure heterogeneity, for the comparison it would have supported
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixedPoisson:
    """Gamma-mixed Poisson counts: patients differ in rate, events are Poisson.

    The null against which self-excitation would be tested. Fitted from the
    annual count summaries, it *predicts* the short-interval statistics with
    nothing left free -- which is what would make the comparison a test rather
    than a fit, if the inputs were commensurable.
    """

    shape: float
    scale: float

    @classmethod
    def from_summaries(cls, *, p_zero: float, mean_per_year: float) -> MixedPoisson:
        """Solve for the Gamma mixing that reproduces ``P(N=0)`` and ``E[N]``."""
        if not 0.0 < p_zero < 1.0:
            raise ValueError("p_zero must lie strictly between 0 and 1")
        if mean_per_year <= 0.0:
            raise ValueError("mean_per_year must be positive")
        if p_zero <= np.exp(-mean_per_year):
            raise ValueError(
                "p_zero is below the Poisson value for this mean; no Gamma "
                "mixing is under-dispersed enough"
            )

        def residual(log_params):
            shape, scale = np.exp(log_params)
            return [
                shape * scale - mean_per_year,
                (1.0 + scale) ** (-shape) - p_zero,
            ]

        solved = optimize.fsolve(residual, [np.log(0.1), np.log(10.0)], full_output=True)
        params, _, status, message = solved
        if status != 1:
            raise RuntimeError(f"Gamma mixing did not converge: {message}")
        shape, scale = np.exp(params)
        return cls(shape=float(shape), scale=float(scale))

    def p_next_within(self, days: float) -> float:
        """``P(next event within `days` | a randomly chosen event)``.

        Choosing an event rather than a patient size-biases toward high-rate
        patients, which is the whole of what heterogeneity contributes to short
        intervals. For Gamma mixing this is
        ``1 - (1 + scale * t)**-(shape + 1)`` with ``t`` in years.
        """
        t = days / 365.25
        return float(1.0 - (1.0 + self.scale * t) ** (-(self.shape + 1.0)))

    def p_above(self, threshold: int) -> float:
        """``P(N > threshold)`` in a year."""
        p = 1.0 / (1.0 + self.scale)
        return float(stats.nbinom.sf(threshold, self.shape, p))

    def tail_episode_share(self, threshold: int, support: int = 5000) -> float:
        """Share of all episodes contributed by patients above ``threshold``."""
        p = 1.0 / (1.0 + self.scale)
        counts = np.arange(support)
        pmf = stats.nbinom.pmf(counts, self.shape, p)
        weighted = counts * pmf
        total = weighted.sum()
        if total <= 0.0:
            return float("nan")
        return float(weighted[counts > threshold].sum() / total)


def excess_over_heterogeneity(
    model: MixedPoisson, observed: dict[float, PublishedEstimate]
) -> list[tuple[float, float, float, float]]:
    """Observed short-interval rates against what heterogeneity alone predicts.

    Returns ``(days, predicted, observed, ratio)``. A ratio above 1 at every
    horizon would indicate self-excitation on top of heterogeneity. A ratio that
    *crosses* 1 as the horizon grows cannot be produced by adding excitation to
    this null, and so indicates the inputs are measuring different things rather
    than that the mechanism is exotic.
    """
    rows = []
    for days, estimate in sorted(observed.items()):
        predicted = model.p_next_within(days)
        ratio = estimate.value / predicted if predicted > 0 else float("nan")
        rows.append((days, predicted, estimate.value, ratio))
    return rows
