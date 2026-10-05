"""Does the free route work? Run this to see the derivation, not the opinion.

Asks whether published summary statistics alone can tell heterogeneity from
self-excitation -- the cheapest possible version of the study, costing nothing
and needing no data access.

They cannot, and the reason is not a shortage of numbers. Run it.
"""

from __future__ import annotations

from voc.summaries import (
    PUBLISHED,
    MixedPoisson,
    check_consistency,
    excess_over_heterogeneity,
)

SHORT_INTERVAL = {
    3.0: "revisit_3d",
    7.0: "revisit_7d",
    14.0: "revisit_14d",
    30.0: "readmit_30d",
}


def main() -> None:
    print("=" * 74)
    print("CAN PUBLISHED SUMMARIES SETTLE THE MECHANISM WITHOUT BUYING DATA?")
    print("=" * 74)
    print()
    print("Figures used (gathered from secondary sources -- verify at the primary")
    print("publications before any of this reaches a protocol):")
    print()
    for estimate in PUBLISHED.values():
        print(f"  {estimate.value:>7.4g}  {estimate.quantity}")
        print(f"           {estimate.source}  |  {estimate.cohort}")
        print(f"           definition: {estimate.definition}")
    print()

    print("-" * 74)
    print("STEP 1 -- fit heterogeneity to the annual count summaries")
    print("-" * 74)
    model = MixedPoisson.from_summaries(
        p_zero=PUBLISHED["p_zero"].value,
        mean_per_year=PUBLISHED["mean_per_year"].value,
    )
    print(f"  Gamma mixing: shape {model.shape:.4f}, scale {model.scale:.3f}")
    print("  Very small shape means very strong heterogeneity, which is expected:")
    print("  a plain Poisson with this mean would leave far fewer patients at zero.")
    print()

    print("-" * 74)
    print("STEP 2 -- predict short intervals from that heterogeneity alone")
    print("-" * 74)
    print("  Picking a random EVENT favours high-rate patients, and that alone")
    print("  produces short intervals. Self-excitation would push the observed")
    print("  rate ABOVE this prediction -- at every horizon, since excitation")
    print("  can only add events.")
    print()
    print(f"  {'window':>8} {'predicted':>11} {'observed':>10} {'ratio':>8}")
    rows = excess_over_heterogeneity(
        model, {days: PUBLISHED[key] for days, key in SHORT_INTERVAL.items()}
    )
    for days, predicted, observed, ratio in rows:
        print(f"  {days:>6.0f}d {predicted:>11.1%} {observed:>10.1%} {ratio:>8.2f}")
    print()
    ratios = [ratio for *_, ratio in rows]
    if ratios[0] > 1.0 > ratios[-1]:
        print("  The ratio CROSSES 1 as the window widens. Adding excitation to this")
        print("  null cannot do that, so the figures are measuring different things.")
        print("  Note the definitions above: the 3-day number counts raw ED revisits")
        print("  with no episode collapsing, and the 30-day number counts only")
        print("  inpatient-to-inpatient readmissions. Neither matches the other, or")
        print("  the event definition behind the counts.")
    print()

    print("-" * 74)
    print("STEP 3 -- over-identification: a summary the fit never saw")
    print("-" * 74)
    predicted_above = model.p_above(3)
    predicted_share = model.tail_episode_share(3)
    print(f"  patients with >3 episodes/yr : predicted {predicted_above:6.1%}   "
          f"published {PUBLISHED['p_high_utilizer'].value:5.0%}")
    print(f"  share of episodes they supply: predicted {predicted_share:6.1%}   "
          f"published {PUBLISHED['tail_episode_share'].value:5.0%}")
    print("  Off by roughly a factor of two in both. Either the mixing is badly")
    print("  misspecified or the cohorts genuinely differ -- and summaries alone")
    print("  cannot distinguish those.")
    print()

    print("-" * 74)
    print("STEP 4 -- the check that needs no model at all")
    print("-" * 74)
    result = check_consistency(
        p_zero=PUBLISHED["p_zero"].value,
        mean_per_year=PUBLISHED["mean_per_year"].value,
        p_above=PUBLISHED["p_high_utilizer"].value,
        tail_episode_share=PUBLISHED["tail_episode_share"].value,
    )
    print(result.report())
    print()

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    if result.feasible:
        print("  The summaries are mutually consistent; the free route is worth")
        print("  pushing further before spending anything.")
        return

    print("  The free route does not work, and not for want of numbers.")
    print()
    print("  The published figures cannot all describe one population, so they")
    print("  cannot be pooled -- and the gaps between them are as large as the")
    print("  effect being tested. More published summaries would not help: each")
    print("  new one arrives with its own cohort and its own event definition.")
    print()
    print("  What individual-level data buys is precisely what is missing here:")
    print("  ONE cohort, ONE event definition, with the count distribution and")
    print("  the interval structure measured on the SAME patients. That makes")
    print("  the comparison in STEP 2 a test instead of an equivocation.")
    print()
    print("  It does not have to be expensive. The comparison needs a count")
    print("  distribution and a short-interval distribution from one cohort --")
    print("  not multi-year follow-up. A one-year inpatient source supports that,")
    print("  and the full interval distribution can wait until it is shown to be")
    print("  the thing worth paying more for.")


if __name__ == "__main__":
    main()
