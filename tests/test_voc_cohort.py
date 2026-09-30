"""Building the crisis-encounter table, and the feasibility verdict."""

from __future__ import annotations

import pandas as pd
import pytest

import voc_fixture
from voc.cohort import build_crisis_encounters, feasibility
from voc.episodes import collapse_episodes


def _encounters(include_ed=True, **kwargs):
    tables = voc_fixture.build()
    return build_crisis_encounters(
        tables["diagnoses"],
        tables["admissions"],
        ed_diagnoses=tables["ed_diagnoses"] if include_ed else None,
        ed_stays=tables["ed_stays"] if include_ed else None,
        **kwargs,
    )


def test_only_crisis_coded_encounters_are_included():
    """Patient 5 has sickle cell coded, but only without-crisis and trait codes.

    Sickle cell with crisis and without have materially different readmission
    profiles, so pooling them would measure something other than crisis
    recurrence -- and trait is not the disease at all.
    """
    encounters = _encounters()
    assert 5 not in set(encounters["subject_id"])


def test_without_crisis_codes_can_be_opted_in():
    encounters = _encounters(include_no_crisis=True)
    assert 5 in set(encounters["subject_id"])


def test_an_ed_visit_that_became_an_admission_is_counted_once():
    """The fixture gives every inpatient episode a preceding ED stay.

    Both carry a crisis code and the ED stay carries the admission's hadm_id.
    Counted naively that doubles every encounter and halves every interval.
    """
    with_ed = _encounters(include_ed=True)
    without_ed = _encounters(include_ed=False)

    inpatient = with_ed[with_ed["source"] == "inpatient"]
    assert len(inpatient) == len(without_ed)

    # The only ED rows surviving are genuine ED-only visits.
    emergency = with_ed[with_ed["source"] == "ed"]
    expected = sum(len(plan.ed_only) for plan in voc_fixture.build()["plans"])
    assert len(emergency) == expected


def test_ed_only_visits_are_included():
    """Patient 4's second episode exists only as an ED visit.

    Dropping ED-only encounters would lose that patient's only interval --
    which is the systematic loss that makes an inpatient-only source the wrong
    instrument for recurrence.
    """
    encounters = _encounters()
    patient = encounters[encounters["subject_id"] == 4]
    assert set(patient["source"]) == {"inpatient", "ed"}
    assert len(patient) == 2


def test_encounters_are_sorted_per_patient():
    encounters = _encounters()
    for _, group in encounters.groupby("subject_id"):
        assert group["start"].is_monotonic_increasing


def test_end_to_end_recovers_the_declared_cohort():
    tables = voc_fixture.build()
    encounters = _encounters()
    episodes = collapse_episodes(encounters)
    counts = episodes.groupby("subject_id").size().to_dict()

    for plan in tables["plans"]:
        expected = plan.n_episodes + len(plan.ed_only)
        if expected == 0:
            assert plan.subject_id not in counts
        else:
            assert counts[plan.subject_id] == expected, plan.subject_id


def test_feasibility_counts_and_verdict():
    episodes = collapse_episodes(_encounters())
    result = feasibility(episodes, threshold_episodes=3, threshold_patients=1)

    # Patient 1 has three episodes; 2 and 3 have one; 4 has two; 5 is absent.
    assert result.n_patients == 4
    assert result.n_with_at_least[1] == 4
    assert result.n_with_at_least[2] == 2
    assert result.n_with_at_least[3] == 1
    assert result.passes
    assert "PASSES" in result.report()


def test_feasibility_fails_against_a_realistic_threshold():
    """The default demands 150 patients with 3+ episodes. A toy cohort must fail.

    The verdict is meant to bind: a threshold chosen after seeing the count is
    a rationalisation, so the failure path has to be the loud one.
    """
    episodes = collapse_episodes(_encounters())
    result = feasibility(episodes)

    assert result.threshold_patients == 150
    assert not result.passes
    report = result.report()
    assert "FAILS" in report
    assert "severity-proxy work only" in report


def test_feasibility_on_an_empty_cohort():
    empty = collapse_episodes(pd.DataFrame({"subject_id": [], "start": [], "end": []}))
    result = feasibility(empty)
    assert result.n_patients == 0
    assert result.n_episodes == 0
    assert not result.passes


def test_feasibility_buckets_the_long_tail():
    episodes = pd.DataFrame(
        {
            "subject_id": [1] * 25 + [2],
            "episode_start": list(range(0, 25 * 30, 30)) + [0],
        }
    )
    result = feasibility(episodes)
    # A patient with 25 episodes is reported in the 10+ bucket, not as its own row.
    assert max(result.episodes_per_patient) == 10
    assert result.n_with_at_least[10] == 1


@pytest.mark.parametrize("include_ed", [True, False])
def test_encounter_table_has_the_expected_shape(include_ed):
    encounters = _encounters(include_ed=include_ed)
    assert list(encounters.columns) == [
        "subject_id",
        "encounter_id",
        "source",
        "start",
        "end",
    ]
    assert not encounters.empty
