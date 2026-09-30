"""Episode collapsing and interval extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import voc_fixture
from voc.episodes import (
    DEFAULT_MIN_GAP_DAYS,
    collapse_episodes,
    interevent_intervals,
    patient_summary,
    to_days,
)


def _encounters(rows):
    """rows: (subject, start_day, length_days)."""
    return pd.DataFrame([{"subject_id": s, "start": a, "end": a + b} for s, a, b in rows])


def test_the_default_threshold_is_the_trial_convention():
    assert DEFAULT_MIN_GAP_DAYS == 3.0


def test_short_revisit_collapses_into_one_episode():
    """~17% of crisis encounters are followed by a revisit within 3 days.

    Those revisits are usually the same under-treated crisis. Counting them as
    separate events manufactures a spike of near-zero intervals that mimics
    exactly the temporal clustering the study is looking for.
    """
    episodes = collapse_episodes(_encounters([(1, 0.0, 2.0), (1, 3.0, 1.0)]))
    assert len(episodes) == 1
    assert episodes.loc[0, "n_encounters"] == 2
    assert episodes.loc[0, "episode_start"] == 0.0
    assert episodes.loc[0, "episode_end"] == 4.0


def test_encounters_beyond_the_threshold_stay_separate():
    episodes = collapse_episodes(_encounters([(1, 0.0, 2.0), (1, 10.0, 1.0)]))
    assert len(episodes) == 2
    assert list(episodes["n_encounters"]) == [1, 1]


def test_collapsing_chains_transitively():
    """Encounters two days apart, repeatedly, are one continuing episode.

    The gap is measured from the running episode's end, not from the first
    encounter, so a long chain of short gaps stays a single episode instead of
    fragmenting.
    """
    episodes = collapse_episodes(
        _encounters([(1, 0.0, 1.0), (1, 3.0, 1.0), (1, 6.0, 1.0), (1, 9.0, 1.0)])
    )
    assert len(episodes) == 1
    assert episodes.loc[0, "n_encounters"] == 4
    assert episodes.loc[0, "episode_end"] == 10.0


def test_gap_measured_from_admission_differs_from_discharge():
    """A long stay changes which encounters count as one episode.

    Start-to-start ignores how long the patient was in hospital, so a 9-day
    admission followed 10 days later by another reads as two episodes; measured
    from discharge, the second begins one day after the first ended and is the
    same episode.
    """
    rows = _encounters([(1, 0.0, 9.0), (1, 10.0, 1.0)])
    assert len(collapse_episodes(rows, gap_from="admission")) == 2
    assert len(collapse_episodes(rows, gap_from="discharge")) == 1


def test_zero_threshold_disables_collapsing():
    episodes = collapse_episodes(
        _encounters([(1, 0.0, 1.0), (1, 1.0, 1.0)]), min_gap_days=0.0
    )
    assert len(episodes) == 2


def test_threshold_travels_with_the_output():
    """A result read without its threshold is uninterpretable, so it ships with it."""
    episodes = collapse_episodes(_encounters([(1, 0.0, 1.0)]), min_gap_days=5.0)
    assert episodes.loc[0, "min_gap_days"] == 5.0


def test_patients_do_not_bleed_into_each_other():
    episodes = collapse_episodes(_encounters([(1, 0.0, 1.0), (2, 1.0, 1.0)]))
    assert len(episodes) == 2
    assert set(episodes["subject_id"]) == {1, 2}
    assert list(episodes["n_encounters"]) == [1, 1]


def test_unsorted_input_is_handled():
    ordered = collapse_episodes(_encounters([(1, 0.0, 1.0), (1, 20.0, 1.0)]))
    shuffled = collapse_episodes(_encounters([(1, 20.0, 1.0), (1, 0.0, 1.0)]))
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_missing_end_time_is_treated_as_instantaneous():
    """An ED stay with no recorded end must not silently swallow the next event.

    Treating it as instantaneous can only lengthen the measured gap, so it can
    under-collapse but never over-collapse -- the safe direction.
    """
    frame = pd.DataFrame(
        {"subject_id": [1, 1], "start": [0.0, 4.0], "end": [np.nan, 5.0]}
    )
    episodes = collapse_episodes(frame)
    assert len(episodes) == 2


def test_end_before_start_does_not_corrupt_the_episode():
    frame = pd.DataFrame({"subject_id": [1], "start": [10.0], "end": [2.0]})
    episodes = collapse_episodes(frame)
    assert episodes.loc[0, "episode_end"] == 10.0


def test_timestamps_and_day_offsets_agree():
    """The same pipeline must serve MIMIC-IV timestamps and HCUP DaysToEvent.

    Intervals are planned from HCUP SID/SEDD and severity from MIMIC-IV, so a
    divergence here would silently misalign the two halves of the study.
    """
    numeric = collapse_episodes(
        _encounters([(1, 0.0, 2.0), (1, 3.0, 1.0), (1, 40.0, 1.0)])
    )

    origin = pd.Timestamp("2150-01-01")
    stamped = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "start": [
                origin,
                origin + pd.Timedelta(days=3),
                origin + pd.Timedelta(days=40),
            ],
            "end": [
                origin + pd.Timedelta(days=2),
                origin + pd.Timedelta(days=4),
                origin + pd.Timedelta(days=41),
            ],
        }
    )
    from_stamps = collapse_episodes(stamped)

    assert len(numeric) == len(from_stamps) == 2
    np.testing.assert_allclose(
        numeric["episode_start"].to_numpy(), from_stamps["episode_start"].to_numpy()
    )


def test_to_days_passes_numeric_through_and_converts_datetimes():
    np.testing.assert_allclose(to_days([0.0, 1.5, 3.0]), [0.0, 1.5, 3.0])
    stamps = pd.to_datetime(["2150-01-01", "2150-01-03"])
    np.testing.assert_allclose(to_days(stamps), [0.0, 2.0])


def test_intervals_are_start_to_start():
    """Not end-to-start, which would build the study's own hypothesis into the data.

    Subtracting length of stay would make the interval depend on a severity
    measure by construction, manufacturing exactly the frequency-severity
    correlation the analysis is meant to test.
    """
    episodes = collapse_episodes(_encounters([(1, 0.0, 9.0), (1, 20.0, 1.0)]))
    intervals = interevent_intervals(episodes)
    assert len(intervals) == 1
    assert intervals.loc[0, "interval_days"] == 20.0


def test_single_episode_patients_contribute_no_interval():
    episodes = collapse_episodes(
        _encounters([(1, 0.0, 1.0), (2, 0.0, 1.0), (2, 30.0, 1.0)])
    )
    intervals = interevent_intervals(episodes)
    assert set(intervals["subject_id"]) == {2}
    assert len(intervals) == 1


def test_interval_order_is_recorded():
    episodes = collapse_episodes(
        _encounters([(1, 0.0, 1.0), (1, 30.0, 1.0), (1, 100.0, 1.0)])
    )
    intervals = interevent_intervals(episodes)
    assert list(intervals["order"]) == [1, 2]
    assert list(intervals["interval_days"]) == [30.0, 70.0]


def test_patient_summary_rates():
    episodes = collapse_episodes(
        _encounters([(1, 0.0, 1.0), (1, 365.25, 1.0), (2, 0.0, 1.0)])
    )
    summary = patient_summary(episodes).set_index("subject_id")
    assert summary.loc[1, "n_episodes"] == 2
    assert summary.loc[1, "rate_per_year"] == pytest.approx(1.0)
    # One episode gives no span, so no rate -- not a rate of zero.
    assert summary.loc[2, "n_episodes"] == 1
    assert np.isnan(summary.loc[2, "rate_per_year"])


def test_empty_inputs_round_trip():
    empty = pd.DataFrame({"subject_id": [], "start": [], "end": []})
    episodes = collapse_episodes(empty)
    assert episodes.empty
    assert interevent_intervals(episodes).empty
    assert patient_summary(episodes).empty


def test_rejects_bad_arguments():
    frame = _encounters([(1, 0.0, 1.0)])
    with pytest.raises(ValueError, match="non-negative"):
        collapse_episodes(frame, min_gap_days=-1.0)
    with pytest.raises(ValueError, match="gap_from"):
        collapse_episodes(frame, gap_from="sideways")
    with pytest.raises(ValueError, match="missing columns"):
        collapse_episodes(pd.DataFrame({"subject_id": [1]}))


def test_fixture_episode_structure_is_recovered_exactly():
    """End-to-end against the synthetic cohort's declared ground truth."""
    tables = voc_fixture.build()
    rows = []
    for plan in tables["plans"]:
        for episode in plan.episodes:
            for start, length in episode:
                rows.append((plan.subject_id, start, length))
    if not rows:
        pytest.skip("fixture has no inpatient episodes")

    episodes = collapse_episodes(_encounters(rows))
    counts = episodes.groupby("subject_id").size().to_dict()
    for plan in tables["plans"]:
        if plan.n_episodes:
            assert counts[plan.subject_id] == plan.n_episodes, plan.subject_id
