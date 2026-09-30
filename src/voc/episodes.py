"""Collapsing encounters into episodes, and extracting inter-event intervals.

The rule this module exists to enforce
--------------------------------------
A hospital encounter is not an event. Roughly **17% of vaso-occlusive crisis ED
encounters are followed by a revisit within 3 days** (Walsh et al., *Am J
Hematol* 2023, 40 US emergency departments, 13,847 index encounters), and a
revisit that fast is usually the *same* under-treated crisis rather than a new
one. Vaso-occlusion trials accordingly count a new crisis only when it begins
at least 3 days after the previous one resolved.

Skipping that rule does not add noise, it manufactures signal: a spike of
near-zero intervals that looks exactly like temporal clustering, at roughly the
magnitude anyone would be hoping to find. Any clustering result computed on raw
encounters is uninterpretable. So collapsing is the default here, the threshold
is explicit, and it is recorded in the output.

Collapsing chains transitively: encounters two days apart, repeatedly, are one
continuing episode, not several. The gap is measured from the *running
episode's* end, which is what makes that work.

Two time conventions, deliberately
----------------------------------
Encounter times may be timestamps (MIMIC-IV) or integer day offsets from a
per-patient random origin (HCUP's ``DaysToEvent``). Both are accepted and
normalised to days internally, because the plan splits across both sources --
intervals from HCUP SID/SEDD, which follows patients across facilities and care
settings, and severity from MIMIC-IV, which has the labs and administered
doses. The same code must run on both or the two halves will not line up.

What the intervals are, and are not
-----------------------------------
These are *gap times* in a recurrent-event process observed only through
encounters at participating facilities. There is no enrollment and no exit: a
patient who moves away, or who uses a hospital outside the data, produces a gap
that looks long and is not. Patients with a single episode contribute no
interval at all, so any distribution fitted here is conditioned on having at
least two -- which is a selected subset, since around three quarters of sickle
cell patients have no crisis encounter in a given year. Treat the output as
observed gaps, not as the inter-crisis distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_MIN_GAP_DAYS",
    "EpisodeSpec",
    "collapse_episodes",
    "interevent_intervals",
    "patient_summary",
    "to_days",
]

# The convention in vaso-occlusion trials: events less than this far apart are
# one episode. Explicit rather than buried, because every downstream number
# moves with it.
DEFAULT_MIN_GAP_DAYS = 3.0


@dataclass(frozen=True)
class EpisodeSpec:
    """Column names in the encounter table, so callers need not rename theirs."""

    subject: str = "subject_id"
    start: str = "start"
    end: str = "end"


def to_days(values, origin=None) -> np.ndarray:
    """Normalise timestamps or day offsets to float days.

    Accepts datetime-like values (converted relative to ``origin``, or to the
    minimum if none is given) or numeric day offsets, which pass through. This
    is what lets one pipeline serve both MIMIC-IV timestamps and HCUP
    ``DaysToEvent`` integers.
    """
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)

    converted = pd.to_datetime(series)
    base = pd.to_datetime(origin) if origin is not None else converted.min()
    return (converted - base).dt.total_seconds().to_numpy(dtype=float) / 86400.0


def collapse_episodes(
    encounters: pd.DataFrame,
    *,
    min_gap_days: float = DEFAULT_MIN_GAP_DAYS,
    spec: EpisodeSpec | None = None,
    gap_from: str = "discharge",
) -> pd.DataFrame:
    """Collapse encounters less than ``min_gap_days`` apart into single episodes.

    Parameters
    ----------
    encounters
        One row per encounter, with subject, start and end columns as named by
        ``spec``. Times may be timestamps or numeric day offsets. The end
        column may be missing values -- an encounter with no recorded end (an
        ED stay, say) is treated as ending when it started, which is the
        conservative choice: it can only *lengthen* the measured gap and so can
        only under-collapse, never over-collapse.
    min_gap_days
        Encounters starting less than this many days after the running
        episode's end join that episode. Set to 0 to disable collapsing, which
        you should only do to demonstrate what it changes.
    gap_from
        ``"discharge"`` measures the gap from the previous encounter's end,
        which matches the trial convention of a new crisis beginning after the
        previous one resolves. ``"admission"`` measures start-to-start, which
        is the only option when the source has no discharge time.

    Returns
    -------
    One row per episode with ``subject_id``, ``episode_start``, ``episode_end``,
    ``n_encounters`` and ``min_gap_days``, sorted by subject and start. The
    threshold travels with the data so a downstream result can never be read
    without it.
    """
    spec = spec or EpisodeSpec()
    if min_gap_days < 0:
        raise ValueError("min_gap_days must be non-negative")
    if gap_from not in {"discharge", "admission"}:
        raise ValueError("gap_from must be 'discharge' or 'admission'")

    required = {spec.subject, spec.start}
    missing = required - set(encounters.columns)
    if missing:
        raise ValueError(f"encounters is missing columns: {sorted(missing)}")

    if encounters.empty:
        return pd.DataFrame(
            {
                "subject_id": pd.Series(dtype=encounters[spec.subject].dtype),
                "episode_start": pd.Series(dtype=float),
                "episode_end": pd.Series(dtype=float),
                "n_encounters": pd.Series(dtype=int),
                "min_gap_days": pd.Series(dtype=float),
            }
        )

    frame = encounters.copy()
    origin = None
    if not pd.api.types.is_numeric_dtype(frame[spec.start]):
        origin = pd.to_datetime(frame[spec.start]).min()
    frame["_start"] = to_days(frame[spec.start], origin=origin)

    if spec.end in frame.columns:
        ends = to_days(frame[spec.end], origin=origin)
        # An encounter with no recorded end is treated as instantaneous.
        frame["_end"] = np.where(np.isnan(ends), frame["_start"], ends)
        # Guard against end-before-start in source data rather than trusting it.
        frame["_end"] = np.maximum(frame["_end"], frame["_start"])
    else:
        frame["_end"] = frame["_start"]

    frame = frame.sort_values([spec.subject, "_start"], kind="mergesort")

    rows = []
    for subject, group in frame.groupby(spec.subject, sort=True):
        starts = group["_start"].to_numpy()
        ends = group["_end"].to_numpy()

        episode_start = starts[0]
        episode_end = ends[0]
        count = 1

        for start, end in zip(starts[1:], ends[1:], strict=True):
            reference = episode_end if gap_from == "discharge" else episode_start
            if start - reference < min_gap_days:
                # Same continuing episode: extend it and keep chaining.
                episode_end = max(episode_end, end)
                count += 1
            else:
                rows.append((subject, episode_start, episode_end, count))
                episode_start, episode_end, count = start, end, 1

        rows.append((subject, episode_start, episode_end, count))

    episodes = pd.DataFrame(
        rows, columns=["subject_id", "episode_start", "episode_end", "n_encounters"]
    )
    episodes["min_gap_days"] = float(min_gap_days)
    return episodes.sort_values(
        ["subject_id", "episode_start"], kind="mergesort"
    ).reset_index(drop=True)


def interevent_intervals(episodes: pd.DataFrame) -> pd.DataFrame:
    """Gap times between consecutive episodes, per patient.

    Returns one row per interval with ``subject_id``, ``interval_days``,
    ``order`` (1 for the first gap), and the ``episode_start`` it ends at.
    Patients with a single episode contribute nothing, which is the selection
    to keep in view: the resulting distribution is conditioned on having at
    least two episodes.

    Gaps are measured start-to-start. The alternative, end-to-start, subtracts
    a length of stay that is itself a severity measure, which would build a
    correlation between interval and severity into the data by construction --
    precisely the correlation the study is meant to test.
    """
    required = {"subject_id", "episode_start"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"episodes is missing columns: {sorted(missing)}")

    if episodes.empty:
        return pd.DataFrame(
            {
                "subject_id": pd.Series(dtype=episodes["subject_id"].dtype),
                "interval_days": pd.Series(dtype=float),
                "order": pd.Series(dtype=int),
                "episode_start": pd.Series(dtype=float),
            }
        )

    frame = episodes.sort_values(["subject_id", "episode_start"], kind="mergesort")
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        starts = group["episode_start"].to_numpy(dtype=float)
        for order, (previous, current) in enumerate(
            zip(starts[:-1], starts[1:], strict=True), start=1
        ):
            rows.append((subject, float(current - previous), order, float(current)))

    return pd.DataFrame(
        rows, columns=["subject_id", "interval_days", "order", "episode_start"]
    )


def patient_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per-patient episode count, observed span and a crude rate.

    ``span_days`` runs from first to last episode start, so it is zero for a
    patient with one episode and understates follow-up for everyone: the data
    say nothing about the time before a patient's first encounter or after
    their last. ``rate_per_year`` is ``(n_episodes - 1) / span`` and is
    therefore defined only for patients with at least two episodes and biased
    upward -- a patient is observed *because* they had encounters. It is a
    triage number, not an incidence estimate.
    """
    required = {"subject_id", "episode_start"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"episodes is missing columns: {sorted(missing)}")

    if episodes.empty:
        return pd.DataFrame(
            columns=["subject_id", "n_episodes", "span_days", "rate_per_year"]
        )

    grouped = episodes.groupby("subject_id")["episode_start"]
    summary = pd.DataFrame(
        {
            "n_episodes": grouped.size(),
            "first_start": grouped.min(),
            "last_start": grouped.max(),
        }
    ).reset_index()

    summary["span_days"] = summary["last_start"] - summary["first_start"]
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["rate_per_year"] = np.where(
            summary["span_days"] > 0,
            (summary["n_episodes"] - 1) * 365.25 / summary["span_days"],
            np.nan,
        )
    return summary[["subject_id", "n_episodes", "span_days", "rate_per_year"]]
