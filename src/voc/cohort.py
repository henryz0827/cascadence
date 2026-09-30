"""Building the crisis-encounter table, and the go/no-go feasibility count.

Run :func:`feasibility` before writing any analysis. The number that decides
this project is **how many distinct patients have at least three crisis
episodes**, and it is unpublished for MIMIC-IV, so it has to be counted.

Two reasons to expect it to be small, both structural rather than fixable:

* The source hospital is not a sickle cell centre. Adult sickle cell care in
  Boston concentrates at Boston Medical Center and Brigham and Women's, and the
  Massachusetts sickle cell population is only a few thousand people.
* MIMIC-IV admits patients only via an ED or ICU touch, excludes anyone under
  18 at first visit, and has no day-hospital encounter type -- which is exactly
  where milder crises are treated. The per-patient event series is therefore
  non-randomly incomplete, on top of being small.

Decide the threshold before looking. A number chosen after seeing the count is
not a decision, it is a rationalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .codes import classify

__all__ = ["Feasibility", "build_crisis_encounters", "feasibility"]


def build_crisis_encounters(
    diagnoses: pd.DataFrame,
    admissions: pd.DataFrame,
    *,
    ed_diagnoses: pd.DataFrame | None = None,
    ed_stays: pd.DataFrame | None = None,
    include_no_crisis: bool = False,
) -> pd.DataFrame:
    """Assemble one row per sickle-cell-crisis encounter, inpatient and ED.

    Parameters
    ----------
    diagnoses
        ``subject_id``, ``hadm_id``, ``icd_code``, ``icd_version``
        (MIMIC-IV ``mimiciv_hosp.diagnoses_icd``).
    admissions
        ``subject_id``, ``hadm_id``, ``admittime``, ``dischtime``
        (MIMIC-IV ``mimiciv_hosp.admissions``).
    ed_diagnoses, ed_stays
        The MIMIC-IV-ED counterparts, optional. ``ed_stays`` needs
        ``subject_id``, ``stay_id``, ``intime``, ``outtime`` and ``hadm_id``.
    include_no_crisis
        Also keep encounters coded as sickle cell disease *without* crisis.
        Off by default: the two have materially different readmission profiles,
        so pooling them measures something other than crisis recurrence.

    Returns
    -------
    ``subject_id``, ``encounter_id``, ``source`` (``inpatient``/``ed``),
    ``start``, ``end``.

    An ED visit that resulted in admission carries the admission's ``hadm_id``;
    those are dropped so the same clinical encounter is not counted twice, once
    as an ED stay and again as an inpatient stay.
    """
    wanted = {"crisis"} | ({"no_crisis"} if include_no_crisis else set())

    def _matching_ids(frame: pd.DataFrame, id_column: str) -> set:
        labels = [
            classify(code, version)
            for code, version in zip(frame["icd_code"], frame["icd_version"], strict=True)
        ]
        keep = pd.Series(labels, index=frame.index).isin(wanted)
        return set(frame.loc[keep, id_column].unique())

    inpatient_ids = _matching_ids(diagnoses, "hadm_id")
    inpatient = admissions[admissions["hadm_id"].isin(inpatient_ids)].copy()
    inpatient = inpatient.rename(
        columns={"hadm_id": "encounter_id", "admittime": "start", "dischtime": "end"}
    )
    inpatient["source"] = "inpatient"
    frames = [inpatient[["subject_id", "encounter_id", "source", "start", "end"]]]

    if ed_diagnoses is not None and ed_stays is not None:
        ed_ids = _matching_ids(ed_diagnoses, "stay_id")
        emergency = ed_stays[ed_stays["stay_id"].isin(ed_ids)].copy()
        if "hadm_id" in emergency.columns:
            # An ED visit that became an admission is already counted above.
            admitted = emergency["hadm_id"].isin(inpatient_ids)
            emergency = emergency[~admitted]
        emergency = emergency.rename(
            columns={"stay_id": "encounter_id", "intime": "start", "outtime": "end"}
        )
        emergency["source"] = "ed"
        frames.append(emergency[["subject_id", "encounter_id", "source", "start", "end"]])

    encounters = pd.concat(frames, ignore_index=True)
    return encounters.sort_values(["subject_id", "start"], kind="mergesort").reset_index(
        drop=True
    )


@dataclass(frozen=True)
class Feasibility:
    """The counts that decide whether the dataset can carry the study."""

    n_patients: int
    n_episodes: int
    episodes_per_patient: dict[int, int]
    n_with_at_least: dict[int, int]
    threshold_episodes: int
    threshold_patients: int

    @property
    def passes(self) -> bool:
        return (
            self.n_with_at_least.get(self.threshold_episodes, 0)
            >= self.threshold_patients
        )

    def report(self) -> str:
        lines = [
            f"patients with >=1 crisis episode:  {self.n_patients}",
            f"crisis episodes in total:          {self.n_episodes}",
            "",
            "episodes per patient:",
        ]
        for count in sorted(self.episodes_per_patient):
            label = f"{count}" if count < 10 else "10+"
            lines.append(
                f"  {label:>4} episodes: {self.episodes_per_patient[count]:>6} patients"
            )
        lines.append("")
        for threshold in sorted(self.n_with_at_least):
            count = self.n_with_at_least[threshold]
            lines.append(f"  patients with >={threshold} episodes: {count:>6}")
        lines.append("")
        verdict = "PASSES" if self.passes else "FAILS"
        lines.append(
            f"pre-registered threshold: >={self.threshold_patients} patients with "
            f">={self.threshold_episodes} episodes -> {verdict}"
        )
        if not self.passes:
            lines.append(
                "  this dataset cannot support per-patient interval or "
                "frequency-severity analysis; use it for severity-proxy work only"
            )
        return "\n".join(lines)


def feasibility(
    episodes: pd.DataFrame,
    *,
    threshold_episodes: int = 3,
    threshold_patients: int = 150,
    report_at: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> Feasibility:
    """Count the repeated-event cohort against a pre-registered threshold.

    The defaults -- at least 150 patients with at least 3 episodes each -- are a
    floor for fitting a recurrent-event model with a frequency-severity term,
    not a guarantee of power. Set them deliberately and write them down before
    running this, then treat the verdict as binding.
    """
    if episodes.empty:
        return Feasibility(
            n_patients=0,
            n_episodes=0,
            episodes_per_patient={},
            n_with_at_least=dict.fromkeys(report_at, 0),
            threshold_episodes=threshold_episodes,
            threshold_patients=threshold_patients,
        )

    per_patient = episodes.groupby("subject_id").size()
    # Bucket the long tail so the histogram stays readable on real data.
    bucketed = np.minimum(per_patient.to_numpy(), 10)
    values, counts = np.unique(bucketed, return_counts=True)

    return Feasibility(
        n_patients=int(per_patient.size),
        n_episodes=int(per_patient.sum()),
        episodes_per_patient={
            int(v): int(c) for v, c in zip(values, counts, strict=True)
        },
        n_with_at_least={int(t): int((per_patient >= t).sum()) for t in report_at},
        threshold_episodes=threshold_episodes,
        threshold_patients=threshold_patients,
    )
