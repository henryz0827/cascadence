"""A synthetic MIMIC-IV-shaped dataset with known ground truth.

The pipeline has to be correct before credentialed access exists, and it has to
stay correct afterwards without the tests depending on data that cannot be
committed. So the fixture builds tables with MIMIC-IV's column names and
shapes, from an explicit episode structure the tests can assert against
exactly.

Every trap the real extraction has to survive is planted here on purpose:
sickle cell *trait* codes that must not be counted, disease-without-crisis
codes that must not be counted by default, unrelated diagnoses, short revisits
that must collapse, an ED visit that became an admission and must not be
counted twice, and an ED-only visit that must be counted once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ORIGIN = pd.Timestamp("2150-01-01")


@dataclass(frozen=True)
class PatientPlan:
    """One synthetic patient, described by the episodes they should produce.

    ``episodes`` is a list of episodes; each episode is a list of
    ``(start_day, length_days)`` encounters relative to the study origin. For
    the ground truth to hold, encounters within an episode must start less than
    3 days after the previous one *ends*, and consecutive episodes must be
    separated by at least 3 days.
    """

    subject_id: int
    episodes: list[list[tuple[float, float]]]
    ed_only: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def episode_starts(self) -> list[float]:
        return [min(start for start, _ in episode) for episode in self.episodes]

    @property
    def intervals(self) -> list[float]:
        starts = self.episode_starts
        return [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)]


# A deliberately awkward cohort:
#  1 -- three clean episodes, one of them a two-encounter episode (collapses)
#  2 -- a single episode built from a three-encounter chain (transitive collapse)
#  3 -- one episode only: contributes no interval
#  4 -- two episodes, the second reached only through an ED-only visit
#  5 -- no crisis at all; trait and no-crisis codes only. Must not appear.
DEFAULT_PLANS = [
    PatientPlan(1, [[(0.0, 4.0)], [(30.0, 3.0), (34.0, 2.0)], [(90.0, 5.0)]]),
    PatientPlan(2, [[(10.0, 2.0), (13.0, 2.0), (16.0, 1.0)]]),
    PatientPlan(3, [[(5.0, 3.0)]]),
    PatientPlan(4, [[(20.0, 2.0)]], ed_only=[(200.0, 0.3)]),
    PatientPlan(5, []),
]

# Codes the extraction must treat differently. Kept as literals so a change to
# the code sets in voc.codes shows up as a test failure rather than silently
# altering what the fixture means.
CRISIS_ICD10 = "D5700"
CRISIS_ICD9 = "28262"
NO_CRISIS_ICD10 = "D571"
TRAIT_ICD10 = "D573"
UNRELATED_ICD10 = "I10"
UNKNOWN_FAMILY_ICD10 = "D57999"  # a family code no explicit set covers


def _stamp(day: float) -> pd.Timestamp:
    return ORIGIN + pd.Timedelta(days=float(day))


def build(plans: list[PatientPlan] | None = None) -> dict[str, pd.DataFrame]:
    """Build the synthetic tables. Returns a dict of MIMIC-IV-shaped frames."""
    plans = list(DEFAULT_PLANS if plans is None else plans)

    admissions, diagnoses, ed_stays, ed_diagnoses = [], [], [], []
    hadm = 1000
    stay = 5000

    for plan in plans:
        for episode in plan.episodes:
            for index, (start, length) in enumerate(episode):
                hadm += 1
                admissions.append(
                    {
                        "subject_id": plan.subject_id,
                        "hadm_id": hadm,
                        "admittime": _stamp(start),
                        "dischtime": _stamp(start + length),
                    }
                )
                # Alternate the coding system, as a 2008-2022 span really does.
                code = CRISIS_ICD9 if index % 2 else CRISIS_ICD10
                version = 9 if index % 2 else 10
                diagnoses.append(
                    {
                        "subject_id": plan.subject_id,
                        "hadm_id": hadm,
                        "icd_code": code,
                        "icd_version": version,
                    }
                )
                # Comorbidity that must not affect anything.
                diagnoses.append(
                    {
                        "subject_id": plan.subject_id,
                        "hadm_id": hadm,
                        "icd_code": UNRELATED_ICD10,
                        "icd_version": 10,
                    }
                )
                # This ED stay became this admission: it must not double-count.
                stay += 1
                ed_stays.append(
                    {
                        "subject_id": plan.subject_id,
                        "stay_id": stay,
                        "hadm_id": hadm,
                        "intime": _stamp(start - 0.2),
                        "outtime": _stamp(start),
                    }
                )
                ed_diagnoses.append(
                    {
                        "subject_id": plan.subject_id,
                        "stay_id": stay,
                        "icd_code": CRISIS_ICD10,
                        "icd_version": 10,
                    }
                )

        for start, length in plan.ed_only:
            stay += 1
            ed_stays.append(
                {
                    "subject_id": plan.subject_id,
                    "stay_id": stay,
                    "hadm_id": None,
                    "intime": _stamp(start),
                    "outtime": _stamp(start + length),
                }
            )
            ed_diagnoses.append(
                {
                    "subject_id": plan.subject_id,
                    "stay_id": stay,
                    "icd_code": CRISIS_ICD10,
                    "icd_version": 10,
                }
            )

        if not plan.episodes and not plan.ed_only:
            # A patient who has sickle cell coded, but never a crisis, plus a
            # trait code. Neither may put them in the cohort.
            hadm += 1
            admissions.append(
                {
                    "subject_id": plan.subject_id,
                    "hadm_id": hadm,
                    "admittime": _stamp(50.0),
                    "dischtime": _stamp(52.0),
                }
            )
            for code in (NO_CRISIS_ICD10, TRAIT_ICD10):
                diagnoses.append(
                    {
                        "subject_id": plan.subject_id,
                        "hadm_id": hadm,
                        "icd_code": code,
                        "icd_version": 10,
                    }
                )

    dictionary = pd.DataFrame(
        [
            {
                "icd_code": CRISIS_ICD10,
                "icd_version": 10,
                "long_title": "Hb-SS disease with crisis, unspecified",
            },
            {
                "icd_code": CRISIS_ICD9,
                "icd_version": 9,
                "long_title": "Sickle-cell disease with crisis",
            },
            {
                "icd_code": NO_CRISIS_ICD10,
                "icd_version": 10,
                "long_title": "Sickle-cell disease without crisis",
            },
            {
                "icd_code": TRAIT_ICD10,
                "icd_version": 10,
                "long_title": "Sickle-cell trait",
            },
            {
                "icd_code": UNKNOWN_FAMILY_ICD10,
                "icd_version": 10,
                "long_title": "Invented sickle-cell subcategory",
            },
            {
                "icd_code": UNRELATED_ICD10,
                "icd_version": 10,
                "long_title": "Essential (primary) hypertension",
            },
        ]
    )

    return {
        "admissions": pd.DataFrame(admissions),
        "diagnoses": pd.DataFrame(diagnoses),
        "ed_stays": pd.DataFrame(ed_stays),
        "ed_diagnoses": pd.DataFrame(ed_diagnoses),
        "d_icd_diagnoses": dictionary,
        "plans": plans,
    }
