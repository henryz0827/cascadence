"""The go/no-go count, and what the 3-day rule costs if you skip it.

Usage
-----
Against a CSV exported from ``src/voc/sql/02_crisis_encounters.sql``::

    python examples/voc_feasibility.py --encounters crisis_encounters.csv

Against the synthetic fixture, to see the pipeline run with no data access::

    python examples/voc_feasibility.py --demo

Order of operations, once credentialed
--------------------------------------
1. Run ``src/voc/sql/01_audit_codes.sql`` and reconcile anything it lists that
   ``voc.codes.classify`` calls ``unknown``. An unclassified code is data being
   dropped without anyone noticing.
2. **Write down the threshold before running this.** The defaults are at least
   150 patients with at least 3 episodes. A threshold picked after seeing the
   count is not a decision.
3. Run ``02_crisis_encounters.sql``, export, and point this script at it.
4. Believe the verdict. If MIMIC-IV fails, it is still the best source for
   severity proxies -- labs, administered opioid doses, transfusion, oxygen --
   which no claims database carries. The interval analysis then belongs on
   HCUP SID/SEDD, which follows a patient across facilities and care settings
   within a state over multiple years.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from voc.cohort import build_crisis_encounters, feasibility
from voc.episodes import collapse_episodes, interevent_intervals, patient_summary

SENSITIVITY_THRESHOLDS = (0.0, 1.0, 3.0, 7.0, 14.0)


def load_encounters(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"subject_id", "start"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    return frame


def demo_encounters() -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import voc_fixture

    tables = voc_fixture.build()
    return build_crisis_encounters(
        tables["diagnoses"],
        tables["admissions"],
        ed_diagnoses=tables["ed_diagnoses"],
        ed_stays=tables["ed_stays"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--encounters", type=Path, help="CSV from 02_crisis_encounters.sql"
    )
    source.add_argument(
        "--demo", action="store_true", help="run on the synthetic fixture"
    )
    parser.add_argument("--min-gap-days", type=float, default=3.0)
    parser.add_argument("--threshold-episodes", type=int, default=3)
    parser.add_argument("--threshold-patients", type=int, default=150)
    args = parser.parse_args()

    if args.demo:
        encounters = demo_encounters()
        print("SYNTHETIC FIXTURE -- these numbers mean nothing clinically.\n")
    else:
        encounters = load_encounters(args.encounters)

    print(f"crisis encounters:  {len(encounters)}")
    print(f"distinct patients:  {encounters['subject_id'].nunique()}")
    if "source" in encounters.columns:
        for label, count in encounters["source"].value_counts().items():
            print(f"  {label}: {count}")
    print()

    episodes = collapse_episodes(encounters, min_gap_days=args.min_gap_days)
    verdict = feasibility(
        episodes,
        threshold_episodes=args.threshold_episodes,
        threshold_patients=args.threshold_patients,
    )
    print(f"--- episodes at the {args.min_gap_days:g}-day rule ---")
    print(verdict.report())

    # What the rule is worth. Roughly 17% of crisis encounters are followed by
    # a revisit within 3 days, and those revisits are usually the same
    # under-treated crisis; counting them separately manufactures a spike of
    # near-zero intervals indistinguishable from real clustering.
    print("\n--- sensitivity to the collapsing rule ---")
    print(f"{'min gap':>9} {'episodes':>9} {'intervals':>10} {'median':>9} {'<7d':>7}")
    for threshold in SENSITIVITY_THRESHOLDS:
        collapsed = collapse_episodes(encounters, min_gap_days=threshold)
        intervals = interevent_intervals(collapsed)
        if intervals.empty:
            print(f"{threshold:>9.0f} {len(collapsed):>9} {0:>10} {'-':>9} {'-':>7}")
            continue
        days = intervals["interval_days"]
        print(
            f"{threshold:>9.0f} {len(collapsed):>9} {len(intervals):>10} "
            f"{days.median():>9.1f} {(days < 7).mean():>7.1%}"
        )
    print("  a large '<7d' share at min gap 0 that shrinks as the rule tightens is")
    print("  the artefact, not a finding: those are revisits for one crisis.")

    summary = patient_summary(episodes)
    repeat = summary[summary["n_episodes"] >= 2]
    if not repeat.empty:
        print(f"\npatients contributing >=1 interval: {len(repeat)}")
        print(f"median episodes among them:        {repeat['n_episodes'].median():.0f}")
        print(f"median observed span (days):       {repeat['span_days'].median():.0f}")
        print(
            "  span runs first encounter to last, so it understates follow-up for\n"
            "  everyone and the implied rates are biased upward: a patient is in\n"
            "  the data because they had encounters."
        )


if __name__ == "__main__":
    main()
