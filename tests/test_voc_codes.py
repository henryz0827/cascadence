"""ICD classification: trait stays out, and unknown family codes get reported."""

from __future__ import annotations

import pandas as pd
import pytest

import voc_fixture
from voc.codes import audit, classify, is_sickle_family, normalize


@pytest.mark.parametrize(
    ("code", "version", "expected"),
    [
        ("D5700", 10, "crisis"),
        ("D5701", 10, "crisis"),
        ("D57219", 10, "crisis"),
        ("D57419", 10, "crisis"),
        ("D57819", 10, "crisis"),
        ("28262", 9, "crisis"),
        ("28264", 9, "crisis"),
        ("28242", 9, "crisis"),
        ("D571", 10, "no_crisis"),
        ("D5720", 10, "no_crisis"),
        ("28260", 9, "no_crisis"),
        ("28261", 9, "no_crisis"),
        ("I10", 10, "other"),
        ("4019", 9, "other"),
    ],
)
def test_classification(code, version, expected):
    assert classify(code, version) == expected


@pytest.mark.parametrize(("code", "version"), [("D573", 10), ("2825", 9)])
def test_sickle_cell_trait_is_never_disease(code, version):
    """Trait is a carrier state with no vaso-occlusive crises.

    It sits in the same code family as the disease and is the classic
    contaminant of sickle cell cohorts, so it gets its own label and is
    excluded from both disease buckets.
    """
    assert classify(code, version) == "trait"


def test_unrecognised_family_codes_are_flagged_not_absorbed():
    """A family code no set covers must surface, not be guessed at.

    The D57 subcategories have been extended repeatedly, and MIMIC-IV spans
    many annual revisions. A code quietly binned as disease, or quietly
    dropped, is a silent data error; 'unknown' forces a human decision.
    """
    assert classify("D57999", 10) == "unknown"
    assert is_sickle_family("D57999", 10)


def test_codes_are_normalised():
    # MIMIC pads some codes, and dotted forms appear in other sources.
    assert normalize(" d5700 ") == "D5700"
    assert normalize("282.62") == "28262"
    assert classify(" D5700 ", 10) == "crisis"
    assert classify("282.62", 9) == "crisis"


def test_rejects_unknown_icd_version():
    with pytest.raises(ValueError, match="icd_version must be 9 or 10"):
        classify("D5700", 11)


def test_audit_reports_the_unclassified_code_in_the_dictionary():
    tables = voc_fixture.build()
    result = audit(tables["d_icd_diagnoses"])

    assert not result.clean
    unknown_codes = {code for code, _, _ in result.unknown}
    assert unknown_codes == {voc_fixture.UNKNOWN_FAMILY_ICD10}
    assert "UNCLASSIFIED" in result.report()

    assert {c for c, _, _ in result.counted_crisis} == {
        voc_fixture.CRISIS_ICD10,
        voc_fixture.CRISIS_ICD9,
    }
    assert {c for c, _, _ in result.counted_trait} == {voc_fixture.TRAIT_ICD10}


def test_audit_is_clean_when_every_family_code_is_covered():
    dictionary = pd.DataFrame(
        [
            {"icd_code": "D5700", "icd_version": 10, "long_title": "crisis"},
            {"icd_code": "D571", "icd_version": 10, "long_title": "no crisis"},
            {"icd_code": "D573", "icd_version": 10, "long_title": "trait"},
            {"icd_code": "I10", "icd_version": 10, "long_title": "unrelated"},
        ]
    )
    result = audit(dictionary)
    assert result.clean
    assert "no unclassified" in result.report()


def test_audit_requires_the_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        audit(pd.DataFrame({"icd_code": ["D5700"]}))
