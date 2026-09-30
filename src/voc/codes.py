"""ICD code sets for sickle cell disease, and an audit that refuses to guess.

Getting these wrong is the single cheapest way to ruin the study, in two
directions that fail differently:

**Sickle cell trait is not sickle cell disease.** ICD-9 282.5 and ICD-10 D57.3
denote heterozygous carriers, who do not have vaso-occlusive crises. They sit
inside the same code family as the disease and are the classic contaminant.
They are excluded here, explicitly, and a test pins that.

**"With crisis" is a coded distinction, not an inference.** ICD separates
sickle cell disease *with* crisis from *without*, and HCUP reports the two have
materially different readmission rates. An analysis of crisis recurrence that
pools them is measuring something else.

Why this module audits instead of trusting its own list
-------------------------------------------------------
The D57 subcategories have been extended repeatedly (acute chest syndrome,
splenic sequestration, cerebral vascular involvement and the beta-thalassemia
splits all arrived at different revisions), and MIMIC-IV spans 2008-2022 --
ICD-9 and ICD-10 both, across many annual revisions. Any hardcoded list written
from memory will be missing codes, and a missing code is silently dropped data.

So :func:`classify` works from explicit sets, and :func:`audit` takes the
dataset's own code dictionary and reports every sickle-cell-family code that
the explicit sets do not cover. Run the audit **before** trusting a cohort
count; unclassified codes are for a human to resolve, not for this module to
assume.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CRISIS_ICD10",
    "CRISIS_ICD9",
    "NO_CRISIS_ICD10",
    "NO_CRISIS_ICD9",
    "TRAIT_ICD10",
    "TRAIT_ICD9",
    "CodeAudit",
    "audit",
    "classify",
    "is_sickle_family",
    "normalize",
]

# ---------------------------------------------------------------------------
# ICD-9 (MIMIC stores codes without the decimal point)
# ---------------------------------------------------------------------------

# 282.42 sickle-cell thalassemia with crisis; 282.62/.64/.69 Hb-SS, Hb-C and
# other sickle-cell disease with crisis.
CRISIS_ICD9 = frozenset({"28242", "28262", "28264", "28269"})

# 282.41 thalassemia without crisis; 282.60 unspecified; 282.61/.63/.68 without
# crisis. 282.60 is "unspecified" -- it is disease, but the crisis status is not
# coded, so it lands here rather than being counted as a crisis.
NO_CRISIS_ICD9 = frozenset({"28241", "28260", "28261", "28263", "28268"})

# 282.5 sickle-cell trait -- carriers, excluded.
TRAIT_ICD9 = frozenset({"2825"})

# ---------------------------------------------------------------------------
# ICD-10
# ---------------------------------------------------------------------------

CRISIS_ICD10 = frozenset(
    {
        # D57.0- Hb-SS disease with crisis
        "D5700",
        "D5701",
        "D5702",
        "D5703",
        "D5709",
        # D57.21- sickle-cell/Hb-C disease with crisis
        "D57211",
        "D57212",
        "D57213",
        "D57218",
        "D57219",
        # D57.41- sickle-cell thalassemia with crisis (and the beta-zero /
        # beta-plus splits introduced later)
        "D57411",
        "D57412",
        "D57413",
        "D57418",
        "D57419",
        "D57431",
        "D57432",
        "D57433",
        "D57438",
        "D57439",
        "D57451",
        "D57452",
        "D57453",
        "D57458",
        "D57459",
        # D57.81- other sickle-cell disorders with crisis
        "D57811",
        "D57812",
        "D57813",
        "D57818",
        "D57819",
    }
)

NO_CRISIS_ICD10 = frozenset(
    {
        "D571",  # sickle-cell disease without crisis
        "D5720",  # sickle-cell/Hb-C without crisis
        "D5740",  # sickle-cell thalassemia without crisis
        "D5742",  # beta-zero thalassemia without crisis
        "D5744",  # beta-plus thalassemia without crisis
        "D5780",  # other sickle-cell disorders without crisis
    }
)

# D57.3 sickle-cell trait -- carriers, excluded.
TRAIT_ICD10 = frozenset({"D573"})

_ICD9_FAMILY_PREFIXES = ("2824", "2825", "2826")
_ICD10_FAMILY_PREFIXES = ("D57",)


def normalize(code: str) -> str:
    """Strip whitespace and upper-case. MIMIC pads some codes; comparisons fail on it."""
    return str(code).strip().upper().replace(".", "")


def is_sickle_family(code: str, icd_version: int | str) -> bool:
    """Whether a code belongs to the sickle-cell family at all, trait included.

    Deliberately broad: the audit needs to see every family member, including
    ones no explicit set covers, so they can be reported rather than dropped.
    """
    value = normalize(code)
    version = int(icd_version)
    prefixes = _ICD9_FAMILY_PREFIXES if version == 9 else _ICD10_FAMILY_PREFIXES
    return value.startswith(prefixes)


def classify(code: str, icd_version: int | str) -> str:
    """Classify a code: ``crisis``, ``no_crisis``, ``trait``, ``unknown`` or ``other``.

    ``unknown`` means the code is in the sickle-cell family but not in any
    explicit set -- new or rare subcategories. These are never silently treated
    as disease or as crisis: :func:`audit` surfaces them for a decision.
    ``other`` means the code is outside the family entirely.
    """
    value = normalize(code)
    version = int(icd_version)

    if version == 9:
        crisis, no_crisis, trait = CRISIS_ICD9, NO_CRISIS_ICD9, TRAIT_ICD9
    elif version == 10:
        crisis, no_crisis, trait = CRISIS_ICD10, NO_CRISIS_ICD10, TRAIT_ICD10
    else:
        raise ValueError(f"icd_version must be 9 or 10, got {icd_version!r}")

    if value in crisis:
        return "crisis"
    if value in no_crisis:
        return "no_crisis"
    if value in trait:
        return "trait"
    if is_sickle_family(value, version):
        return "unknown"
    return "other"


@dataclass(frozen=True)
class CodeAudit:
    """What the dataset's own dictionary contains that the explicit sets do not."""

    unknown: tuple[tuple[str, int, str], ...]
    counted_crisis: tuple[tuple[str, int, str], ...]
    counted_no_crisis: tuple[tuple[str, int, str], ...]
    counted_trait: tuple[tuple[str, int, str], ...]

    @property
    def clean(self) -> bool:
        """Whether every sickle-cell-family code in the dictionary is classified."""
        return not self.unknown

    def report(self) -> str:
        lines = [
            f"crisis codes:     {len(self.counted_crisis)}",
            f"no-crisis codes:  {len(self.counted_no_crisis)}",
            f"trait (excluded): {len(self.counted_trait)}",
        ]
        if self.unknown:
            lines.append("")
            lines.append(
                f"UNCLASSIFIED ({len(self.unknown)}) -- resolve these before "
                "trusting any count:"
            )
            lines.extend(
                f"  ICD-{version} {code}  {title}"
                for code, version, title in self.unknown
            )
        else:
            lines.append("")
            lines.append("no unclassified sickle-cell-family codes")
        return "\n".join(lines)


def audit(dictionary) -> CodeAudit:
    """Check the explicit code sets against the dataset's own ICD dictionary.

    Parameters
    ----------
    dictionary
        A dataframe of the dataset's diagnosis dictionary, with columns
        ``icd_code``, ``icd_version`` and ``long_title`` -- in MIMIC-IV that is
        ``mimiciv_hosp.d_icd_diagnoses``.

    Call this first, every time, on a new data release. A code the sets do not
    cover is data being dropped without anyone noticing, and the D57
    subcategories have grown repeatedly over the years MIMIC-IV spans.
    """
    required = {"icd_code", "icd_version", "long_title"}
    missing = required - set(dictionary.columns)
    if missing:
        raise ValueError(f"dictionary is missing columns: {sorted(missing)}")

    buckets: dict[str, list[tuple[str, int, str]]] = {
        "unknown": [],
        "crisis": [],
        "no_crisis": [],
        "trait": [],
    }
    for code, version, title in zip(
        dictionary["icd_code"],
        dictionary["icd_version"],
        dictionary["long_title"],
        strict=True,
    ):
        label = classify(code, version)
        if label in buckets:
            buckets[label].append((normalize(code), int(version), str(title)))

    return CodeAudit(
        unknown=tuple(sorted(buckets["unknown"])),
        counted_crisis=tuple(sorted(buckets["crisis"])),
        counted_no_crisis=tuple(sorted(buckets["no_crisis"])),
        counted_trait=tuple(sorted(buckets["trait"])),
    )
