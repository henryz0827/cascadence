"""voc -- turning clinical encounter records into vaso-occlusive event series.

A separate concern from :mod:`cascadence`, which is the flow-network cascade
library, and deliberately a sibling package rather than a submodule of it:
nothing here involves a network. The two meet only in the paper.

What it does
------------
:mod:`voc.codes`
    ICD code sets for sickle cell disease, with sickle cell *trait* excluded
    and an audit that reports family codes the sets do not cover instead of
    silently dropping them.
:mod:`voc.episodes`
    Collapsing encounters into episodes under the 3-day rule, and extracting
    gap times. The rule is not optional -- see the module docstring.
:mod:`voc.cohort`
    Assembling the crisis-encounter table and counting the repeated-event
    cohort against a pre-registered threshold.

Where the data comes from
-------------------------
Nothing here ships data, and ``data/`` is unconditionally git-ignored; see
``data/README.md``. The pipeline is written against MIMIC-IV's schema and
tested against synthetic fixtures of the same shape, so it can be developed and
checked before credentialed access exists.

The intended division of labour, which is why :mod:`voc.episodes` accepts both
timestamps and integer day offsets:

* **intervals and clustering** from HCUP SID/SEDD, which follows a patient
  across facilities *and* care settings within a state over multiple years;
* **severity proxies** from MIMIC-IV, which has the labs, administered opioid
  doses and transfusions that no claims source carries.

MIMIC-IV is not expected to carry the interval analysis on its own: the source
hospital is not a sickle cell centre, entry requires an ED or ICU touch,
under-18s are excluded, and day-hospital encounters -- where milder crises are
treated -- are absent entirely. :func:`voc.cohort.feasibility` exists to settle
that with a count rather than an assumption.
"""

from __future__ import annotations

from . import codes, cohort, episodes

__all__ = ["codes", "cohort", "episodes"]
