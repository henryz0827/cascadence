-- Extract one row per sickle-cell-CRISIS encounter, inpatient and ED.
--
-- Deliberately stops at encounters. Collapsing encounters into episodes under
-- the 3-day rule happens in voc.episodes, where it is tested -- burying that
-- rule in SQL would put the single most consequential analytic choice in
-- untested code, and it changes every downstream number.
--
-- Output matches voc.cohort.build_crisis_encounters exactly:
--   subject_id, encounter_id, source, start, end
-- so the same Python pipeline runs on a CSV export of this query or on frames
-- read straight from the database.
--
-- Codes: sickle cell disease WITH crisis only. Sickle cell trait (ICD-9 282.5,
-- ICD-10 D57.3) is a carrier state with no vaso-occlusive crises and is
-- excluded; disease WITHOUT crisis is excluded too, because the two have
-- materially different readmission profiles and pooling them measures
-- something other than crisis recurrence.
--
-- Keep this list in step with voc/codes.py, and run 01_audit_codes.sql first
-- to find anything neither covers.
--
-- BigQuery: prefix each table with `physionet-data.`

WITH crisis_hadm AS (
    SELECT DISTINCT hadm_id
    FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 9 AND TRIM(icd_code) IN (
               '28242', '28262', '28264', '28269'))
       OR (icd_version = 10 AND UPPER(TRIM(icd_code)) IN (
               'D5700', 'D5701', 'D5702', 'D5703', 'D5709',
               'D57211', 'D57212', 'D57213', 'D57218', 'D57219',
               'D57411', 'D57412', 'D57413', 'D57418', 'D57419',
               'D57431', 'D57432', 'D57433', 'D57438', 'D57439',
               'D57451', 'D57452', 'D57453', 'D57458', 'D57459',
               'D57811', 'D57812', 'D57813', 'D57818', 'D57819'))
),

crisis_ed_stay AS (
    SELECT DISTINCT stay_id
    FROM mimiciv_ed.diagnosis
    WHERE (icd_version = 9 AND TRIM(icd_code) IN (
               '28242', '28262', '28264', '28269'))
       OR (icd_version = 10 AND UPPER(TRIM(icd_code)) IN (
               'D5700', 'D5701', 'D5702', 'D5703', 'D5709',
               'D57211', 'D57212', 'D57213', 'D57218', 'D57219',
               'D57411', 'D57412', 'D57413', 'D57418', 'D57419',
               'D57431', 'D57432', 'D57433', 'D57438', 'D57439',
               'D57451', 'D57452', 'D57453', 'D57458', 'D57459',
               'D57811', 'D57812', 'D57813', 'D57818', 'D57819'))
),

inpatient AS (
    SELECT
        a.subject_id,
        CAST(a.hadm_id AS STRING) AS encounter_id,
        'inpatient'               AS source,
        a.admittime               AS start,
        a.dischtime               AS end
    FROM mimiciv_hosp.admissions AS a
    INNER JOIN crisis_hadm AS c ON c.hadm_id = a.hadm_id
),

-- An ED visit that became an admission is the same clinical encounter as the
-- inpatient row above. Counting both doubles every such encounter and halves
-- the apparent interval, so admitted ED stays are dropped here.
emergency AS (
    SELECT
        e.subject_id,
        CAST(e.stay_id AS STRING) AS encounter_id,
        'ed'                      AS source,
        e.intime                  AS start,
        e.outtime                 AS end
    FROM mimiciv_ed.edstays AS e
    INNER JOIN crisis_ed_stay AS c ON c.stay_id = e.stay_id
    WHERE e.hadm_id IS NULL
       OR e.hadm_id NOT IN (SELECT hadm_id FROM crisis_hadm)
)

SELECT * FROM inpatient
UNION ALL
SELECT * FROM emergency
ORDER BY subject_id, start;
