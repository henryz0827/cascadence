-- Run this FIRST, before any cohort query.
--
-- Lists every sickle-cell-family code that actually occurs in the data, with
-- how many admissions and distinct patients carry it. Compare the output
-- against the explicit sets in voc/codes.py: any code here that classify()
-- returns 'unknown' for is data that would otherwise be silently dropped.
--
-- This matters because the D57 subcategories have been extended repeatedly
-- (acute chest syndrome, splenic sequestration, cerebral vascular involvement
-- and the beta-thalassemia splits each arrived at different revisions), and
-- MIMIC-IV spans 2008-2022 under both ICD-9 and ICD-10.
--
-- BigQuery: prefix each table with `physionet-data.`
-- Postgres: the schema-qualified names below work as written.

SELECT
    d.icd_version,
    d.icd_code,
    dict.long_title,
    COUNT(*)                        AS n_diagnosis_rows,
    COUNT(DISTINCT d.hadm_id)       AS n_admissions,
    COUNT(DISTINCT d.subject_id)    AS n_patients
FROM mimiciv_hosp.diagnoses_icd AS d
LEFT JOIN mimiciv_hosp.d_icd_diagnoses AS dict
       ON dict.icd_code = d.icd_code
      AND dict.icd_version = d.icd_version
WHERE (d.icd_version = 10 AND UPPER(TRIM(d.icd_code)) LIKE 'D57%')
   OR (d.icd_version = 9  AND TRIM(d.icd_code) LIKE '2824%')
   OR (d.icd_version = 9  AND TRIM(d.icd_code) LIKE '2825%')
   OR (d.icd_version = 9  AND TRIM(d.icd_code) LIKE '2826%')
GROUP BY d.icd_version, d.icd_code, dict.long_title
ORDER BY n_patients DESC, d.icd_version, d.icd_code;
