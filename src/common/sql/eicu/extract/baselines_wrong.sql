-- PURPOSE: extract traditional risk score baseline, evaluate on bugged version of OASIS and SAPS II implementation
DROP TABLE IF EXISTS baselines_wrong; CREATE TABLE baselines_wrong AS

WITH apache_iv AS(
    SELECT pa.uniquepid, apa.*
    FROM patient pa
    LEFT JOIN apachepatientresult apa
        ON apa.patientunitstayid = pa.patientunitstayid
    WHERE apa.apacheversion = 'IV'
)
, apache_iva AS(
    SELECT pa.uniquepid, apa.*
    FROM patient pa
    LEFT JOIN apachepatientresult apa
        ON apa.patientunitstayid = pa.patientunitstayid
    WHERE apa.apacheversion = 'IVa'
)
, base AS(
    SELECT pa.*
    , iv.predictedhospitalmortality::NUMERIC AS apache_iv_prob
    , CASE
        WHEN iv.actualhospitalmortality = 'EXPIRED' THEN 1
        WHEN iv.actualhospitalmortality = 'ALIVE' THEN 0
        ELSE NULL
      END AS actualhospitalmortality
    , iva.predictedhospitalmortality::NUMERIC AS apache_iva_prob
    , oa.oasis_prob::NUMERIC
    , sap.sapsii_prob::NUMERIC

    FROM patient pa

    LEFT JOIN apache_iv iv
        ON iv.patientunitstayid = pa.patientunitstayid
    LEFT JOIN apache_iva iva
        ON iva.patientunitstayid = pa.patientunitstayid
    LEFT JOIN oasis_wrong oa
        ON oa.patientunitstayid = pa.patientunitstayid
    LEFT JOIN sapsii_wrong sap
        ON sap.patientunitstayid = pa.patientunitstayid
)
, first_icu_stay AS(
	SELECT patientunitstayid
	, ROW_NUMBER() OVER (PARTITION BY uniquepid ORDER BY hospitaladmitoffset DESC) as rn
	FROM patient
)
SELECT base.uniquepid, base.patientunitstayid, base.actualhospitalmortality
, apache_iv_prob
, apache_iva_prob
, oasis_prob
, sapsii_prob

FROM base

LEFT JOIN first_icu_stay
    ON first_icu_stay.patientunitstayid = base.patientunitstayid

WHERE first_icu_stay.rn = 1             -- select first ICU stay
AND apache_iv_prob IS NOT NULL          -- select patients with APACHE IV score available (this filter the patient population)
AND apache_iv_prob != -1                
AND apache_iva_prob IS NOT NULL         -- similar to above
AND apache_iva_prob != -1