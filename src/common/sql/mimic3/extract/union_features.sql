-- AUTHOR: TONY and CHLOE
DROP TABLE IF EXISTS union_features; CREATE TABLE union_features AS
WITH surgflag AS
(
  SELECT ie.icustay_id
    , MAX(CASE
        WHEN LOWER(curr_service) LIKE '%surg%' THEN 1
        WHEN curr_service = 'ORTHO' THEN 1
    ELSE 0 END) AS surgical
  FROM icustays ie
  LEFT JOIN services se
    ON ie.hadm_id = se.hadm_id
    AND se.transfertime < DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  GROUP BY ie.icustay_id
)
, comorbidity AS(
  SELECT hadm_id	-- do at admission level
		-- take MAX since if patient have any of these, it's considered as YES
	  , MAX(CASE
		WHEN SUBSTR(icd9_code,1,3) BETWEEN '042' AND '044' THEN 1
		ELSE 0
		END) AS aids      /* HIV AND AIDS */
	
	  , MAX(CASE
		WHEN icd9_code BETWEEN '20000' AND '20238' THEN 1 -- lymphoma
		WHEN icd9_code BETWEEN '20240' AND '20248' THEN 1 -- leukemia
		WHEN icd9_code BETWEEN '20250' AND '20302' THEN 1 -- lymphoma
		WHEN icd9_code BETWEEN '20310' AND '20312' THEN 1 -- leukemia
		WHEN icd9_code BETWEEN '20302' AND '20382' THEN 1 -- lymphoma
		WHEN icd9_code BETWEEN '20400' AND '20522' THEN 1 -- chronic leukemia
		WHEN icd9_code BETWEEN '20580' AND '20702' THEN 1 -- other myeloid leukemia
		WHEN icd9_code BETWEEN '20720' AND '20892' THEN 1 -- other myeloid leukemia
		WHEN SUBSTR(icd9_code,1,4) = '2386' THEN 1 -- lymphoma
		WHEN SUBSTR(icd9_code,1,4) = '2733' THEN 1 -- lymphoma
		ELSE 0
		END) AS hem    /* Hematologic Cancer */
	
	  , MAX(CASE
		WHEN SUBSTR(icd9_code,1,4) BETWEEN '1960' AND '1991' THEN 1
		WHEN icd9_code BETWEEN '20970' AND '20975' THEN 1
		WHEN icd9_code = '20979' THEN 1
		WHEN icd9_code = '78951' THEN 1
		ELSE 0
		END) AS mets      /* Metastatic cancer */

	  FROM diagnoses_icd
  GROUP BY hadm_id
)
, cpap AS
(
  SELECT ie.icustay_id
    , MIN(DATETIME_SUB(charttime, INTERVAL '1' HOUR)) AS starttime
    , MAX(DATETIME_ADD(charttime, INTERVAL '4' HOUR)) AS endtime
    , MAX(CASE
          WHEN LOWER(ce.value) LIKE '%cpap%' THEN 1
          WHEN LOWER(ce.value) LIKE '%bipap mask%' THEN 1
        ELSE 0 END) AS cpap
  FROM icustays ie
  INNER JOIN chartevents ce
    ON ie.icustay_id = ce.icustay_id
    AND ce.charttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  WHERE itemid IN (467, 469, 226732)
  AND (LOWER(ce.value) LIKE '%cpap%' OR LOWER(ce.value) LIKE '%bipap mask%')
  -- exclude rows marked as error
  AND (ce.error IS NULL OR ce.error = 0)
  GROUP BY ie.icustay_id
)
, pafi1 AS
(
  -- join blood gas to ventilation durations to determine if patient was vent
  -- also join to cpap table for the same purpose
  SELECT bg.icustay_id, bg.charttime
  , pao2fio2
  , CASE WHEN vd.icustay_id IS NOT NULL THEN 1 ELSE 0 END AS vent
  , CASE WHEN cp.icustay_id IS NOT NULL THEN 1 ELSE 0 END AS cpap
  FROM blood_gas_first_day_arterial bg
  LEFT JOIN ventilation_durations vd
    ON bg.icustay_id = vd.icustay_id
    AND bg.charttime >= vd.starttime
    AND bg.charttime <= vd.endtime
  LEFT JOIN cpap cp
    ON bg.icustay_id = cp.icustay_id
    AND bg.charttime >= cp.starttime
    AND bg.charttime <= cp.endtime
)
, pafi2 AS
(
  -- get the minimum PaO2/FiO2 ratio *only for ventilated/cpap patients*, adapted from SAPS-II features
  SELECT icustay_id
  , MIN(pao2fio2) AS pao2fio2_vent_min
  FROM pafi1
  WHERE vent = 1 OR cpap = 1
  GROUP BY icustay_id
)
, surgflag_helper AS			-- to help with generating admission type, which is a SAPS-II feature
(
  SELECT adm.hadm_id
    , CASE WHEN LOWER(curr_service) LIKE '%surg%' THEN 1 ELSE 0 END AS surgical
    , ROW_NUMBER() OVER
    (
      PARTITION BY adm.hadm_id
      ORDER BY transfertime
    ) AS serviceOrder
  FROM admissions adm
  LEFT JOIN services se
    ON adm.hadm_id = se.hadm_id
)
, collections AS(		-- collect even more features
	SELECT icustay_id
	  , MAX(po2) AS pao2_max
	  , MIN(po2) AS pao2_min
	  , MAX(pco2) AS paco2_max
	  , MIN(pco2) AS paco2_min
	  , MAX(ph) AS ph_max
	  , MIN(ph) AS ph_min
	  , MAX(aado2) AS aado2_max
	  , MIN(aado2) AS aado2_min
	FROM blood_gas_first_day
	GROUP BY icustay_id
)
SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, ie.intime AS icu_intime, adm.hospital_expire_flag
	, ROUND(DATETIME_DIFF(ie.intime, adm.admittime, 'MINUTE')::NUMERIC) AS preiculos
	, ROUND(DATETIME_DIFF(ie.intime, pat.dob, 'YEAR')::NUMERIC) AS age
	, gcs.mingcs AS gcs_min
	, vent.vent AS mechvent
	, uo.urineoutput
	-- from vitals_first_day
	, vitals.heartrate_min
	, vitals.heartrate_max
	, vitals.meanbp_min
	, vitals.meanbp_max
	, vitals.resprate_min
	, vitals.resprate_max
	, vitals.tempc_min
	, vitals.tempc_max
	, vitals.sysbp_min
	, vitals.sysbp_max
	-- from labs_first_day
	, labs.bun_min
	, labs.bun_max
	, labs.wbc_min
	, labs.wbc_max
	, labs.potassium_min
	, labs.potassium_max
	, labs.sodium_min
	, labs.sodium_max
	, labs.bicarbonate_min
	, labs.bicarbonate_max
	, labs.bilirubin_min
	, labs.bilirubin_max
	, labs.hematocrit_min
	, labs.hematocrit_max
	, labs.creatinine_min
	, labs.creatinine_max
	, labs.albumin_min
	, labs.albumin_max
    , CASE		-- use values from both vitals_first_day AND labs_first_day for glucose
        WHEN labs.glucose_max IS NULL AND vitals.glucose_max IS NULL
          THEN NULL
        WHEN labs.glucose_max IS NULL OR vitals.glucose_max > labs.glucose_max
		  THEN vitals.glucose_max
        WHEN vitals.glucose_max IS NULL OR labs.glucose_max > vitals.glucose_max
          THEN labs.glucose_max
        ELSE labs.glucose_max -- if equal, just pick labs
      END AS glucose_max
    , CASE
        WHEN labs.glucose_min IS NULL AND vitals.glucose_min IS NULL
          THEN NULL
        WHEN labs.glucose_min IS NULL OR vitals.glucose_min < labs.glucose_min
	      THEN vitals.glucose_min
        WHEN vitals.glucose_min IS NULL OR labs.glucose_min < vitals.glucose_min
          THEN labs.glucose_min
        ELSE labs.glucose_min -- if equal, just pick labs
      END AS glucose_min
	-- comorbidity
	, comorb.aids
	, comorb.hem
	, comorb.mets
	-- elective surgery
	, CASE
		WHEN adm.ADMISSION_TYPE = 'ELECTIVE' AND sf.surgical = 1
			THEN 1
		WHEN adm.ADMISSION_TYPE IS NULL OR sf.surgical IS NULL
			THEN NULL
		ELSE 0
      END AS electivesurgery
	
	, pf.pao2fio2_vent_min
	
	, CASE	-- to obtain admission_type feature, note that we use a different surgical flag without time constraint here
		WHEN adm.ADMISSION_TYPE = 'ELECTIVE' AND sf_help.surgical = 1
            THEN 'ScheduledSurgical'
		WHEN adm.ADMISSION_TYPE != 'ELECTIVE' AND sf_help.surgical = 1
            THEN 'UnscheduledSurgical'
		ELSE 'Medical'
	  END AS admissiontype
	
	, collections.pao2_max
	, collections.pao2_min
	, collections.paco2_max
	, collections.paco2_min
	, collections.ph_max
	, collections.ph_min
	, collections.aado2_max
	, collections.aado2_min
	
FROM icustays ie
-- base tables, provided by dataset
INNER JOIN admissions adm
	ON ie.hadm_id = adm.hadm_id
INNER JOIN patients pat
	ON ie.subject_id = pat.subject_id
-- first day tables, from official MIMIC-III code
LEFT JOIN ventilation_first_day vent
	ON ie.icustay_id = vent.icustay_id
LEFT JOIN vitals_first_day vitals
	ON ie.icustay_id = vitals.icustay_id
LEFT JOIN labs_first_day labs
	ON ie.icustay_id = labs.icustay_id
LEFT JOIN urine_output_first_day uo
	ON ie.icustay_id = uo.icustay_id
LEFT JOIN gcs_first_day gcs
	ON ie.icustay_id = gcs.icustay_id
-- join views
LEFT JOIN comorbidity comorb
	ON ie.hadm_id = comorb.hadm_id
LEFT JOIN surgflag sf
	ON ie.icustay_id = sf.icustay_id
LEFT JOIN pafi2 pf
	ON ie.icustay_id = pf.icustay_id
LEFT JOIN surgflag_helper sf_help
	ON adm.hadm_id = sf_help.hadm_id AND sf_help.serviceOrder = 1
LEFT JOIN collections
	ON ie.icustay_id = collections.icustay_id