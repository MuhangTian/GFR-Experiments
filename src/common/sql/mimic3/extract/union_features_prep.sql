-- PURPOSE: to select appropriate population
DROP TABLE IF EXISTS union_features_prep; CREATE TABLE union_features_prep AS
WITH burn_flag AS( -- create flags for patients with small issues
	SELECT hadm_id	-- do at admission level
		-- take MAX since if patient have any of these, it's considered as YES
	  , MAX(CASE
		  WHEN SUBSTR(icd9_code,1,3) BETWEEN '940' AND '946' THEN 1			-- burns
		  ELSE 0
	  END) as burn_flag
	FROM diagnoses_icd
  	GROUP BY hadm_id
)
, icu_los_flag AS(
	SELECT icustay_id
	, CASE
		WHEN los < 1 THEN 1
		ELSE 0
	END AS icu_los_flag
	FROM icustays
)
, hosp_los_flag AS(
	SELECT hadm_id
	, CASE
		WHEN DATETIME_DIFF(dischtime, admittime, 'DAY') > 365 THEN 1
		ELSE 0
	END AS hosp_los_flag
	FROM admissions
)
, first_icu_stay AS(
	SELECT *
	, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY icu_intime ASC) as rn
	FROM union_features
)
, cohort AS(
	SELECT ie.*
	, bf.burn_flag
	, lf.icu_los_flag
	, hf.hosp_los_flag
	FROM first_icu_stay ie
	LEFT JOIN burn_flag bf
		ON ie.hadm_id = bf.hadm_id
	LEFT JOIN icu_los_flag lf
		ON ie.icustay_id = lf.icustay_id
	LEFT JOIN hosp_los_flag hf
		ON ie.hadm_id = hf.hadm_id
)
SELECT subject_id, hadm_id, icustay_id, hospital_expire_flag
	, preiculos
	, age
	, gcs_min
	, mechvent
	, urineoutput
	, heartrate_min
	, heartrate_max
	, meanbp_min
	, meanbp_max
	, resprate_min
	, resprate_max
	, tempc_min
	, tempc_max
	, sysbp_min
	, sysbp_max
	, bun_min
	, bun_max
	, wbc_min
	, wbc_max
	, potassium_min
	, potassium_max
	, sodium_min
	, sodium_max
	, bicarbonate_min
	, bicarbonate_max
	, bilirubin_min
	, bilirubin_max
	, hematocrit_min
	, hematocrit_max
	, creatinine_min
	, creatinine_max
	, albumin_min
	, albumin_max
	, glucose_max
	, glucose_min
	, aids
	, hem
	, mets
	, electivesurgery
	, pao2fio2_vent_min
	, admissiontype
	, pao2_max
	, pao2_min
	, paco2_max
	, paco2_min
	, ph_min
	, ph_max
	, aado2_min
	, aado2_max

FROM cohort

WHERE 
rn = 1									-- use first ICU stay
AND age >= 16 AND age <= 89				-- filter age
AND urineoutput >= 0					-- exclude negative values
AND burn_flag != 1						-- no burns
AND icu_los_flag != 1					-- at least 24 hours of stay
AND hosp_los_flag != 1					-- stay for at most 1 year

