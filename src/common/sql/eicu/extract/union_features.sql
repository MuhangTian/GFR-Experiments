-- PURPOSE: to extract union of features for fasterrisk out of distribution testing on eICU
-- Variables store in apache relevant tables (apachePredVar, apacheApsVar...etc) do not need to filter by 24 hours since they are already filtered by the first 24 hours
DROP TABLE IF EXISTS union_features; CREATE TABLE union_features AS 
WITH age_extract AS (  
  SELECT patientunitstayid 
  , CASE
    WHEN age = '> 89' THEN 91
    WHEN age = '' THEN NULL
    ELSE CAST(age AS INTEGER) 
    END AS age
    FROM patient
)
, merged_gcs AS (
  SELECT pat_gcs.patientunitstayid, physicalexam.gcs1, pivoted_gcs.gcs2
  FROM patient AS pat_gcs
   
    LEFT JOIN(
      SELECT patientunitstayid, MIN(CAST(physicalexamvalue AS NUMERIC)) AS gcs1
      FROM physicalexam
      WHERE  (
      (physicalExamPath LIKE 'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/_' OR
       physicalExamPath LIKE 'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/__')
      AND (physicalexamoffset > 0 AND physicalexamoffset <= 1440) -- consider only first 24h
      AND physicalexamvalue IS NOT NULL)
  GROUP BY patientunitstayid
  )
  AS physicalexam
  ON physicalexam.patientunitstayid = pat_gcs.patientunitstayid

    LEFT JOIN(
      SELECT pivoted_gcs.patientunitstayid, pivoted_gcs.gcs as gcs2
      FROM pivoted_gcs
      WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    )
  AS pivoted_gcs
  ON pivoted_gcs.patientunitstayid = pat_gcs.patientunitstayid
)
, gcs_min_extract AS (
	SELECT tmp_min.patientunitstayid, MIN(tmp_min.gcs_min) as gcs_min
	FROM (
		SELECT patientunitstayid, COALESCE(gcs1, gcs2) AS gcs_min 
		FROM merged_gcs
	) tmp_min
	GROUP BY tmp_min.patientunitstayid
)
, merged_vent AS (

    -- 1: use patient table as base 
    SELECT pat.patientunitstayid, vent_1, vent_2, vent_3, vent_4
    FROM patient AS pat

    -- 2: ventilation_events table
      LEFT JOIN(
        SELECT patientunitstayid,
        MAX( CASE WHEN event = 'mechvent start' OR event = 'mechvent end' THEN 1
        ELSE NULL
        END) as vent_1
        FROM ventilation_events AS vent_events
        GROUP BY patientunitstayid
  )
  AS vent_events
  ON vent_events.patientunitstayid = pat.patientunitstayid 

    -- 3: apacheapsvar table
    LEFT JOIN(
      SELECT patientunitstayid, intubated as vent_2
      FROM apacheapsvar AS apacheapsvar
      WHERE (intubated = 1)
  )
  AS apacheapsvar
  ON apacheapsvar.patientunitstayid = pat.patientunitstayid 
  
    -- 4: apachepredvar table
    LEFT JOIN(
      SELECT patientunitstayid, oobintubday1 as vent_3
      FROM apachepredvar AS apachepredvar
      WHERE (oobintubday1 = 1)
  )
  AS apachepredvar
  ON apachepredvar.patientunitstayid = pat.patientunitstayid 
    
    
    -- 5: respiratory care table 
    LEFT JOIN(
      SELECT patientunitstayid, 
      CASE
      WHEN COUNT(airwaytype) >= 1 THEN 1
      WHEN COUNT(airwaysize) >= 1 THEN 1
      WHEN COUNT(airwayposition) >= 1 THEN 1
      WHEN COUNT(cuffpressure) >= 1 THEN 1
      WHEN COUNT(setapneatv) >= 1 THEN 1
      ELSE NULL
      END AS vent_4
      FROM respiratorycare AS resp_care
      WHERE (respCareStatusOffset > 0 AND respCareStatusOffset <= 1440)
      GROUP BY patientunitstayid
  )
  AS resp_care
  ON resp_care.patientunitstayid = pat.patientunitstayid 
)
, mechvent_extract AS(
	SELECT patientunitstayid
    , CASE
    WHEN vent_1 = 1 THEN 1
    WHEN vent_2 = 1 THEN 1
    WHEN vent_3 = 1 THEN 1
    WHEN vent_4 = 1 THEN 1
    ELSE 0
    END AS mechvent
    FROM merged_vent
)
, urineoutput_extract AS (
  
  -- pat table as base for patientunitstayid 
  SELECT pat.patientunitstayid, COALESCE(pivoted_uo.urineoutput, apache_urine.urine) AS urineoutput -- consider pivoted_uo first, if missing -> apacheapsvar
  FROM patient AS pat
  
  -- Join information from pivoted_uo table
  LEFT JOIN(
  SELECT patientunitstayid AS pid_uo, SUM(urineoutput) AS urineoutput
  FROM pivoted_uo
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
  AND urineoutput > 0 AND urineoutput IS NOT NULL -- ignore biologically implausible values <0
  GROUP BY pid_uo
  ) AS pivoted_uo
  ON pivoted_uo.pid_uo = pat.patientunitstayid

  -- Join information from apacheapsvar table
  LEFT JOIN(
  SELECT patientunitstayid AS pid_auo, urine
  FROM apacheapsvar
  WHERE urine > 0 AND urine IS NOT NULL -- ignore biologically implausible values <0
  ) AS apache_urine
  ON apache_urine.pid_auo = pat.patientunitstayid
)
, merged_electivesurgery AS (

    -- 1: pat table as base for patientunitstayid  
    SELECT pat.patientunitstayid, electivesurgery1
      , CASE
      WHEN unitAdmitSource LIKE 'Emergency Department' THEN 0
      ELSE 1
      END AS adm_elective1
      FROM patient AS pat

    -- 2: apachepredvar table
    LEFT JOIN (
    SELECT apache.patientunitstayid, electivesurgery AS electivesurgery1
    FROM apachepredvar AS apache
    )
    AS apache
    ON pat.patientunitstayid = apache.patientunitstayid
)
, electivesurgery_extract AS (
  SELECT patientunitstayid

  , CASE    -- For feature
    WHEN electivesurgery1 = 0 THEN 0
    WHEN electivesurgery1 IS NULL THEN 0
    WHEN adm_elective1 = 0 THEN 0
    ELSE 1
    END AS electivesurgery
  FROM merged_electivesurgery
)
, resprate_extract AS (
SELECT patientunitstayid
  , MIN(respiratoryrate)::INTEGER as resprate_min
  , MAX(respiratoryrate)::INTEGER as resprate_max
  FROM pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    AND respiratoryrate IS NOT NULL
  GROUP BY patientunitstayid
)
, tempc_extract AS (
  SELECT patientunitstayid
  , MIN(temperature) as tempc_min
  , MAX(temperature) as tempc_max
  FROM pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    AND temperature IS NOT NULL
  GROUP BY patientunitstayid
)
, nursevars AS (
  SELECT patientunitstayid
  , MIN(temperature) AS tempc_min
  , MAX(temperature) AS tempc_max
  , MIN(heartrate) as heartrate_min
  , MAX(heartrate) as heartrate_max
	
  , CASE 
    WHEN MIN(ibp_systolic) IS NOT NULL THEN MIN(ibp_systolic)
    WHEN MIN(nibp_systolic) IS NOT NULL THEN MIN(nibp_systolic)
    ELSE NULL
    END AS sysbp_min
	
  , CASE
    WHEN MAX(ibp_systolic) IS NOT NULL THEN MAX(ibp_systolic)
    WHEN MAX(nibp_systolic) IS NOT NULL THEN MAX(nibp_systolic)
    ELSE NULL
    END AS sysbp_max
  
  , CASE
    WHEN MIN(ibp_mean) IS NOT NULL THEN MIN(ibp_mean)
    WHEN MIN(nibp_mean) IS NOT NULL THEN MIN(nibp_mean)
    ELSE NULL
    END AS meanbp_min
  
  , CASE
    WHEN MAX(ibp_mean) IS NOT NULL THEN MAX(ibp_mean)
    WHEN MAX(nibp_mean) IS NOT NULL THEN MAX(nibp_mean)
    ELSE NULL
    END AS meanbp_max
	
  FROM pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440)
  GROUP BY patientunitstayid
)
, vitalperiodicvars AS (
	SELECT patientunitstayid, 
	  MIN(heartrate) AS heartrate_min, 
	  MAX(heartrate) AS heartrate_max, 
	  MIN(systemicsystolic) AS sysbp_min, 
	  MAX(systemicsystolic) AS sysbp_max,
	  MIN(systemicmean) AS meanbp_min,
	  MAX(systemicmean) AS meanbp_max,
      MIN(respiration) AS resprate_min,
	  MAX(respiration) AS resprate_max,
	  MIN(temperature) AS tempc_min,
	  MAX(temperature) AS tempc_max
	FROM vitalperiodic
	WHERE observationoffset <= 1440
	GROUP BY patientunitstayid
)
, combine_extract AS(		-- coalesce with vitalPeriodic values to reduce NULLs
	SELECT pa.patientunitstayid,
	COALESCE(nv.heartrate_min, vv.heartrate_min) AS heartrate_min,
	COALESCE(nv.heartrate_max, vv.heartrate_max) AS heartrate_max,
	COALESCE(nv.sysbp_min, vv.sysbp_min) AS sysbp_min,
	COALESCE(nv.sysbp_max, vv.sysbp_max) AS sysbp_max,
	COALESCE(nv.meanbp_min, vv.meanbp_min) AS meanbp_min,
	COALESCE(nv.meanbp_max, vv.meanbp_max) AS meanbp_max,
	COALESCE(re.resprate_min, vv.resprate_min) AS resprate_min,
	COALESCE(re.resprate_max, vv.resprate_max) AS resprate_max,
	COALESCE(nv.tempc_min, vv.tempc_min) AS tempc_min,
	COALESCE(nv.tempc_max, vv.tempc_max) AS tempc_max
	
	FROM patient pa
	LEFT JOIN nursevars nv
		ON pa.patientunitstayid = nv.patientunitstayid
	LEFT JOIN vitalperiodicvars vv
		ON pa.patientunitstayid = vv.patientunitstayid
	LEFT JOIN resprate_extract re
		ON pa.patientunitstayid = re.patientunitstayid
)
, labs_extract AS(
	SELECT patientunitstayid
		, MIN(bun) AS bun_min
		, MAX(bun) AS bun_max
		, MIN(wbc) AS wbc_min
		, MAX(wbc) AS wbc_max
		, MIN(potassium) AS potassium_min
		, MAX(potassium) AS potassium_max
		, MIN(sodium) AS sodium_min
		, MAX(sodium) AS sodium_max
		, MIN(bicarbonate) AS bicarbonate_min
		, MAX(bicarbonate) AS bicarbonate_max
		, MIN(bilirubin) AS bilirubin_min
		, MAX(bilirubin) AS bilirubin_max
		, MIN(hematocrit) AS hematocrit_min
		, MAX(hematocrit) AS hematocrit_max
		, MIN(creatinine) AS creatinine_min
		, MAX(creatinine) AS creatinine_max
		, MIN(albumin) AS albumin_min
		, MAX(albumin) AS albumin_max
		, MIN(glucose) AS glucose_min
		, MAX(glucose) AS glucose_max
	
	FROM lab_firstday
	GROUP BY patientunitstayid
)
, comorb AS 
(SELECT p.patientunitstayid, p.hem, 
 CASE WHEN a.aids IS NULL THEN 0
 ELSE a.aids END AS aids,
 
 CASE WHEN a.met IS NULL THEN 0
 ELSE a.met END AS mets
FROM (
  SELECT patientunitstayid,
    CASE WHEN patientunitstayid IN (
        SELECT patientunitstayid 
        FROM pasthistory
        WHERE pasthistorypath LIKE '%notes/Progress Notes/Past History/Organ Systems/Hematology/Oncology (R)/Cancer/Hematologic Malignancy%'
    ) THEN 1 ELSE 0
    END AS hem
  FROM patient
) p
LEFT JOIN (
  SELECT patientunitstayid, aids, metastaticcancer AS met
  FROM apachepredvar
) a
ON p.patientunitstayid = a.patientunitstayid)

, admissiontypevar AS (
SELECT
  p.patientunitstayid,
  CASE
    WHEN ap.electivesurgery = 1 THEN 'ScheduledSurgical'
    WHEN p.unitadmitsource IN ('operating room', 'recoveryroom', 'PACU') THEN 'UnscheduledSurgical'
    WHEN dxids.operative = 1 THEN 'UnscheduledSurgical'
    ELSE 'Medical'
  END AS admissiontype
FROM
  patient p
  LEFT JOIN apachepredvar ap ON p.patientunitstayid = ap.patientunitstayid
  LEFT JOIN (
    SELECT patientunitstayid,
    MAX(CASE WHEN admitdxpath LIKE '%All Diagnosis|Operative%' THEN 1 ELSE 0 END) AS operative
    FROM admissiondx
    GROUP BY patientunitstayid
  ) dxids ON p.patientunitstayid = dxids.patientunitstayid
)
, pao2fio2_vent_min_extract AS (
  SELECT bg_firstday.patientunitstayid,
         CASE WHEN vent_firstday.ventcpap = 1 AND MIN(bg_firstday.pao2 / bg_firstday.fio2) IS NOT NULL 
              THEN MIN(bg_firstday.pao2 / bg_firstday.fio2) ELSE NULL END AS pao2fio2_vent_min
  FROM bg_firstday
  LEFT JOIN vent_firstday ON bg_firstday.patientunitstayid = vent_firstday.patientunitstayid
  GROUP BY bg_firstday.patientunitstayid, vent_firstday.ventcpap
)
, admissiontype_extract AS (
SELECT
  p.patientunitstayid,
  CASE
    WHEN ap.electivesurgery = 1 THEN 'ScheduledSurgical'
    WHEN p.unitadmitsource IN ('operating room', 'recoveryroom', 'PACU') THEN 'UnscheduledSurgical'
    WHEN dxids.operative = 1 THEN 'UnscheduledSurgical'
    ELSE 'Medical'
  END AS admissiontype
FROM
  patient p
  LEFT JOIN apachepredvar ap ON p.patientunitstayid = ap.patientunitstayid
  LEFT JOIN (
    SELECT patientunitstayid,
    MAX(CASE WHEN admitdxpath LIKE '%All Diagnosis|Operative%' THEN 1 ELSE 0 END) AS operative
    FROM admissiondx
    GROUP BY patientunitstayid
  ) dxids ON p.patientunitstayid = dxids.patientunitstayid
)
, bg_extract AS (
	SELECT patientunitstayid
	, MIN(pao2) AS pao2_min
	, MAX(pao2) AS pao2_max
	, MIN(aado2) AS aado2_min
	, MAX(aado2) AS aado2_max
	, MIN(ph) AS ph_min
	, MAX(ph) AS ph_max
	, MIN(paco2) AS paco2_min
	, MAX(paco2) AS paco2_max
	
	FROM bg_firstday
	GROUP BY patientunitstayid
)
SELECT patient.uniquepid, patient.patientunitstayid
	, CASE 
	  WHEN patient.hospitalDischargeStatus = 'Expired' THEN 1
	  WHEN patient.hospitalDischargeLocation = 'Death' THEN 1
	  WHEN patient.hospitalDischargeStatus IS NULL THEN NULL
	  WHEN patient.hospitalDischargeLocation IS NULL THEN NULL 
	  ELSE 0
	  END AS hospital_expire_flag
	, patient.hospitalAdmitOffset * (-1) AS preiculos		-- do negation here since subtraction is the other way around
  , age_extract.age
	, gcs_min_extract.gcs_min
	, mechvent_extract.mechvent
	, urineoutput_extract.urineoutput

  , combine_extract.heartrate_min
	, combine_extract.heartrate_max
  , combine_extract.meanbp_min
	, combine_extract.meanbp_max
	, combine_extract.resprate_min
	, combine_extract.resprate_max
	, combine_extract.tempc_min
	, combine_extract.tempc_max
  , combine_extract.sysbp_min
	, combine_extract.sysbp_max

  , labs_extract.bun_min
	, labs_extract.bun_max
	, labs_extract.wbc_min
	, labs_extract.wbc_max
	, labs_extract.potassium_min
	, labs_extract.potassium_max
	, labs_extract.sodium_min
	, labs_extract.sodium_max
	, labs_extract.bicarbonate_min
	, labs_extract.bicarbonate_max
	, labs_extract.bilirubin_min
	, labs_extract.bilirubin_max
	, labs_extract.hematocrit_min
	, labs_extract.hematocrit_max
	, labs_extract.creatinine_min
	, labs_extract.creatinine_max
	, labs_extract.albumin_min
	, labs_extract.albumin_max
	, labs_extract.glucose_max
	, labs_extract.glucose_min

  , comorb.aids
	, comorb.hem
	, comorb.mets

	, electivesurgery_extract.electivesurgery
	
	, pao2fio2_vent_min_extract.pao2fio2_vent_min
	, admissiontype_extract.admissiontype
	
	, bg_extract.pao2_max
	, bg_extract.pao2_min
  , bg_extract.paco2_max
	, bg_extract.paco2_min
  , bg_extract.ph_min
	, bg_extract.ph_max
	, bg_extract.aado2_min
	, bg_extract.aado2_max
	
FROM patient

LEFT JOIN gcs_min_extract
	ON gcs_min_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN age_extract
	ON age_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN mechvent_extract
	ON mechvent_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN urineoutput_extract
	ON urineoutput_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN electivesurgery_extract
	ON electivesurgery_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN combine_extract
	ON combine_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN labs_extract
	ON labs_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN comorb
	ON comorb.patientunitstayid = patient.patientunitstayid
LEFT JOIN pao2fio2_vent_min_extract
	ON pao2fio2_vent_min_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN admissiontype_extract
	ON admissiontype_extract.patientunitstayid = patient.patientunitstayid
LEFT JOIN bg_extract
	ON bg_extract.patientunitstayid = patient.patientunitstayid

WHERE age_extract.age <= 89   -- filter age