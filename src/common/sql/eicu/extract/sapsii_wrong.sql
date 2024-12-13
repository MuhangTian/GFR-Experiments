-- PURPOSE: to see how bugged version of SAPS II implementation in MIMIC performs on eICU

DROP TABLE IF EXISTS sapsii_wrong; 
CREATE TABLE sapsii_wrong AS 

WITH 
pre_icu_los_data AS (
SELECT patientunitstayid

  , hospitaladmitoffset * (-1) as preiculos     -- do negation here since subtraction is the other way around
    FROM patient
)
, cohort AS (
SELECT uniquepid, patientunitstayid, 
  CASE 
  WHEN patient.unitDischargeStatus = 'Expired' THEN 1
  WHEN patient.unitDischargeLocation = 'Death' THEN 1
  WHEN patient.unitDischargeStatus IS NULL THEN NULL
  WHEN patient.unitDischargeLocation IS NULL THEN NULL 
  ELSE 0
  END AS icustay_expire_flag, 
  CASE 
  WHEN patient.hospitalDischargeStatus = 'Expired' THEN 1
  WHEN patient.hospitalDischargeLocation = 'Death' THEN 1
  WHEN patient.hospitalDischargeStatus IS NULL THEN NULL
  WHEN patient.hospitalDischargeLocation IS NULL THEN NULL 
  ELSE 0
  END AS hospital_expire_flag, 
  CASE
  WHEN age = '> 89' THEN 91
  WHEN age = '' THEN NULL
  ELSE CAST(age AS INTEGER) 
  END AS age
FROM patient 
)

,labvars AS (
SELECT patientunitstayid, 
      MIN(bicarbonate) AS bicarbonate_min, 
      MAX(bicarbonate) AS bicarbonate_max,
      AVG(bicarbonate) AS bicarbonate_mean,
      STDDEV(bicarbonate) AS bicarbonate_std, 
      COUNT(bicarbonate) AS bicarbonate_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY bicarbonate) AS bicarbonate_10p, 
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY bicarbonate) AS bicarbonate_90p,

	  MIN(bilirubin) AS bilirubin_min,
      MAX(bilirubin) AS bilirubin_max,
      AVG(bilirubin) AS bilirubin_mean,
      STDDEV(bilirubin) AS bilirubin_std,
      COUNT(bilirubin) AS bilirubin_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY bilirubin) AS bilirubin_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY bilirubin) AS bilirubin_90p,

      MIN(bun) AS bun_min,
      MAX(bun) AS bun_max,
      AVG(bun) AS bun_mean,
      STDDEV(bun) AS bun_std,
      COUNT(bun) AS bun_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY bun) AS bun_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY bun) AS bun_90p,

      MIN(potassium) AS potassium_min,
      MAX(potassium) AS potassium_max,
      AVG(potassium) AS potassium_mean,
      STDDEV(potassium) AS potassium_std,
      COUNT(potassium) AS potassium_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY potassium) AS potassium_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY potassium) AS potassium_90p,

      MIN(sodium) AS sodium_min,
      MAX(sodium) AS sodium_max,
      AVG(sodium) AS sodium_mean,
      STDDEV(sodium) AS sodium_std,
      COUNT(sodium) AS sodium_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY sodium) AS sodium_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY sodium) AS sodium_90p,

      MIN(wbc) AS wbc_min,
      MAX(wbc) AS wbc_max,
      AVG(wbc) AS wbc_mean,
      STDDEV(wbc) AS wbc_std,
      COUNT(wbc) AS wbc_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY wbc) AS wbc_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY wbc) AS wbc_90p

FROM lab_firstday
GROUP BY patientunitstayid
)

, bgvars AS (
  SELECT bg_firstday.patientunitstayid,
         CASE WHEN vent_firstday.ventcpap = 1 AND MIN(bg_firstday.pao2 / bg_firstday.fio2) IS NOT NULL 
              THEN MIN(bg_firstday.pao2 / bg_firstday.fio2) ELSE NULL END AS pao2fio2_vent_min
  FROM bg_firstday
  LEFT JOIN vent_firstday ON bg_firstday.patientunitstayid = vent_firstday.patientunitstayid
  GROUP BY bg_firstday.patientunitstayid, vent_firstday.ventcpap
)

-- Note that pivotedvital does NOT come from the vitalperiodic or aperiodic tables, but from nursecharting
, nursingvars AS (
  SELECT patientunitstayid
  , MIN(temperature) as tempc_min
  , MAX(temperature) as tempc_max
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
    WHEN AVG(ibp_systolic) IS NOT NULL THEN AVG(ibp_systolic)
    WHEN AVG(nibp_systolic) IS NOT NULL THEN AVG(nibp_systolic)
    ELSE NULL
    END AS sysbp_mean
  , CASE 
    WHEN STDDEV(ibp_systolic) IS NOT NULL THEN STDDEV(ibp_systolic)
    WHEN STDDEV(nibp_systolic) IS NOT NULL THEN STDDEV(nibp_systolic)
    ELSE NULL
	END AS sysbp_std
  , CASE 
    WHEN COUNT(ibp_systolic) IS NOT NULL THEN COUNT(ibp_systolic)
    WHEN COUNT(nibp_systolic) IS NOT NULL THEN COUNT(nibp_systolic)
    ELSE NULL
    END AS sysbp_count
  , CASE 
    WHEN PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY ibp_systolic) IS NOT NULL THEN PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY ibp_systolic) 
    WHEN PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY nibp_systolic) IS NOT NULL THEN PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY nibp_systolic) 
    ELSE NULL
    END AS sysbp_10p
  , CASE 
    WHEN PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ibp_systolic) IS NOT NULL THEN PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ibp_systolic) 
    WHEN PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY nibp_systolic) IS NOT NULL THEN PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY nibp_systolic) 
    ELSE NULL
    END AS sysbp_90p
  FROM pivoted_vital
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
    AND temperature IS NOT NULL
  GROUP BY patientunitstayid
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
, comb_gcs AS (
    SELECT patientunitstayid, COALESCE(gcs1, gcs2) AS gcs_min 
    FROM merged_gcs
)
-- Only keep minimal gcs from merged_gcs table
, minimal_gcs AS (
    SELECT patientunitstayid, MIN(gcs_min) as gcs_min
    FROM comb_gcs
    GROUP BY patientunitstayid
)


, merged_uo AS (
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

, chronicvars AS 
(SELECT p.patientunitstayid, p.hem, a.aids, a.met
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

, highresvitalvars AS (

SELECT patientunitstayid, 
  MIN(temperature) AS min_temperature, 
  MAX(temperature) AS max_temperature,
  MIN(heartrate) AS min_heartrate, 
  MAX(heartrate) AS max_heartrate, 
  MIN(systemicsystolic) AS sysbp_min, 
  MAX(systemicsystolic) AS sysbp_max, 
  AVG(systemicsystolic) AS sysbp_mean,
  STDDEV(systemicsystolic) AS sysbp_std,  
  COUNT(systemicsystolic) AS sysbp_count, 
  PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY systemicsystolic) AS sysbp_10p, 
  PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY systemicsystolic) AS sysbp_90p
FROM vitalperiodic
WHERE observationoffset <= 1440
GROUP BY patientunitstayid
)

, combfeatures AS (
SELECT 
       cohort.patientunitstayid,
       cohort.uniquepid,		
	   cohort.icustay_expire_flag,
       cohort.hospital_expire_flag,
	   cohort.age,
       labvars.bicarbonate_min,
       labvars.bicarbonate_max,
       labvars.bicarbonate_mean,
       labvars.bicarbonate_std,
       labvars.bicarbonate_count,
       labvars.bicarbonate_10p,
       labvars.bicarbonate_90p,
       labvars.bilirubin_min,
       labvars.bilirubin_max,
       labvars.bilirubin_mean,
       labvars.bilirubin_std,
       labvars.bilirubin_count,
       labvars.bilirubin_10p,
       labvars.bilirubin_90p,
       labvars.bun_min,
       labvars.bun_max,
       labvars.bun_mean,
       labvars.bun_std,
       labvars.bun_count,
       labvars.bun_10p,
       labvars.bun_90p,
       labvars.potassium_min,
       labvars.potassium_max,
       labvars.potassium_mean,
       labvars.potassium_std,
       labvars.potassium_count,
       labvars.potassium_10p,
       labvars.potassium_90p,
       labvars.sodium_min,
       labvars.sodium_max,
       labvars.sodium_mean,
       labvars.sodium_std,
       labvars.sodium_count,
       labvars.sodium_10p,
       labvars.sodium_90p,
       labvars.wbc_min,
       labvars.wbc_max,
       labvars.wbc_mean,
       labvars.wbc_std,
       labvars.wbc_count,
       labvars.wbc_10p,
       labvars.wbc_90p,
	   bgvars.pao2fio2_vent_min,
       nursingvars.tempc_min, 
       nursingvars.tempc_max, 
       CASE 
       WHEN nursingvars.heartrate_min IS NULL THEN highresvitalvars.min_heartrate
       ELSE nursingvars.heartrate_min
       END AS heartrate_min, 

       CASE 
       WHEN nursingvars.heartrate_max IS NULL THEN highresvitalvars.max_heartrate
       ELSE nursingvars.heartrate_max
       END AS heartrate_max,


       CASE 
       WHEN nursingvars.sysbp_min IS NULL THEN highresvitalvars.sysbp_min
       ELSE nursingvars.sysbp_min
       END AS sysbp_min,

       CASE 
       WHEN nursingvars.sysbp_max IS NULL THEN highresvitalvars.sysbp_max
       ELSE nursingvars.sysbp_max
       END AS sysbp_max,


       nursingvars.sysbp_mean,
	   nursingvars.sysbp_std,
       nursingvars.sysbp_count,
       nursingvars.sysbp_10p,
       nursingvars.sysbp_90p,
       minimal_gcs.gcs_min AS gcs,
       merged_uo.urineoutput,
       chronicvars.aids, 
       chronicvars.hem, 
       chronicvars.met AS mets, 
       admissiontypevar.admissiontype,
       pre_icu_los_data.preiculos

FROM cohort
LEFT JOIN labvars ON cohort.patientunitstayid = labvars.patientunitstayid
LEFT JOIN bgvars ON cohort.patientunitstayid = bgvars.patientunitstayid
LEFT JOIN nursingvars ON cohort.patientunitstayid = nursingvars.patientunitstayid
LEFT JOIN minimal_gcs ON cohort.patientunitstayid = minimal_gcs.patientunitstayid
LEFT JOIN merged_uo ON cohort.patientunitstayid = merged_uo.patientunitstayid
LEFT JOIN chronicvars ON cohort.patientunitstayid = chronicvars.patientunitstayid
LEFT JOIN admissiontypevar ON cohort.patientunitstayid = admissiontypevar.patientunitstayid
LEFT JOIN highresvitalvars ON cohort.patientunitstayid = highresvitalvars.patientunitstayid
LEFT JOIN pre_icu_los_data ON cohort.patientunitstayid = pre_icu_los_data.patientunitstayid
ORDER BY cohort.uniquepid, cohort.patientunitstayid
)

, scorecomp as
(
select
  combfeatures.*
  -- Below code calculates the component scores needed for SAPS
  , case
      when age is null then null
      when age <  40 then 0
      when age <  60 then 7
      when age <  70 then 12
      when age <  75 then 15
      when age <  80 then 16
      when age >= 80 then 18
    end as age_score

  , case
      when heartrate_max is null then null
      when heartrate_min < 40 then 11
      when heartrate_max >= 160 then 7
      when heartrate_max >= 120 then 4
      when heartrate_min  <  70 then 2
      when  heartrate_max >= 70 and heartrate_max < 120
        and heartrate_min >= 70 and heartrate_min < 120
      then 0
    end as hr_score

  , case
      when  sysbp_min is null then null
      when  sysbp_min <   70 then 13
      when  sysbp_min <  100 then 5
      when  sysbp_max >= 200 then 2
      when  sysbp_max >= 100 and sysbp_max < 200
        and sysbp_min >= 100 and sysbp_min < 200
        then 0
    end as sysbp_score

  , case
      when tempc_max is null then null
      when tempc_min <  39.0 then 0
      when tempc_max >= 39.0 then 3
    end as temp_score

  , case
      when pao2fio2_vent_min is null then null
      when pao2fio2_vent_min <  100 then 11
      when pao2fio2_vent_min <  200 then 9
      when pao2fio2_vent_min >= 200 then 6
    end as pao2fio2_score

  , case
      when urineoutput is null then null
      when urineoutput <   500.0 then 11
      when urineoutput <  1000.0 then 4
      when urineoutput >= 1000.0 then 0
    end as uo_score

  , case
      when bun_max is null then null
      when bun_max <  28.0 then 0
      when bun_max <  84.0 then 6
      when bun_max >= 84.0 then 10
    end as bun_score

  , case
      when wbc_max is null then null
      when wbc_min <   1.0 then 12
      when wbc_max >= 20.0 then 3
      when wbc_max >=  1.0 and wbc_max < 20.0
       and wbc_min >=  1.0 and wbc_min < 20.0
        then 0
    end as wbc_score

  , case
      when potassium_max is null then null
      when potassium_min <  3.0 then 3
      when potassium_max >= 5.0 then 3
      when potassium_max >= 3.0 and potassium_max < 5.0
       and potassium_min >= 3.0 and potassium_min < 5.0
        then 0
      end as potassium_score

  , case
      when sodium_max is null then null
      when sodium_min  < 125 then 5
      when sodium_max >= 145 then 1
      when sodium_max >= 125 and sodium_max < 145
       and sodium_min >= 125 and sodium_min < 145
        then 0
      end as sodium_score

  , case
      when bicarbonate_max is null then null
      -- NOTE: this is MIMIC III SAPSII implementation bug, which uses 5 (it should be 6, see sapsii.sql for correct implementation)
      when bicarbonate_min <  15.0 then 5
      when bicarbonate_min <  20.0 then 3
      when bicarbonate_max >= 20.0
       and bicarbonate_min >= 20.0
          then 0
      end as bicarbonate_score

  , case
      when bilirubin_max is null then null
      when bilirubin_max  < 4.0 then 0
      when bilirubin_max  < 6.0 then 4
      when bilirubin_max >= 6.0 then 9
      end as bilirubin_score

   , case
      when gcs is null then null
        when gcs <  3 then null -- erroneous value/on trach
        when gcs <  6 then 26
        when gcs <  9 then 13
        when gcs < 11 then 7
        when gcs < 14 then 5
        when gcs >= 14
         and gcs <= 15
          then 0
        end as gcs_score

    , case
        when aids = 1 then 17
        when hem  = 1 then 10
        when mets = 1 then 9
        else 0
      end as comorbidity_score

    , case
        when admissiontype = 'ScheduledSurgical' then 0
        when admissiontype = 'Medical' then 6
        when admissiontype = 'UnscheduledSurgical' then 8
        else null
      end as admissiontype_score

from combfeatures
)
-- Calculate SAPS II here so we can use it in the probability calculation below
-- , score as
, score as
(
  select s.*
  -- coalesce statements impute normal score of zero if data element is missing
  , coalesce(age_score,0)
  + coalesce(hr_score,0)
  + coalesce(sysbp_score,0)
  + coalesce(temp_score,0)
  + coalesce(pao2fio2_score,0)
  + coalesce(uo_score,0)
  + coalesce(bun_score,0)
  + coalesce(wbc_score,0)
  + coalesce(potassium_score,0)
  + coalesce(sodium_score,0)
  + coalesce(bicarbonate_score,0)
  + coalesce(bilirubin_score,0)
  + coalesce(gcs_score,0)
  + coalesce(comorbidity_score,0)
  + coalesce(admissiontype_score,0)
    as sapsii
  from scorecomp s
)
select uniquepid, patientunitstayid, hospital_expire_flag
, 1 / (1 + exp(- (-7.7631 + 0.0737*(sapsii) + 0.9971*(ln(sapsii + 1))) )) as sapsii_prob
, age
, preiculos
, heartrate_max
, heartrate_min
, sysbp_min
, sysbp_max
, sysbp_mean
, sysbp_std
, sysbp_count
, sysbp_10p
, sysbp_90p

, tempc_max
, tempc_min
, pao2fio2_vent_min
, urineoutput

, bun_max
, bun_min
, bun_mean
, bun_std
, bun_count
, bun_10p
, bun_90p

, wbc_min
, wbc_max
, wbc_mean
, wbc_std
, wbc_count
, wbc_10p
, wbc_90p

, potassium_min
, potassium_max
, potassium_mean
, potassium_std
, potassium_count
, potassium_10p
, potassium_90p

, sodium_min
, sodium_max
, sodium_mean
, sodium_std
, sodium_count
, sodium_10p
, sodium_90p

, bicarbonate_min
, bicarbonate_max
, bicarbonate_mean
, bicarbonate_std
, bicarbonate_count
, bicarbonate_10p
, bicarbonate_90p

, bilirubin_min
, bilirubin_max
, bilirubin_mean
, bilirubin_std
, bilirubin_count
, bilirubin_10p
, bilirubin_90p
, gcs
, aids
, hem
, mets
, admissiontype
FROM score