-- SOFA Script for eICU
-- This script creates the sofa table, which contains the sofa score for each patient


DROP TABLE IF EXISTS sofa; 
CREATE TABLE sofa AS 

WITH 
cohort AS (
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

-- GCS

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

-- Lab Variables: 

,labvars AS (
SELECT patientunitstayid, 

	  MIN(bilirubin) AS bilirubin_min,
      MAX(bilirubin) AS bilirubin_max,
      AVG(bilirubin) AS bilirubin_mean,
      STDDEV(bilirubin) AS bilirubin_std,
      COUNT(bilirubin) AS bilirubin_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY bilirubin) AS bilirubin_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY bilirubin) AS bilirubin_90p,

      MIN(creatinine) AS creatinine_min,
      MAX(creatinine) AS creatinine_max,
      AVG(creatinine) AS creatinine_mean,
      STDDEV(creatinine) AS creatinine_std,
      COUNT(creatinine) AS creatinine_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY creatinine) AS creatinine_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY creatinine) AS creatinine_90p,

      MIN(platelets) AS platelet_min,
      MAX(platelets) AS platelet_max,
      AVG(platelets) AS platelet_mean,
      STDDEV(platelets) AS platelet_std,
      COUNT(platelets) AS platelet_count,
      PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY platelets) AS platelet_10p,
      PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY platelets) AS platelet_90p


FROM lab_firstday
GROUP BY patientunitstayid
)

, bloodpressure AS (
  SELECT patientunitstayid
  , CASE      -- use IBP as higher prioity following Joey's advice, TONY
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
  WHERE (chartoffset > 0 AND chartoffset <= 1440) -- consider only first 24h
  GROUP BY patientunitstayid
)

, vaso_preprocess AS (
    SELECT
        patientunitstayid,
		drugname, 
        infusionoffset,
        CAST(NULLIF(drugamount, '') AS NUMERIC) AS drugamount,
        CAST(NULLIF(volumeoffluid, '') AS NUMERIC) AS volumeoffluid,
        CASE
            WHEN drugrate IN ('Documentation undone', 'Date\Time Correction', 'OFF', 'ERROR', 'UD', 'OFF\.br\\.br\', '30\.br\') THEN 0.00000001
            ELSE CAST(drugrate AS NUMERIC)
        END AS drugrate
    FROM infusiondrug
    WHERE drugname IN (
        'Dopamine (mcg/kg/min)',
        'Dopamine (mcg/kg/hr)',
        'Dopamine (mcg/min)',
        'Dopamine (nanograms/kg/min)', 
        'Dopamine (mg/kg/min)',
        'Dopamine (mcg/hr)',
        'Dopamine (mg/min)',
        'Dopamine (mg/hr)',
        'Dopamine (ml/hr)',
        'Dopamine Volume (ml)',
        'Dopamine Volume (ml) (ml/hr)'
        'Dobutamine (mcg/kg/min)',
        'Dobutamine (mcg/kg/hr)',
        'Dobutamine (mcg/min)',
        'Dobutamine (nanograms/kg/min)', 
        'Dobutamine (mg/kg/min)',
        'Dobutamine (mcg/hr)',
        'Dobutamine (mg/min)',
        'Dobutamine (mg/hr)',
        'Dobutamine (ml/hr)',
        'Dobutamine Volume (ml)',
        'Dobutamine Volume (ml) (ml/hr)'
        'Norepinephrine (mcg/kg/min)',
        'Norepinephrine (mcg/kg/hr)',
        'Norepinephrine (mcg/min)',
        'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)',
        'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)',
        'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)',
        'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)',
        'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)',
        'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)',
        'Norepinephrine (mg/kg/min)',
        'Norepinephrine (mcg/hr)',
        'Norepinephrine (mg/min)',
        'Norepinephrine (mg/hr)',
        'Norepinephrine (ml/hr)',
        'norepinephrine Volume (ml)',
        'norepinephrine Volume (ml) (ml/hr)'
        'Epinephrine (mcg/kg/min)',
        'Epinephrine (mcg/kg/hr)',
        'Epinephrine (mcg/min)',
        'Epinephrine (mg/kg/min)',
        'Epinephrine (mcg/hr)',
        'Epinephrine (mg/min)',
        'Epinephrine (mg/hr)',
        'Epinephrine (ml/hr)',
        'Epinephrine Volume (ml)',
        'Epinephrine Volume (ml) (ml/hr)'
    )
	AND drugrate <> ''
    -- Check timing
    AND infusionoffset >= 0 AND infusionoffset <= 1440
)
, admissionweight AS (
    -- Usually, admission weight is close to discharge weight, so we use discharge as a proxy for admit weight if not available.
        SELECT
        patientunitstayid,
        CASE
            WHEN (admissionweight IS NOT NULL AND admissionweight != 0) THEN admissionweight
            WHEN (dischargeweight IS NOT NULL AND dischargeweight != 0) THEN dischargeweight
			-- Average weight of all patients is 83.998. Round to 2 decimal places
			ELSE 84.00
        END AS patient_weight
    FROM patient
)

, vasopresser_series AS (
SELECT
    vaso_preprocess.patientunitstayid
    , CASE
        WHEN drugname LIKE '%opamine%' THEN 'dopamine'
        WHEN drugname LIKE '%obutamine%' THEN 'dobutamine'
        WHEN drugname LIKE '%orepinephrine%' THEN 'norepinephrine'
        WHEN drugname LIKE '%pinephrine%' THEN 'epinephrine'
        ELSE NULL
    END AS drugname
    , infusionoffset
    , CASE
        -- Dopamine
        WHEN drugname = 'Dopamine (mcg/kg/min)' THEN drugrate
        WHEN drugname = 'Dopamine (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Dopamine (mcg/hr)' THEN drugrate / 60 / patient_weight
        WHEN drugname = 'Dopamine (ml/hr)' 
        THEN ((drugrate * CASE WHEN drugamount IS NOT NULL THEN drugamount ELSE 400 END) / (CASE WHEN volumeoffluid IS NOT NULL THEN volumeoffluid ELSE 250 END) * 1000) / (60 * patient_weight)
        -- Dobutamine
        WHEN drugname = 'Dobutamine (mcg/kg/hr)' THEN drugrate / 60
        WHEN drugname = 'Dobutamine (mcg/kg/min)' THEN drugrate
        WHEN drugname = 'Dobutamine (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Dobutamine (mcg/hr)' THEN drugrate / 60 / patient_weight
        WHEN drugname = 'Dobutamine (ml/hr)' 
        THEN ((drugrate * CASE WHEN drugamount IS NOT NULL THEN drugamount ELSE 250 END) / (CASE WHEN volumeoffluid IS NOT NULL THEN volumeoffluid ELSE 500 END) * 1000) / (60 * patient_weight)
        -- Norepinephrine
        WHEN drugname = 'Norepinephrine (mcg/kg/min)' THEN drugrate
        WHEN drugname = 'Norepinephrine (mcg/kg/hr)' THEN drugrate / 60
        WHEN drugname = 'Norepinephrine (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine MAX 32 mg Dextrose 5% 250 ml (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine STD 32 mg Dextrose 5% 282 ml (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine STD 4 mg Dextrose 5% 500 ml (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine STD 8 mg Dextrose 5% 250 ml (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine STD 8 mg Dextrose 5% 500 ml (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Norepinephrine (mg/kg/min)' THEN drugrate * 1000
        WHEN drugname = 'Norepinephrine (mcg/hr)' THEN drugrate / 60 / patient_weight
        WHEN drugname = 'Norepinephrine (mg/min)' THEN drugrate * 1000 / patient_weight
        WHEN drugname = 'Norepinephrine (mg/hr)' THEN drugrate * 1000 / patient_weight / 60
        WHEN drugname = 'Norepinephrine (ml/hr)' 
            OR drugname = 'norepinephrine Volume (ml)'
            OR drugname = 'norepinephrine Volume (ml) (ml/hr)' 
        THEN ((drugrate * CASE WHEN drugamount IS NOT NULL THEN drugamount ELSE 4 END) / (CASE WHEN volumeoffluid IS NOT NULL THEN volumeoffluid ELSE 250 END) * 1000) / (60 * patient_weight)
        -- Epinephrine
        WHEN drugname = 'Epinephrine (mcg/kg/min)' THEN drugrate
        WHEN drugname = 'Epinephrine (mcg/kg/hr)' THEN drugrate / 60
        WHEN drugname = 'Epinephrine (mcg/min)' THEN drugrate / patient_weight
        WHEN drugname = 'Epinephrine (mg/kg/min)' THEN drugrate
        WHEN drugname = 'Epinephrine (mcg/hr)' THEN drugrate / 60 / patient_weight
        WHEN drugname = 'Epinephrine (mg/min)' THEN drugrate * 1000 / patient_weight
        WHEN drugname = 'Epinephrine (mg/hr)' THEN drugrate * 1000 / patient_weight / 60
        WHEN drugname = 'Epinephrine (ml/hr)' 
        THEN ((drugrate * CASE WHEN drugamount IS NOT NULL THEN drugamount ELSE 1 END) / (CASE WHEN volumeoffluid IS NOT NULL THEN volumeoffluid ELSE 250 END) * 1000) / (60 * patient_weight)
        ELSE 0
    END AS drugrate_mcgkgmin
FROM vaso_preprocess
LEFT JOIN admissionweight
    ON vaso_preprocess.patientunitstayid = admissionweight.patientunitstayid
ORDER BY patientunitstayid, infusionoffset
)

, vaso_maxvalues AS (

SELECT patientunitstayid,
    MAX(CASE WHEN drugname = 'dopamine' THEN drugrate_mcgkgmin ELSE NULL END) AS max_dopamine,
    MAX(CASE WHEN drugname = 'dobutamine' THEN drugrate_mcgkgmin ELSE NULL END) AS max_dobutamine,
    MAX(CASE WHEN drugname = 'norepinephrine' THEN drugrate_mcgkgmin ELSE NULL END) AS max_norepinephrine,
    MAX(CASE WHEN drugname = 'epinephrine' THEN drugrate_mcgkgmin ELSE NULL END) AS max_epinephrine
FROM vasopresser_series
WHERE drugname IN ('dopamine', 'dobutamine', 'norepinephrine', 'epinephrine')
GROUP BY patientunitstayid


)


, resp_chart AS (

  SELECT 
  patientunitstayid, 
  1 AS vent_yes,
  MIN(respchartvaluelabel) AS event,
  
  MIN(CASE 
  WHEN LOWER(respchartvaluelabel) LIKE '%endotracheal%'
  OR LOWER(respchartvaluelabel) LIKE '%ett%'
  OR LOWER(respchartvaluelabel) LIKE '%ET Tube%'
  THEN respchartoffset
 ELSE 0
  END) AS vent_start_delta,

  MAX(CASE 
  WHEN LOWER(respchartvaluelabel) LIKE '%endotracheal%' 
  OR LOWER(respchartvaluelabel) LIKE '%ett%' 
  OR LOWER(respchartvaluelabel) LIKE '%ET Tube%'
  THEN respchartoffset
 ELSE NULL
  END) AS vent_stop_delta,

  MAX(offset_discharge) AS offset_discharge

  FROM respiratorycharting AS rc

  LEFT JOIN(
  SELECT patientunitstayid AS pat_pid, unitdischargeoffset AS offset_discharge
  FROM patient
  )
  AS pat
  ON pat.pat_pid = rc.patientunitstayid

  WHERE LOWER(respchartvaluelabel) LIKE '%endotracheal%' 
  OR LOWER(respchartvaluelabel) LIKE '%ett%' 
  OR LOWER(respchartvaluelabel) LIKE '%ET Tube%'

  GROUP BY patientunitstayid
)

, vent_nc AS (

  SELECT nc.patientunitstayid AS nc_pid, 
  1 AS vent_yes,
  MIN(cellattribute) AS event,
  
  MIN(CASE 
  WHEN (cellattribute = 'Airway Size' OR cellattribute = 'Airway Type') THEN nursecareentryoffset
 ELSE 0
  END) AS vent_start_delta,

  MAX(CASE 
  WHEN (cellattribute = 'Airway Size' OR cellattribute = 'Airway Type') THEN nursecareentryoffset
 ELSE NULL
  END) AS vent_stop_delta,

  MAX(offset_discharge) AS offset_discharge

  FROM nursecare AS nc

  LEFT JOIN(
  SELECT patientunitstayid AS pat_pid, unitdischargeoffset AS offset_discharge
  FROM patient
  )
  AS pat
  ON pat.pat_pid = nc.patientunitstayid

  WHERE cellattribute = 'Airway Size' 
  OR cellattribute = 'Airway Type'

  GROUP BY patientunitstayid
)

, vent_note AS (

  SELECT patientunitstayid AS note_pid,
  1 AS vent_yes,
  MIN(notetype) AS event,

  MIN(CASE 
  WHEN notetype = 'Intubation' THEN noteoffset
 ELSE 0
  END) AS vent_start_delta,

  MIN(CASE 
  WHEN notetype = 'Extubation' THEN noteoffset
  ELSE NULL
  END) AS vent_stop_delta,

  MAX(offset_discharge) AS offset_discharge

  FROM note AS note

  LEFT JOIN(
  SELECT patientunitstayid AS pat_pid, unitdischargeoffset AS offset_discharge
  FROM patient
  )
  AS pat
  ON pat.pat_pid = note.patientunitstayid

  WHERE notetype = 'Intubation' OR notetype = 'Extubation'

  GROUP BY patientunitstayid

) 

, vent_vente AS (

  SELECT patientunitstayid AS vent_pid, 
  1 AS vent_yes,
  MIN(event), 

  MIN(CASE 
  WHEN (event = 'mechvent start' ) THEN (hrs*60)
 ELSE 0
  END) AS vent_start_delta,

  MAX(CASE 
  WHEN (event = 'mechvent end' ) THEN (hrs*60)
  ELSE NULL
  END) AS vent_stop_delta,

  MAX(CASE 
  WHEN (event = 'ICU Discharge' ) THEN (hrs*60)
 ELSE NULL
  END) AS offset_discharge

  FROM ventilation_events

  WHERE event = 'ICU Discharge' 
  OR event = 'mechvent start'
  OR event = 'mechvent end'

  GROUP BY patientunitstayid
) 

/*
airwaytype -> Oral ETT, Nasal ETT, Tracheostomy, Double-Lumen Tube (do not use -> Cricothyrotomy)
airwaysize -> all unless ''
airwayposition -> all unless: Other (Comment), deflated, mlt, Documentation undone

Heuristic for times 
use ventstartoffset for start of ventilation
use priorventendoffset for end of ventilation
*/

, resp_care AS (
  SELECT 
    patientunitstayid,
    1 AS vent_yes,
    MIN(airwaytype) AS event,
    MIN(CASE 
      WHEN LOWER(airwaytype) LIKE '%ETT%' 
        OR LOWER(airwaytype) LIKE '%Tracheostomy%' 
        OR LOWER(airwaytype) LIKE '%Tube%'
        OR LOWER(airwaysize) NOT LIKE ''
        OR LOWER(airwayposition) NOT LIKE 'Other (Comment)'
        OR LOWER(airwayposition) NOT LIKE 'deflated'
        OR LOWER(airwayposition) NOT LIKE 'mlt'
        OR LOWER(airwayposition) NOT LIKE 'Documentation undone'
      THEN ventstartoffset
      ELSE NULL
    END) AS vent_start_delta,
    MAX(CASE 
      WHEN LOWER(airwaytype) LIKE '%ETT%' 
        OR LOWER(airwaytype) LIKE '%Tracheostomy%' 
        OR LOWER(airwaytype) LIKE '%Tube%'
        OR LOWER(airwaysize) NOT LIKE ''
        OR LOWER(airwayposition) NOT LIKE 'Other (Comment)'
        OR LOWER(airwayposition) NOT LIKE 'deflated'
        OR LOWER(airwayposition) NOT LIKE 'mlt'
        OR LOWER(airwayposition) NOT LIKE 'Documentation undone'
      THEN priorventendoffset
      ELSE NULL
    END) AS vent_stop_delta,
    MAX(offset_discharge) AS offset_discharge
  FROM respiratorycare AS rcare
  LEFT JOIN (
    SELECT patientunitstayid AS pat_pid, unitdischargeoffset AS offset_discharge
    FROM patient
  ) AS pat ON pat.pat_pid = rcare.patientunitstayid
  WHERE LOWER(airwaytype) LIKE '%ETT%' 
    OR LOWER(airwaytype) LIKE '%Tracheostomy%' 
    OR LOWER(airwaytype) LIKE '%Tube%'
    OR LOWER(airwaysize) NOT LIKE ''
    OR LOWER(airwayposition) NOT LIKE 'Other (Comment)'
    OR LOWER(airwayposition) NOT LIKE 'deflated'
    OR LOWER(airwayposition) NOT LIKE 'mlt'
    OR LOWER(airwayposition) NOT LIKE 'Documentation undone'
  GROUP BY patientunitstayid
)

, care_plan AS (
  SELECT 
    patientunitstayid, 
    1 AS vent_yes,
    STRING_AGG(cplitemvalue, '') AS event,
    MIN(CASE 
      WHEN cplitemvalue LIKE 'Intubated%' OR cplitemvalue LIKE 'Ventilated%'
      THEN cplitemoffset
      ELSE NULL
    END) AS vent_start_delta,
    CAST(NULL AS INT) AS vent_stop_delta,
    MAX(offset_discharge) AS offset_discharge
  FROM careplangeneral AS cpg
  LEFT JOIN (
    SELECT patientunitstayid AS pat_pid, unitdischargeoffset AS offset_discharge
    FROM patient
  ) AS pat ON pat.pat_pid = cpg.patientunitstayid
  WHERE (cplgroup = 'Airway' OR cplgroup = 'Ventilation')
    AND cplitemvalue NOT LIKE ''
  GROUP BY patientunitstayid
)


, union_table AS (

  SELECT * FROM resp_chart

  UNION DISTINCT

  SELECT * FROM vent_nc

  UNION DISTINCT

  SELECT * FROM vent_note

  UNION DISTINCT

  SELECT * FROM vent_vente
  
  UNION DISTINCT

  SELECT * FROM resp_care

  UNION DISTINCT

  SELECT * FROM care_plan
)

, vent_durations AS (
SELECT 
    patientunitstayid,
    -- MAX(vent_yes) AS vent_yes,
    -- STRING_AGG(event, ',') AS event,
    MIN(vent_start_delta) AS vent_start_delta,
    MAX(
		CASE WHEN vent_stop_delta IS NULL THEN offset_discharge
		ELSE vent_stop_delta
		END
	)
	 AS vent_stop_delta,
    MAX(offset_discharge) AS offset_discharge,
    MAX(CASE 
        WHEN vent_stop_delta <> 0 OR vent_stop_delta IS NOT NULL
        THEN vent_stop_delta - min_vent_start_delta
        ELSE offset_discharge - min_vent_start_delta
    END) AS vent_duration
FROM (
    SELECT 
        patientunitstayid,
        vent_yes,
        event,
        vent_start_delta,
        vent_stop_delta,
        offset_discharge,
        MIN(vent_start_delta) OVER (PARTITION BY patientunitstayid) AS min_vent_start_delta
    FROM union_table
    WHERE vent_start_delta IS NOT NULL
) sub
GROUP BY patientunitstayid
ORDER BY patientunitstayid
)

, pao2fio2divide as
(
  -- join blood gas to ventilation durations to determine if patient was vent
  select bg.patientunitstayid, chartoffset, pao2/fio2 AS pao2fio2
  , case when vd.patientunitstayid is not null then 1 else 0 end as isvent
  from bg_firstday bg
  left join vent_durations vd
    on bg.patientunitstayid = vd.patientunitstayid
    and bg.chartoffset >= vd.vent_start_delta
    and bg.chartoffset <= vd.vent_stop_delta
  order by bg.patientunitstayid, bg.chartoffset
)

, pao2fio2values AS (
  select patientunitstayid
  , min(case when isvent = 0 then pao2fio2 else null end) as pao2fio2_novent_min
  , min(case when isvent = 1 then pao2fio2 else null end) as pao2fio2_vent_min
  from pao2fio2divide
  group by patientunitstayid
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


-- Compute SOFA Score
-- Aggregate the components for the score
, scorecomp as
(
select cohort.patientunitstayid
  , cohort.uniquepid	
  , cohort.icustay_expire_flag
  , cohort.hospital_expire_flag
  , cohort.age
  , bloodpressure.meanbp_min
  , vaso_maxvalues.max_norepinephrine as rate_norepinephrine_max
  , vaso_maxvalues.max_epinephrine as rate_epinephrine_max
  , vaso_maxvalues.max_dopamine as rate_dopamine_max
  , vaso_maxvalues.max_dobutamine as rate_dobutamine_max
  , labvars.creatinine_max
  , labvars.bilirubin_max
  , labvars.platelet_min
  , pao2fio2values.pao2fio2_novent_min
  , pao2fio2values.pao2fio2_vent_min
  , merged_uo.urineoutput
  , minimal_gcs.gcs_min AS mingcs

FROM cohort
LEFT JOIN labvars ON cohort.patientunitstayid = labvars.patientunitstayid
LEFT JOIN bloodpressure ON cohort.patientunitstayid = bloodpressure.patientunitstayid
LEFT JOIN pao2fio2values ON cohort.patientunitstayid = pao2fio2values.patientunitstayid
LEFT JOIN vaso_maxvalues ON cohort.patientunitstayid = vaso_maxvalues.patientunitstayid
LEFT JOIN minimal_gcs ON cohort.patientunitstayid = minimal_gcs.patientunitstayid
LEFT JOIN merged_uo ON cohort.patientunitstayid = merged_uo.patientunitstayid
)

, scorecalc as
(
  -- Calculate the final score
  -- note that if the underlying data is missing, the component is null
  -- eventually these are treated as 0 (normal), but knowing when data is missing is useful for debugging
  select uniquepid
  , patientunitstayid
  , icustay_expire_flag
  , hospital_expire_flag
  , age
  , meanbp_min
  , rate_dobutamine_max
  , rate_dopamine_max
  , rate_epinephrine_max
  , rate_norepinephrine_max
  , creatinine_max
  , bilirubin_max
  , platelet_min
  , pao2fio2_novent_min
  , pao2fio2_vent_min 
  , urineoutput
  , mingcs

  -- Respiration
  , case
      when pao2fio2_vent_min   < 100 then 4
      when pao2fio2_vent_min   < 200 then 3
      when pao2fio2_novent_min < 300 then 2
      when pao2fio2_novent_min < 400 then 1
      when coalesce(pao2fio2_vent_min, pao2fio2_novent_min) is null then null
      else 0
    end as respiration

  -- Coagulation
  , case
      when platelet_min < 20  then 4
      when platelet_min < 50  then 3
      when platelet_min < 100 then 2
      when platelet_min < 150 then 1
      when platelet_min is null then null
      else 0
    end as coagulation

  -- Liver
  , case
      -- Bilirubin checks in mg/dL
        when bilirubin_max >= 12.0 then 4
        when bilirubin_max >= 6.0  then 3
        when bilirubin_max >= 2.0  then 2
        when bilirubin_max >= 1.2  then 1
        when bilirubin_max is null then null
        else 0
      end as liver

  -- Cardiovascular
  , case
      when rate_dopamine_max > 15 or rate_epinephrine_max >  0.1 or rate_norepinephrine_max >  0.1 then 4
      when rate_dopamine_max >  5 or rate_epinephrine_max <= 0.1 or rate_norepinephrine_max <= 0.1 then 3
      when rate_dopamine_max >  0 or rate_dobutamine_max > 0 then 2
      when meanbp_min < 70 then 1
      when coalesce(meanbp_min, rate_dopamine_max, rate_dobutamine_max, rate_epinephrine_max, rate_norepinephrine_max) is null then null
      else 0
    end as cardiovascular

  -- Neurological failure (GCS)
  , case
      when (mingcs >= 13 and mingcs <= 14) then 1
      when (mingcs >= 10 and mingcs <= 12) then 2
      when (mingcs >=  6 and mingcs <=  9) then 3
      when  mingcs <   6 then 4
      when  mingcs is null then null
  else 0 end
    as cns

  -- Renal failure - high creatinine or low urine output
  , case
    when (creatinine_max >= 5.0) then 4
    when  urineoutput < 200 then 4
    when (creatinine_max >= 3.5 and creatinine_max < 5.0) then 3
    when  urineoutput < 500 then 3
    when (creatinine_max >= 2.0 and creatinine_max < 3.5) then 2
    when (creatinine_max >= 1.2 and creatinine_max < 2.0) then 1
    when coalesce(urineoutput, creatinine_max) is null then null
  else 0 end
    as renal
  from scorecomp
)
, score AS (
select uniquepid, patientunitstayid, hospital_expire_flag, icustay_expire_flag
  , meanbp_min
  , rate_dobutamine_max
  , rate_dopamine_max
  , rate_epinephrine_max
  , rate_norepinephrine_max
  , creatinine_max
  , bilirubin_max
  , platelet_min
 -- , platelet_max
  , urineoutput
  , pao2fio2_novent_min
  , pao2fio2_vent_min
  , mingcs
    -- Calculate the final SOFA score
  , COALESCE(s.respiration, 0) +
  COALESCE(s.coagulation, 0) +
  COALESCE(s.liver, 0) +
  COALESCE(s.cardiovascular, 0) +
  COALESCE(s.cns, 0) +
  COALESCE(s.renal, 0) AS sofa_score
FROM scorecalc AS s
order by uniquepid, patientunitstayid
)
-- We use results of this paper for SOFA Score:
-- Vincent, Jean-Louis MD, PhD, FCCM; de Mendonca, Arnaldo MD; Cantraine, Francis MD; Moreno, Rui MD; Takala, Jukka MD, PhD; Suter, Peter M. MD, FCCM; Sprung, Charles L. MD, JD, FCCM; Colardyn, Francis MD; Blecher, Serge MD. Use of the SOFA score to assess the incidence of organ dysfunction/failure in intensive care units: Results of a multicenter, prospective study. Critical Care Medicine 26(11):p 1793-1800, November 1998.
-- https://journals.lww.com/ccmjournal/Fulltext/1998/11000/Use_of_the_SOFA_score_to_assess_the_incidence_of.16.aspx
-- To find exact numerical values, interactive calculator produces them:
-- https://clincalc.com/IcuMortality/SOFA.aspx

SELECT score.*, 
-- AUC Derived Prob
CASE WHEN sofa_score > 5 THEN 1 ELSE 0 END AS sofa_prob, 
CASE WHEN sofa_score = 0 THEN 0.033
WHEN sofa_score = 1 THEN 0.058
WHEN sofa_score = 2 THEN 0.038
WHEN sofa_score = 3 THEN 0.033
WHEN sofa_score = 4 THEN 0.070
WHEN sofa_score = 5 THEN 0.100
WHEN sofa_score = 6 THEN 0.045

WHEN sofa_score = 7 THEN 0.153
WHEN sofa_score = 8 THEN 0.225
WHEN sofa_score = 9 THEN 0.225

WHEN sofa_score = 10 THEN 0.458
WHEN sofa_score = 11 THEN 0.400
WHEN sofa_score = 12 THEN 0.458

WHEN sofa_score = 13 THEN 0.600
WHEN sofa_score = 14 THEN 0.515

WHEN sofa_score = 15 THEN 0.820
-- The mortality rate was > 90% in patients whose maximum SOFA score was >15.
WHEN sofa_score > 15 THEN 0.901
ELSE NULL
END AS sofa_emp_prob, 
-- Binned SOFA score (use max for each cluster)
CASE WHEN sofa_score <= 6 THEN 0.100
WHEN sofa_score <= 9 THEN 0.225
WHEN sofa_score <= 12 THEN 0.458
WHEN sofa_score <= 14 THEN 0.600
WHEN sofa_score = 15 THEN 0.820
WHEN sofa_score > 15 THEN 0.901
ELSE NULL
END AS sofa_binned_emp_prob
FROM score