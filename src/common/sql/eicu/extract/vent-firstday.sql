-- Ventiliation: Calculated whether the patient was ventilated ... this should include CPAP for eICU

DROP TABLE IF EXISTS vent_firstday; CREATE TABLE vent_firstday AS 

WITH merged_vent AS (

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

, is_vent AS (
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
 ELSE 0
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
 ELSE 0
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
 ELSE 0
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
 ELSE 0
  END) AS vent_stop_delta,

  MAX(CASE 
  WHEN (event = 'ICU Discharge' ) THEN (hrs*60)
 ELSE 0
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

, comb1 AS (
SELECT patientunitstayid,
MAX(vent_yes) AS vent_yes

FROM union_table 

WHERE vent_start_delta IS NOT NULL
-- Add limitation for first day
AND vent_start_delta <= 1440
GROUP BY patientunitstayid
ORDER BY patientunitstayid
 )
 
SELECT is_vent.patientunitstayid, CASE 
WHEN mechvent = 1 THEN 1
WHEN mechvent = 0 AND vent_yes = 1 THEN 1
ELSE 0
END ventcpap
FROM is_vent
LEFT JOIN comb1 ON is_vent.patientunitstayid = comb1.patientunitstayid;