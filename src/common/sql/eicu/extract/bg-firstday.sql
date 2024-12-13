DROP TABLE IF EXISTS bg_firstday CASCADE;
CREATE TABLE bg_firstday as

-- get blood gas measures
with vw0 as
(
  select
      patientunitstayid
    , labname
    , labresultoffset
    , labresultrevisedoffset
  from lab
  where labname in
  (
        'paO2'
      , 'paCO2'
      , 'pH'
      , 'FiO2'
      , 'anion gap'
      , 'Base Deficit'
      , 'Base Excess'
      , 'PEEP'
  )
  and labresultoffset <= 1440
  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
  having count(distinct labresult)<=1
)
-- get the last lab to be revised
, vw1 as
(
  select
      lab.patientunitstayid
    , lab.labname
    , lab.labresultoffset
    , lab.labresultrevisedoffset
    , lab.labresult
    , ROW_NUMBER() OVER
        (
          PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
          ORDER BY lab.labresultrevisedoffset DESC
        ) as rn
  from lab
  inner join vw0
    ON  lab.patientunitstayid = vw0.patientunitstayid
    AND lab.labname = vw0.labname
    AND lab.labresultoffset = vw0.labresultoffset
    AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
  WHERE
     (lab.labname = 'paO2' and lab.labresult >= 15 and lab.labresult <= 720)
  OR (lab.labname = 'paCO2' and lab.labresult >= 5 and lab.labresult <= 250)
  OR (lab.labname = 'pH' and lab.labresult >= 6.5 and lab.labresult <= 8.5)
  OR (lab.labname = 'FiO2' and lab.labresult >= 0.2 and lab.labresult <= 1.0)
  -- we will fix fio2 units later
  OR (lab.labname = 'FiO2' and lab.labresult >= 20 and lab.labresult <= 100)
  OR (lab.labname = 'anion gap' and lab.labresult >= 0 and lab.labresult <= 300)
  OR (lab.labname = 'Base Deficit' and lab.labresult >= -100 and lab.labresult <= 100)
  OR (lab.labname = 'Base Excess' and lab.labresult >= -100 and lab.labresult <= 100)
  OR (lab.labname = 'PEEP' and lab.labresult >= 0 and lab.labresult <= 60)
)
, comb AS (
select
    patientunitstayid
  , labresultoffset as chartoffset
  -- the aggregate (max()) only ever applies to 1 value due to the where clause
  , MAX(case
        when labname != 'FiO2' then null
        when labresult >= 20 then labresult/100.0
      else labresult end) as fio2
  , MAX(case when labname = 'paO2' then labresult else null end) as pao2
  , MAX(case when labname = 'paCO2' then labresult else null end) as paco2
  , MAX(case when labname = 'pH' then labresult else null end) as pH
  , MAX(case when labname = 'anion gap' then labresult else null end) as aniongap
  , MAX(case when labname = 'Base Deficit' then labresult else null end) as basedeficit
  , MAX(case when labname = 'Base Excess' then labresult else null end) as baseexcess
  , MAX(case when labname = 'PEEP' then labresult else null end) as peep
from vw1
group by patientunitstayid, labresultoffset, rn
order by patientunitstayid, labresultoffset, rn
)

  -- https://ccmdb.kuality.ca/index.php?title=AaDO2#:~:text=To%20calculate%20the%20AaDO2%3A,85)%20%2D%2040))%20%3D%20231.5
  -- https://github.com/MIT-LCP/eicu-code/issues/87 (Less applicable formula but relevant)
SELECT patientunitstayid, chartoffset, fio2, pao2, paco2, pH, aniongap, basedeficit, baseexcess, peep, ((fio2 * 713) - pao2) - paco2 as aado2
FROM comb
