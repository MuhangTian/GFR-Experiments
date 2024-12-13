DROP TABLE IF EXISTS disease_flag CASCADE; CREATE TABLE disease_flag AS
SELECT patientUnitStayID
		-- Consider whether to include information about diagnosisOffset
	, MAX(CASE
		WHEN (icd9code LIKE '%995.92%') OR (icd9code LIKE '%995.91%') OR (icd9code LIKE '%785.52%') OR icd9code LIKE '%038%' THEN 1
		ELSE 0
		END) AS sepsis

	, MAX(CASE
		WHEN icd9code LIKE '%410%' THEN 1
		ELSE 0
		END) AS AMI

	, MAX(CASE
		WHEN icd9code LIKE '%428%' THEN 1
		ELSE 0
		END) AS heart_failure
	
	, MAX(CASE
		WHEN icd9code LIKE '%584%' THEN 1
		ELSE 0
		END) AS AKF

	FROM diagnosis
GROUP BY patientUnitStayID