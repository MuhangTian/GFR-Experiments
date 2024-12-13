DROP TABLE IF EXISTS disease_flag; CREATE TABLE disease_flag AS
SELECT hadm_id
		-- take MAX since if patient have any of these, it's considered as YES
	, MAX(CASE
		WHEN (icd9_code = '99591') OR (icd9_code = '99592') OR (icd9_code = '78552') OR (icd9_code LIKE '038%') THEN 1
		ELSE 0
		END) AS sepsis

	, MAX(CASE
		WHEN icd9_code LIKE '410%' THEN 1
		ELSE 0
		END) AS AMI

	, MAX(CASE
		WHEN icd9_code LIKE '428%' THEN 1
		ELSE 0
		END) AS heart_failure
	
	, MAX(CASE
		WHEN icd9_code LIKE '584%' THEN 1
		ELSE 0
		END) AS AKF
	
	, MAX(CASE
		WHEN icd9_code LIKE '401%' THEN 1
		ELSE 0
		END) AS hypertension
	
	, MAX(CASE
		WHEN icd9_code LIKE '272%' THEN 1
		ELSE 0
		END) AS hyperlipidemia
	
	, MAX(CASE
		WHEN icd9_code LIKE '157%' THEN 1
		ELSE 0
		END) AS pancreatic_cancer

	FROM diagnoses_icd
GROUP BY hadm_id