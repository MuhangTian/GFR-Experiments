# important columns
BASE_COLS = ['subject_id', 'hadm_id', 'icustay_id']
INVARIANT_COLS = ['subject_id', 'hadm_id', 'icustay_id', 'icustay_expire_flag', 'icustay_age_group', 'hospital_expire_flag', 'oasis', 'icu_mort_proba', 'age', 'urineoutput', 'preiculos']
ADM_COLS = ['subject_id', 'hadm_id']
SEED = 474