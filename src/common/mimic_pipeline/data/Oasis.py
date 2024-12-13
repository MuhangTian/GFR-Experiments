import collections 

ScoreInfo = collections.namedtuple("ScoreInfo", ["numerical_cols", "bool_cols", "categorical_cols", "score_cols", "label_cols", "proba_cols", "coefs"])

# oasis score
# References:
# - https://journals.lww.com/ccmjournal/fulltext/2013/07000/A_New_Severity_of_Illness_Scale_Using_a_Subset_of.15.aspx
oasis_info = ScoreInfo(numerical_cols=['age', 'preiculos', 'gcs', 'heartrate_min', 'heartrate_max', 'meanbp_min',
                                       'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 'tempc_max', 'urineoutput'],
                       bool_cols=['mechvent', 'electivesurgery'],
                       categorical_cols=[],
                       score_cols=['oasis', 'age_score', 'preiculos_score', 'gcs_score', 'heartrate_score', 'meanbp_score',
                                   'resprate_score', 'temp_score', 'urineoutput_score', 'mechvent_score', 'electivesurgery_score'],
                       label_cols=['hospital_expire_flag', 'icustay_expire_flag'],
                       proba_cols=['hosp_mort_proba', 'icu_mort_proba'],
                       coefs = {"hosp_mort_proba": [-6.1746, 0.1275], 'icu_mort_proba': [-7.4225, 0.1434]})
