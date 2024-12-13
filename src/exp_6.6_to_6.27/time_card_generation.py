import argparse
import os
import time

import pandas as pd
from mimic_pipeline.feature import BinBinarizer
from mimic_pipeline.model import FasterRisk

REPLACE_DICT = {
    'preiculos': 'Pre-ICU LOS',
    'age': 'Age', 
    'gcs_min': 'Min GCS', 
    'mechvent': 'Ventilation', 
    'urineoutput': 'Urine Output', 
    
    'heartrate_min': 'Min Heart Rate',
    'heartrate_max': 'Max Heart Rate',
    
    'meanbp_min': 'Min MBP',
    'meanbp_max': 'Max MBP',
    
    'resprate_min': 'Min Respiratory Rate',
    'resprate_max': 'Max Respiratory Rate',
    
    'tempc_min': 'Min Temperature',
    'tempc_max': 'Max Temperature',
    
    'sysbp_min': 'Min SBP',
    'sysbp_max': 'Max SBP',
    
    'bun_min': 'Min BUN',
    'bun_max': 'Max BUN',
    
    'wbc_min': 'Min WBC',
    'wbc_max': 'Max WBC',
    
    'potassium_min': 'Min Potassium',
    'potassium_max': 'Max Potassium',
    
    'sodium_min': 'Min Sodium',
    'sodium_max': 'Max Sodium',
    
    'bicarbonate_min': 'Min Bicarbonate',
    'bicarbonate_max': 'Max Bicarbonate',
    
    'bilirubin_min': 'Min Bilirubin',
    'bilirubin_max': 'Max Bilirubin',
    
    'hematocrit_min': 'Min Hematocrit',
    'hematocrit_max': 'Max Hematocrit',
    
    'creatinine_min': 'Min Creatinine',
    'creatinine_max': 'Max Creatinine',
    
    'albumin_min': 'Min Albumin',
    'albumin_max': 'Max Albumin',
    
    'glucose_max': 'Max Glucose',
    'glucose_min': 'Min Glucose',
    
    'aids': 'AIDS/HIV',
    'hem': 'Hematologic Cancer',
    'mets': 'Metastatic Cancer',
    
    'electivesurgery': 'Elective Surgery',
    'pao2fio2_vent_min': 'Min P/F Ratio',
    'admissiontype': 'Admission Type',
    
    'pao2_max': 'Max PaO2',
    'pao2_min': 'Min PaO2',
    
    'paco2_max': 'Max PaCO2',
    'paco2_min': 'Min PaCO2',
    
    'ph_min': 'Min pH',
    'ph_max': 'Max pH',
    
    'aado2_min': 'Min A-aO2',
    'addo2_max': 'Max A-aO2',
}

if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("--m", type=int, required=True)
    prs.add_argument("--k", type=int, default=30)
    prs.add_argument("--lb", type=int, default=-50)
    prs.add_argument("--ub", type=int, default=50)
    prs.add_argument("--gap_tolerance", type=float, default=0.3)
    prs.add_argument("--gp", type=int, required=True)
    args = prs.parse_args()

    entire = pd.read_csv("src/exp_6.6_to_6.27/data/MIMIC-WHOLE.csv")
    entire = entire.rename(columns=REPLACE_DICT)

    print(f"Columns: {entire.columns}\nShape: {entire.shape}\nArguments: {args}")

    X_train, y_train = entire.drop('hospital_expire_flag', axis=1), entire['hospital_expire_flag']
    binarizer = BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)
    X_train, group_idx = binarizer.fit_transform(X_train)

    start = time.time()
    fasterrisk = FasterRisk(
        gap_tolerance=args.gap_tolerance, 
        k=args.k, 
        lb=args.lb, 
        ub=args.ub, 
        select_top_m=args.m,
        group_sparsity=args.gp,
        featureIndex_to_groupIndex=group_idx,
    )
    fasterrisk.fit(X_train, y_train)
    print("Time (mins):", (time.time() - start) / 60)

    os.makedirs("src/exp_6.6_to_6.27/results/time_card", exist_ok=True)
    for model_idx in range(args.m):
        fasterrisk.visualize_risk_card(
            list(X_train.columns), 
            X_train, 
            model_idx=model_idx,
            save_path=f"src/exp_6.6_to_6.27/results/time_card/m={args.m}_gp={args.gp}_model={model_idx}.png"
        )