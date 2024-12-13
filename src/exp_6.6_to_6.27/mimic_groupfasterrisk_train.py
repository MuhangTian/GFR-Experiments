import argparse
import os

import matplotlib.pyplot as plt
import mimic_pipeline as mmp
import mimic_pipeline.utils as utils
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

import wandb

if __name__ == '__main__':
    prs = argparse.ArgumentParser()
    prs.add_argument("--group_sparsity", dest="group_sparsity", type=str)
    prs.add_argument("--interval_width", dest="interval_width", type=int, default=1)
    prs.add_argument("--save_path", dest="save_path", type=str, default="src/exp_6.6_to_6.27/models")
    prs.add_argument("--log", dest="log", action="store_true")
    prs.add_argument("--train_path", dest="train_path", type=str, default="src/exp_6.6_to_6.27/data/TRAIN-union-features-excluded-cmo.csv")
    prs.add_argument("--test_path", dest="test_path", type=str, default="src/exp_6.6_to_6.27/data/TEST-union-features-excluded-cmo.csv")
    args = prs.parse_args()
    
    utils.seed_everything()
    
    if args.log:
        wandb.init(
            entity='dukeds-mimic2023',
            project='MIMIC (6.15)',
            name=f'FasterRisk-{args.group_sparsity}-all',
            group='FasterRisk MIMIC',
            save_code=True,
        )
    else:
        wandb = None
    
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    
    binarizer = mmp.feature.BinBinarizer(interval_width=args.interval_width, whether_interval=False, group_sparsity=True)
    # X_train, y_train = train.drop('hospital_expire_flag', axis=1), train['hospital_expire_flag']
    # X_test, y_test = test.drop('hospital_expire_flag', axis=1), test['hospital_expire_flag']
    entire = pd.concat([train, test], axis=0)
    X_train, y_train = entire.drop('hospital_expire_flag', axis=1), entire['hospital_expire_flag']
    if args.group_sparsity == 'oasis':
        X_train = X_train[['heartrate_min', 'heartrate_max', 'meanbp_min', 'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 'tempc_max', 'urineoutput', 'mechvent', 'electivesurgery', 'age', 'gcs_min', 'preiculos']]
    
    gfr_path = f'{args.save_path}/fasterrisk-{args.group_sparsity}'
    os.makedirs(gfr_path, exist_ok=True)

    X_train, group_idx = binarizer.fit_transform(X_train)
    joblib.dump(binarizer, f'{gfr_path}/binarizer.joblib')
    print(f"Binarizer saved as {gfr_path}/binarizer.joblib")
    
    fasterrisk_dict = {
        '10': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=10, k=40, lb=-70, select_top_m=1, ub=30, featureIndex_to_groupIndex=group_idx),
        '14': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=14, k=50, lb=-90, select_top_m=1, ub=90, featureIndex_to_groupIndex=group_idx),
        '15': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=50, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        
        '15_k49': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=49, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k48': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=48, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k47': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=47, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k46': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=46, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k45': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=45, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k44': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=44, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k43': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=43, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k42': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=42, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k41': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=41, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_k40': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=40, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '15_remove_55.5': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=50, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        
        '16': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=16, k=50, lb=-90, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '17': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=17, k=50, lb=-90, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '18': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=18, k=50, lb=-90, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '19': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=19, k=50, lb=-30, select_top_m=1, ub=30, featureIndex_to_groupIndex=group_idx),
        '20': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=20, k=60, lb=-90, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '25': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=25, k=80, lb=-70, select_top_m=1, ub=50, featureIndex_to_groupIndex=group_idx),
        '30': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=30, k=100, lb=-70, select_top_m=1, ub=90, featureIndex_to_groupIndex=group_idx),     # NOTE: best among all group sparsity
        '35': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=35, k=80, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        '40': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=40, k=80, lb=-70, select_top_m=1, ub=50, featureIndex_to_groupIndex=group_idx),
        '45': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=45, k=100, lb=-90, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
        'oasis': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=14, k=50, lb=-50, select_top_m=1, ub=120, featureIndex_to_groupIndex=group_idx),
    }

    print(f"Training size: {X_train.shape}")
    print(f"Group sparsity: {args.group_sparsity}")
    fasterrisk = fasterrisk_dict[args.group_sparsity]
    
    if args.group_sparsity == '15_remove_55.5':
        print("***** REMOVING 55.5 *****")
        train = pd.read_csv("src/exp_6.6_to_6.27/data/MIMIC-WHOLE-no-55.5.csv")
        X_train, y_train = train.drop('hospital_expire_flag', axis=1), train['hospital_expire_flag']
    
    mmp.evaluate.train_evaluate_model(
        fasterrisk, X_train, y_train, X_test=None, y_test=None,
        save_path=gfr_path, card_path=gfr_path,
        card_title=f'Risk Score Card with Group Sparsity of {args.group_sparsity}',
        wandb=wandb,
    )