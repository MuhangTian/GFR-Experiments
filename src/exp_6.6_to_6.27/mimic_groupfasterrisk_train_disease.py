import argparse

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
    prs.add_argument("--disease", dest="disease", type=str)
    args = prs.parse_args()
    
    utils.seed_everything()
    
    binarizer = mmp.feature.BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)
    entire = pd.read_csv(f"src/exp_6.6_to_6.27/data/mimic-disease/{args.disease}-union-features.csv")
    X_train, y_train = entire.drop('hospital_expire_flag', axis=1), entire['hospital_expire_flag']
    X_train, group_idx = binarizer.fit_transform(X_train)
    joblib.dump(binarizer, f'src/exp_6.6_to_6.27/models/disease/{args.disease}/fasterrisk-{args.group_sparsity}-binarizer')       # save the model
    print(f"Binarizer saved as src/exp_6.6_to_6.27/models/disease/{args.disease}/fasterrisk-{args.group_sparsity}-binarizer")
    
    fasterrisk_dict = {
        '10': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=10, k=40, lb=-70, select_top_m=1, ub=30, featureIndex_to_groupIndex=group_idx),
        '14': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=14, k=50, lb=-90, select_top_m=1, ub=90, featureIndex_to_groupIndex=group_idx),
        '15': mmp.model.FasterRisk(gap_tolerance=0.3, group_sparsity=15, k=50, lb=-50, select_top_m=1, ub=70, featureIndex_to_groupIndex=group_idx),
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
    
    mmp.evaluate.train_evaluate_model(
        fasterrisk, X_train, y_train, X_test=None, y_test=None,
        save_path=f'src/exp_6.6_to_6.27/models/disease/{args.disease}/fasterrisk-{args.group_sparsity}', card_path=f'src/exp_6.6_to_6.27/models/disease/{args.disease}/fasterrisk-{args.group_sparsity}',
        card_title=f'Risk Score Card with Group Sparsity of {args.group_sparsity} For {args.disease} Patients',
        # wandb=wandb,
    )