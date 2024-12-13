import argparse
import os
import sys

import mimic_pipeline.utils as utils
import numpy as np
import pandas as pd
from mimic_pipeline.metric import get_calibration_curve
from mimic_pipeline.model import FasterRisk
from mimic_pipeline.feature import BinBinarizer
from sklearn.metrics import (auc, brier_score_loss, precision_recall_curve,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def parse_arguments():
    prs = argparse.ArgumentParser()
    prs.add_argument("--outcome", dest="outcome", type=str, default="akf")
    return prs.parse_args()

def cross_validation(X, y, save_path=None):
    stats = {'auroc': [], 'auprc': [], 'precision': [], 'recall': [], 'h-stat': [], 'fpr': [], 'tpr': [], 'true_prob': [], 'pred_prob': [], 'h-p-value': [], 'c-stat': [], 'c-p-value': [], 'brier': [], "smr": [], 'complexity': []}
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=utils.SEED)
    pbar = tqdm(enumerate(kfold.split(X, y)), desc="Fold 1...")

    for fold, (train_idx, test_idx) in pbar:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        binarizer = BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)
        
        X_train, _ = binarizer.fit_transform(X_train)
        X_test, group_idx_arr = binarizer.transform(X_test)
        
        model = FasterRisk(gap_tolerance=0.3, group_sparsity=40, k=70, lb=-70, ub=70, select_top_m=1, featureIndex_to_groupIndex=group_idx_arr)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auprc = auc(recall, precision)
        prob_true, prob_pred, h_stat, p_h = get_calibration_curve(y_test, y_prob)
        _, _, c_stat, p_c = get_calibration_curve(y_test, y_prob, strategy='quantile')
        brier = brier_score_loss(y_test, y_prob)
        smr = np.sum(y_test.replace({-1: 0})) / np.sum(y_prob)

        stats['auroc'].append(auroc)
        stats['auprc'].append(auprc)
        stats['h-stat'].append(h_stat)
        stats['h-p-value'].append(p_h)
        stats['c-stat'].append(c_stat)
        stats['c-p-value'].append(p_c)
        stats['brier'].append(brier)
        stats['smr'].append(smr)

        pbar.set_description(desc=f"Fold {fold + 1}...")
    
    if save_path is not None:
        terminal = sys.stdout
        sys.stdout = open(f"{save_path}.txt", "wt")
    print(f"AUROC: {np.asarray(stats['auroc']).mean():.3f} $\pm$ {np.asarray(stats['auroc']).std():.3f}")
    print(f"AUPRC: {np.asarray(stats['auprc']).mean():.3f} $\pm$ {np.asarray(stats['auprc']).std():.3f}")
    print(f"Hosmer Lemeshow H stat: {np.asarray(stats['h-stat']).mean():.3f} $\pm$ {np.asarray(stats['h-stat']).std():.3f}")
    print(f"Hosmer Lemeshow C stat: {np.asarray(stats['c-stat']).mean():.3f} $\pm$ {np.asarray(stats['c-stat']).std():.3f}")
    print(f"Brier Score: {np.asarray(stats['brier']).mean():.3f} $\pm$ {np.asarray(stats['brier']).std():.3f}")
    print(f"SMR: {np.asarray(stats['smr']).mean():.3f} $\pm$ {np.asarray(stats['smr']).std():.3f}")
    if save_path is not None:
        sys.stdout = terminal

if __name__ == "__main__":
    args = parse_arguments()
    utils.seed_everything()

    outcome_set = ["sepsis", "pancreatic_cancer", "hyperlipidemia", "hypertension", "akf", "ami", "heart_failure"]
    assert args.outcome in outcome_set, f"Invalid outcome: {args.outcome}"
    data = pd.read_csv(f"src/exp_6.6_to_6.27/data/other_outcomes/mimic.csv")
    X, y = data.drop(["hospital_expire_flag", "hadm_id", "subject_id", "icustay_id", *outcome_set], axis=1), data[args.outcome]

    save_path = "src/exp_6.6_to_6.27/results/other-outcomes"
    os.makedirs(save_path, exist_ok=True)
    cross_validation(X, y.replace({0: -1}), save_path=f"{save_path}/{args.outcome}")

    