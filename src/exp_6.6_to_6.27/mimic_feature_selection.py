import argparse

import joblib
import mimic_pipeline as mmp
import mimic_pipeline.utils as utils
import numpy as np
import pandas as pd
from mimic_pipeline.model import (AdaBoostClassifier, DecisionTreeClassifier,
                                  ExplainableBoostingClassifier, FasterRisk,
                                  LogisticRegression, RandomForestClassifier,
                                  XGBClassifier)
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

MODELS = {
    # "fasterrisk-14": FasterRisk,
    "nonlinear-logreg-l1": LogisticRegression,
    "nonlinear-logreg-l2": LogisticRegression,
    "ebm": ExplainableBoostingClassifier,
    "random-forest": RandomForestClassifier,
    "adaboost": AdaBoostClassifier,
    "xgboost": XGBClassifier,
}

PARAMS_OASIS_PLUS = {   # recorded best hyper-parameters from grid/bayes search to benchmark with OASIS+ approaches
    "fasterrisk-oasis": {
        "fold1": dict(gap_tolerance= 0.3, group_sparsity=14, k=30, lb=-70, select_top_m=1, ub=70),
        "fold2": dict(gap_tolerance= 0.3, group_sparsity=14, k=60, lb=-90, select_top_m=1, ub=70),
        "fold3": dict(gap_tolerance= 0.3, group_sparsity=14, k=60, lb=-90, select_top_m=1, ub=120),
        "fold4": dict(gap_tolerance= 0.3, group_sparsity=14, k=60, lb=-50, select_top_m=1, ub=70),
        "fold5": dict(gap_tolerance= 0.3, group_sparsity=14, k=40, lb=-120, select_top_m=1, ub=50),
    },
    "fasterrisk-14": {
        "fold1": dict(gap_tolerance=0.3, group_sparsity=14, k=50, lb=-70, select_top_m=1, ub=120),
        "fold2": dict(gap_tolerance=0.3, group_sparsity=14, k=40, lb=-30, select_top_m=1, ub=30),
        "fold3": dict(gap_tolerance=0.3, group_sparsity=14, k=30, lb=-50, select_top_m=1, ub=120),
        "fold4": dict(gap_tolerance=0.3, group_sparsity=14, k=50, lb=-90, select_top_m=1, ub=50),
        "fold5": dict(gap_tolerance=0.3, group_sparsity=14, k=40, lb=-50, select_top_m=1, ub=70),
    },
    "nonlinear-logreg-l1": {    # NOTE: these are the non-linear case where binarization is applied
        "fold1": dict(C=0.1, max_iter=1000, penalty="l1", solver="saga"),
        "fold2": dict(C=0.1, max_iter=1000, penalty="l1", solver="saga"),
        "fold3": dict(C=0.1, max_iter=1000, penalty="l1", solver="saga"),
        "fold4": dict(C=0.1, max_iter=1000, penalty="l1", solver="saga"),
        "fold5": dict(C=0.1, max_iter=1000, penalty="l1", solver="saga"),
    },
    "nonlinear-logreg-l2": {
        "fold5": dict(C=0.01, max_iter=1000, penalty="l2", solver="sag"),
        "fold4": dict(C=0.01, max_iter=1000, penalty="l2", solver="saga"),
        "fold3": dict(C=0.01, max_iter=1000, penalty="l2", solver="newton-cg"),
        "fold2": dict(C=0.01, max_iter=1000, penalty="l2", solver="newton-cg"),
        "fold1": dict(C=0.01, max_iter=1000, penalty="l2", solver="saga"),
    },
    "ebm": {
        "fold1": dict(inner_bags=7, interactions=0, learning_rate=0.1, outer_bags=6),
        "fold2": dict(inner_bags=7, interactions=0, learning_rate=0.1, outer_bags=4),
        "fold3": dict(inner_bags=10, interactions=0, learning_rate=0.05, outer_bags=2),
        "fold4": dict(inner_bags=5, interactions=0, learning_rate=0.01, outer_bags=4),
        "fold5": dict(inner_bags=3, interactions=0, learning_rate=0.01, outer_bags=6),
    },
    "random-forest": {
        "fold1": dict(criterion="log_loss", max_depth=16, min_samples_split=10, n_estimators=400, n_jobs=-1),
        "fold2": dict(criterion="entropy", max_depth=16, min_samples_split=10, n_estimators=400, n_jobs=-1),
        "fold3": dict(criterion="entropy", max_depth=16, min_samples_split=8, n_estimators=500, n_jobs=-1),
        "fold4": dict(criterion="log_loss", max_depth=16, min_samples_split=10, n_estimators=250, n_jobs=-1),
        "fold5": dict(criterion="entropy", max_depth=12, min_samples_split=6, n_estimators=500, n_jobs=-1),
    },
    "adaboost": {
        "fold1": dict(learning_rate=0.01, estimator=DecisionTreeClassifier(max_depth=4), n_estimators=250),
        "fold2": dict(learning_rate=0.1, estimator=DecisionTreeClassifier(max_depth=2), n_estimators=180),
        "fold3": dict(learning_rate=0.5, estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50),
        "fold4": dict(learning_rate=0.05, estimator=DecisionTreeClassifier(max_depth=2), n_estimators=300),
        "fold5": dict(learning_rate=0.05, estimator=DecisionTreeClassifier(max_depth=2), n_estimators=400),
    },
    "xgboost": {
        "fold1": dict(eta=0.1, gamma=0.8, max_depth=5, n_estimators=100),
        "fold2": dict(eta=0.1, gamma=0.4, max_depth=5, n_estimators=100),
        "fold3": dict(eta=0.1, gamma=0.6, max_depth=5, n_estimators=150),
        "fold4": dict(eta=0.1, gamma=0.8, max_depth=5, n_estimators=100),
        "fold5": dict(eta=0.1, gamma=0.4, max_depth=5, n_estimators=100),
    },
}

def train_ml_with_fasterrisk_features(model_name, mode, param_dict):
    assert mode in ['oasis', 'fasterrisk']
    stats = {'auroc': [], 'auprc': []}
    model = MODELS[model_name]
    pbar = tqdm(range(1, 6), desc=f"MODE: {mode} | Fold 1 for {model_name}...")
    for fold in pbar:
        train = pd.read_csv(f"src/exp_6.6_to_6.27/data/k-fold/TRAIN-union-features-fold{fold}.csv")
        test = pd.read_csv(f"src/exp_6.6_to_6.27/data/k-fold/TEST-union-features-fold{fold}.csv")
        params = param_dict[f'fold{fold}']
        
        if mode == 'fasterrisk':
            fasterrisk_features = [
                'hospital_expire_flag', 'tempc_max', 'bilirubin_max', 'urineoutput',  'age', 'gcs_min', 'sysbp_min', 'ph_min', 
                'heartrate_max', 'mechvent', 'mets', 'resprate_min', 'bun_max', 'glucose_min', 'pao2_max',
            ]
            train = train[fasterrisk_features]
            test = test[fasterrisk_features]
        elif mode == 'oasis':
            oasis_features = [
                'hospital_expire_flag', 'heartrate_min', 'heartrate_max', 'meanbp_min', 'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 
                'tempc_max', 'urineoutput', 'mechvent', 'electivesurgery', 'age', 'gcs_min', 'preiculos'
            ]
            train = train[oasis_features]
            test = test[oasis_features]
        
        X_train, y_train = train.drop('hospital_expire_flag', axis=1), train['hospital_expire_flag']
        X_test, y_test = test.drop('hospital_expire_flag', axis=1), test['hospital_expire_flag']
        pbar.set_description(desc=f"MODE: {mode} | Fold {fold} for {model_name}, apply operations...")
        
        if isinstance(model(), (FasterRisk, LogisticRegression)):
            binarizer = mmp.feature.BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)
            X_train, group_idx = binarizer.fit_transform(X_train)
            if isinstance(model(), FasterRisk):
                params["featureIndex_to_groupIndex"] = group_idx
            X_test, _ = binarizer.transform(X_test)
        else:
            X_train, X_test = utils.run_MICE(X_train, X_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        init_model = model(**params)
        
        pbar.set_description(desc=f"MODE: {mode} | Fold {fold} for {model_name}, calculating metrics...")
        if isinstance(init_model, XGBClassifier):
            y_train = y_train.replace({-1: 0})
            y_test = y_test.replace({-1: 0})
        
        init_model.fit(X_train, y_train)
        y_prob = utils.adapt_proba(init_model.predict_proba(X_test))
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auroc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auprc = auc(recall, precision)
        
        stats['auroc'].append(auroc)
        stats['auprc'].append(auprc)
    
    joblib.dump(stats, f"src/exp_6.6_to_6.27/results/compare-feature/{mode}-{model_name}")
    
if __name__ == "__main__":
    for mode in ['fasterrisk', 'oasis']:
        for algo in MODELS.keys():
            train_ml_with_fasterrisk_features(algo, mode, PARAMS_OASIS_PLUS[algo])
        