import argparse
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import mimic_pipeline as mmp
import mimic_pipeline.utils as utils
import numpy as np
import pandas as pd
import seaborn as sns
from mimic_pipeline.feature import BinBinarizer
from mimic_pipeline.model import (AdaBoostClassifier,
                                  ExplainableBoostingClassifier, FasterRisk,
                                  LogisticRegression, RandomForestClassifier,
                                  XGBClassifier)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import wandb

ml_ops = {'imputation': "MICE", 'standardize': StandardScaler()}
OPS = {
    'linear-logreg-l1': ml_ops,
    "linear-logreg-l2": ml_ops,
    'nonlinear-logreg-l1': {"onehot": BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=False)},
    "nonlinear-logreg-l2": {"onehot": BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=False)},
    'ebm': ml_ops,
    'random-forest': ml_ops,
    'adaboost': ml_ops,
    'xgboost': ml_ops
}

MODELS = {
    "linear-logreg-l1": LogisticRegression,
    "linear-logreg-l2": LogisticRegression,
    "nonlinear-logreg-l1": LogisticRegression,
    "nonlinear-logreg-l2": LogisticRegression,
    "ebm": ExplainableBoostingClassifier,
    "random-forest": RandomForestClassifier,
    "adaboost": AdaBoostClassifier,
    "xgboost": XGBClassifier,
}

PARAMS_OASIS_PLUS = {
    "linear-logreg-l1": dict(C=100, max_iter=1200, penalty='l1', solver='saga'),
    "linear-logreg-l2": dict(C=1, max_iter=800, penalty='l2', solver='liblinear'),
    "nonlinear-logreg-l1": dict(C=0.1, max_iter=1000, penalty="l1", solver="liblinear"),
    "nonlinear-logreg-l2": dict(C=0.01, max_iter=1000, penalty="l2", solver="newton-cholesky"),
    "ebm": dict(inner_bags=10, interactions=0,learning_rate=0.1, outer_bags=6),
    "random-forest": dict(criterion='log_loss', max_depth=16, min_samples_split=2, n_estimators=450, n_jobs=-1),
    "adaboost": dict(learning_rate=0.05, estimator=DecisionTreeClassifier(max_depth=3), n_estimators=150),
    "xgboost": dict(eta=0.1, gamma=0.4, max_depth=5, n_estimators=100),
}

PARAMS_UNION_49 = {
    "linear-logreg-l1": dict(C=10000, max_iter=1000, penalty='l1', solver='liblinear'),
    "linear-logreg-l2": dict(C=0.01, max_iter=200, penalty='l2', solver='liblinear'),
    "nonlinear-logreg-l1": dict(C=0.1, max_iter=1000, penalty="l1", solver="liblinear"),
    "nonlinear-logreg-l2": dict(C=0.01, max_iter=1000, penalty="l2", solver="lbfgs"),
    "ebm": dict(inner_bags=10, interactions=3, learning_rate=0.05, outer_bags=2),
    "random-forest": dict(criterion='log_loss', max_depth=20, min_samples_split=2, n_estimators=500, n_jobs=-1),
    "adaboost": dict(learning_rate=0.1, estimator=DecisionTreeClassifier(max_depth=2), n_estimators=250),
    "xgboost": dict(eta=0.1, gamma=0, max_depth=19, n_estimators=400),
}

def train_on_mimic(model, X_train, y_train, param_dict: dict, operations: dict, save_path: str=None, mode: str=None) -> None:
    if mode == 'oasis':
        oasis_features = [
            'heartrate_min', 'heartrate_max', 'meanbp_min', 'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 
            'tempc_max', 'urineoutput', 'mechvent', 'electivesurgery', 'age', 'gcs_min', 'preiculos'
        ]
        X_train = X_train[oasis_features]
    
    print(f"Apply operations...")
    for op_name, op in operations.items():
        if op_name == 'imputation' and op == 'MICE':
            print(f"Imputing with MICE...")
            columns = list(X_train.columns)
            imputer = IterativeImputer(max_iter=10, random_state=utils.SEED)
            X_train = imputer.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns=columns)
            joblib.dump(imputer, f"{save_path}-imputer")
        elif op_name == 'standardize':
            print(f"Standardizing...")
            X_train = op.fit_transform(X_train)
            joblib.dump(op, f"{save_path}-scaler")
        elif op_name == 'onehot':
            print(f"Binarizing...")
            X_train, _ = op.fit_transform(X_train)
            joblib.dump(op, f"{save_path}-binarizer")
        else: 
            raise ValueError(f"Unknown operation: {op_name}")
    init_model = model(**param_dict)
    print(f"Training {init_model.__class__.__name__}...")
    if isinstance(init_model, XGBClassifier):
        y_train = y_train.replace({-1: 0})
    init_model.fit(X_train, y_train)
    joblib.dump(init_model, f"{save_path}")
    print(f"Transformations (operations) and model saved as {save_path}\n")
            

if __name__ == '__main__':
    prs = argparse.ArgumentParser()
    prs.add_argument("--algo", dest="algo", type=str, default='xgboost')
    prs.add_argument("--save_path", dest="save_path", type=str, default='auto')
    prs.add_argument("--mode", dest="mode", type=str, default=None)
    prs.add_argument("--exp", dest="exp", type=str, default='oasis+')
    args = prs.parse_args()
    
    utils.seed_everything()
    train = pd.read_csv('src/exp_6.6_to_6.27/data/TRAIN-union-features.csv')
    test = pd.read_csv('src/exp_6.6_to_6.27/data/TEST-union-features.csv')
    entire = pd.concat([train, test], axis=0)
    X_train, y_train = entire.drop('hospital_expire_flag', axis=1), entire['hospital_expire_flag']
    
    EXP = {
        "union49": PARAMS_UNION_49,
        "oasis+": PARAMS_OASIS_PLUS,
    }
    
    if args.exp == 'oasis+':
        args.mode = 'oasis'
    
    for model_name, model in MODELS.items():
        print(f"Training {model_name}...")
        if args.save_path == 'auto':
            save_path = f"src/exp_6.6_to_6.27/models/{args.exp}/{model_name}"
        train_on_mimic(model, X_train, y_train, param_dict=EXP[args.exp][model_name], operations=OPS[model_name], save_path=save_path, mode=args.mode)