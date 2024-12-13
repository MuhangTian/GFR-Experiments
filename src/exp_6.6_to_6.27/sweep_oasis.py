'''
To initiate hyper-parameter tuning using wandb with command line arguments
'''
import argparse

import numpy as np
import pandas as pd
from mimic_pipeline.data import SEED
from mimic_pipeline.feature import BinBinarizer
from mimic_pipeline.model import SHORTHAND
from mimic_pipeline.train import WandbTuner
from mimic_pipeline.utils import given_model_get_op, get_logger
from sklearn.preprocessing import StandardScaler

ml_ops = {
    'imputation': "MICE",
    'standardize': StandardScaler(),
}

OPS = {             # define operations to be applied on each model's training set
    'FasterRisk': {'onehot': BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)},
    'LogisticRegression': {'onehot': BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=False)},      # logistic regression use binarization to have non-linear functions
    'ExplainableBoostingClassifier': ml_ops,
    'RandomForestClassifier': ml_ops,          
    'AdaBoostClassifier': ml_ops,
    'XGBClassifier': ml_ops,
}

np.random.seed(SEED)

def sweep(args, model_name: str, X_train, y_train, resample_ratio: float, oasis: bool, KF: bool, K: int):
    """
    Function to start hyper-parameter sweep using wandb

    Parameters
    ----------
    model_name : str
        name of model, {"fasterrisk", "logreg", "ebm", "adaboost", "xgboost", "random forest"}
    disease : str
        {"all", "heart attack", "heart failure", "sepsis"}
    resample_ratio : float
        resampling ratio, if don't want to do resampling, set it to -1
    oasis : bool
        whether to record training and validation metrics for oasis
    KF : bool
        whether to do KFold
    K : int
        number of folds
    """
    logger = get_logger()
    model = SHORTHAND[model_name]
    tuner = WandbTuner(model=model, wdb=True, logger=logger)
    logger.info(f"{'*'*50} MODEL: {model().__class__.__name__} {'*'*50}")
    logger.info(X_train.columns)
    logger.info(X_train.shape)
    if args.path is not None:
        logger.info(f"Path to tuning data: {args.path}")
    resample_ratio = False if resample_ratio == -1 else True
    ops = given_model_get_op(model(), OPS, resample=resample_ratio)
    tuner.fit(X_train, y_train, operations=ops, KF=KF, K=K, oasis=oasis)

if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("--model_name", dest="model_name", default='logreg')
    prs.add_argument("--disease", dest="disease", default='all')
    prs.add_argument("--resample_ratio", dest="resample_ratio", type=float, default=-1)
    prs.add_argument("--oasis", dest="oasis", action=argparse.BooleanOptionalAction, default=False)
    prs.add_argument("--KF", dest="KF", action=argparse.BooleanOptionalAction, default=True)
    prs.add_argument("--K", dest='K', type=int, default=5)
    prs.add_argument("--fold", dest='fold', type=int, default=None)
    args, unknown = prs.parse_known_args()
    
    # ----------------------------------------- DATA SOURCE -------------------------------------------
    # train = pd.read_csv('src/exp_6.6_to_6.27/data/TRAIN-union-features.csv')
    # train = train[[     # for OASIS features
    #     'hospital_expire_flag', 'heartrate_min', 'heartrate_max', 'meanbp_min', 'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 'tempc_max',
    #     'urineoutput', 'mechvent', 'electivesurgery', 'age', 'gcs_min', 'preiculos'
    # ]]
    if args.fold is not None:
        args.path = f"src/exp_6.6_to_6.27/data/k-fold/TRAIN-union-features-fold{args.fold}.csv"
        train = pd.read_csv(args.path)
        train = train[[     # for OASIS features
            'hospital_expire_flag', 'heartrate_min', 'heartrate_max', 'meanbp_min', 'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 'tempc_max',
            'urineoutput', 'mechvent', 'electivesurgery', 'age', 'gcs_min', 'preiculos'
        ]]
    else:
        args.path = f"src/exp_6.6_to_6.27/data/k-fold/TRAIN-union-features.csv\nsrc/exp_6.6_to_6.27/data/TEST-union-features.csv"
        train = pd.read_csv('src/exp_6.6_to_6.27/data/TRAIN-union-features.csv')
        test = pd.read_csv('src/exp_6.6_to_6.27/data/TEST-union-features.csv')
        train = pd.concat([train, test], axis=0)            # use entire MIMIC-III dataset to tune parameter for out of distribution evaluation
        train = train[[     # for OASIS features
            'hospital_expire_flag', 'heartrate_min', 'heartrate_max', 'meanbp_min', 'meanbp_max', 'resprate_min', 'resprate_max', 'tempc_min', 'tempc_max',
            'urineoutput', 'mechvent', 'electivesurgery', 'age', 'gcs_min', 'preiculos'
        ]]
    X_train, y_train = train.drop('hospital_expire_flag', axis=1), train['hospital_expire_flag']
    # ------------------------------------------------------------------------------------------------
    
    sweep(
        args=args,
        X_train=X_train, y_train=y_train,
        model_name=args.model_name,
        resample_ratio=args.resample_ratio,
        oasis=args.oasis,
        KF=args.KF,
        K=args.K,
    )