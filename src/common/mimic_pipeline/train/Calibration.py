import joblib
import mimic_pipeline.utils as utils
import numpy as np
import pandas as pd
from mimic_pipeline.model import FasterRisk
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def fit_calibrator(model, X: pd.DataFrame, y: pd.Series, operations: dict, parameters: dict=None, k: int=5, save_path: str=None) -> IsotonicRegression:
    if isinstance(model(), FasterRisk): assert parameters is not None, "parameters must be provided for FasterRisk!"
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=utils.SEED)
    y_true_kf, y_prob_kf = np.array([]), np.array([])       # for calibration
    
    for counter, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"{'+'*30} FOLD {counter} {'+'*30}")
        X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
        X_train, y_train, X_test, GroupIdx = utils.apply_ops(X_train, y_train, X_test, operations, logger=None)
        
        if isinstance(model(), FasterRisk):
            parameters['featureIndex_to_groupIndex'] = GroupIdx
            init_model = model(**parameters)
        else:
            init_model = model(**parameters)
        
        print(f"Fitting {init_model.__class__.__name__}...")
        init_model.fit(X_train, y_train)
        y_prob = init_model.predict_proba(X_test)
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:, 1]
        y_true_kf = np.concatenate((y_true_kf, y_test))
        y_prob_kf = np.concatenate((y_prob_kf, y_prob))
    
    print(f"{'-'*30} Fitting calibration {'-'*30}")
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    print(calibrator)
    calibrator.fit(y_prob_kf, y_true_kf)
    
    if save_path is not None:
        joblib.dump(calibrator, save_path)
        print(f"Calibrator saved as {save_path}")
    else:
        return calibrator
    
        
        
        