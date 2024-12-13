import sys
import time

import joblib
import numpy as np
import pandas as pd
from mimic_pipeline.data import SEED
from mimic_pipeline.model.CommonModels import *
from mimic_pipeline.utils import load_object
from numpy.typing import *
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             f1_score, precision_recall_curve, roc_auc_score)

np.random.seed(SEED)

def evaluate_oasis(X_train, y_train, X_test, y_test) -> tuple:
    print("=================== EVALUTE OASIS ===================")
    y_prob = X_test['icu_mort_proba']
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_test, y_prob)
    
    y_prob = X_train['icu_mort_proba']
    precision, recall, _ = precision_recall_curve(y_train, y_prob)
    auprc_train = auc(recall, precision)
    auroc_train = roc_auc_score(y_train, y_prob)
    
    print(f"Train AUROC: {round(auroc_train, 3)} Train AUPRC: {round(auprc_train, 3)}\nTest AUROC: {round(auroc, 3)} Test AUPRC: {round(auprc, 3)}")
    

def train_evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame=None, y_test: pd.Series=None,
                   save_path: str=None, card_path: str=None, card_title: str=None, get_stat: bool=False, wandb=None):
    """
    Train first, and then evaluate trained model

    Parameters
    ----------
    model : object
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test : pd.DataFrame, optional
    y_test : pd.Series, optional
    save_path : str, optional
        path to save the trained model, by default None
    card_path : str, optional
        path to store fasterrisk score card, useful only when model is FasterRisk, by default None
    get_stat : bool, optional
        whether to return stats, by default False
    wandb : bool, optional
        whether to log stats to wandb, by default None

    Returns
    -------
    dict
        return a dictionary of stats if get_stat = True
    """
    if isinstance(model, FasterRisk):         # for FasterRisk where feature names are needed to produce risk cards
        names = list(X_train.columns)
    elif isinstance(model, XGBClassifier):    # for XGBoost where negative labels are not allowed
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    
    start_time = time.time()
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print(f"Time taken to train: {(time.time()-start_time)/60} minutes")
    if save_path is not None:
        joblib.dump(model, f"{save_path}/model.joblib")       # save the model
        print(f"Trained model saved as: \"{save_path}/model.joblib\"")
    
    print(f"Evaluating {model.__class__.__name__}...")
    y_prob_train = model.predict_proba(X_train)
    if len(y_prob_train.shape) == 2:
        y_prob_train = y_prob_train[:, 1]
    
    auroc_train = round(roc_auc_score(y_train, y_prob_train), 3)
    precision, recall, _ = precision_recall_curve(y_train, y_prob_train)
    auprc_train = round(auc(recall, precision), 3)
    
    if X_test is not None and y_test is not None:
        y_prob = model.predict_proba(X_test)
        if len(y_prob.shape) == 2:        # for some scikit-learn models where probas is 2D
            y_prob = y_prob[:, 1]
        
        auroc_test = round(roc_auc_score(y_test, y_prob), 3)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auprc_test = round(auc(recall, precision), 3)
    else:
        auroc_test, auprc_test = 0, 0
    
    print(f"{'*'*15} RESULT {'*'*15} ")
    print(f"Model: {model}\n\nHyper-parameters: {model.get_params()}\n\nTrain AUPRC: {auprc_train} Train AUROC: {auroc_train}\n\nTest AUPRC: {auprc_test} Test AUROC: {auroc_test}")
    if wandb is not None:
        wandb.log({'Train AUPRC': auprc_train, 'Train AUROC': auroc_train, 'Test AUPRC': auprc_test, 'Test AUROC': auroc_test})
    
    if isinstance(model, FasterRisk):       # if model is FasterRisk, store the risk score cards
        store_fasterrisk_cards(model=model, card_path=card_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, names=names, card_title=card_title)
    
    if get_stat:
        return {'Train AUPRC': auprc_train, 'Train AUROC': auroc_train, 'Test AUPRC': auprc_test, 'Test AUROC': auroc_test}


def tune_decision_threshold(X_test: pd.DataFrame, y_test: pd.Series, model=None, load_path:str=None, score:ArrayLike=None) -> tuple:
    if model is None:
        assert isinstance(load_path, str), "must have load path when \"model\" is None!"
        model = load_object(load_path)
    if load_path is None:
        assert model is not None, "must have \"model\" when load path is None!"
    if score is None:
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = score
        
    if len(y_prob.shape) == 2:
        y_prob = y_prob[:,1]
        
    best_threshold, max_f1 = 0, 0
    for threshold in np.arange(0.001, 1.001, 0.001):
        y = np.where(y_prob >= threshold, 1, -1)
        f1 = f1_score(y_test, y)
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold
    
    y_pred = np.where(y_prob >= best_threshold, 1, -1)
    accuracy = accuracy_score(y_test, y_pred)
    
    return best_threshold, accuracy, max_f1


def store_fasterrisk_cards(model: FasterRisk, card_path: str, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series, names: list, K: int=None, card_title: str='Risk Score Card'):
    print(f"\nStoring {K} FasterRisk Risk Card(s)...")
    terminal = sys.stdout
    sys.stdout = open(f"{card_path}/card.txt", "wt")
    if K is not None:
        model.print_topK_risk_cards(names, X_train, y_train, X_test, y_test, K)
    else:
        model.print_risk_card(names, X_train, 0)        # if no top K, use the model with minimum logistic loss    
    sys.stdout = terminal
    print(f".txt risk score card(s) saved at \"{card_path}/card.txt\"")
    
    model.visualize_risk_card(names, X_train, title=card_title, save_path=f"{card_path}/card.png")
    print(f"Risk score card image saved at \"{card_path}/card.png\"")