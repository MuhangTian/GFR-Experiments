'''
Pipeline to standardize hyper-parameter tuning
'''
import io
import sys
import time

import joblib
import numpy as np
import pandas as pd
from mimic_pipeline.data import SEED
from mimic_pipeline.feature import BinBinarizer
from mimic_pipeline.model.CommonModels import *
from mimic_pipeline.utils import apply_ops, get_logger, load_yaml
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)
from tqdm import tqdm
from xgboost import XGBClassifier

import wandb

# NOTE: DO NOT SET np.random.seed(474) inside MODULES, this should be done at the top level

class Tuner:
    """
    General tuner for all models with scikit-learn interface (basically a wrapper for GridSearchCV)
    
    Parameters
    ----------
    model
        the model you would like to tune
    config_path : str
        path to the .yaml file containing the setting of gridsearch
    save_path : str
        path to where you want the end result to be stored at; an instance of GridSearchCV()
        is saved at this path
    """
    def __init__(self, model, config_path: str, save_path: str) -> None:
        self.config = load_yaml(config_path)
        self.model = model
        self.save_path = save_path
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: int=3):
        if self.config['mode'] == 'KFold':
            cv = KFold(n_splits=self.config['csv'])
        elif self.config['mode'] == 'StratefiedKFold':
            cv = StratifiedKFold(n_splits=self.config['cv'])
        else: pass
        gs = GridSearchCV(
            estimator=self.model,
            param_grid=self.config['parameters'],
            scoring=self.config['scoring'],
            n_jobs=self.config['n_jobs'],
            refit=self.config['refit'],
            cv=cv,
            verbose=verbose,
        )
        gs.fit(X, y)
        joblib.dump(gs, self.save_path)
        print('DONE')
        

class WandbTuner():
    """
    Tuner class to perform hyper-parameter tuning using Weights & Biases.
    This tuner is only useful if we want to speed up tuning using CS cluster, speed up
    can be achieved by running 30-50 nodes (computers) to tune the hyper-parameter combinations
    at the same time (so we are tuning 30-50 different combinations at the same time), thus approximately
    30-50 times faster.
    
    Parameters
    ----------
    model : Object
        the model you want to tune that obeys Scikit-learn API
    entity : str
        name of the entity for your wandb platform, default to "dukeds-mimic2023"
    wdb : bool
        whether to allow wandb interface, default to True
    
    Example
    -------
    >>> from imblearn.over_sampling import SMOTE
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> tuner = WandbTuner(model=AdaBoostClassifier, wdb=True)
    >>> X_train, y_train = get_data(disease='all', mode='TRAIN')
    >>> operations = {
    >>>    'resampling' : SMOTE(random_state=474, sampling_strategy=1),
    >>>    'onehot': IntervalBinarizer(interval_width=2, categorical_cols=['mechvent', 'electivesurgery'])}
    >>> tuner.fit(X_train, y_train, operations=operations, KF=True)
    """
    def __init__(self, model, wdb: bool, logger: None):
        if wdb: wandb.init()
        self.config = wandb.config
        self.model = model
        self.logger = logger
    
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, KF: bool=False, K: int=5, 
            operations: dict={}, mode: str='StratefiedKFold', oasis: bool=False) -> None:
        """
        Boilerplate code to perform hyper-parameter tuning and log results to Weights & Biases (https://wandb.ai/site)
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.DataFrame
            Training data labels
        KF : bool, optional
            Whether to use KFold, by default False
        K : int, optional
            Number of folds, by default 5
        operations : dict, optional
            operations to perform on the training data, such as scaling, resampline, and onehot, by default {}
        mode : str, optional
            Which mode to use for KFold, either "KFold" or "StratefiedKFold", by default 'StratefiedKFold'
        oasis : bool, optional
            Whether data contains oasis probability, by default False
        Example
        -------
        >>> tuner = WandbTuner(model=FasterRisk)    # do not initialize model here
        >>> X_train, y_train = get_data(disease='all', mode='TRAIN')    # or whatever data you have
        >>> operations = {  # initialize inside the dictionary
        >>>    'resampling' : SMOTE(random_state=474, resampling_strategy=1),
        >>>    'onehot': IntervalBinarizer(interval_width=1, categorical_cols=['mechvent', 'electivesurgery'], whether_interval=False),
        >>> }
        >>> tuner.fit(X_train, y_train, operations=operations)
        """
        assert type(operations) == dict, "'operations' must be a dictionary"
        assert isinstance(X, pd.DataFrame) and (isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)), "X, y must be pd.DataFrame!"
        assert type(KF) == bool, "'KF' must be boolean!"
        assert type(K) == int, "'K' must be integer!"
        assert mode in ['KFold', 'StratefiedKFold'], "'mode' must be either 'KFold' or 'StratefiedKFold'!"
        
        self.ops = list(operations.keys())     # list for all operations
        self.logger.info(f"OPERATIONS: {operations}\n")
        if KF == True:
            self.logger.info(f"{'='*45} START TUNING {'='*45}")
            start_time = time.time()
            
            kf = self.__mode_selection(mode, K)
            y, negative_label = self.__compatibility_adjustments(y)

            auc_train_total, pr_train_total, auc_val_total, pr_val_total = [], [], [], []
            auc_train_oasis_total, pr_train_oasis_total, auc_val_oasis_total, pr_val_oasis_total = [], [], [], []
            
            for counter, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                self.logger.info(f"{'+'*30} FOLD {counter} {'+'*30}")
                # ------------------------------------------ PREPARATIONS ----------------------------------------
                X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
                
                if oasis:   # if true, get oasis probabilities from training and validation set, and seperate them from the sets
                    oasis_prob_train, oasis_prob_test = X_train["icu_mort_proba"].to_numpy(), X_test["icu_mort_proba"].to_numpy()
                    X_train, X_test = X_train.drop("icu_mort_proba", axis=1), X_test.drop("icu_mort_proba", axis=1)
                    y_train_oasis = y_train.to_numpy()     # record y_train since it's modified in resampling
                
                if isinstance(operations['onehot'], BinBinarizer) and 'interval_width' in dict(self.config).keys():
                    self.config = dict(self.config)
                    operations['onehot'] = BinBinarizer(interval_width=self.config['interval_width'], whether_interval=False, group_sparsity=True)
                    self.config.pop('interval_width')
                
                X_train, y_train, X_test, GroupIdx = apply_ops(X_train, y_train, X_test, operations, logger=self.logger)
                names = list(X_train.columns)
                self.logger.info(f"Training set size: {X_train.shape}")
                self.logger.info(f"Proportion of validation set: {round(100*len(X_test)/(len(X_train)+len(X_test)), 3)}%")
                
                if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
                    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
                
                if isinstance(self.model(), FasterRisk):        # if FasterRisk, add GroupIdx as the argument
                    self.config = dict(self.config)             # due to the nature of wandb, turn to dict first
                    if 'group_sparsity' in self.config.keys():
                        self.config['featureIndex_to_groupIndex'] = GroupIdx
                        assert (self.config['featureIndex_to_groupIndex'] == GroupIdx).all(), 'GroupIdx is wrong!'
                        self.logger.info("######### WITH GROUP SPARSITY FOR FASTERRISK #########")
                    
                model = self.model(**self.config)       # initialize model
                
                self.logger.info('Training...')
                a = time.time()
                model.fit(X_train, y_train)
                self.logger.info(f"FINISH, timke taken to train: {(time.time()-a)/60:.2f} minutes\n")
                if isinstance(model, FasterRisk):
                    self.record_fasterrisk(model, names=names, X_train=X_train)

                # ------------------------------------------ TRAINING METICS ----------------------------------------
                y_train_prob = model.predict_proba(X_train)     # need proba in all cases
                if len(y_train_prob.shape) == 2:        # for some scikit-learn models where probas is 2D
                    y_train_prob = y_train_prob[:, 1]
                auc_train = roc_auc_score(y_train, y_train_prob)
                precision, recall, _ = precision_recall_curve(y_train, y_train_prob)
                pr_train = auc(recall, precision)
                
                if oasis:
                    auc_train_oasis = roc_auc_score(y_train_oasis, oasis_prob_train)
                    precision, recall, _ = precision_recall_curve(y_train_oasis, oasis_prob_train)
                    pr_train_oasis = auc(recall, precision)
                
                # ------------------------------------------ VALIDATION METICS ----------------------------------------
                y_val_prob = model.predict_proba(X_test)
                if len(y_val_prob.shape) == 2:          # for some scikit-learn models where probas is 2D
                    y_val_prob = y_val_prob[:, 1]
                auc_val = roc_auc_score(y_test, y_val_prob)
                precision, recall, _ = precision_recall_curve(y_test, y_val_prob)
                pr_val = auc(recall, precision)
                
                if oasis:
                    auc_val_oasis = roc_auc_score(y_test, oasis_prob_test)
                    precision, recall, _ = precision_recall_curve(y_test, oasis_prob_test)
                    pr_val_oasis = auc(recall, precision)
                
                # ------------------------------------------ RECORD METICS ----------------------------------------
                auc_train_total.append(auc_train)
                pr_train_total.append(pr_train)
                auc_val_total.append(auc_val)
                pr_val_total.append(pr_val)
                
                if oasis:
                    auc_train_oasis_total.append(auc_train_oasis)
                    pr_train_oasis_total.append(pr_train_oasis)
                    auc_val_oasis_total.append(auc_val_oasis)
                    pr_val_oasis_total.append(pr_val_oasis)

            self.logger.info(f"\nFINISHED, time taken: {(time.time()-start_time)/60} minutes")
            
            if oasis:
                wandb.log({     # log everything
                    "Mean Training AUROC": round(np.mean(auc_train_total), 3),
                    "Std Training AUROC": round(np.std(auc_train_total), 3),
                    
                    "Mean Training AUPRC": round(np.mean(pr_train_total), 3),
                    "Std Training AUPRC": round(np.std(pr_train_total), 3),
                    
                    "Mean Validation AUROC": round(np.mean(auc_val_total), 3),
                    "Std Validation AUROC": round(np.std(auc_val_total), 3),
                    
                    "Mean Validation AUPRC": round(np.mean(pr_val_total), 3),
                    "Std Validation AUPRC": round(np.std(pr_val_total), 3),
                    
                    "OASIS Mean Training AUROC": round(np.mean(auc_train_oasis_total), 3),
                    "OASIS Std Training AUROC": round(np.std(auc_train_oasis_total), 3),
                    
                    "OASIS Mean Training AUPRC": round(np.mean(pr_train_oasis_total), 3),
                    "OASIS Std Training AUPRC": round(np.std(pr_train_oasis_total), 3),
                    
                    "OASIS Mean Validation AUROC": round(np.mean(auc_val_oasis_total), 3),
                    "OASIS Std Validation AUROC": round(np.std(auc_val_oasis_total), 3),
                    
                    "OASIS Mean Validation AUPRC": round(np.mean(pr_val_oasis_total), 3),
                    "OASIS Std Validation AUPRC": round(np.std(pr_val_oasis_total), 3),
                })
            else:
                wandb.log({     # log everything
                    "Mean Training AUROC": round(np.mean(auc_train_total), 3),
                    "Std Training AUROC": round(np.std(auc_train_total), 3),
                    
                    "Mean Training AUPRC": round(np.mean(pr_train_total), 3),
                    "Std Training AUPRC": round(np.std(pr_train_total), 3),
                    
                    "Mean Validation AUROC": round(np.mean(auc_val_total), 3),
                    "Std Validation AUROC": round(np.std(auc_val_total), 3),
                    
                    "Mean Validation AUPRC": round(np.mean(pr_val_total), 3),
                    "Std Validation AUPRC": round(np.std(pr_val_total), 3),
                })
                
        else:   # if no kfold, just do train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)   # test is validation set
            self.logger.info(len(y_train[y_train == 1])/len(y_train))
            self.logger.info(len(y_test[y_test == 1])/len(y_test))
            
            # apply operations on training
            X_train, y_train, X_test = self.apply_ops(X_train, y_train, X_test, operations)
            if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
                X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
            self.logger.info('Operations COMPLETE\n')
            
            model = self.model(**self.config)       # initialize model

            if isinstance(model, XGBClassifier):
                y_train[y_train == -1] = 0
                y_test[y_test == -1] = 0
            
            self.logger.info('Fitting...')
            a = time.time()
            model.fit(X_train, y_train)
            self.logger.info(f"Timke taken to fit: {time.time()-a}\n")
            
            y_train_prob = model.predict_proba(X_train)     # need proba in all cases
            if len(y_train_prob.shape) == 2:        # for some scikit-learn models where probas is 2D
                y_train_prob = y_train_prob[:, 1]
            y_train_pred = model.predict(X_train)      
            
            f1_train = f1_score(y_train, y_train_pred)      # training metrics
            auc_train = roc_auc_score(y_train, y_train_prob)
            precision, recall, _ = precision_recall_curve(y_train, y_train_prob)
            pr_train = auc(recall, precision)
            
            y_val_prob = model.predict_proba(X_test)
            if len(y_val_prob.shape) == 2:          # for some scikit-learn models where probas is 2D
                y_val_prob = y_val_prob[:, 1]
            y_val_pred = model.predict(X_test)      # if not tuning, use default
            
            f1_val = f1_score(y_test, y_val_pred)      # validation metrics
            auc_val = roc_auc_score(y_test, y_val_prob)
            precision, recall, _ = precision_recall_curve(y_test, y_val_prob)
            pr_val = auc(recall, precision)
            
            wandb.log({  # log stuff
                "Validation F1" : round(f1_val, 3),
                "Validation AUC": round(auc_val, 3),
                "Validation PR AUC": round(pr_val, 3),
                "Training F1" : round(f1_train ,3),
                "Training AUC": round(auc_train, 3),
                "Training PR AUC": round(pr_train, 3),
            })
    
    def record_fasterrisk(self, model: FasterRisk, names: list, X_train: pd.DataFrame):
        output = io.StringIO()
        sys.stdout = output
        model.print_risk_card(names, X_train, 0)
        sys.stdout = sys.__stdout__
        card_prints = output.getvalue()
        card_img = model.visualize_risk_card(card_prints, score_card_name='', title="Card generated with sweep")
        wandb.log({'Risk Score Card': wandb.Image(card_img, caption='Risk Score Card generated with sweep')})
        model.print_risk_card(names, X_train, 0)        # print again to be logged into wandb
        
        
    def __compatibility_adjustments(self, y: pd.Series) -> tuple:
        """
        to make compatibility adjustments for models, so things still work

        Parameters
        ----------
        y : pd.Series
            label

        Returns
        -------
        tuple
            y, negative_label (specifies what value is negative label)
        """
        negative_label = -1          # default to -1
        if isinstance(self.model(), AdaBoostClassifier):
            self.config = dict(self.config)   # scikit-learn implementation uses another estimator for AdaBoost to control max depth
            self.config["estimator"] = DecisionTreeClassifier(max_depth=self.config["max_depth"])
            self.config.pop("max_depth")
        elif isinstance(self.model(), XGBClassifier):
            y[y == -1] = 0
            negative_label = 0          # XGBoost only allows non-negative labels, so do a switch here
        
        return y, negative_label

    def __mode_selection(self, mode: str, K: int):
        '''choose appropriate mode'''
        if mode == 'StratefiedKFold':
            kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
        elif mode == 'KFold':
            kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
        else: 
            raise ValueError('Invalid mode!')
        self.logger.info(f"KFold Strategy: {kf}")

        return kf