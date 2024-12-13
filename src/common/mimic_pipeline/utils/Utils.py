import inspect
import logging
import random
import time
from collections import namedtuple
from typing import *

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mimic_pipeline.data import SEED
from numpy.typing import *
from sklearn.experimental import \
    enable_iterative_imputer  # NOTE: this line is necessary for MICE, don't remove!
from sklearn.impute import IterativeImputer

def verbose_print(content: str, verbose: bool):
    if verbose:
        return print(content)
    else:
        return None

def smart_print(content: str, verbose: bool, logger):
    if not verbose:
        return None
    if logger is None:
        print(content)
    else:
        logger.info(content)
        
def adapt_proba(y_prob: ArrayLike) -> ArrayLike:
    if len(y_prob.shape) == 2:        # for some scikit-learn models where probas is 2D
        y_prob = y_prob[:, 1]
    return y_prob

def apply_ops(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, operations: dict, verbose: bool=True, logger=None) -> tuple:
    """
    helper function for fit(), applying the transformations specified in operations
    dictionary onto TRAINING set (with the same order as the order in which operation is passed). 
    The ONLY possible operation applied to TEST set is onehot

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.DataFrame
    X_test : pd.DataFrame
        validation set data, not used except for one-hot.
    operations : dict
        dictionary holding the list of operations to do on TRAINING (or onehot on testing).
    verbose : bool, optional
        whether to print out the operations, by default True

    Returns
    -------
    tuple
        X_train, y_train, X_test, GroupIdxArray
    """
    GroupIdxArray = None
    for key, ops in operations.items():
        if key == "resampling":
            smart_print(f"RESAMPLE using {ops}", verbose, logger)
            X_train, y_train = ops.fit_resample(X_train, y_train)
            smart_print(f"Proportion of positive class after resample: {len(y_train[y_train == 1])/len(y_train)}", verbose, logger)
        elif key == "onehot":
            smart_print(f"ONEHOT using {ops}...", verbose, logger)
            X_train, GroupIdx2 = ops.fit_transform(X_train)
            X_test, GroupIdxArray = ops.transform(X_test)
            if GroupIdxArray is not None and GroupIdxArray is not None:
                assert (GroupIdx2 == GroupIdxArray).all(), "idx is wrong!"  
            assert list(X_train.columns) == list(X_test.columns), "Bad onehotting!"
        elif key == 'standardize':
            smart_print(f"STANDARDIZE using {ops}...", verbose, logger)
            X_train = ops.fit_transform(X_train)
            X_test = ops.transform(X_test)
        elif key == "imputation":
            smart_print(f"IMPUTING using {ops}...", verbose, logger)
            X_train, X_test = run_MICE(X_train, X_test)
        else: 
            raise ValueError("Operation not supported!")
        
    smart_print("OPERATIONS COMPLETE\n", verbose, logger)
    
    return X_train, y_train, X_test, GroupIdxArray


def given_model_get_op(model, OPS:dict, resample: bool=True) -> dict:
    """
    Get operations specific for a given model

    Parameters
    ----------
    model
        the model want to get operations for
    OPS : dict
        dictionary of operations to apply
    resample : bool, optional
        whether to include resampling in the operations, by default True

    Returns
    -------
    dict
        dictionary of operations
    """
    op = OPS[model.__class__.__name__].copy()
    if resample:
        assert "resampling" in op.keys(), "\"resampling\" is not included in OPS in the first place!"
    else:
        if 'resampling' in op.keys():
            del op["resampling"]
    
    return op


def unique(X) -> ArrayLike:
    '''helper to get unique values'''
    if isinstance(X, np.ndarray) or isinstance(X, list):
        result = np.unique(X)
    elif isinstance(X, pd.Series):
        result = X.unique()
    else:
        raise ValueError(f"Unsupported type: {type(X)}")

    return result


def load_yaml(path: str):
    with open(path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    return config


def drop_columns_for_training(X: pd.DataFrame, y: pd.DataFrame, keepScore: bool, removeSedated:bool) -> tuple:
    """
    remove columns that model doesn't need for training and evaluation, such as subject ID, length of stay, sedation dummy...etc
    """
    if removeSedated:
        data = pd.concat([X, y], axis=1)
        data = data[data["is_sedated"] != 1]
        X = data.drop("icustay_expire_flag", axis=1)
        y = data["icustay_expire_flag"]
        
    if not keepScore:
        X = X.drop(["subject_id", "is_sedated", "icu_mort_proba"], axis=1)
    else:
        X = X.drop(["subject_id", "is_sedated"], axis=1)
    
    return X, y


def load_object(path: str):
    """
    To load an object stored using joblib library
    """
    return joblib.load(path)


def get_data(load_path: str, split: str, intoXY: bool=True, removeCol: bool=False, keepScore: bool=False, removeSedated:bool=False) -> tuple or pd.DataFrame:
    df = pd.read_csv(f"{load_path}/{split}.csv")
   
    X = df.drop('icustay_expire_flag', axis=1)
    y = df['icustay_expire_flag']
    if removeCol == True:
        X, y = drop_columns_for_training(X, y, keepScore, removeSedated)
    
    if intoXY:
        return X, y
    else:
        return pd.concat([X, y], axis=1)
    

def load_results(path: str, suffix: str, names: list=["fasterrisk", "adaboost", "ebm", "logreg", "random-forest", "xgboost"]) -> list:
    '''load trained models in results folder'''
    ModelArray = []
    for name in names:
        model = load_object(f"{path}/{name}-{suffix}")
        ModelArray.append(model)
    
    return ModelArray


def assert_dataframes_equal(df1:pd.DataFrame, df2:pd.DataFrame) -> None:
    '''check whether two dataframes are exactly same'''
    pd.testing.assert_frame_equal(df1, df2)
    print("Both frames are EQUAL")


def save_data(data, save_path:str) -> None:
    if isinstance(data, pd.DataFrame):
        filetype = "csv"
        data.to_csv(f"{save_path}.{filetype}")
    elif isinstance(data, np.ndarray) or isinstance(data, dict):
        filetype = "npy"
        np.save(f"{save_path}.{filetype}", data)
    else:
        raise ValueError("Unsupported datatype!")
    
    print(f"Saved as \"{save_path}.{filetype}\"")


def load_data(load_path:str, dtype=None):
    filetype = load_path.split(".")[1]      # assuming we only have one dot
    if filetype == "csv":
        if dtype == None:
            data = pd.read_csv(load_path)
        else:
            data = pd.read_csv(load_path, dtype=dtype)
    elif filetype == "npy":
        data = np.load(load_path, allow_pickle=True)
        try:
            if isinstance(data.item(), dict):
                data = data.item()
        except: pass
    else:
        raise ValueError("Unsupported filetype!")
    
    return data


def plt_save_or_show(save_path:str, dpi:int=200, verbose: bool=True):
    '''save or show plt.plot() visualizations'''
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        if verbose:
            print(f"Figure saved as {save_path}")
        

def merge_all(invariant:pd.DataFrame, DataArray:ArrayLike, same_cols:list) -> pd.DataFrame:
    """
    merge an array of dataframes together
    Parameters
    ----------
    invariant : pd.DataFrame
        dataframe columns which should be included in the union
    DataArray : ArrayLike
        array of pd.DataFrame
    same_cols : list
        columns which needs to be the same across all dataframes, such as hadm_id, icustay_id, and subject_id
    Returns
    -------
    pd.DataFrame
    """
    result = invariant
    for element in DataArray:
        pd.testing.assert_frame_equal(result[same_cols], element[same_cols])
        result = pd.concat([result, element.drop(same_cols, axis=1)], axis=1)
    
    return result


def run_MICE(X_train:pd.DataFrame, X_test:pd.DataFrame=None) -> tuple:
    """
    run MICE imputation based on scikit learn implementation

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple
        X_train, X_test
    """
    assert isinstance(X_train, pd.DataFrame), "must be pd.DataFrame!"
    if X_test is not None:
        assert isinstance(X_test, pd.DataFrame), 'must be a pd.DataFrame!'
        assert list(X_train.columns) == list(X_test.columns), "columns must be the same!"
    columns = list(X_train.columns)
    imputer = IterativeImputer(max_iter=10, random_state=SEED)
    X_train = imputer.fit_transform(X_train)
    if X_test is not None:
        X_test = imputer.transform(X_test)      # NOTE: don't fit transform on test data
    X_train = pd.DataFrame(X_train, columns=columns)
    if X_test is not None:
        X_test = pd.DataFrame(X_test, columns=columns)
    
    if X_test is not None:
        return X_train, X_test
    else:
        return X_train

def get_logger(log_file: str=None):
    logging.basicConfig(
        filename = log_file, 
        format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)
    
    return logging

def check_type_hints(func):
    sig = inspect.signature(func)
    params = sig.parameters

    for param_name, param in params.items():
        if param.annotation is not inspect.Parameter.empty:
            if not isinstance(param.default, type(param.annotation)):
                print(f"Error: Parameter '{param_name}' does not match the type hint.")
        else:
            print(f"Warning: No type hint specified for parameter '{param_name}'.")

def seed_everything(seed_value: int=SEED):
    random.seed(seed_value)
    np.random.seed(seed_value)

def create_id() -> int:
    return int(time.time())