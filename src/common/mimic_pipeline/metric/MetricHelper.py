import sys
import warnings
from collections import namedtuple
from typing import *

import numpy as np
from mimic_pipeline.model import (AdaBoostClassifier,
                                  ExplainableBoostingClassifier, FasterRisk,
                                  LogisticRegression, RandomForestClassifier,
                                  XGBClassifier)
from numpy.typing import ArrayLike
from pympler import asizeof
from scipy.stats import chi2
from sklearn.metrics import (auc, brier_score_loss, precision_recall_curve,
                             roc_curve)

Metrics = namedtuple("Metrics", ['auroc', 'auprc', 'brier', 'precision', 'recall', 'fpr', 'tpr', 'prob_true', 'prob_pred', 'H', 'p_h', 'C', 'p_c', 'smr'])


def HosmerLemeshow(bin_true: ArrayLike, bin_sums: ArrayLike, bin_total: ArrayLike) -> tuple[float, float]:
    """
    Hosmer-Lemeshow goodness of fit test, type of statistics (H or C) depends on how data is binned. C statistics uses deciles, H statistics uses fixed intervals for predicted probabilities
    
    Reference
    ---------
    Lemeshow, S., & Hosmer Jr, D. W. (1982). A review of goodness of fit statistics for use in the development of logistic regression models. American journal of epidemiology, 115(1), 92-106.
    """
    observed, not_observed = bin_true, bin_total - bin_true
    expected, not_expected = bin_sums, bin_total - bin_sums
    nonzero = (expected != 0) & (not_expected != 0)
    h_stat = np.sum( ((observed[nonzero] - expected[nonzero])**2 / expected[nonzero]) + ((not_observed[nonzero] - not_expected[nonzero])**2 / not_expected[nonzero]) )
    p_value = 1 - chi2.cdf(h_stat, len(bin_true)-2)

    return h_stat, p_value

def get_calibration_curve(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int=10, strategy: str='uniform') -> tuple[ArrayLike, ArrayLike, float, float]:
    """
    visualize calibration curve

    Parameters
    ----------
    y_true : ArrayLike
    y_prob : ArrayLike
    n_bins : int, optional
        bins to obtain an estimate, by default 10
    strategy : str, optional
        {'quantile', 'uniform'}, by default 'uniform'

    Returns
    -------
    tuple[ArrayLike, ArrayLike, float, float]
        prob_true, prob_pred, Hosmer-Lemeshow statistic, p_value
    
    Note that Hosmer-Lemeshow statistic depends on how data is binned, if strategy = 'uniform', equivalent to H statistics, if strategy = 'quantile', equivalent to C statistics

    Reference
    ---------
    Lemeshow, S., & Hosmer Jr, D. W. (1982). A review of goodness of fit statistics for use in the development of logistic regression models. American journal of epidemiology, 115(1), 92-106.
    """
    y_prob, y_true = np.asarray(y_prob), np.asarray(y_true)
    assert len(y_prob) == len(y_true), 'y_prob and y must have the same length.'
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")
    labels = list(np.unique(y_true))
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    if labels == [-1, 1]:           # Convert negative label to 0 for Hosmer Lemeshow test
        y_true = np.where(y_true == -1, 0, y_true)
        
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )
    
    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    h_stat, p_value = HosmerLemeshow(bin_true[nonzero], bin_sums[nonzero], bin_total[nonzero])
    
    return prob_true, prob_pred, h_stat, p_value

def get_model_size(trained_model: object) -> int:
    """
    Calculate model size in bytes or complexity (parameters involved for prediction)
    - FasterRisk: number of non-zero elements in multipliers, beta0s, betas (for FasterRisk, features that are not used have coefficients of zero, so only count non-zero coefficients)
    - AdaBoostClassifier/XGBClassifier: number of splits in all weak learners (decision trees) + weights for all weak learners
    - RandomForestClassifier: number of splits in all decision trees
    - LogisticRegression: number of non-zero elements in coefficients and intercept (if there exist one); this is similar to FasterRisk, where features not selected have zero coefficients
    - ExplainableBoostingClassifier: number of steps in all feature functions + intercept (if there exist one); we still count zero coefficients here because they are used in inference and are part of feature functions

    Parameters
    ----------
    trained_model : object
        fitted model

    Returns
    -------
    int
        model size in bytes or complexity
    """
    if isinstance(trained_model, FasterRisk):
        multipliers, beta0s, betas = trained_model.get_model_params()
        model_size =  np.count_nonzero(multipliers[0]) + np.count_nonzero(beta0s[0]) + np.count_nonzero(betas[0])       # use first model with lowest logistic loss since that's the only one used in inference
    elif isinstance(trained_model, AdaBoostClassifier):
        model_size = 0
        for tree in trained_model.estimators_:
            tree = tree.tree_
            model_size += tree.node_count - tree.n_leaves
        model_size += len(trained_model.estimator_weights_)
    elif isinstance(trained_model, RandomForestClassifier):
        model_size = 0
        for tree in trained_model.estimators_:
            tree = tree.tree_
            model_size += tree.node_count - tree.n_leaves
    elif isinstance(trained_model, LogisticRegression):
        model_size = np.count_nonzero(trained_model.coef_) + np.count_nonzero(trained_model.intercept_)
    elif isinstance(trained_model, XGBClassifier):
        trees_df = trained_model.get_booster().trees_to_dataframe()
        model_size = len(trees_df[trees_df['Feature'] != 'Leaf'])
        model_size += len(trees_df['Tree'].unique())            # each tree has a weight of eta, so add this to model size
    elif isinstance(trained_model, ExplainableBoostingClassifier):
        model_size = 0
        for sub_score in trained_model.term_scores_:
            model_size += sub_score.size
        if hasattr(trained_model, 'intercept_'):
            model_size += len(trained_model.intercept_)                                         # add intercept term
    
    return model_size

def compute_all_metrics(y_true: ArrayLike, y_prob: ArrayLike) -> NamedTuple:
    """
    Calculate performance metrics for binary classification tasks

    Parameters
    ----------
    y_true : ArrayLike
        true labels
    y_prob : ArrayLike
        predicted probabilities

    Returns
    -------
    NamedTuple
        namedtuple containing performance metrics, 'auroc', 'auprc', 'brier', 'precision', 'recall', 'fpr', 'tpr', 'prob_true', 'prob_pred', 'H', 'p', 'smr'
        - brier: Brier score
        - prob_true: fraction of positives for calibration curve
        - prob_pred: mean of predicted probabilities for calibration curve
        - H: Hosmer-Lemeshow Chi statistic (H-statistic)
        - p: p-value for Hosmer-Lemeshow test
        - smr: Standardized Mortality Ratio
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    prob_true, prob_pred, h, p_h = get_calibration_curve(y_true, y_prob)
    _, _, c, p_c = get_calibration_curve(y_true, y_prob, strategy='quantile')
    smr = np.sum(y_true.replace({-1: 0})) / np.sum(y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    return Metrics(fpr=fpr, tpr=tpr, auroc=auroc, precision=precision, recall=recall, auprc=auprc, prob_true=prob_true, prob_pred=prob_pred, H=h, p_h=p_h, C=c, p_c=p_c, smr=smr, brier=brier)
    