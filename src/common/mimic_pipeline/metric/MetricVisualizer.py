import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from mimic_pipeline.evaluate.Evaluate import tune_decision_threshold
from mimic_pipeline.feature import FeatureNaNPercentage
from mimic_pipeline.utils import (COLOR, apply_ops, given_model_get_op, tfont,
                                  tsfont)
from numpy.typing import *
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

sns.set_theme()     # set global theme

def plot_curve_between_models(models: list, names: list, title: str, metric: str, 
                              X_test: list, y_test: list, COLOR:dict=None,
                              X_train: list=None, y_train: list=None, legend_fontsize:int=10, score_name: str='OASIS',
                              oasis_proba: np.ndarray=None, save_path:str=None, fig_size:tuple=None) -> None:
    sns.set_style("ticks")
    
    if COLOR is None:
        COLOR = {
        "XGBClassifier": ("darkviolet", 1, 0.5),
        "ExplainableBoostingClassifier": ("darkorange", 1, 0.5),
        "RandomForestClassifier": ("green", 1, 0.5),
        "AdaBoostClassifier": ("black", 1, 0.5),
        "LogisticRegression": ("saddlebrown", 1, 0.5),
        score_name: ("blue", 1.5, 1),
        "FasterRisk": ("red", 1.5, 1),
        }
    placeholder = "AUROC" if metric == "roc" else "AUPRC"
    AucArray, NameArray, CoordArray = [], [], []
    pbar = tqdm(range(len(models)))
    for i in pbar:
        model, name = models[i], names[i]
        pbar.set_description(f'Evaluating {name}...')
        if X_train is not None and y_train is not None:
            X_train_tmp, y_train_tmp = X_train[i], y_train[i]
            if isinstance(X_train_tmp, pd.DataFrame):
                X_train_tmp, y_train_tmp = X_train_tmp.to_numpy(), y_train_tmp.to_numpy()
        else:
            X_test_tmp, y_test_tmp = X_test[i], y_test[i]
            if isinstance(X_test_tmp, pd.DataFrame):
                X_test_tmp, y_test_tmp = X_test_tmp.to_numpy(), y_test_tmp.to_numpy()
        
        if X_test is not None:      # do it on test set
            y_prob = model.predict_proba(X_test_tmp)
            y = y_test_tmp
        else:       # if not given, do it on training set
            y_prob = model.predict_proba(X_train_tmp)
            y = y_train_tmp
    
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:,1]
        if metric == 'roc':
            x_num, y_num, _ = roc_curve(y, y_prob)
        elif metric == 'pr':
            y_num, x_num, _ = precision_recall_curve(y, y_prob)
        else: raise ValueError("Only ROC or PR supported!")
        auc_num = auc(x_num, y_num)
        AucArray.append(auc_num)
        NameArray.append(name)
        CoordArray.append((x_num, y_num))
    
    if oasis_proba is not None:     # lastly, if given, plot oasis_proba as well
        y = y_test_tmp
        if metric == 'roc':
            x_num, y_num, _ = roc_curve(y, oasis_proba)
        elif metric == 'pr':
            y_num, x_num, _ = precision_recall_curve(y, oasis_proba)
        auc_num = auc(x_num, y_num)
        AucArray.append(auc_num)
        NameArray.append(score_name)
        CoordArray.append((x_num, y_num))
    
    indices = np.flip(np.argsort(AucArray))
    for idx in tqdm(indices, desc=f'Plotting {metric.upper()} Curves...'):     # plot in descending order of AUC
        x_num, y_num = CoordArray[idx]
        auc_num = AucArray[idx]
        name = NameArray[idx]
        ax = sns.lineplot(
            x=x_num, y=y_num, 
            label=f"{name}, {placeholder}={round(auc_num, 3)}", 
            linewidth=COLOR[name][1], color=COLOR[name][0], alpha=COLOR[name][2], errorbar=None,
        )
        if fig_size is not None:
            ax.figure.set_size_inches(fig_size[0], fig_size[1])
    
    if metric == "roc":
        sns.lineplot(x=np.linspace(0,1), y=np.linspace(0,1), color="grey", linestyle="--", linewidth=1, label="Random")
        plt.xlabel('FPR')
        plt.ylabel("TPR")
        plt.title(f"{metric.upper()} Curves: {title}", **tfont)
    elif metric == "pr":
        plt.xlabel('Recall')
        plt.ylabel("Precision")
        plt.title(f"{metric.upper()} Curves: {title}", **tfont)
    plt.legend(fontsize=legend_fontsize, bbox_to_anchor=(1, 1), loc='upper left')
    
    if save_path is not None:
        plt.savefig(f"{save_path}", dpi=600, bbox_inches="tight")
        print(f"SAVE COMPLETE, saved at {save_path}")
    else:
        plt.show()
        plt.close()
        
    
def plot_curve_for_features(
    title: str, X_train: pd.DataFrame, y_train: pd.DataFrame, save_path:str=None, topK:int=None, negate:bool=False, separate:int=None,
    mode:str="auroc", fig_size=None, X_test: pd.DataFrame=None, y_test: pd.DataFrame=None, fontsize:int=20) -> None: 

    sns.set_style('white') 
    assert isinstance(X_train, pd.DataFrame), "must be pd.DataFrame!"
    assert mode in ["auroc", "auprc"], "only these are allowed!"
    if separate is not None:
        assert topK % separate == 0, "\"topK\" must be divisible by \"separate\"!"
        
    if X_test is not None:
        assert isinstance(X_test, pd.DataFrame), "must be pd.DataFrame!"
        assert list(X_train.columns) == list(X_test.columns), "features don't match!"
        y_test = y_test.to_numpy()
    
    y_train = y_train.to_numpy()
    columns = list(X_train.columns)
    ColumnArray, AucArray, CoordArray = [], [], []
    
    if X_test is not None:
        data_tmp = X_test
        y_actual = y_test
    else:
        data_tmp = X_train
        y_actual = y_train
    
    containNaN = True if data_tmp.isnull().values.any() == True else False
    if containNaN:
        print('Contain NaN! Will plot ROCs using non-NaN values...')
    
    for i in tqdm(range(len(columns)), desc="Processing features..."):     # plot individual features
        col = columns[i]
            
        if containNaN:
            idx = data_tmp[col].notnull()
            feature_score = data_tmp[col][idx]
            y = y_actual[idx]
        else:
            feature_score = data_tmp[col]
            y = y_actual
        
        if mode == "auroc":
            x_num, y_num, _ = roc_curve(y, feature_score)
            tmp_auc = auc(x_num, y_num)
            if negate:
                if tmp_auc < 0.5:
                    x_num, y_num = 1 - x_num, 1 - y_num     # negate the prediction label when worse than random guessing
        elif mode == "auprc":
            y_num, x_num, _ = precision_recall_curve(y, feature_score)
        
        auc_num = auc(x_num, y_num)
        ColumnArray.append(col)
        AucArray.append(auc_num)
        CoordArray.append((x_num, y_num))
    
    print("\n")
    palette = sns.color_palette("tab10", 10)
    index = np.flip(np.argsort(AucArray))       # rank by decreasing order of AUC
    
    if topK is not None:
        index = index[:topK]
        
    pbar = tqdm(range(1, len(index)+1), desc="Plotting...")
    
    for k in pbar:
        idx = index[k-1]
        name = ColumnArray[idx]
        auc_num = AucArray[idx]
        coord = CoordArray[idx]
            
        ax = sns.lineplot(
            x=coord[0], y=coord[1], label=f"{name}, {mode.upper()}={round(auc_num, 3)}", 
            linewidth=1, color=palette[int( (k-1) % 10)], errorbar=None
        )
        if fig_size is not None:
            ax.figure.set_size_inches(fig_size[0], fig_size[1])
        
        if separate is not None:
            if k % int(topK/separate) == 0:
                if mode == "auroc":
                    sns.lineplot(x=np.linspace(0,1), y=np.linspace(0,1), color="grey", linestyle="--", linewidth=1, label="Random")
                
                if mode == "auroc":
                    plt.xlabel('FPR')
                    plt.ylabel("TPR")
                else:
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    
                plt.title(f"Individual Features: {title}", **tfont)
                plt.legend(fontsize=fontsize, bbox_to_anchor=(1, 1), loc='upper left')
                plt.show()
                plt.close()
    
    if separate is None:
        if mode == "auroc":
            sns.lineplot(x=np.linspace(0,1), y=np.linspace(0,1), color="grey", linestyle="--", linewidth=1, label="Random")
        
        if mode == "auroc":
            plt.xlabel('FPR')
            plt.ylabel("TPR")
        else:
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            
        plt.title(f"Individual Features: {title}", **tfont)
        plt.legend(fontsize=fontsize, bbox_to_anchor=(1, 1), loc='upper left')
        
        if save_path is not None:
            plt.savefig(f"{save_path}", dpi=300, bbox_inches="tight")
            print(f"SAVE COMPLETE, saved at {save_path}")
        else:
            plt.show()
            plt.close()

def plot_calibration_between_models(
    models: list, names: list, title: str, 
    X_test: list, y_test: list, COLOR:dict=None,
    X_train: list=None, y_train: list=None, legend_fontsize:int=10, score_name: str='OASIS',
    oasis_proba: np.ndarray=None, save_path:str=None, fig_size:tuple=None, n_bins=None, marker: str='s', strategy: str='uniform',
    ) -> None:
    
    sns.set_style("ticks")
    if isinstance(n_bins, int):
        n_bins = {name: n_bins for name in [*names, score_name]}
    elif n_bins is None:
        n_bins = {name: 10 if name != "AdaBoostClassifier" else 30 for name in [*names, score_name]}           # need more bins for AdaBoost
    else:
        assert isinstance(n_bins, dict), "n_bins must be int or dict or None!"
        assert len(n_bins) == len(names), "n_bins must have same length as names!"
        assert list(n_bins.keys()) == names, "n_bins must have same keys as names!"
    
    if COLOR is None:
        COLOR = {
        "XGBClassifier": ("darkviolet", 1, 0.5),
        "ExplainableBoostingClassifier": ("darkorange", 1, 0.5),
        "RandomForestClassifier": ("green", 1, 0.5),
        "AdaBoostClassifier": ("black", 1, 0.5),
        "LogisticRegression": ("saddlebrown", 1, 0.5),
        score_name: ("blue", 1.5, 1),
        "FasterRisk": ("red", 1.5, 1),
        }
    NameArray, CoordArray = [], []
    pbar = tqdm(range(len(models)))
    for i in pbar:
        model, name = models[i], names[i]
        pbar.set_description(f'Evaluating {name}...')
        if X_train is not None and y_train is not None:
            X_train_tmp, y_train_tmp = X_train[i], y_train[i]
            if isinstance(X_train_tmp, pd.DataFrame):
                X_train_tmp, y_train_tmp = X_train_tmp.to_numpy(), y_train_tmp.to_numpy()
        else:
            X_test_tmp, y_test_tmp = X_test[i], y_test[i]
            if isinstance(X_test_tmp, pd.DataFrame):
                X_test_tmp, y_test_tmp = X_test_tmp.to_numpy(), y_test_tmp.to_numpy()
        
        if X_test is not None:      # do it on test set
            y_prob = model.predict_proba(X_test_tmp)
            y = y_test_tmp
        else:       # if not given, do it on training set
            y_prob = model.predict_proba(X_train_tmp)
            y = y_train_tmp
    
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:,1]
        
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins[name], strategy=strategy)
        NameArray.append(name)
        CoordArray.append((prob_pred, prob_true))
    
    if oasis_proba is not None:     # lastly, if given, plot oasis_proba as well
        y = y_test_tmp
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins[score_name], strategy=strategy)
        NameArray.append(score_name)
        CoordArray.append((prob_pred, prob_true))
    
    for idx in tqdm(range(len(NameArray)), desc=f'Plotting calibration curves...'):
        x_num, y_num = CoordArray[idx]
        name = NameArray[idx]
        ax = sns.lineplot(
            x=x_num, y=y_num, 
            label=f"{name}", marker=marker,
            linewidth=COLOR[name][1], color=COLOR[name][0], alpha=COLOR[name][2], errorbar=None,
        )
        if fig_size is not None:
            ax.figure.set_size_inches(fig_size[0], fig_size[1])
    
    sns.lineplot(x=np.linspace(0,1), y=np.linspace(0,1), color="grey", linestyle="--", linewidth=1, label="Perfectly Calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curves: {title}", **tfont)
    plt.legend(fontsize=legend_fontsize, bbox_to_anchor=(1, 1), loc='upper left')
    
    if save_path is not None:
        plt.savefig(f"{save_path}", dpi=600, bbox_inches="tight")
        print(f"SAVE COMPLETE, saved at {save_path}")
    else:
        plt.show()
        plt.close()


def plot_metric_gender_and_race(model, names: list, title: str, metric: str, X_list: list, y_list: list, 
                                X_test: pd.DataFrame=None, y_test: pd.DataFrame=None, save: bool=False) -> None:
    assert isinstance(X_list, list) and isinstance(y_list, list), "'X' and 'y' must be lists!"
    assert len(names) == len(X_list) == len(y_list), "must have same length!"
    if X_test is not None: assert isinstance(X_test, pd.DataFrame), "test data must be pd.DataFrame!"
    
    if metric == "roc":
        sns.lineplot(x=np.linspace(0,1), y=np.linspace(0,1), color="grey", linestyle="--", linewidth=1, label="Random")
    for i in range(len(X_list)):
        name, X, y = names[i], X_list[i].to_numpy(), y_list[i].to_numpy()
        y_prob = model.predict_proba(X)
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:,1]
        if metric == 'roc':
            x_num, y_num, _ = roc_curve(y, y_prob)
        elif metric == 'pr':
            y_num, x_num, _ = precision_recall_curve(y, y_prob)
        else: raise ValueError("Only ROC or PR supported!")
        auc_num = auc(x_num, y_num)
        sns.lineplot(x=x_num, y=y_num, label=f"{name}, AUC={round(auc_num, 3)}", linewidth=1)
    
    if X_test is not None and y_test is not None:
        y_prob = model.predict_proba(X_test.to_numpy())
        if len(y_prob.shape) == 2:
            y_prob = y_prob[:,1]
        if metric == 'roc':
            x_num, y_num, _ = roc_curve(y_test, y_prob)
        elif metric == 'pr':
            y_num, x_num, _ = precision_recall_curve(y_test, y_prob)
        auc_num = auc(x_num, y_num)
        sns.lineplot(x=x_num, y=y_num, label=f"Entire Test Set, AUC={round(auc_num, 3)}", linewidth=1, color="black")
        
    if metric == "roc":
        plt.xlabel('FPR')
        plt.ylabel("TPR")
        plt.title(f"{metric.upper()} Curve: {title}", **tfont)
    elif metric == "pr":
        plt.xlabel('Recall')
        plt.ylabel("Precision")
        plt.title(f"{metric.upper()} Curve: {title}", **tfont)
    plt.legend(fontsize=10)
    if save == True:
        plt.savefig(f"visuals/oasis/metrics/{metric.upper()}-{title}.png", dpi=200)
        print("SAVE COMPLETE, saved at visuals/oasis/metrics")
    else:
        plt.show()


# def plot_confusion_matrix(models, names, X_train:pd.DataFrame, y_train:pd.DataFrame, X_test: pd.DataFrame, 
#                    y_test: pd.DataFrame, save: bool=False, resample:bool=True, score:ArrayLike=None) -> None:
#     if score is not None:
#         best_threshold, accuracy, best_f1 = tune_decision_threshold(X_test, y_test, "", score=score)
#         y_pred = np.where(score >= best_threshold, 1, -1)
#         mat = confusion_matrix(y_test, y_pred)
#         ax = sns.heatmap(mat/np.sum(mat), annot=True, fmt='.2%', cmap='magma')
#         ax.xaxis.set_ticklabels(["Alive", "Deceased"])
#         ax.yaxis.set_ticklabels(["Alive", "Deceased"])
#         plt.title(f"OASIS, with F1={round(best_f1, 3)}, Accuracy={round(accuracy, 3)}, Best Threshold={round(best_threshold, 3)}", **tsfont)
#         plt.xlabel("Predicted Label")
#         plt.ylabel("True Label")
#         if save == True:
#             plt.savefig(f"visuals/oasis/metrics/ConfusioMatrix-OASIS.png", dpi=200)
#             plt.close()
#         else:
#             plt.show()
        
#     # X, y = check_and_turn_numpy(X, y)
#     for i in tqdm(range(len(models)), desc="Plotting..."):
#         model, name = models[i], names[i]
#         op = given_model_get_op(model, resample)
#         X_train_tmp, y_train_tmp, X_test_tmp = apply_ops(X_train, y_train, X_test, op)
#         y_prob = model.predict_proba(X_test_tmp)
#         best_threshold, accuracy, best_f1 = tune_decision_threshold(X_test_tmp, y_test, model)
#         if len(y_prob.shape) == 2:
#             y_prob = y_prob[:,1]
#         y_pred = np.where(y_prob >= best_threshold, 1, -1)
#         mat = confusion_matrix(y_test, y_pred)
#         ax = sns.heatmap(mat/np.sum(mat), annot=True, fmt='.2%', cmap='magma')
#         ax.xaxis.set_ticklabels(["Alive", "Deceased"])
#         ax.yaxis.set_ticklabels(["Alive", "Deceased"])
#         plt.title(f"{name}, with F1={round(best_f1, 3)}, Accuracy={round(accuracy, 3)}, Best Threshold={round(best_threshold, 3)}", **tsfont)
#         plt.xlabel("Predicted Label")
#         plt.ylabel("True Label")
#         if save == True:
#             plt.savefig(f"visuals/oasis/metrics/ConfusioMatrix-{name}.png", dpi=200)
#             plt.close()
#         else:
#             plt.show()


def plot_roc_curve(X_train:pd.DataFrame, y_train:pd.DataFrame, columns:list, title:str, negate:bool=False):
    X_train = X_train[columns]
    containNaN = True if X_train.isnull().values.any() == True else False
    CoordArray, AucArray, NameArray = [], [], []
    
    for col in columns:
        
        if containNaN:
            idx = X_train[col].notnull()
            feature_score = X_train[col][idx]
            y = y_train[idx]
        else:
            feature_score = X_train[col]
            y = y_train
        
        fpr, tpr, _ = roc_curve(y, feature_score)
        if negate:
            if auc(fpr, tpr) < 0.5:
                fpr, tpr = 1 - fpr, 1 - tpr
        tmp_auc = auc(fpr, tpr)
        
        AucArray.append(tmp_auc)
        CoordArray.append((fpr, tpr))
        NameArray.append(col)
    
    palette = sns.color_palette("tab10", 10)
    
    index = np.flip(np.argsort(AucArray))
    
    for i in range(len(index)):
        idx = index[i]
        name = NameArray[idx]    
        auc_tmp = AucArray[idx]
        x, y = CoordArray[idx]
        
        ax = sns.lineplot(
            x=x, y=y, label=f"{name}, AUROC={round(auc_tmp, 3)}", 
            linewidth=1, color=palette[i], errorbar=None,
        )
        ax.figure.set_size_inches(11, 9)
    
    sns.lineplot(x=np.linspace(0,1), y=np.linspace(0,1), color="grey", linestyle="--", linewidth=1, label="Random")
    plt.xlabel('FPR')
    plt.ylabel("TPR")
    plt.title(f"Individual Features: {title}", **tfont)
    plt.legend(fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
    plt.close()