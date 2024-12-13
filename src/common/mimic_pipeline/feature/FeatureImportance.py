import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mimic_pipeline.data import SEED
from mimic_pipeline.metric.MetricVisualizer import plot_curve_for_features
from mimic_pipeline.model.CommonModels import *
from mimic_pipeline.utils import plt_save_or_show, tfont
from sklearn.inspection import permutation_importance

sns.set_theme()

class FeatureImportance:

    def __init__(self, X_train:pd.DataFrame, y_train:pd.Series, X_test:pd.DataFrame=None, y_test:pd.Series=None) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.names = list(X_train.columns)
    
    def plot_model_reliance(self, model=None, scoring:str=None, save_path:str=None, topK:int=12):
        '''evaluate feature importance by permutation of the individual features'''        
        self.X_train["random"] = np.random.uniform(0, 5000, len(self.X_train))
        self.X_test["random"] = np.random.uniform(0, 5000, len(self.X_test))
        
        if isinstance(model, XGBClassifier):        # XGB is annoying :(
            y_train = self.y_train.replace({-1:0})
            y_test = self.y_test.replace({-1:0})
        else:
            y_train, y_test = self.y_train, self.y_test
        
        names = list(self.X_train.columns)
            
        model.fit(self.X_train, y_train)
        result = permutation_importance(
            model, self.X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1, scoring=scoring,
        )
        importances = np.asarray(result.importances_mean)
        std = np.asarray(result.importances_std)
        
        sorted_idx = np.flip(np.argsort(importances))
        names = np.asarray(names)[sorted_idx]
        std = std[sorted_idx]
        
        if topK is not None:
            data = pd.DataFrame.from_dict({
                "Mean Decrease": np.flip(np.sort(importances))[:topK],
                "Feature Name": names[:topK],
                "std": std[:topK],
            })
        else:
            data = pd.DataFrame.from_dict({
                "Mean Decrease": np.flip(np.sort(importances)),
                "Feature Name": names,
                "std": std,
            })
        
        ax = sns.barplot(data, y="Feature Name", x="Mean Decrease", palette="rocket_r", orient="h")
        ax.errorbar(data=data, y="Feature Name", x="Mean Decrease", xerr="std", ls="", lw=2, color="black", capsize=10, capthick=2)
        ax.figure.set_size_inches(15, 20)
        
        if scoring == "roc_auc":
            prefix = "AUROC"
        elif scoring == None:
            prefix = "loss"
        plt.title(f"Model Reliance for {model.__class__.__name__}, scored by {prefix} on validtion", **tfont)
        
        plt_save_or_show(save_path=save_path)

    def plot_mean_decrease_impurity(self, save_path:str, topK:int=10):
        """
        Plot mean decrease impurity based on random forest

        Parameters
        ----------
        save_path : str
        topK : int, optional, by default 10
        """
        names = list(self.X_train.columns)
        self.X_train["random"] = np.random.uniform(0, 5000, len(self.X_train))
        
        forest = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        forest.fit(self.X_train, self.y_train)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        sorted_idx = np.flip(np.argsort(importances))
        names = np.asarray(names)[sorted_idx]
        std = std[sorted_idx]
        
        data1 = pd.DataFrame.from_dict({
            "Mean Decrease in Impurity": np.flip(np.sort(importances))[:topK],
            "Feature Name": names[:topK],
            "std": std[:topK],
        })
 
        ax = sns.barplot(data1, y="Feature Name", x="Mean Decrease in Impurity", palette="crest", orient="h")
        ax.errorbar(data=data1, y="Feature Name", x="Mean Decrease in Impurity", xerr="std", ls="", lw=2, color="black")
        ax.figure.set_size_inches(20, 8)
        plt.title(f"Top {topK} Features Ranked by MDI", **tfont)
        
        plt_save_or_show(save_path)
    
    
    def plot_curve_per_feature(self, title:str, save_path:str, mode:str, topK=None, filterNaN:bool=False):
        plot_curve_for_features(
            title=title,
            X_train=self.X_train, y_train=self.y_train,
            mode=mode,
            X_test=self.X_test, y_test=self.y_test,
            fig_size=(16, 13),
            save_path=save_path,
            topK=topK,
            filterNaN=filterNaN,
        )
