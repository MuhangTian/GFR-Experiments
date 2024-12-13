from typing import *

import numpy as np
import pandas as pd
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing._encoders import _BaseEncoder


class UserBins:
    '''to hold user defined thresholds for binarization, mostly just an immutable dictionary'''
    def __init__(self, **kwargs) -> None:
        for thresholds in kwargs.values():
            assert isinstance(thresholds, (tuple, list, np.ndarray)), f"threshold must be tuple, list or np.ndarray! Detected type: {type(threshold)}"
            for threshold in thresholds:
                assert type(threshold) == int or type(threshold) == float or np.issubdtype(type(threshold), np.number), f"threshold must be numeric! Detected type: {type(threshold)}"
        self._data = dict(kwargs)
    
    def __str__(self) -> str:
        string = ""
        for feature_name, thresholds in self._data.items():
            for threshold in thresholds:
                string += f"{feature_name}<={threshold}\n"
            string += "\n"
        return string
    
    def __len__(self) -> int:
        value = 0
        for thresholds in self._data.values():
            value += len(thresholds)
        return value
    
    def __contains__(self, feature_name) -> bool:
        return feature_name in self._data.keys()
    
    def __getitem__(self, feature_name) -> Sequence[Union[int, float]]:
        return self._data[feature_name]
    
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{self.__class__.__name__} is immutable!")
    
    def __delattr__(self, name) -> None:
        raise AttributeError(f"{self.__class__.__name__} is immutable!")
    
    def __iter__(self) -> Iterator[Tuple[str, Sequence[Union[int, float]]]]:
        return iter(self._data.items())
    
    def features(self) -> Sequence[str]:
        return tuple(self._data.keys())
    
    def bins(self) -> Sequence[Sequence[Union[int, float]]]:
        return tuple(self._data.values())
    
    def pairs(self) -> Sequence[Tuple[str, Sequence[Union[int, float]]]]:
        return tuple(self._data.items())
    
    
def nan_onehot_single_column(column:pd.Series) -> np.ndarray:
     onehot = np.zeros(len(column))
     onehot[column.isnull()] = 1

     return onehot

class BinBinarizer(_BaseEncoder):
     """
     Binarize variables into binary variables based on percentile or user defined thresholds.
     
     Parameters
     ----------
     interval_width : int
         width of the interval measured by percentiles. For instance, if interval_width=10, then
         each interval will be between nth and (n+10)th percentile
     categorical_cols : list
         list of names for categorical variables
     wheter_interval : bool
         whether to one hot based on intervals or based on less thans, by default False (use less thans)
     """
     def __init__(self, interval_width: int, whether_interval: bool=False, group_sparsity: bool=False) -> None:
         assert type(interval_width) == int, "'interval_width' must be integer!"
         assert 100 % interval_width == 0, "'interval_width' must divide 100!"
         self.interval_width = interval_width
         self.whether_interval = whether_interval
         self.group_sparsity = group_sparsity

     def fit(self, df: pd.DataFrame) -> None:
         '''fit IntervalBinarizer'''
         assert type(df) == pd.DataFrame, 'must be a pd.DataFrame!'

         self.cols = list(df.columns)
         tiles = range(self.interval_width, 100+self.interval_width, self.interval_width)
         binarizers= []
         if self.group_sparsity:
             GroupMap = {}

         for col_idx in range(len(self.cols)):
             col = self.cols[col_idx]
             col_value = df[col]

             binarizers.append({                 # need to keep track of NaN for every column
                 'name': f"{col}_isNaN",
                 'col': col,
                 'threshold': np.nan,
             })                      

             col_value = col_value.dropna()      # drop NaN
             vals = col_value.unique()         # count unique values

             if len(vals) == 1: 
                 if self.group_sparsity:
                     GroupMap[col] = col_idx
                 continue
             elif len(vals) > (100/self.interval_width):     # if more than number of bins, do percentiles
                 perctile = np.percentile(vals, tiles, method="closest_observation")
             else:   # else just use the unique values in sorted order
                 perctile = np.sort(vals)

             if self.whether_interval:       # do it in intervals
                 for i in range(0, len(perctile)):
                     if i == 0:
                         name = f"{col}<={perctile[i]}"
                         threshold = perctile[i]
                     else:
                         name = f"{perctile[i-1]}<{col}<={perctile[i]}"
                         threshold = (perctile[i-1], perctile[i])
                     binarizers.append({
                         "name": name,
                         "col": col,
                         "threshold": threshold,           
                     })
                     if self.group_sparsity and col not in GroupMap.keys():
                         GroupMap[col] = col_idx
             else:       # do it in <=
                 for i in range(0, len(perctile)):
                     binarizers.append({
                         "name": f"{col}<={perctile[i]}",
                         "col": col,
                         "threshold": perctile[i],           
                     })
                     if self.group_sparsity and col not in GroupMap.keys():
                         GroupMap[col] = col_idx

         if self.group_sparsity:
             assert len(GroupMap) == len(self.cols), "invalid GroupMap!"
             self.GroupMap = GroupMap
         self.binarizers = binarizers                # save binarizer and GroupMap for transform() function

     def transform(self, df: pd.DataFrame) -> tuple:
         """
         Transform data using percentiles found in fitting
         
         Parameters
         ----------
         df : pd.DataFrame
             data to transform
             
         Returns
         -------
         tuple
             transformed data, group sparsity index
         """
         assert hasattr(self, "binarizers"), 'IntervalBinarizer not fitted yet!'
         assert type(df) == pd.DataFrame, "must be a pd.DataFrame"
         assert list(self.cols) == list(df.columns), "data used for fitting and transforming must have same columns and same order!"

         n, result, GroupIdx = len(df), {}, []

         for bin in self.binarizers:
             feature = np.zeros(n, dtype=int)
             if self.whether_interval:
                 if type(bin["threshold"]) != tuple:
                     feature[df[bin["col"]] <= bin["threshold"]] = 1
                 else:   # apply a < variable <= b, threshold determined by .fit()
                     feature[(bin["threshold"][0] < df[bin["col"]]) & (df[bin["col"]] <= bin["threshold"][1])] = 1

                 if self.group_sparsity:
                     GroupIdx.append(self.GroupMap[bin["col"]])
                 result[bin["name"]] = feature
             else:
                 if np.isnan(bin['threshold']):                      # if threshold is NaN, calculate NaN dummy
                     feature = nan_onehot_single_column(df[bin['col']])
                     result[bin['name']] = feature
                 else:
                     feature[df[bin["col"]] <= bin["threshold"]] = 1
                     result[bin["name"]] = feature

                 if self.group_sparsity:
                     GroupIdx.append(self.GroupMap[bin["col"]])

         result = pd.DataFrame.from_dict(result)

         if self.group_sparsity:
             assert len(np.unique(GroupIdx)) == len(df.columns), "GroupIdx is wrong!"
             assert len(GroupIdx) == result.shape[1], "GroupIdx is wrong!"
             return result, np.asarray(GroupIdx)
         else:
             return result, None

     def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
         '''fit and transform on same dataframe'''
         self.fit(df)
         return self.transform(df)


def onehot_categorical(df: pd.DataFrame, category_cols: list):
    """
    Perform one hot encoding on categorical data

    Parameters
    ----------
    df : pd.DataFrame
        data frame to perform one hot on
    category_cols : list
        name of categorical features (variables)
    
    Returns
    -------
    pd.DataFrame
        onehot encoded dataframe
    """
    # new_df = df[all_cols].dropna()
    hot = OneHotEncoder()
    # get sub columns with proceed in OneHotEnc
    df_category = pd.DataFrame(hot.fit_transform(df[category_cols].astype(int)).todense(), columns=hot.get_feature_names_out())

    return df_category

def onehot_numerical(df: pd.DataFrame, numerical_cols: list):
    """
    Peform one hot on all available numerical data (suitable for FasterRisk)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of interest
    numerical_cols : list
        name of numerical features (variables)
    
    Returns
    -------
    pd.DataFrame
        onehot encoded dataframe
    """
    return convert_continuous_df_to_binary_df(df[numerical_cols])

def fasterrisk_onehot(X, categorical_cols: list, numerical_cols: list, labels: list):
    """
    one hot encoder for fasterrisk

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of interest
    categorical_cols : list
        name of categorical variables
    numerical_cols : list
        name of numerical variables
    labels : list
        name of labels
    
    Returns
    -------
    pd.DataFrame
        one hot encoded dataframe with all numerical and categorical features, with labels included as well
    """
    df1 = onehot_categorical(X, categorical_cols)
    df2 = onehot_numerical(X, numerical_cols)
    X = pd.concat([df1, df2], axis=1)

    return X
