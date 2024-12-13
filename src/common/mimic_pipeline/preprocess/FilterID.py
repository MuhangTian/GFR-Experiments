import numpy as np
import pandas as pd
from mimic_pipeline.data import BASE_COLS

# Hey tony I refactored this into loader so it's done automatically
# def capitalize(df_arr: list, mode: str):
    
#     """
#     To turn a list of pd.DataFrame's columns into capital letters or lower case letters.
#     This helps to deal with the scenario where the columns we want to merge have unmatched
#     letters (uppercase and lowercase)

#     Parameters
#     ----------
#     df_arr : list
#         list of pd.DataFrame
#     mode : str
#         {'lower', 'upper'}

#     Returns
#     -------
#     same list of pd.DataFrame but with columns changed according to mode
#     """
#     for i in range(len(df_arr)):
#         if mode == 'lower':
#             df_arr[i].columns = map(str.lower, df_arr[i].columns)
#         elif mode == 'upper':
#             df_arr[i].columns = map(str.upper, df_arr[i].columns)
#         else: raise ValueError('Only upper case or lower case supported')
#     return df_arr

# def filter_admits(to_be_filtered: pd.DataFrame, adm_df: pd.DataFrame, 
#                   icustay_df: pd.DataFrame, how="first") -> pd.DataFrame:
#     # Tony: hi jack I changed the next line because if we merge with on=_BASE_COLS, there will be two columns in
#     # filter_df for admittime, namely admittime_x and admittime_y, and the next line of the code would not function properly
#     # so I added 'admittime' to solve this bug, just let me know if I didn't do this correctly.
#     # filter_df = pd.merge(to_be_filtered, adm_df[[*__ADM_COLS, 'admittime']], on=__ADM_COLS)
#     filter_df = pd.merge(to_be_filtered, adm_df[[*__ADM_COLS, 'admittime']], on=[*__ADM_COLS, 'admittime'])
    
#     filter_df = filter_df.sort_values(by="admittime")
#     gb = filter_df.groupby(__ADM_COLS, as_index=False)
#     if how == "first":
#         filter_df = gb.first()
#     elif how == "last":
#         filter_df = gb.last()
#     else:
#         print("invalid filter " + how)

#     return filter_df.drop('admittime', axis=1)

# def filter_icu_stays(to_be_filtered: pd.DataFrame, adm_df: pd.DataFrame, 
#                      icustay_df: pd.DataFrame, how="first") -> pd.DataFrame:
#     # Tony: hi jack I changed the next line because if we merge with on=_BASE_COLS, there will be two columns in
#     # filter_df for intime, namely intime_x and intime_y, and the next line of the code would not function properly
#     # so I added 'intime' to solve this bug, just let me know if I didn't do this correctly.
#     # filter_df = pd.merge(to_be_filtered, icustay_df[[*__BASE_COLS, 'intime']], on=__BASE_COLS)
#     filter_df = pd.merge(to_be_filtered, icustay_df[[*__BASE_COLS, 'intime']], on=[*__BASE_COLS, 'intime'])
    
#     filter_df = filter_df.sort_values(by="intime")
#     gb = filter_df.groupby(__BASE_COLS, as_index=False)
    
#     if how == "first":
#         filter_df = gb.first()
#     elif how == "last":
#         filter_df = gb.last()
#     else:
#         print("invalid filter " + how)
    
#     return filter_df.drop('intime', axis=1)


def filter_by_adm(to_be_filtered: pd.DataFrame, adm_df: pd.DataFrame, icu_df: pd.DataFrame, 
              by="admission", how="first"):
    ...
    
def filter_by_icu(to_be_filtered: pd.DataFrame, adm_df: pd.DataFrame, icu_df: pd.DataFrame, 
              by="admission", how="first"):
    """
    Gets a patient's first or last ICU
    """
    
    copy_df = to_be_filtered.copy()
    if not 'intime' in to_be_filtered.columns:
        copy_df = pd.merge(copy_df, icu_df, on=[*BASE_COLS])
    
    copy_df.sort_values('intime', inplace=True)
    
    gb = copy_df.groupby('subject_id')
    if how=='first':
        copy_df = gb.first()
    elif how=='last':
        copy_df = gb.last()
        
    copy_df.reset_index(inplace=True)
    # NOTE: this last step is needed since groupby.last() or groupby.first() in pandas returns NONE NAN VALUE for each column, which
    # is not what we want since it messes with the original data. Given that "icustay_id" is unique, we can just do 
    # an inner join on the BASE_COLS based on the original data ("to_be_filtered")
    return pd.merge(to_be_filtered, copy_df[BASE_COLS], on=BASE_COLS, how='inner')