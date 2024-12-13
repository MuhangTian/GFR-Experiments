import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
from mimic_pipeline.data import BASE_COLS, SEED
from mimic_pipeline.preprocess.FilterID import filter_by_icu
from mimic_pipeline.utils import DataBaseLoader, unique
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

np.random.seed(SEED)

class DataSaver():
    """
    To save pre-processed data to a directory

    Parameters
    ----------
    load_path : str
        path to load raw MIMIC data
    save_path : str
        path to save pre-processed data
    disease : str
        {"all", "heart attack", "sepsis", "heart failure"}
    
    Example
    -------
    >>> saver = DataSaver(load_path="data/full", save_path="data/processed/oasis")
    >>> saver.save_data()
    """
    def __init__(self, **kwargs) -> None:
        # self.PreProcessor = PreProcessor(**kwargs)
        self.loader = DataBaseLoader(**kwargs)
    
    
    def preprocess(self, df:pd.DataFrame, target: str) -> pd.DataFrame:
        """
        pre-process raw MIMIC III data

        Returns
        -------
        pd.DataFrame
            pre-processed data
        """
        loader = self.loader
        adm = loader['admissions']
        icu = loader['icustays']
        
        df = pd.merge(df, icu[[*BASE_COLS, 'los']], on=BASE_COLS)       # obtain LOS from icustays table
        df['los'] = df['los'].astype(float)
        df = df[df['los'] >= 1]                                         # select patients who have stayed for at least 24 hours
        df = df.drop("los", axis=1)
        # this line is for removing wrong data: like 300 years old patients, negative preiculos, and negative urineoutput
        df = df[ (df['age'] > 15) & (df["age"] <= 90) & (df["preiculos"] >= 0) & (df["urineoutput"] >= 0) ]
        
        df = filter_by_icu(df, adm, icu, how="first")                    # select each patient's FIRST ICU stay, to ensure uniqueness
        assert (len(df['subject_id']) == len(df['subject_id'].unique())), "subject IDs not unique!"
    
        # df = self.PreProcessor.remove_zero_mortality_icd(df, target)            # search and drop ICD9 codes with zero mortality
        
        return df
    
    
    def train_test_split(self, ProcessedData: pd.DataFrame, target: str) -> tuple:
        """
        perform train test split on pre-processed data

        Parameters
        ----------
        ProcessedData : pd.DataFrame
            pre-processed data

        Returns
        -------
        tuple
            train, test
        """
        X = ProcessedData.drop(target, axis=1)
        y = ProcessedData[target].astype(int).replace(0, -1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)
        print("\n ************************ NaN Summary ************************")
        for col in list(X_train.columns):
            train_nan_num = X_train[col].isna().sum()
            test_nan_num = X_test[col].isna().sum()
            print(f"Feature {col}: Train has {round(train_nan_num/len(X_train), 3)}    |    Test has {round(test_nan_num/len(X_test), 3)}")
        assert not (y_train.isna().values.any()), "label contain NaNs!"
        assert not (y_test.isna().values.any()), "label contain NaNs!"
        print(f"% of positives (deaths) in TRAIN: {round(len(y_train[y_train==1])/len(y_train)*100, 3)}%")
        print(f"% of positives (deaths) in TEST: {round(len(y_test[y_test==1])/len(y_test)*100, 3)}%")
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        
        return train, test


    def save_data(self, df:pd.DataFrame, save_path:str, suffix:str=None, KFold: bool=False, K: int=5,
                  label: str='icustay_expire_flag') -> None:
        """
        pre-process and generate TRAIN and TEST split, and save to "save_path"

        Parameters
        ----------
        KFold : bool, optional
            whether to use KFold, by default False (only needed for nested cross validation)
        K : int, optional
            number of folds, by default 5
        """
        ProcessedData = self.preprocess(df, target=label)
        suffix = '' if suffix is None else suffix
        if KFold:
            kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
            k = 1
            X = ProcessedData.drop(label, axis=1)
            y = ProcessedData[label].astype(int).replace(0, -1)
            for train_idx, test_idx in kf.split(X, y):
                X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
                print(f"For Fold {k}...")
                print(f"% of positives (deaths) in TRAIN: {round(100*len(y_train[y_train==1])/len(y_train), 3)}%")
                print(f"% of positives (deaths) in TEST: {round(100*len(y_test[y_test==1])/len(y_test), 3)}%\n")
                train = pd.concat([X_train, y_train], axis=1)
                test = pd.concat([X_test, y_test], axis=1)
            
                train.to_csv(f'{save_path}/TRAIN-{suffix}-fold{k}.csv', index=False)
                test.to_csv(f'{save_path}/TEST-{suffix}-fold{k}.csv', index=False)
                k += 1
        else:
            train, test = self.train_test_split(ProcessedData=ProcessedData, target=label)
            train.to_csv(f'{save_path}/TRAIN-{suffix}.csv', index=False)
            test.to_csv(f'{save_path}/TEST-{suffix}.csv', index=False)
            
        print(f"DONE, files saved at '{save_path}'")


# class PreProcessor():
#     """
#     for pre-processing data

#     Parameters
#     ----------
#     load_path : str
#         path to where raw MIMIC III data is stored
#     """
#     def __init__(self, **kwargs) -> None:
#         self.loader = DataBaseLoader(**kwargs)


    # def search_zero_mortality_icd(self, ProcessedData: pd.DataFrame, target: str) -> dict:
    #     """
    #     search for first three digits of ICD9 codes and their associated patients with zero mortalty

    #     Parameters
    #     ----------
    #     ProcessedData : pd.DataFrame
    #         processed dataframe

    #     Returns
    #     -------
    #     dict
    #         a dictionary keyed by ICD9 codes, with an array of IDs associated with that ICD9 code as the value
    #     """
    #     diagnoses = self.loader["diagnoses_icd"]
    #     diagnoses = diagnoses[diagnoses["seq_num"] == 1]      # filter by SEQ_NUM = 1
    #     diagnoses = diagnoses.drop_duplicates(subset=["subject_id"], keep=False)        # NOTE: drop patients with multiple ICD9 codes, FOR NOW
    #     assert diagnoses["subject_id"].is_unique, "Subject IDs not unique!"
        
    #     zero_mortality, patient_num = {}, 0     # dictionary, keyed by ICD9, with value as array of subjectIDs belong to that ICD9
    #     for code in tqdm(diagnoses["icd9_code"].str[:3].unique(), desc="Counting mortality for each ICD9 code..."): # use first three digits
    #         tmp = diagnoses[diagnoses["icd9_code"].str.startswith(code)]
    #         tmp2 = ProcessedData[ProcessedData["subject_id"].astype(int).isin(tmp["subject_id"].astype(int))]
    #         if len(tmp2[tmp2[target].astype(int) == 1]) == 0 and len(tmp2) != 0:
    #             zero_mortality[code] = list(tmp2['subject_id'])     # if no mortality, store subject ID and ICD9
    #             patient_num += len(tmp2["subject_id"])
    #         else:
    #             continue
        
    #     print(f"COMPLETE\nFound {len(zero_mortality.keys())} ICD9 codes with zero mortality\nAssociated with {patient_num} patients\nTOTAL number of patients: {len(ProcessedData)}")
        
    #     return zero_mortality


    # def remove_zero_mortality_icd(self, ProcessedData: pd.DataFrame, target: str) -> pd.DataFrame:
    #     """
    #     remove patients with zero mortality ICD9 codes

    #     Parameters
    #     ----------
    #     ProcessedData : pd.DataFrame
    #         processed dataframe

    #     Returns
    #     -------
    #     pd.DataFrame
    #         ProcessedData, where patients with zero mortality ICD9 codes are removed
    #     """
    #     ZeroMortalityDict = self.search_zero_mortality_icd(ProcessedData, target)
    #     IDs = np.array([], dtype=str)
    #     for v in ZeroMortalityDict.values():
    #         IDs = np.concatenate((IDs, v))
    #     IDs = unique(IDs)
    #     ProcessedData = ProcessedData[~ProcessedData["subject_id"].isin(IDs)]

    #     return ProcessedData
        
    # def search_sedated(self, save_path: str=None) -> np.ndarray:
    #     """
    #     search for sedated patients

    #     Parameters
    #     ----------
    #     save_path : str
    #         path to save the result, by default None. If given, save the subject IDs to this path, else return the
    #         subject IDs

    #     Returns
    #     -------
    #     np.ndarray
    #         array of unique subject IDs for patients who have been sedated
    #     """
    #     if save_path is not None:
    #         assert isinstance(save_path, str), "'save_path' must be a string!"
    #     loader = self.loader
    #     cptevents = loader['cptevents']     # these tables contain information we need about sedation
    #     ditems = loader['d_items']
    #     dcpt = loader['d_cpt']
    #     loader.set_mode("dd")           # CHARTEVENTS too big, use dask.dataframe to speed up
    #     chartevents = loader["chartevents"]
    #     print(f"{'*'*25} Begin searching... {'*'*25}")
    #     a = time.time()
    #     subject_id1 = self.__cptevents_search_sedated(cptevents, dcpt)
    #     subject_id2 = self.__ditems_search_sedated(ditems, chartevents)
    #     subject_id = np.concatenate((subject_id1, subject_id2))
    #     subject_id = np.unique(subject_id)
    #     print('*'*80)
    #     print(f"Search COMPLETE\nTotal time taken: {(time.time()-a)/60} mins")
        
    #     if save_path == None:
    #         return subject_id
    #     else:
    #         np.save(f"{save_path}/sedated-subject-id.npy", subject_id)
    #         print(f"DONE\nResult saved at {save_path}")
    
    
    # def add_sedated_dummy(self, ProcessedData: pd.DataFrame, load_path: str=None, loc: int=15) -> pd.DataFrame:
    #     """
    #     Add dummy variable for sedated patients to the dataframe

    #     Parameters
    #     ----------
    #     ProcessedData : pd.DataFrame
    #         processed dataframe which we want to add dummy for
    #     load_path : str, optional
    #         path to the directory to load array of sedated patients' subject IDs, by default None
    #     loc : int
    #         index at which to insert the dummy column

    #     Returns
    #     -------
    #     pd.DataFrame
    #         ProcessedData with sedated dummies added
    #     """
    #     if load_path is not None:
    #         assert isinstance(load_path, str), "'load_path' must be a string!"
    #         SedatedSubjectID = np.load(load_path, allow_pickle=True)
    #     else:
    #         SedatedSubjectID = self.search_sedated()
    #     dummy = np.zeros(shape=len(ProcessedData), dtype=int)
    #     dummy[ProcessedData["subject_id"].isin(SedatedSubjectID)] = 1
    #     ProcessedData.insert(loc, "sedation_dummy", dummy)
        
    #     return ProcessedData

    
    # def __cptevents_search_sedated(self, cptevents: pd.DataFrame, dcpt: pd.DataFrame) -> np.ndarray:
    #     """
    #     search for sedated patients in CPTEVENTS table

    #     Parameters
    #     ----------
    #     cptevents : pd.DataFrame
    #         CPTEVENTS table in pd.DataFrame
    #     dcpt : pd.DataFrame
    #         D_CPT table in pd.DataFrame

    #     Returns
    #     -------
    #     np.ndarray
    #         a np.array containing unique subject IDs for sedate patients
    #     """
    #     print("- Search in CPTEVENTS...")
    #     a = time.time()
    #     dcpt = dcpt[(dcpt['sectionheader'] == "Anesthesia") & (dcpt["subsectionheader"].isin(["Conscious sedation (deleted codes)", "Moderate (conscious) sedation"]))]
    #     cptevents["cpt_cd"] = cptevents["cpt_cd"].str.replace("[a-zA-Z]", "99999").astype(int)  # replace letters, doesn't matter what the value is since not used in here
    #     subsection_range1 = dcpt[["mincodeinsubsection", "maxcodeinsubsection"]].astype(int).to_numpy()
    #     min, max = np.min(subsection_range1), np.max(subsection_range1)     # since there is no gap in range, use min and max to filter
    #     cptevents = cptevents[(cptevents["cpt_cd"] <= max) & (cptevents["cpt_cd"] >= min)]      # apply filter
    #     subject_id = cptevents["subject_id"].unique()
    #     print(f"- COMPLETE search in CPTEVENTS\nTime taken: {(time.time() - a)/60} mins")
        
    #     return subject_id
    
    
    # def __ditems_search_sedated(self, ditems: pd.DataFrame, chartevents: dd) -> np.ndarray:
    #     """
    #     search for sedated patients in CHARTEVENTS table using information from D_ITEMS table

    #     Parameters
    #     ----------
    #     ditems : pd.DataFrame
    #         D_ITEMS table
    #     chartevents : dask.dataframe
    #         CHARTEVENTS table

    #     Returns
    #     -------
    #     np.ndarray
    #         unique array of subject IDs of sedated patients
    #     """
    #     print("- Search in CHARTEVENTS...")
    #     a = time.time()
    #     assert isinstance(chartevents, dd.core.DataFrame), "CHARTEVENTS table must be in dask.dataframe!"
    #     ditems = ditems[ditems["label"].isin(["PAR-Remain sedated", "Conscious sedation used (Bronch)", "Conscious sedation (THCEN)"])]
    #     assert ditems["linksto"].unique() == "chartevents", "the linked table(s) don't just involve CHARTEVENTS!"
    #     item_ids = ditems["itemid"]
    #     chartevents = chartevents[chartevents["itemid"].isin(item_ids)]
    #     subject_id = chartevents["subject_id"].compute().unique()       # get actual pd.DataFrame by calling compute(), return unique IDs
    #     print(f"- COMPLETE search in CHARTEVENTS\nTime taken: {(time.time() - a)/60} mins")
        
    #     return subject_id

        
    