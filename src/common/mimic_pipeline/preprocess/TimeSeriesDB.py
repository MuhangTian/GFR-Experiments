"""
For extraction of time-series data stored in local PostgreSQL database, for feature analysis.

IMPORTANT: we assume you have already set up the local database by following the tutorial, and ran 
"\i sql/concepts_postgres/firstday-intervals/generate-intervals-oasis.sql" 
and 
"\i sql/concepts_postgres/firstday-intervals/generate-intervals-oasis.sql" in "psql" to generate the required tables.
"""
import numpy as np
import pandas as pd
from mimic_pipeline.data import BASE_COLS
from mimic_pipeline.preprocess.FilterID import filter_by_icu
from mimic_pipeline.utils import DataBaseLoader, Table
from tqdm import tqdm

from mimic_pipeline.feature import (APSIIIFeatures, OASISFeatures,
                                    SAPSIIFeatures, SOFAFeatures)


class TimeSeriesBase:
    
    def __init__(self, **kwargs) -> None:
        self.dbloader = DataBaseLoader(**kwargs)
        # self.PreProcessor = PreProcessor(**kwargs)


class TimeSeriesExtract(TimeSeriesBase):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def oasis_timeseries(self, save_path:str=None) -> dict:
        """
        generate a dictionary holding timeseries data for analysis (for OASIS features)
        NOTE: NaNs are dropped here

        Parameters
        ----------
        save_path : str, optional
            path to save the dictionary, if not given, return the dictionary, by default None

        Returns
        -------
        dict
        """
        if save_path is not None: assert save_path.split(".")[-1] == "npy", "must save as .npy!"
        result = {}
        lost_total, c = 0, 0
        oasis_original = self.dbloader["oasis"]
        for interval in tqdm(range(0, 24, 2), desc="Extracting OASIS..."):
            suffix = f"{interval}_to_{interval+2}"
            oasis_df = self.dbloader[f"oasis_{suffix}"]
            
            pd.testing.assert_frame_equal(      # to check that our operations are valid
                oasis_original[["subject_id", "hadm_id", "icustay_id"]],
                oasis_df[["subject_id", "hadm_id", "icustay_id"]],
            )
            
            oasis_df = oasis_df[oasis_df["icustay_age_group"] == 'adult']       # NOTE: we drop NaN inside the inner loop to avoid dropping too much
            oasis_df = pd.merge(oasis_df, self.dbloader["icustays"][[*BASE_COLS, 'los']], on=BASE_COLS)
            oasis_df['los'] = oasis_df['los'].astype(float)
            oasis_df = oasis_df[oasis_df['los'] >= 1]
            
            oasis_df = filter_by_icu(oasis_df, self.dbloader['admissions'], self.dbloader["icustays"], how="last")
            assert (len(oasis_df['subject_id']) == len(oasis_df['subject_id'].unique())), "subject IDs not unique!"
            # oasis_df = self.PreProcessor.remove_zero_mortality_icd(oasis_df)
            
            for column in list(oasis_df.columns):
                if column in ["subject_id", "hadm_id", "icustay_id", "icustay_age_group", "icustay_expire_flag"]:
                    continue
                tmp_df = oasis_df[[column, "icustay_expire_flag"]]
                lost_total += 1 - len(tmp_df.dropna(how='any')) / len(tmp_df)
                c += 1
                tmp_df = tmp_df.dropna(how="any")
                try:
                    result[column][f"{interval}-{interval+2}"] = tmp_df
                except:
                    result[column] = {f"{interval}-{interval+2}": tmp_df}
                    
        print(f"Average % lost due to dropping NaNs: {lost_total/c}")
        
        if save_path is None:
            return result
        else:
            np.save(save_path, result)
            print(f"result saved as \"{save_path}\"")
    
    
    def sapsii_timeseries(self, save_path:str=None):
        """
        generate a dictionary holding timeseries data for analysis (for SAPS-II features)
        NOTE: NaNs are dropped here

        Parameters
        ----------
        save_path : str, optional
            path to save the dictionary, if not given, return the dictionary, by default None

        Returns
        -------
        dict
        """
        if save_path is not None: assert save_path.split(".")[-1] == "npy", "must save as .npy!"
        result = {}
        oasis_df = self.dbloader["oasis"]
        lost_total, c = 0, 0
        for interval in tqdm(range(0, 24, 2), desc="Extracting SAPS-II..."):
            suffix = f"{interval}_to_{interval+2}"
            sapsii_df = self.dbloader[f"sapsii_{suffix}"]
            
            pd.testing.assert_frame_equal(      # to check that they have a valid merge
                sapsii_df[["subject_id", "hadm_id", "icustay_id"]],
                oasis_df[["subject_id", "hadm_id", "icustay_id"]],
            )
            sapsii_df["icustay_age_group"] = oasis_df["icustay_age_group"]
            sapsii_df["icustay_expire_flag"] = oasis_df["icustay_expire_flag"]

            sapsii_df = sapsii_df[sapsii_df["icustay_age_group"] == 'adult']
            sapsii_df = pd.merge(sapsii_df, self.dbloader["icustays"][[*BASE_COLS, 'los']], on=BASE_COLS)
            sapsii_df['los'] = sapsii_df['los'].astype(float)
            sapsii_df = sapsii_df[sapsii_df['los'] >= 1]
            
            sapsii_df = filter_by_icu(sapsii_df, self.dbloader['admissions'], self.dbloader["icustays"], how="last")
            assert (len(sapsii_df['subject_id']) == len(sapsii_df['subject_id'].unique())), "subject IDs not unique!"
            # sapsii_df = self.PreProcessor.remove_zero_mortality_icd(sapsii_df)
            
            for column in list(sapsii_df.columns):
                if column in ["subject_id", "hadm_id", "icustay_id", "icustay_expire_flag"]:
                    continue
                tmp_df = sapsii_df[[column, "icustay_expire_flag"]]
                lost_total += 1 - len(tmp_df.dropna(how='any')) / len(tmp_df)
                c += 1
                tmp_df = tmp_df.dropna(how="any")
                try:
                    result[column][f"{interval}-{interval+2}"] = tmp_df
                except:
                    result[column] = {f"{interval}-{interval+2}": tmp_df}
                    
        print(f"Average % lost due to dropping NaNs: {lost_total/c}")
        
        if save_path is None:
            return result
        else:
            np.save(save_path, result)
            print(f"result saved as \"{save_path}\"")
            
    
    def apsiii_timeseries(self, save_path:str=None):
        if save_path is not None: assert save_path.split(".")[-1] == "npy", "must save as .npy!"
        result = {}
        oasis_df = self.dbloader["oasis"]
        lost_total, c = 0, 0
        for interval in tqdm(range(0, 24, 2), desc="Extracting APS-III..."):
            suffix = f"{interval}_to_{interval+2}"
            apsiii_df = self.dbloader[f"apsiii_{suffix}"]
            
            pd.testing.assert_frame_equal(      # to check that they have a valid merge
                apsiii_df[["subject_id", "hadm_id", "icustay_id"]],
                oasis_df[["subject_id", "hadm_id", "icustay_id"]],
            )
            apsiii_df["icustay_age_group"] = oasis_df["icustay_age_group"]
            apsiii_df["icustay_expire_flag"] = oasis_df["icustay_expire_flag"]

            apsiii_df = apsiii_df[apsiii_df["icustay_age_group"] == 'adult']
            apsiii_df = pd.merge(apsiii_df, self.dbloader["icustays"][[*BASE_COLS, 'los']], on=BASE_COLS)
            apsiii_df['los'] = apsiii_df['los'].astype(float)
            apsiii_df = apsiii_df[apsiii_df['los'] >= 1]
            
            apsiii_df = filter_by_icu(apsiii_df, self.dbloader['admissions'], self.dbloader["icustays"], how="last")
            assert (len(apsiii_df['subject_id']) == len(apsiii_df['subject_id'].unique())), "subject IDs not unique!"
            # apsiii_df = self.PreProcessor.remove_zero_mortality_icd(apsiii_df)
            
            for column in list(apsiii_df.columns):
                if column in ["subject_id", "hadm_id", "icustay_id", "icustay_expire_flag"]:
                    continue
                tmp_df = apsiii_df[[column, "icustay_expire_flag"]]
                lost_total += 1 - len(tmp_df.dropna(how='any')) / len(tmp_df)
                c += 1
                tmp_df = tmp_df.dropna(how="any")
                try:
                    result[column][f"{interval}-{interval+2}"] = tmp_df
                except:
                    result[column] = {f"{interval}-{interval+2}": tmp_df}
                    
        print(f"Average % lost due to dropping NaNs: {lost_total/c}")
        
        if save_path is None:
            return result
        else:
            np.save(save_path, result)
            print(f"result saved as \"{save_path}\"")
    
    
    def sofa_timeseries(self, save_path:str=None):
        if save_path is not None: assert save_path.split(".")[-1] == "npy", "must save as .npy!"
        result = {}
        oasis_df = self.dbloader["oasis"]
        lost_total, c = 0, 0
        for interval in tqdm(range(0, 24, 2), desc="Extracting APS-III..."):
            suffix = f"{interval}_to_{interval+2}"
            sofa_df = self.dbloader[f"sofa_{suffix}"]
            
            pd.testing.assert_frame_equal(      # to check that they have a valid merge
                sofa_df[["subject_id", "hadm_id", "icustay_id"]],
                oasis_df[["subject_id", "hadm_id", "icustay_id"]],
            )
            sofa_df["icustay_age_group"] = oasis_df["icustay_age_group"]
            sofa_df["icustay_expire_flag"] = oasis_df["icustay_expire_flag"]

            sofa_df = sofa_df[sofa_df["icustay_age_group"] == 'adult']
            sofa_df = pd.merge(sofa_df, self.dbloader["icustays"][[*BASE_COLS, 'los']], on=BASE_COLS)
            sofa_df['los'] = sofa_df['los'].astype(float)
            sofa_df = sofa_df[sofa_df['los'] >= 1]
            
            sofa_df = filter_by_icu(sofa_df, self.dbloader['admissions'], self.dbloader["icustays"], how="last")
            assert (len(sofa_df['subject_id']) == len(sofa_df['subject_id'].unique())), "subject IDs not unique!"
            # sofa_df = self.PreProcessor.remove_zero_mortality_icd(sofa_df)
            
            for column in list(sofa_df.columns):
                if column in ["subject_id", "hadm_id", "icustay_id", "icustay_expire_flag"]:
                    continue
                tmp_df = sofa_df[[column, "icustay_expire_flag"]]
                lost_total += 1 - len(tmp_df.dropna(how='any')) / len(tmp_df)
                c += 1
                tmp_df = tmp_df.dropna(how="any")
                try:
                    result[column][f"{interval}-{interval+2}"] = tmp_df
                except:
                    result[column] = {f"{interval}-{interval+2}": tmp_df}
                    
        print(f"Average % lost due to dropping NaNs: {lost_total/c}")
        
        if save_path is None:
            return result
        else:
            np.save(save_path, result)
            print(f"result saved as \"{save_path}\"")


class TimeSeriesDifference(TimeSeriesBase):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def calculate_change(self, bin_width:int, save_path:str=None, mode:str="diff", div_zero_columns:list=None) -> pd.DataFrame:
        """
        calculate change for between each time intervals for time series data
        Parameters
        ----------
        bin_width : int
            length of the interval
        save_path : str, optional
            path to save the final data, by default None, if None, return the actual pd.DataFrame
        mode : str, optional
            mode to calculate the change, {"diff", "pct diff", "pct change"}, which corresponds to difference, percentage difference, and percentage change, by default "diff"
        div_zero_columns : list, optional
            columns which would have division by zero problem, and numpy would give NaN for those values, if you think this is a problem, 
            specify those columns here, and actual difference would be used for those columns instead, by default None
        Returns
        -------
        pd.DataFrame
            containing a union of all the features with their change values between every interval
        """
        assert mode in ["diff", "pct diff", "pct change"], "mode not supported!"
        result = self.dbloader["oasis"][[*BASE_COLS, "icustay_age_group", 'icustay_expire_flag', 'age', 'urineoutput', 'preiculos']] 
        # use subject_id, icustay_age_group from oasis table as a starting place to merge
        
        for prefix in tqdm(["oasis", "sapsii", "apsiii", 'sofa'], desc="Calculating..."):
            for interval in range(0, 24-bin_width, bin_width):
                
                suffix1 = f"{interval}_to_{interval+bin_width}"
                suffix2 = f"{interval+bin_width}_to_{interval+bin_width*2}"
                df1 = self.dbloader[f"{prefix}_{suffix1}"]          # read the table at one interval
                df2 = self.dbloader[f"{prefix}_{suffix2}"]          # read the other table at one interval after
                
                if prefix == "oasis":           # select appropriate columns based on scoring systems
                    oasis_features = [e for e in OASISFeatures().all() if e not in ["age", "preiculos", "electivesurgery"]]
                    df1 = df1[[*BASE_COLS, *oasis_features]]
                    df2 = df2[[*BASE_COLS, *oasis_features]]
                elif prefix == "sapsii":
                    df1 = df1[[*BASE_COLS, *SAPSIIFeatures().all()]]
                    df2 = df2[[*BASE_COLS, *SAPSIIFeatures().all()]]
                elif prefix == "apsiii":
                    df1 = df1[[*BASE_COLS, *APSIIIFeatures().all()]]
                    df2 = df2[[*BASE_COLS, *APSIIIFeatures().all()]]
                elif prefix == 'sofa':
                    df1 = df1[[*BASE_COLS, *SOFAFeatures().all()]]
                    df2 = df2[[*BASE_COLS, *SOFAFeatures().all()]]
                    
                table1, table2 = Table(df1), Table(df2)     # initialize table for vectorized operations
                suffix3 = f"({interval}-{interval+bin_width})_to_({interval+bin_width}-{interval+bin_width*2})"     # suffix for the column
                
                change = table2-table1                                      # calculate difference
                change_df = change.df.drop(BASE_COLS, axis=1)               # add suffix for non-ID columns
                change_df = change_df.add_suffix(f"_diff_{suffix3}")
                
                if mode == 'pct diff':                          # perform different operations based on "mode"
                    change = change/(table2+table1)*200         # (b-a) / (0.5*(a+b)) * 100
                    if div_zero_columns is not None:
                        div_zero_columns_suffix = [f"{element}_diff_{suffix3}" for element in div_zero_columns if element in change.df.columns]
                        change_df = change_df[div_zero_columns_suffix]              # get the difference column for division by zero columns
                        div_zero_columns_tmp = [element for element in div_zero_columns if element in change.df.columns]
                        change_df2 = change.df.drop([*BASE_COLS, *div_zero_columns_tmp], axis=1)            # do not select division by zero columns
                        change_df2 = change_df2.add_suffix(f"_pct_diff_{suffix3}")
                        change_df = pd.concat([change_df, change_df2], axis=1)                  # concatenate difference for division by zero columns with the other columns
                    else:                                       # if None, then just use all the columns
                        change_df = change.df.drop([*BASE_COLS], axis=1)
                        change_df = change_df.add_suffix(f"_pct_diff_{suffix3}")
                elif mode == 'pct change':                    # same operations for percentage change, but append a different suffix
                    change = change/(table1)*100
                    if div_zero_columns is not None:
                        div_zero_columns_suffix = [f"{element}_diff_{suffix3}" for element in div_zero_columns if element in change.df.columns]
                        change_df = change_df[div_zero_columns_suffix]
                        div_zero_columns_tmp = [element.replace(f"_diff_{suffix3}", "") for element in div_zero_columns_suffix]
                        change_df2 = change.df.drop([*BASE_COLS, *div_zero_columns_tmp], axis=1)
                        change_df2 = change_df2.add_suffix(f"_pct_chg_{suffix3}")
                        change_df = pd.concat([change_df, change_df2], axis=1)
                    else:
                        change_df = change.df.drop([*BASE_COLS], axis=1)
                        change_df = change_df.add_suffix(f"_pct_chg_{suffix3}")
                
                if div_zero_columns is not None:
                    for column in div_zero_columns_tmp:
                        div_zero_columns.remove(column)     # remove used columns
                
                    if len(div_zero_columns) == 0:      # if all are done, set it back to None
                        div_zero_columns = None
                
                for i in range(len(BASE_COLS)):     # lastly, append BASE COLS, NOTE: BASE COLS are guaranteed to be the same in Table() calculation (see Gadgets.py), thus this operation is valid
                    change_df.insert(i, BASE_COLS[i], change.df[BASE_COLS[i]])
    
                pd.testing.assert_frame_equal(change_df[BASE_COLS], result[BASE_COLS], check_dtype=False)       # one last check :)
                result = pd.concat([result, change_df.drop(BASE_COLS, axis=1)], axis=1)
        
        if save_path is not None:
            result.to_csv(save_path, index=False)
            print(f"File saved as: {save_path}")
        else:
            return result
                
    
    