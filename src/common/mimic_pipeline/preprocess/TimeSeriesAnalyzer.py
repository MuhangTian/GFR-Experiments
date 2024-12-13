import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from mimic_pipeline.feature import (APSIIIRawFeatures, OASISRawFeatures,
                                    SAPSIIRawFeatures)
from mimic_pipeline.preprocess.Constants import *
from mimic_pipeline.utils import (DataBaseLoader, check_type_hints, get_logger,
                                  plt_save_or_show, tfont, SEED)
from numpy.typing import *
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm

sns.set_theme()

class TimeSeriesViolinPlot:
    """
    To analyze timeseries data using violin plots and t-statistics
    NOTE: we assume you saved timeseries data as a dictionary by using the TimeSeriesExtract from TimeSeriesDB.py

    Parameters
    ----------
    ExtractedData : dict
        extracted timeseries data from TimeSeriesExtract
    positive : int, optional
        label for the positive class, by default 1
    negative : int, optional
        label for the negative class, by default 0
    LabelColumn : str, optional
        name of the label column, by default "icustay_expire_flag"
    """
    def __init__(self, ExtractedData:dict, positive:int=1, negative:int=0, LabelColumn:str="icustay_expire_flag") -> None:
        self.ExtractedData = ExtractedData
        self.positive = positive
        self.negative = negative
        self.LabelColumn = LabelColumn
        self.__test_data_validity()
    
    
    def plot_timeseries_distributions(self, save:bool=False) -> None:
        """
        whether to plot timeseries distributions for visualization and analysis, t-stat is also included

        Parameters
        ----------
        save : bool, optional
            whether to save the visualization, by default False, if True, save to "visuals/timeseries"
        """
        for feature, contents in tqdm(self.ExtractedData.items(), desc="Processing..."):
            self.__violin_t_stat(feature, contents, save)
        
    
    def __test_data_validity(self) -> None:
        for key, value in self.ExtractedData.items():
            print(f"Checking {key}...")
            interval_check = 0
            for interval, df in value.items():
                assert f"{interval_check}-{interval_check+2}" == interval, "name is not good!"
                assert self.LabelColumn in list(df.columns), "label is not in the data!"
                assert len(df[self.LabelColumn].unique()) == 2, "too many labels!"
                assert self.negative in df[self.LabelColumn].unique() and self.positive in df[self.LabelColumn].unique(), "label doesn't fit!"
                interval_check += 2
        print("ALL PASS")
    
    
    def __violin_t_stat(self, feature:str, contents:dict, save:bool=False) -> None:
        '''plot violin plot and t-stat for the given feature at the time intervals for timeseries data stored in contents dictionary'''
        assert isinstance(feature, str) and isinstance(contents, dict), "type doesn't match!"
        
        GroupedData = self.__preprocess(feature, contents)
        fig, axs = plt.subplots(2, figsize=(15,9))
        ax = sns.violinplot(
            data=GroupedData, 
            y="Value", 
            x="Interval (2 hours)", 
            hue="Label", 
            palette=['.8', '.3'], 
            split=True, 
            inner="quartile",
            ax=axs[0],
            legned=False,
        )
        colors = sns.color_palette('Paired')
        tmp = len(ax.findobj(PolyCollection))
        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            idx = ind
            rgb = to_rgb(colors[ind//2])
            if idx % 2 != 0:
                rgb = 0.7 * np.array(rgb)
            else:
                rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
            violin.set_facecolor(rgb)
        
        tStat = self.__calculate_test_statistics(contents)
        bar = sns.barplot(
            data=tStat,
            x="Interval (2 hours)",
            y="p value",
            errorbar=None,
            palette="Paired",
            ax=axs[1],
        )
        bar.axhline(0.05, color="grey", linestyle="--", linewidth=2, label="0.05 Cutoff")
        fig.suptitle(f"{feature} in 24 Hours", **tfont) 
        plt.tight_layout()
        plt.legend()
        if save:
            plt.savefig(f"visuals/timeseries/{feature}.png", dpi=300)
            print(f"Saved as \"visuals/timeseries/{feature}.png\"")
        else:
            plt.show()
        
    
    def __preprocess(self, feature:str, contents:dict) -> pd.DataFrame:
        '''preprocess the dictionary of timeseries data into pd.DataFrame for visualization and analysis'''
        IntervalColumn, LabelColumn, DataColumn = [], [], []
        for interval, data in contents.items():
            data, label = list(data[feature]), list(data[self.LabelColumn].replace({self.negative: "Alive", self.positive: "Expired"}))
            for i in range(len(data)):
                IntervalColumn.append(interval)
                DataColumn.append(data[i])
                LabelColumn.append(label[i])
        
        result = pd.DataFrame.from_dict({
            "Interval (2 hours)": IntervalColumn,
            "Label": LabelColumn,
            "Value": DataColumn
        })
        
        return result

    
    def __calculate_test_statistics(self, contents:dict, test:str="mann-whitney") -> pd.DataFrame:
        '''calculate test statistics for dictionary of timeseries data'''
        assert test in ["mann-whitney", "t-test"], "\"test\" must be in [\"mann-whitney\", \"t-test\"]!"
        IntervalColumn, PValueColumn = [], []
        for interval, data in contents.items():
            data_alive = data[data[self.LabelColumn] == self.negative]
            data_dead = data[data[self.LabelColumn] == self.positive]
            data_alive = data_alive.drop(self.LabelColumn, axis=1)
            data_dead = data_dead.drop(self.LabelColumn, axis=1)
            
            if test == "t-test":
                t, p = ttest_ind(data_alive, data_dead, equal_var=False)
            elif test == "mann-whitney":
                t, p = mannwhitneyu(data_alive, data_dead)
                
            PValueColumn.append(p.item())
            IntervalColumn.append(interval)
        
        result = pd.DataFrame.from_dict({
            "Interval (2 hours)": IntervalColumn,
            "p value": PValueColumn,
        })

        return result


class TimeSeriesLinePlot:
    def __init__(self, **kwargs) -> None:
        self.dbloader = DataBaseLoader(**kwargs)
        self.timer = timeit.default_timer
        self.feature_table_dict = FEATURE_TABLE_DICT
        self.item_dict = ITEMID_DICT
        self.disease_dict = DISEASE_DICT
        self.logger = get_logger()
        
    def get_correct_table(self, feature_name: str) -> str:
        '''get the correct table name for the given feature'''
        try:
            table_name = self.feature_table_dict[feature_name]
        except KeyError:
            raise ValueError(f"Cannot get table for {feature_name}!")
        return table_name
    
    def add_intime(self, df: pd.DataFrame) -> pd.DataFrame:
        df_icustays = self.dbloader['icustays']
        '''add intime to the given dataframe'''
        df = df.merge(df_icustays[['subject_id', 'intime']], on='subject_id', how='left')
        return df
    
    def calc_hrs_charttime(self, df: pd.DataFrame) -> pd.DataFrame:
        '''calculate the difference between charttime and intime'''
        df['charttime'] = pd.to_datetime(df['charttime'])
        df['intime'] = pd.to_datetime(df['intime'])
        # Calculate the time difference in hours
        df['hours_between_charttime'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600
        return df

    def add_expire_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        '''add whether this patient expired or not'''
        oasis = self.dbloader['oasis']          # use oasis table to get the expire_flag
        df = df.merge(oasis[['subject_id', 'icustay_expire_flag']], on='subject_id', how='left')
        return df
    
    def collect_samples(self, feature_name: str, label: str, time_limit: int) -> pd.DataFrame:
        '''
        collect samples from relevant tables (like chartevents) in MIMIC III for a patient's last icustay
        NOTE: IT'S IMPORTANT TO NOTICE THAT THIS IS ICU BASED RATHER THAN HOSPITAL BASED, MAKES THINGS A BIT DIFFERENT
        '''
        assert isinstance(feature_name, str), f"\"feature_name\" must be a string!"
        assert isinstance(label, str), f"\"label\" must be a string!"
        if time_limit == 'outtime':
            time_limit = int(1e9)
        
        self.logger.info(f"Collecting samples for \"{feature_name}\"...")
        start_time = self.timer()
        table_name = self.get_correct_table(feature_name)
        if feature_name not in self.item_dict.keys():
            raise ValueError(f"\"{feature_name}\" is not in ITEMID_DICT!\nAvailble ones are {list(self.item_dict.keys())}")
        itemid_tuple = self.item_dict[feature_name]
        if table_name == 'chartevents':
            result = self.dbloader.query(f"""
                WITH ranked_icustays AS (
                SELECT 
                    ie.*,
                    ROW_NUMBER() OVER(PARTITION BY ie.subject_id ORDER BY ie.intime DESC) AS rn
                FROM 
                    icustays ie
                ),
                icustay_tmp AS(		-- use last icustay
                    SELECT *
                    FROM ranked_icustays
                    WHERE rn = 1
                ),
                tmp_table AS
                (
                SELECT 
                ie.intime,
                DATETIME_DIFF(ie.outtime, ie.intime, 'HOUR') AS outtime_hour,
                DATETIME_DIFF(ce.charttime, ie.intime, 'HOUR') AS hours_between_charttime,
                ce.*
                FROM chartevents ce 
                LEFT JOIN icustay_tmp ie
                ON ce.icustay_id = ie.icustay_id AND ce.subject_id = ie.subject_id AND ce.hadm_id = ie.hadm_id
                AND ce.charttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                AND DATETIME_DIFF(ce.charttime, ie.intime, 'SECOND') > 0
                AND DATETIME_DIFF(ce.charttime, ie.intime, 'HOUR') <= {time_limit}
                WHERE itemid IN {itemid_tuple}
                )
                SELECT 
                tp.subject_id,
                tp.outtime_hour,
                oa.{label},
                tp.hours_between_charttime,
                tp.value,
                tp.valuenum
                FROM tmp_table tp
                LEFT JOIN oasis oa
                ON tp.icustay_id = oa.icustay_id AND tp.subject_id = oa.subject_id AND tp.hadm_id = oa.hadm_id
                WHERE 
                {label} IS NOT NULL		-- drop NULL since this flag is produced from OASIS implementation, which would give some NULLs
                AND
                hours_between_charttime IS NOT NULL	-- drop NULL since NULL is created from the left join for tmp_table, which gives NULL when condition for charttime is not satisfied
            """)
        elif table_name == 'chartevents' and feature_name == 'fio2':
            result = self.dbloader.query(f"""
                WITH ranked_icustays AS (
                SELECT 
                ie.*,
                ROW_NUMBER() OVER(PARTITION BY ie.subject_id ORDER BY ie.intime DESC) AS rn
                FROM 
                icustays ie
                ),
                icustay_tmp AS(		-- use last icustay
                SELECT *
                FROM ranked_icustays
                WHERE rn = 1
                ),
                tmp_table AS
                (
                SELECT ce.subject_id, ce.hadm_id, ce.icustay_id, ce.charttime
                    -- pre-process the FiO2s to ensure they are between 21-100%
                    , CASE
                        WHEN itemid = 223835
                            THEN CASE
                            WHEN valuenum > 0 AND valuenum <= 1
                                THEN valuenum * 100
                            -- improperly input data - looks like O2 flow in litres
                            WHEN valuenum > 1 AND valuenum < 21
                                THEN null
                            WHEN valuenum >= 21 AND valuenum <= 100
                                THEN valuenum
                            ELSE null END -- unphysiological
                        WHEN itemid IN (3420, 3422)
                        -- all these values are well formatted
                            THEN valuenum
                        WHEN itemid = 190 AND valuenum > 0.20 AND valuenum < 1
                        -- well formatted but not in %
                            THEN valuenum * 100
                    ELSE null END
                    AS fio2
                    , DATETIME_DIFF(ce.charttime, ie.intime, 'HOUR') AS hours_between_charttime
                    , DATETIME_DIFF(ie.outtime, ie.intime, 'HOUR') AS outtime_hour
                FROM chartevents ce
                LEFT JOIN icustay_tmp ie
                ON ie.icustay_id = ce.icustay_id AND ie.subject_id = ce.subject_id AND ie.hadm_id = ce.hadm_id
                AND ce.charttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                AND DATETIME_DIFF(ce.charttime, ie.intime, 'SECOND') > 0
                AND DATETIME_DIFF(ce.charttime, ie.intime, 'HOUR') <= {time_limit}
                WHERE ITEMID IN
                (
                    3420 -- FiO2
                , 190 -- FiO2 set
                , 223835 -- Inspired O2 Fraction (FiO2)
                , 3422 -- FiO2 [measured]
                )
                -- exclude rows marked as error
                AND (error IS NULL OR error = 0)
                )
                SELECT 
                tp.subject_id,
                tp.outtime_hour,
                oa.{label},
                tp.hours_between_charttime,
                tp.fio2 AS valuenum
                FROM tmp_table tp
                LEFT JOIN oasis oa
                ON tp.icustay_id = oa.icustay_id AND tp.subject_id = oa.subject_id AND tp.hadm_id = oa.hadm_id
                WHERE 
                {label} IS NOT NULL		-- drop NULL since this flag is produced from OASIS implementation, which would give some NULLs
                AND
                hours_between_charttime IS NOT NULL	-- drop NULL since NULL is created from the left join for tmp_table, which gives NULL when condition for charttime is not satisfied
            """)
        elif table_name == 'outputevents' and feature_name == 'urineoutput':
            raise NotImplementedError
            # NOTE: this currently only applies to urineoutput, make it more flexible if other features need to use this.
            result = self.dbloader.query(f"""
                WITH tmp_table AS
                (
                SELECT
                -- patient identifiers
                ie.subject_id, ie.hadm_id, ie.icustay_id,
                MAX(DATETIME_DIFF(oe.charttime, ie.intime, 'HOUR')) AS hours_between_charttime
                -- volumes associated with urine output ITEMIDs
                , SUM(
                    -- we consider input of GU irrigant as a negative volume
                    CASE
                        WHEN oe.itemid = 227488 AND oe.value > 0 THEN -1*oe.value
                        ELSE oe.value
                    END) AS valuenum
                FROM icustays ie
                -- Join to the outputevents table to get urine output
                LEFT JOIN outputevents oe
                -- join on all patient identifiers
                ON ie.subject_id = oe.subject_id AND ie.hadm_id = oe.hadm_id AND ie.icustay_id = oe.icustay_id
                -- and ensure the data occurs during the first day
                AND oe.charttime BETWEEN ie.intime AND (DATETIME_ADD(ie.intime, INTERVAL '1' DAY)) -- first ICU day
                WHERE itemid IN
                (
                -- these are the most frequently occurring urine output observations in CareVue
                40055, -- "Urine Out Foley"
                43175, -- "Urine ."
                40069, -- "Urine Out Void"
                40094, -- "Urine Out Condom Cath"
                40715, -- "Urine Out Suprapubic"
                40473, -- "Urine Out IleoConduit"
                40085, -- "Urine Out Incontinent"
                40057, -- "Urine Out Rt Nephrostomy"
                40056, -- "Urine Out Lt Nephrostomy"
                40405, -- "Urine Out Other"
                40428, -- "Urine Out Straight Cath"
                40086,--	Urine Out Incontinent
                40096, -- "Urine Out Ureteral Stent #1"
                40651, -- "Urine Out Ureteral Stent #2"

                -- these are the most frequently occurring urine output observations in MetaVision
                226559, -- "Foley"
                226560, -- "Void"
                226561, -- "Condom Cath"
                226584, -- "Ileoconduit"
                226563, -- "Suprapubic"
                226564, -- "R Nephrostomy"
                226565, -- "L Nephrostomy"
                226567, --	Straight Cath
                226557, -- R Ureteral Stent
                226558, -- L Ureteral Stent
                227488, -- GU Irrigant Volume In
                227489  -- GU Irrigant/Urine Volume Out
                )
                GROUP BY ie.subject_id, ie.hadm_id, ie.icustay_id
                ORDER BY ie.subject_id, ie.hadm_id, ie.icustay_id
                )
                SELECT 
                tp.subject_id,
                oa.{label},
                tp.hours_between_charttime,
                tp.valuenum
                FROM tmp_table tp
                LEFT JOIN oasis oa
                ON tp.icustay_id = oa.icustay_id
            """)
        elif table_name == 'labevents':
            result = self.dbloader.query(f"""
                WITH ranked_icustays AS (
                SELECT 
                    ie.*,
                    ROW_NUMBER() OVER(PARTITION BY ie.subject_id ORDER BY ie.intime DESC) AS rn
                FROM 
                    icustays ie
                ),
                icustay_tmp AS(		-- use last icustay
                    SELECT *
                    FROM ranked_icustays
                    WHERE rn = 1
                ),
                tmp_table AS
                (
                SELECT 
                ie.intime, ie.icustay_id,
                DATETIME_DIFF(le.charttime, ie.intime, 'HOUR') AS hours_between_charttime,
                DATETIME_DIFF(ie.outtime, ie.intime, 'HOUR') AS outtime_hour,
                le.*
                FROM labevents le
                LEFT JOIN icustay_tmp ie
                ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id
                AND le.charttime BETWEEN ie.intime AND ie.outtime           -- labevents is doesn't have icustay_id, so we need to use charttime to select correct icustay times
                AND le.charttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                AND DATETIME_DIFF(le.charttime, ie.intime, 'SECOND') > 0
                AND DATETIME_DIFF(le.charttime, ie.intime, 'HOUR') <= {time_limit}
                WHERE itemid = {itemid_tuple}
                )
                SELECT 
                tp.subject_id,
                tp.outtime_hour,
                oa.{label},
                tp.hours_between_charttime,
                tp.value,
                tp.valuenum
                FROM tmp_table tp
                LEFT JOIN oasis oa
                ON tp.icustay_id = oa.icustay_id AND tp.subject_id = oa.subject_id AND tp.hadm_id = oa.hadm_id
                WHERE 
                {label} IS NOT NULL		-- drop NULL since this flag is produced from OASIS implementation, which would give some NULLs
                AND
                hours_between_charttime IS NOT NULL	-- drop NULL since NULL is created from the left join for tmp_table, which gives NULL when condition for charttime is not satisfied
            """)
        elif table_name == 'inputevents' and feature_name in ['rate_epinephrine', 'rate_dopamine', 'rate_dobutamine', 'rate_norepinephrine']:
            result = self.dbloader.query(f"""
                WITH ranked_icustays AS (
                SELECT 
                ie.*,
                ROW_NUMBER() OVER(PARTITION BY ie.subject_id ORDER BY ie.intime DESC) AS rn
                FROM 
                icustays ie
                ),
                icustay_tmp AS(    -- use last icustay
                SELECT *
                FROM ranked_icustays
                WHERE rn = 1
                ),
                wt AS
                (
                SELECT ie.icustay_id
                -- ensure weight is measured in kg
                , AVG(CASE
                WHEN itemid IN (762, 763, 3723, 3580, 226512)
                THEN valuenum
                -- convert lbs to kgs
                WHEN itemid IN (3581)
                THEN valuenum * 0.45359237
                WHEN itemid IN (3582)
                THEN valuenum * 0.0283495231
                ELSE NULL
                END) AS weight

                FROM icustay_tmp ie
                LEFT JOIN chartevents c
                ON ie.icustay_id = c.icustay_id
                WHERE valuenum IS NOT NULL
                AND itemid IN
                (
                762, 763, 3723, 3580,                     -- Weight Kg
                3581,                                     -- Weight lb
                3582,                                     -- Weight oz
                226512 -- Metavision: Admission Weight (Kg)
                )
                AND valuenum != 0
                AND charttime BETWEEN DATETIME_SUB(ie.intime, INTERVAL '1' DAY) AND DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                -- exclude rows marked as error
                AND (c.error IS NULL OR c.error = 0)
                GROUP BY ie.icustay_id
                )
                , echo2 AS(
                SELECT ie.icustay_id, AVG(weight * 0.45359237) AS weight
                FROM icustay_tmp ie
                LEFT JOIN echo_data echo
                ON ie.hadm_id = echo.hadm_id
                AND echo.charttime > DATETIME_SUB(ie.intime, INTERVAL '7' DAY)
                AND echo.charttime < DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                GROUP BY ie.icustay_id
                )
                , vaso_cv AS
                (
                SELECT ie.icustay_id
                -- case statement determining whether the ITEMID is an instance of vasopressor usage
                , CASE
                WHEN itemid = 30047 THEN rate / COALESCE(wt.weight,ec.weight) -- measured in mcgmin
                WHEN itemid = 30120 THEN rate -- measured in mcgkgmin ** there are clear errors, perhaps actually mcgmin
                ELSE NULL
                END AS rate_norepinephrine

                , CASE
                WHEN itemid =  30044 THEN rate / COALESCE(wt.weight,ec.weight) -- measured in mcgmin
                WHEN itemid IN (30119,30309) THEN rate -- measured in mcgkgmin
                ELSE NULL
                END AS rate_epinephrine

                , CASE WHEN itemid IN (30043,30307) THEN rate END AS rate_dopamine
                , CASE WHEN itemid IN (30042,30306) THEN rate END AS rate_dobutamine
                , DATETIME_DIFF(cv.charttime, ie.intime, 'HOUR') AS hours_between_charttime
                , DATETIME_DIFF(ie.outtime, ie.intime, 'HOUR') AS outtime_hour

                FROM icustay_tmp ie
                INNER JOIN inputevents_cv cv
                ON ie.icustay_id = cv.icustay_id AND cv.charttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                LEFT JOIN wt
                ON ie.icustay_id = wt.icustay_id
                LEFT JOIN echo2 ec
                ON ie.icustay_id = ec.icustay_id
                WHERE itemid IN (30047,30120,30044,30119,30309,30043,30307,30042,30306)
                AND rate IS NOT NULL
                )
                , vaso_mv AS
                (
                SELECT ie.icustay_id
                -- case statement determining whether the ITEMID is an instance of vasopressor usage
                , CASE WHEN itemid = 221906 THEN rate END AS rate_norepinephrine
                , CASE WHEN itemid = 221289 THEN rate END AS rate_epinephrine
                , CASE WHEN itemid = 221662 THEN rate END AS rate_dopamine
                , CASE WHEN itemid = 221653 THEN rate END AS rate_dobutamine
                , DATETIME_DIFF(mv.starttime, ie.intime, 'HOUR') AS hours_between_charttime
                , DATETIME_DIFF(ie.outtime, ie.intime, 'HOUR') AS outtime_hour
                FROM icustay_tmp ie
                INNER JOIN inputevents_mv mv
                ON ie.icustay_id = mv.icustay_id AND mv.starttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '{time_limit}' HOUR)
                WHERE itemid IN (221906,221289,221662,221653)
                -- 'Rewritten' orders are not delivered to the patient
                AND statusdescription != 'Rewritten'
                )
                , tmp_table AS
                (
                SELECT ie.icustay_id, ie.hadm_id, ie.subject_id
                , COALESCE(cv.rate_norepinephrine, mv.rate_norepinephrine) AS rate_norepinephrine
                , COALESCE(cv.rate_epinephrine, mv.rate_epinephrine) AS rate_epinephrine
                , COALESCE(cv.rate_dopamine, mv.rate_dopamine) AS rate_dopamine
                , COALESCE(cv.rate_dobutamine, mv.rate_dobutamine) AS rate_dobutamine
                , COALESCE(cv.hours_between_charttime, mv.hours_between_charttime) AS hours_between_charttime
                , COALESCE(cv.outtime_hour, mv.outtime_hour) AS outtime_hour
                FROM icustay_tmp ie
                LEFT JOIN vaso_cv cv
                ON ie.icustay_id = cv.icustay_id
                LEFT JOIN vaso_mv mv
                ON ie.icustay_id = mv.icustay_id
                WHERE
                COALESCE(cv.hours_between_charttime, mv.hours_between_charttime) IS NOT NULL
                )
                SELECT
                tp.subject_id,
                tp.outtime_hour,
                oa.{label},
                tp.hours_between_charttime,
                tp.rate_epinephrine AS valuenum
                FROM tmp_table tp
                LEFT JOIN oasis oa
                ON tp.icustay_id = oa.icustay_id AND tp.subject_id = oa.subject_id AND tp.hadm_id = oa.hadm_id
                WHERE 
                {label} IS NOT NULL    -- drop NULL since this flag is produced from OASIS implementation, which would give some NULLs
                AND
                hours_between_charttime IS NOT NULL    -- drop NULL since NULL is created from the left join for tmp_table, which gives NULL when condition for charttime is not satisfied
            """)
        
        assert not result.empty, "No data collected!"
        self.logger.info(f"Time elapsed to collect samples: {self.timer() - start_time:.2f} seconds")
        
        return result
    
    def filter_disease(self, df: pd.DataFrame, disease: str) -> pd.DataFrame:
        '''filter patients in a dataframe by disease (using ICD9 codes)'''
        try: 
            icd9_codes = self.disease_dict[disease]
            icd9_codes = tuple(str(code) for code in icd9_codes)
        except KeyError: raise KeyError("Disease not found!")
        
        d_icd_diagnoses_df = self.dbloader['d_icd_diagnoses']
        disease_df = d_icd_diagnoses_df[d_icd_diagnoses_df['icd9_code'].str.startswith(icd9_codes)]
        code_series = disease_df['icd9_code']
        diagnoses_df = self.dbloader['diagnoses_icd']
        filtered = diagnoses_df[diagnoses_df['icd9_code'].isin(code_series)]
        
        subject_id = filtered['subject_id'].unique()
        df = df[df['subject_id'].isin(subject_id)]
        
        return df.copy()
    
    def plot(
        self, feature_name: str, time_limit: int, select: str=None, threshold=None, 
        num_patients: int=None, label: str='icustay_expire_flag', palette=None,
        subject_id : int=None, save_path: str=None, interval: int=2, disease: str=None,
        markersize: int=8, linewidth: int=2, upper_quantile: float=0.99, lower_quantile: float=0.01, fig_size: tuple=(10, 6),
        ) -> None:
                
        if select is not None:
            assert select in ['Alive', 'Expired'], "select must be either Alive or Expired!"
        samples = self.collect_samples(feature_name, label, time_limit)
        
        if disease is not None:
            samples = self.filter_disease(samples, disease)
            disease_placeholder = f"{disease}_"
        else:
            disease_placeholder = ''
            
        samples[label] = samples[label].replace({1:'Expired', 0:'Alive'})
        
        if threshold is not None:
            if isinstance(threshold, float) or isinstance(threshold, int):
                samples = samples[samples['valuenum'] < threshold]
            elif isinstance(threshold, tuple):
                samples = samples[(samples['valuenum'] > threshold[0]) & (samples['valuenum'] < threshold[1])]
            else: raise ValueError("threshold must be either float or tuple!")
            
        samples.rename(columns={label: 'Label', 'valuenum': 'Numeric Value', 'hours_between_charttime': 'Time (Hours)'}, inplace=True)
        samples['Numeric Value'] = samples['Numeric Value'].astype(float)
        if select is not None:
            samples = samples[samples['Label'] == select]
        samples['Label'] = pd.Categorical(samples['Label'], categories=['Alive', 'Expired'], ordered=True)
        palette = sns.color_palette() if palette is None else palette
        
        if subject_id is not None:      # NOTE: this is for checking individual patients
            samples = samples[samples['subject_id'] == subject_id]
            ax = sns.lineplot(
                data=samples, x='Time (Hours)', y='Numeric Value', palette={'Alive': palette[0], 'Expired': palette[1]}, hue='Label', 
                errorbar=None, marker='s', markersize=markersize, linewidth=linewidth
                )
            ax.figure.set_size_inches(fig_size[0], fig_size[1])
            if time_limit == 'outtime':
                assert len(group_df['outtime_hour'].unique()) == 1, "outtime_hour must be unique!"
                max_number = float(group_df['outtime_hour'].iloc[0])
                interval = round(max_number/12, 1)
                try:
                    plt.axvline(x=max_number, color='r', linestyle='--', linewidth=linewidth, label='ICU Out Time')
                    plt.xticks(np.arange(0, np.ceil(max_number+interval), interval))
                    plt.xlim(-1, np.ceil(max_number+1))
                except:
                    self.logger.warn(f"Skipped this patient due to plotting error: subject_id: {subject_id}, outtime: {max_number:.2f}, max_time: {group_df['Time (Hours)'].max():.2f}, interval: {interval}")
            else:
                plt.xticks(np.arange(0, np.ceil(time_limit+interval), interval))
                plt.xlim(-1, np.ceil(time_limit+1))
            plt.title(f"{feature_name} of patient {subject_id}")
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            plt.close()
            return samples
        std = samples['Numeric Value'].std()
        y_min, y_max = samples['Numeric Value'].quantile(lower_quantile) - std*0.5, samples['Numeric Value'].quantile(upper_quantile) + std*0.5
        samples = list(samples.groupby('subject_id'))
        
        self.logger.info(f"Plotting for {len(samples)} patients...")
        if save_path is None:       # if plotting subset, choose subset randomly, else plot all
            iterator = np.random.randint(0, len(samples), num_patients) if num_patients is not None else range(0, len(samples))
        else:
            iterator = tqdm(np.random.randint(0, len(samples), num_patients)) if num_patients is not None else tqdm(range(0, len(samples)))
            
        start_time = self.timer()
        two_sample_count, no_value_count = 0, 0
        alive_pdf, dead_pdf = PdfPages(f"{save_path}/{feature_name}_{disease_placeholder}alive.pdf"), PdfPages(f"{save_path}/{feature_name}_{disease_placeholder}expired.pdf")
        for i in iterator:
            subject_id, group_df = samples[i]
            if len(group_df) < 2: 
                two_sample_count += 1
                continue
            if group_df['Numeric Value'].isnull().all():
                no_value_count += 1
                continue
            ax = sns.lineplot(
                data=group_df, x='Time (Hours)', y='Numeric Value', palette={'Alive': palette[0], 'Expired': palette[1]}, hue='Label', 
                errorbar=None, marker='s', markersize=markersize, linewidth=linewidth
                )
            ax.figure.set_size_inches(fig_size[0], fig_size[1])   
            if time_limit == 'outtime':
                assert len(group_df['outtime_hour'].unique()) == 1, "outtime_hour must be unique!"
                max_number = float(group_df['outtime_hour'].iloc[0])
                interval = round(max_number/12, 1)
                try:
                    plt.axvline(x=max_number, color='r', linestyle='--', linewidth=linewidth, label='ICU Out Time')
                    plt.xticks(np.arange(0, np.ceil(max_number+interval), interval))
                    plt.xlim(-1, np.ceil(max_number+1))
                except:
                    self.logger.warn(f"Skipped this patient due to plotting error: subject_id: {subject_id}, outtime: {max_number:.2f}, max_time: {group_df['Time (Hours)'].max():.2f}, interval: {interval}")
            else:
                plt.xticks(np.arange(0, np.ceil(time_limit+interval), interval))
                plt.xlim(-1, np.ceil(time_limit+1))
            plt.ylim(y_min, y_max)
            plt.title(f"{feature_name} of patient {subject_id}")
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            if all(group_df['Label'] == 'Alive'):
                alive_pdf.savefig()
            elif all(group_df['Label'] == 'Expired'):
                dead_pdf.savefig()
            else:
                raise ValueError("Label must be either Alive or Expired!")
            plt.clf()
        alive_pdf.close()
        dead_pdf.close()
        self.logger.info(f"FINISH, time elapsed to plot: {self.timer() - start_time:.2f} seconds")
        self.logger.info(f"Skipped {two_sample_count} patients with less than two samples!\nSkipped {no_value_count} patients with no value!")
        if save_path is not None:
            self.logger.info(f"Plots saved to {save_path}/{feature_name}/")
        
    def plot_all(
        self, features: list, time_limit: int, num_patients: int, disease: str=None, label: str='icustay_expire_flag', markersize: int=8,
        linewidth: int=2, fig_size: tuple=(10, 6), save_path: str=None, palette=None, threshold: tuple=(-5, 10)
        ) -> None:
        
        samples_dict = {}
        pbar = tqdm(features)
        if threshold is None:
            ymax, ymin = -np.inf, np.inf
        else:
            ymin, ymax = threshold
        subject_ids = list(self.dbloader['oasis'].sample(n=num_patients, random_state=SEED)['subject_id'].unique())
        print(f"Plotting for {len(subject_ids)} patients...")
        
        for feature_name in pbar:
            pbar.set_description(f"Collecting samples for \"{feature_name}\"...")
            samples = self.collect_samples(feature_name, label, time_limit)
            
            if disease is not None:
                samples = self.filter_disease(samples, disease)
                disease_placeholder = f"{disease}_"
            else:
                disease_placeholder = ''
            
            samples[label] = samples[label].replace({1:'Expired', 0:'Alive'})
            samples.rename(columns={label: 'Label', 'valuenum': 'Numeric Value', 'hours_between_charttime': 'Time (Hours)'}, inplace=True)
            samples['Numeric Value'] = samples['Numeric Value'].astype(float)
            samples['Numeric Value'] = ( samples['Numeric Value'] - samples['Numeric Value'].mean() ) / samples['Numeric Value'].std()          # standardize
            sample_max, sample_min = samples['Numeric Value'].max(), samples['Numeric Value'].min()
            
            if threshold is None:
                if sample_max > ymax:
                    ymax = sample_max
                if sample_min < ymin:
                    ymin = sample_min
                
            samples['Label'] = pd.Categorical(samples['Label'], categories=['Alive', 'Expired'], ordered=True)
            samples_dict[feature_name] = samples
        
        palette = sns.color_palette() if palette is None else palette
        alive_pdf, dead_pdf = PdfPages(f"{save_path}/stacked_{disease_placeholder}alive.pdf"), PdfPages(f"{save_path}/stacked_{disease_placeholder}expired.pdf")
        start_time = self.timer()
        
        for subject_id in tqdm(subject_ids, desc='Plotting...'):
            alive, checker, empty_count, outtime_hour = False, None, 0, 0
            for idx, (feature, sample_df) in enumerate(samples_dict.items()):
                sample_df = sample_df[sample_df['subject_id'] == subject_id]
                
                if len(sample_df) < 2:
                    empty_count += 1
                    continue
            
                assert len(sample_df['outtime_hour'].unique()) == 1, "outtime_hour must be unique!"
                if outtime_hour != 0:
                    assert outtime_hour == float(sample_df['outtime_hour'].iloc[0]), "outtime_hour must be the same!"
                else:
                    outtime_hour = float(sample_df['outtime_hour'].iloc[0])
                
                if all(sample_df['Label'] == 'Alive'):
                    alive = True
                    if checker is not None and checker != alive:
                        raise ValueError(f"Patient {subject_id} is alive at some time but expired at other time!")
                    else:
                        checker = alive
                elif all(sample_df['Label'] == 'Expired'):
                    alive = False
                    if checker is not None and checker != alive:
                        raise ValueError(f"Patient {subject_id} is alive at some time but expired at other time!")
                    else:
                        checker = alive
                
                ax = sns.lineplot(
                    data=sample_df, x='Time (Hours)', y='Numeric Value', label=feature, color=palette[idx],
                    errorbar=None, marker='s', markersize=markersize, linewidth=linewidth
                    )
                ax.figure.set_size_inches(fig_size[0], fig_size[1])   
            
            if empty_count == len(samples_dict):
                self.logger.warning(f"Skipped patient {subject_id} due to no value!")
                continue
            else:
                if time_limit == 'outtime':
                    interval = round(outtime_hour/12, 1)
                    try:
                        plt.axvline(x=outtime_hour, color='r', linestyle='--', linewidth=linewidth, label='ICU Out Time')
                        plt.xticks(np.arange(0, np.ceil(outtime_hour+interval), interval))
                        plt.xlim(-1, np.ceil(outtime_hour+1))
                    except:
                        self.logger.warn(f"Skipped this patient due to plotting error: subject_id: {subject_id}, outtime: {outtime_hour:.2f}, max_time: {sample_df['Time (Hours)'].max():.2f}, interval: {interval}")
                else:
                    plt.xticks(np.arange(0, np.ceil(time_limit+interval), interval))
                    plt.xlim(-1, np.ceil(time_limit+1))
                plt.ylim(ymin, ymax)
                plt.title(f"Patient {subject_id}")
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
                plt.tight_layout()
                if alive:
                    alive_pdf.savefig()
                elif not alive:
                    dead_pdf.savefig()
                else:
                    raise ValueError("Label must be either Alive or Expired!")
                plt.clf()
            
        alive_pdf.close()
        dead_pdf.close()
        self.logger.info(f"FINISH, time elapsed to plot: {self.timer() - start_time:.2f} seconds")
        if save_path is not None:
            self.logger.info(f"Plots saved to {save_path}")

if __name__ == "__main__":
    plotter = TimeSeriesLinePlot(user='mt361', password='tian01050417')
    plotter.plot_all(features=['aado2', 'bicarbonate', 'bun', 'creatinine', 'fio2', 'pao2', 'sysbp'], save_path='src/exp_5.24_to_5.30/timeseries_stacked', num_patients=400, time_limit='outtime', disease=None)
