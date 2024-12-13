import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mimic_pipeline.utils import (DataBaseLoader, Loader, plt_save_or_show,
                                  tfont, tsfont)
from numpy.typing import *
from tqdm import tqdm

sns.set_theme()

def plot_gcs_mortality_rate(df: pd.DataFrame, save: bool=False, split: str="TRAIN"):
    '''plot mortality rate for each GCS'''
    gcs_df = df[["gcs", "icustay_expire_flag"]]
    gcs, rate = np.sort(df["gcs"].unique()), []
    for e in gcs:
        tmp = gcs_df[gcs_df["gcs"] == e]
        rate.append(len(tmp[tmp["icustay_expire_flag"] == 1]) / len(tmp))
    sns.barplot(y=rate, x=gcs, palette="magma")
    plt.xlabel('GCS')
    plt.ylabel("Proportion")
    plt.title(f"Proportion of Deaths for Each GCS Group ({split})", **tfont)
    if not save:
        plt.show()
    else:
        plt.savefig(f"visuals/oasis/proportion-deaths-gcs-{split}.png", dpi=200)
        print("SAVE COMPLETE, saved at visuals/oasis")


def plot_gcs_patients(df: pd.DataFrame, title: str):
    '''plot patient numbers for each GCS'''
    gcs_df = df[["gcs", "icustay_expire_flag"]]
    gcs, rate = np.sort(df["gcs"].unique()), []
    for e in gcs:
        tmp = gcs_df[gcs_df["gcs"] == e]
        rate.append(len(tmp))
    sns.barplot(y=rate, x=gcs, palette="magma")
    plt.xlabel('GCS')
    plt.ylabel("Count")
    plt.title(title, **tfont)
    plt.show()


def plot_sedation_by_gcs(df: pd.DataFrame, save: bool=False, split: str="TRAIN"):
    '''plot proportion of sedated patients for each GCS group'''
    gcs_arr = np.sort(df["gcs"].unique())[:-1]
    prop_arr = []
    for gcs in gcs_arr:
        if gcs == "nan": continue
        tmp = df[df["gcs"].astype(float) == gcs]
        prop = len(tmp[tmp["is_sedated"] == 1])/len(tmp)
        prop_arr.append(prop)
    sns.barplot(x=gcs_arr, y=prop_arr, palette="magma")
    plt.title(f"Proportion of Sedated Patients for Each GCS Group ({split})", **tfont)
    plt.xlabel("GCS")
    plt.ylabel("Proportion")
    if not save:
        plt.show()
    else:
        plt.savefig(f"visuals/oasis/proportion-sedated-gcs-{split}.png", dpi=200)
        print("SAVE COMPLETE, saved at visuals/oasis")
        

def plot_patients_number_by_gcs(df: pd.DataFrame, save: bool=False, split: str="TRAIN", verbose: bool=False):
    '''plot number of sedated patients for each GCS group'''
    gcs_arr = np.sort(df["gcs"].astype(int).unique())
    numbers = []
    for gcs in gcs_arr:
        tmp = df[df["gcs"] == gcs]
        numbers.append(len(tmp))
        if verbose == True:
            print(f"With GCS of {gcs}: {len(tmp)} patients")
    sns.barplot(x=gcs_arr, y=numbers, palette="magma")
    plt.title(f"Number of Patients for Each GCS Group ({split})", **tfont)
    plt.xlabel("GCS")
    plt.ylabel("Patients")
    if not save:
        plt.show()
    else:
        plt.savefig(f"visuals/oasis/patients-number-gcs-{split}.png", dpi=200)
        print("SAVE COMPLETE, saved at \"visuals/oasis\"")
    

def plot_mortality_by_icd9(df: pd.DataFrame, ZeroMortalityDict: dict, topK: int, save: bool=False):
    '''plot mortality by ICD9 codes for codes with non-zero mortality'''
    loader = Loader("data/full")
    diagnoses = loader["DIAGNOSES_ICD"]
    diagnoses = diagnoses[diagnoses["seq_num"] == "1"]
    diagnoses = diagnoses.drop_duplicates(subset=["subject_id"], keep=False)
    
    MortalityDict = {}
    ZeroMortalityICD9 = ZeroMortalityDict.keys()
    for code in tqdm(diagnoses["icd9_code"].str[:3].unique(), desc="Counting mortality for each ICD9 code..."):
        if code in ZeroMortalityICD9:
            continue
        else:
            tmp = diagnoses[diagnoses["icd9_code"].str.startswith(code)]
            tmp2 = df[df["subject_id"].isin(tmp["subject_id"].astype(int))]
            if len(tmp2) != 0:
                assert len(tmp2[tmp2["icustay_expire_flag"].astype(int) == 1]) != 0, "must be non-zero mortality!"
                MortalityDict[code] = list(tmp2["subject_id"])
            else:
                continue
    
    MortalityCount, dead_num = [], len(df[df["icustay_expire_flag"].astype(int) == 1])
    for tmp_icd9, tmp_id in MortalityDict.items():
        tmp_df = df[df["subject_id"].isin(tmp_id)]
        mortality = len(tmp_df[tmp_df["icustay_expire_flag"].astype(int) == 1])/dead_num
        MortalityCount.append(mortality)
    
    idx_arr = np.flip(np.argsort(MortalityCount))
    mortality = np.flip(np.sort(MortalityCount))
    codes = [list(MortalityDict.keys())[idx] for idx in idx_arr]
    sns.barplot(y=codes[:topK], x=mortality[:topK], palette="viridis", orient="h")
    plt.xlabel("Death Contribution (%)")
    plt.title(f"Top {topK} ICD9 Codes (First Three Digits)", **tfont)
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig("visuals/oasis/barplot-icd9-mortality-rank.png", dpi=200)
        print("SAVE COMPLETE, saved at visuals/oasis")


def plot_all_distributions(Data:pd.DataFrame, save_path:str=None, label_column: str='icustay_expire_flag') -> None:
    """
    plot all distributions and save the visualizations to "save_path" using plot_distribution() as a helper
    Parameters
    ----------
    Data : pd.DataFrame
    save_path : str
    """
    assert isinstance(Data, pd.DataFrame), "must be pd.DataFrame!"
    pbar = tqdm(Data.columns)
    for column in pbar:
        if column == label_column:
            continue
        pbar.set_description(f"Processing and saving {column}...")
        plot_distribution(Data, column, plot=False, label_column=label_column)
        if save_path is not None:
            plt_save_or_show(f"{save_path}/{column}.png", verbose=False)
        else:
            plt_save_or_show(save_path=None)


def plot_distribution(Data:pd.DataFrame, column:str, label_column: str, save_path:str=None, plot:bool=True, xlim: tuple=None, ylim: tuple=None, stat='density'):
    """
    generate plot for a single column in a dataframe, separatedly for expired and alive patients
    Parameters
    ----------
    Data : pd.DataFrame
    column : str
    save_path : str, optional
        path to save the visualization, by default None
    plot : bool, optional
        whether to plot the visualization, by default True
    """
    sns.set_style("ticks")
    if 0 in np.unique(list(Data[label_column])):
        negative = 0
    elif -1 in np.unique(list(Data[label_column])):
        negative = -1
    
    tmp_data = Data[[column, label_column]]
    nan_percent = round(tmp_data[column].isna().sum()/len(tmp_data), 2)*100
    dead_data = tmp_data[tmp_data[label_column] == 1]
    alive_data = tmp_data[tmp_data[label_column] == negative]
    
    data_column = pd.concat([dead_data[column], alive_data[column]])
    label_column = pd.concat([
        dead_data[label_column].replace({1:'Expired', 0:'Alive', negative:'Alive'}),
        alive_data[label_column].replace({1:'Expired', 0:'Alive', negative:'Alive'}),
    ])
    
    plot_data = pd.DataFrame.from_dict({
        column: data_column,
        'Label': label_column
    })
    
    palette = sns.color_palette('Paired')
    
    ax = sns.histplot(
        data=plot_data, x=column, hue='Label', hue_order=["Alive", "Expired"],
        palette=[palette[1], palette[5]], stat=stat, 
        common_norm=False,
    )
    ax.figure.set_size_inches(10, 8)

    if tmp_data[column].dtype == float or tmp_data[column].dtype == int:    
        plt.title(f"{column}, Max={round(tmp_data[column].max(), 2)}, Min={round(tmp_data[column].min(), 2)}, Mean={tmp_data[column].mean():.2f}, Std={tmp_data[column].std():.2f}, Median={tmp_data[column].median():.2f}, NaN={nan_percent:.1f}%", **tfont)
    else:
        plt.title(f"{column}, NaN={nan_percent}%", **tsfont)
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if plot:
        plt_save_or_show(save_path)


def plot_nan_summary(data:pd.DataFrame, save_path:str=None, tolerance:float=0.05, autosize:bool=True) -> None:
    """
    generate a visualization of percentage of NaNs in each column in the data, ranked by decreasing order
    
    Parameters
    ----------
    data : pd.DataFrame
    save_path : str, optional
        path to save the visualization, by default None
    tolerance : float, optional
        tolerance for the percentage of NaNs, smaller than which is not visualized, by default 0.05
    """
    nan_percent_array = np.asarray([data[col].isnull().sum() / len(data) for col in data.columns])
    columns = np.asarray(list(data.columns))[nan_percent_array > tolerance]
    
    nan_percent_array = nan_percent_array[nan_percent_array > tolerance]
    columns = columns[np.flip(np.argsort(nan_percent_array))]
    nan_percent_array = np.flip(np.sort(nan_percent_array))
    
    ax = sns.barplot(x=nan_percent_array*100, y=columns, orient='h', palette='magma').legend()
    
    plt.xlabel('NaN %')
    if autosize:
        ax.figure.set_size_inches(12, int(0.3*len(columns)))
    plt.tight_layout()
    plt.title(f'NaN Summary, for features with more than {tolerance*100}% of NaNs', **tsfont)
    plt_save_or_show(save_path)


def plot_summary_over_time(dbloader:DataBaseLoader, feature:str, table:str, interval:int, save_path:str=None, figwidth:int=5):
    """
    Generating plots for summary statistics over time

    Parameters
    ----------
    dbloader : DataBaseLoader
    feature : str
        name of the feature of interest
    table : str
        table prefix where the data is from
    interval : int
        length of interval
    save_path : str, optional
        path to save the figure, by default None
    figwidth : int, optional
        width of the figure, by default 5
    """
    fig, axes = plt.subplots(2,1, sharex=True)
    fig.set_figheight(9)
    fig.set_figwidth(figwidth)
    mean_arr, std_arr, nan_arr, time_arr = [], [], [], []
    
    for itvl in range(0, 24, interval):
        suffix = f"{itvl}_to_{itvl+interval}"
        data_tmp = dbloader[f"{table}_{suffix}"]
        mean_arr.append(data_tmp[feature].mean())
        std_arr.append(data_tmp[feature].std())
        nan_arr.append(data_tmp[feature].isnull().sum() / len(data_tmp)*100)
        time_arr.append(suffix)
    
    ax = sns.barplot(x=time_arr, y=mean_arr, ax=axes[0], palette='crest')
    ax.errorbar(x=time_arr, y=mean_arr, yerr=std_arr, ls="", lw=2, color="black", capsize=10, capthick=2)
    ax.set(ylabel='Mean Value')
    
    ax2 = sns.barplot(x=time_arr, y=nan_arr, ax=axes[1], palette='crest')
    ax2.set(xlabel='Intervals', ylabel='NaN %')
    plt.suptitle(f"{feature} Over Time", **tfont)
    plt.tight_layout()
    
    plt_save_or_show(save_path)