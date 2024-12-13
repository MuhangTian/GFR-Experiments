import math
import time
import warnings

import numpy as np
import pandas as pd
from mimic_pipeline.feature import (APSIIIFeatures, OASISFeatures,
                                    SAPSIIFeatures)
from mimic_pipeline.metric import HosmerLemeshow, get_calibration_curve
from mimic_pipeline.model import DecisionTreeClassifier
from mimic_pipeline.utils import DataBaseLoader, Table
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def test_equalness(a, b):
    '''test equalness of two array like objects'''
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Series) and isinstance(b, pd.Series):
        pd.testing.assert_series_equal(a, b)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        print(f"Whether equal: {np.array_equal(a, b)}")
    elif isinstance(a, list) and isinstance(b, list):
        print(f"Whether equal: {a == b}")
    else:
        raise ValueError("Not Supported!")
    
    print("Whether equal: True")        # if all pd.testing passes, then they are equal
    

def test_table_class(df1:pd.DataFrame, df2:pd.DataFrame, rounds:int=1000):
    '''run tests for Table() in Gadgets.py to ensure the behaviors are as expected'''
    assert list(df1.columns) == list(df2.columns), "columns don't match!"
    
    names = list(df1.drop("subject_id", axis=1).columns)
    table1, table2 = Table(df1), Table(df2)
    subject_id = df1["subject_id"]
    df1, df2 = df1.drop("subject_id", axis=1), df2.drop("subject_id", axis=1)
    
    for _ in tqdm(range(rounds), desc="Testing..."):
        randnum = np.random.randint(0, 4, 1)
        randnum_2 = np.random.randint(0, 4, 1)
        if randnum == 0:        # mult
            df1_np, df2_np = df1.to_numpy().astype(float), df2.to_numpy().astype(float)
            if randnum_2 == 0:
                tmp_np  = np.multiply(df1_np, df2_np)
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table1 * table2
            elif randnum_2 == 1:
                tmp_np  = np.multiply(df2_np, df1_np)
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table2 * table1
            elif randnum_2 == 2:
                tmp_np  = np.multiply(df1_np, df2_np)
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table1 * table2
            elif randnum_2 == 3:
                tmp_np  = np.multiply(df2_np, df1_np)
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table2 * table1
            else: raise ValueError("wrong random number!")
            df1.insert(0, "subject_id", subject_id)
            pd.testing.assert_frame_equal(df1, table1.df)
        elif randnum == 1:          # div
            df1_np, df2_np = df1.to_numpy().astype(float), df2.to_numpy().astype(float)
            if randnum_2 == 0:
                tmp_np  = np.divide(df1_np, df2_np)
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table1 / table2
            elif randnum_2 == 1:
                tmp_np  = np.divide(df2_np, df1_np)
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table2 / table1
            elif randnum_2 == 2:
                tmp_np  = np.divide(df1_np, df2_np)
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table1 / table2
            elif randnum_2 == 3:
                tmp_np  = np.divide(df2_np, df1_np)
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table2 / table1
            else: raise ValueError("wrong random number!")
            df1.insert(0, "subject_id", subject_id)
            pd.testing.assert_frame_equal(df1, table1.df)
        elif randnum == 2:         # sub
            df1_np, df2_np = df1.to_numpy().astype(float), df2.to_numpy().astype(float)
            if randnum_2 == 0:
                tmp_np  = df1_np - df2_np
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table1 - table2
            elif randnum_2 == 1:
                tmp_np  = df2_np - df1_np
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table2 - table1
            elif randnum_2 == 2:
                tmp_np  = df1_np - df2_np
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table1 - table2
            elif randnum_2 == 3:
                tmp_np  = df2_np - df1_np
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table2 - table1
            else: raise ValueError("wrong random number!")
            df1.insert(0, "subject_id", subject_id)
            pd.testing.assert_frame_equal(df1, table1.df)
        elif randnum == 3:      # add
            df1_np, df2_np = df1.to_numpy().astype(float), df2.to_numpy().astype(float)
            if randnum_2 == 0:
                tmp_np  = df1_np + df2_np
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table1 + table2
            elif randnum_2 == 1:
                tmp_np  = df2_np + df1_np
                df1 = pd.DataFrame(tmp_np, columns=names)
                table1 = table2 + table1
            elif randnum_2 == 2:
                tmp_np  = df1_np + df2_np
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table1 + table2
            elif randnum_2 == 3:
                tmp_np  = df2_np + df1_np
                df2 = pd.DataFrame(tmp_np, columns=names)
                table2 = table2 + table1
            else: raise ValueError("wrong random number!")
            df1.insert(0, "subject_id", subject_id)
            pd.testing.assert_frame_equal(df1, table1.df)
        else: raise ValueError("wrong random number!")
        
        df1 = df1.drop("subject_id", axis=1)
    
    print("PASS")

def test_binbinarizer_group_idx(rounds:int):
    """
    run tests on group index array returned by BinBinarizer

    Parameters
    ----------
    rounds : int
        number of unique datasets to test for
    """
    from mimic_pipeline.feature.BinBinarizer import BinBinarizer
    binarizer = BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)
    
    pbar = tqdm(range(rounds), desc=f'Testing group index array with {rounds} simulated datasets...')
    
    for i in pbar:
        train = np.random.uniform(0, 5000, size=(10000, 50))
        names = [str(e) for e in range(1, 51)]
        train = pd.DataFrame(train, columns=names)
        transformed, GroupIdx = binarizer.fit_transform(train)
        columns = list(transformed.columns)
        map = {}
        for i in range(len(columns)):
            col = columns[i]
            if 'isNaN' in col:
                feature = col.split('_')[0]
            else:
                feature = col.split("<=")[0]
            try:
                map[feature].append(GroupIdx[i])
            except:
                map[feature] = [GroupIdx[i]]

        for k, v in map.items():
            assert len(np.unique(v)) == 1
    print(f"{'*'*50} ALL PASS {'*'*50}")
    
def test_percentage_change_csv(df:pd.DataFrame, dbloader:DataBaseLoader):
    '''to check that the percentage change csv is indeed what we intend them to be'''
    df1_oasis = dbloader["oasis_0_to_12"]
    df2_oasis = dbloader["oasis_12_to_24"]
    
    df1_sapsii = dbloader["sapsii_0_to_12"]
    df2_sapsii = dbloader["sapsii_12_to_24"]
    
    df1_apsiii = dbloader["apsiii_0_to_12"]
    df2_apsiii = dbloader["apsiii_12_to_24"]

    print("CHECKING...")
    fail = 0
    for col in df.columns:
        if col in ["subject_id", "icustay_age_group"]: 
            continue
        col = col[:-4]
        if col in OASISFeatures().all():
            col1, col2 = df1_oasis[col].to_numpy(), df2_oasis[col].to_numpy()
            tmp_diff = list(2*(col2 - col1) / (col1 + col2) * 100)
            df_diff = list(df[f"{col}_pct"])
            for i in range(len(tmp_diff)):
                val1 = tmp_diff[i]
                val2 = df_diff[i]
                if math.isnan(val1) and math.isnan(val2):
                    continue
                elif val1 != val2:
                    if round(val1, 5) != round(val2, 5):
                        print(f"FAILED CASE: {val1}, {val2}")
                        print(f"Type: {type(val1)}, {type(val2)}")
                        fail = 1
        elif col in SAPSIIFeatures().all():
            col1, col2 = df1_sapsii[col].to_numpy(), df2_sapsii[col].to_numpy()
            tmp_diff = list(2*(col2 - col1) / (col1 + col2) * 100)
            df_diff = list(df[f"{col}_pct"])
            for i in range(len(tmp_diff)):
                val1 = tmp_diff[i]
                val2 = df_diff[i]
                if math.isnan(val1) and math.isnan(val2):
                    continue
                elif val1 != val2:
                    if round(val1, 5) != round(val2, 5):
                        print(f"FAILED CASE: {val1}, {val2}")
                        print(f"Type: {type(val1)}, {type(val2)}")
                        fail = 1
        elif col in APSIIIFeatures().all():
            col1, col2 = df1_apsiii[col].to_numpy(), df2_apsiii[col].to_numpy()
            tmp_diff = list(2*(col2 - col1) / (col1 + col2) * 100)
            df_diff = list(df[f"{col}_pct"])
            for i in range(len(tmp_diff)):
                val1 = tmp_diff[i]
                val2 = df_diff[i]
                if math.isnan(val1) and math.isnan(val2):
                    continue
                elif val1 != val2:
                    if round(val1, 5) != round(val2, 5):
                        print(f"FAILED CASE: {val1}, {val2}")
                        print(f"Type: {type(val1)}, {type(val2)}")
                        fail = 1
        else: raise ValueError()
        
    if fail == 0:
        print("PASS")

def test_binbinarizer_value(rounds:int=10) -> None:
    """
    Test BinBinarizer class with simulated data to ensure implementation is correct

    Parameters
    ----------
    rounds : int, optional
        number of simulated datasets to test, by default 10
    """
    from mimic_pipeline.feature import BinBinarizer
    print(f"Testing BinBinarizer with {rounds} rounds...")
    pbar = tqdm(range(1, rounds+1))
    
    for round in pbar:
        
        nan_percent = np.random.uniform(0.3, 0.8)                                 # initialize random NaN percentage
        pbar.set_description(f"Dataset {round}, {nan_percent*100:.2f}% NaN...")
        
        random_data = np.random.uniform(low=-1000, high=1000, size=(30000, 30))         # simulation data
        random_data = np.hstack((
            random_data,                                                # continuous data
            np.random.randint(low=0, high=2, size=(30000, 10)),         # binary  
            np.random.randint(low=0, high=20, size=(30000, 10)),        # ordinal
        ))
        np.random.shuffle(random_data.transpose())                      # shufflling
        
        nan_mask = np.zeros(random_data.shape, dtype=bool)              # add NaNs
        nan_mask[:int(len(random_data)*nan_percent)] = True
        np.random.shuffle(nan_mask)
        nan_mask = nan_mask.reshape(random_data.shape)
        random_data[nan_mask] = np.nan
        
        original_data = pd.DataFrame(random_data, columns=[str(i) for i in range(1, random_data.shape[1]+1)])       # make into pd.DataFrame
        binned_data, group_idx = BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True).fit_transform(original_data)
        
        for original_column in list(original_data.columns):
            col_tmp = []
            for col in binned_data.columns:
                if f'{original_column}_isNaN' == col:
                    col_tmp.append(col)
                if original_column == col.split('<=')[0]:
                    col_tmp.append(col)
            binned_tmp, original_tmp = binned_data[col_tmp].to_numpy(), original_data[original_column].to_numpy()
            col_val = np.asarray([float(col.split('<=')[1]) for col in col_tmp if 'isNaN' not in col])
            
            for i in range(len(binned_tmp)):
                row, row_original = binned_tmp[i], original_tmp[i]
                if np.isnan(row_original):
                    assert row[0] == 1, 'FAIL'
                    assert np.unique(row[1:]) == 0, 'FAIL'
                else:
                    nonzero_idx = np.where(row[1:] == 1)[0]
                    nonzero_idx = np.insert(nonzero_idx, 0, nonzero_idx[0]-1)
                    thresholds = col_val[nonzero_idx][:2]
                    if -1 not in nonzero_idx:               # when not less than the first bin
                        assert row_original <= thresholds[1] and row_original > thresholds[0], 'FAIL'
                    else:                                   # when less than the first bin
                        assert row_original <= thresholds[0], 'FAIL'
            
        pbar.set_description(f"Dataset {round}: PASS")
        time.sleep(2)
            
    print(f"{'*'*50} ALL PASS {'*'*50}")

def truth_HosmerLemeshow(model, X_test, Y, pihat, bins=4):
    pihatcat=pd.cut(pihat, np.linspace(0.0, 1.0, bins + 1), labels=False, include_lowest=True) #here we've chosen only bins groups

    meanprobs =[0]*bins 
    expevents =[0]*bins
    obsevents =[0]*bins 
    meanprobs2=[0]*bins 
    expevents2=[0]*bins
    obsevents2=[0]*bins 

    warnings.filterwarnings("ignore")
    for i in range(bins):
       meanprobs[i]=np.mean(pihat[pihatcat==i])
       expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
       obsevents[i]=np.sum(Y[pihatcat==i])
       meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
       expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
       obsevents2[i]=np.sum(1-Y[pihatcat==i]) 
    warnings.filterwarnings('default')
    
    non_nan = ~np.isnan(meanprobs)
    meanprobs = np.asarray(meanprobs)
    meanprobs2 = np.asarray(meanprobs2)
    expevents = np.asarray(expevents)
    expevents2 = np.asarray(expevents2)
    obsevents = np.asarray(obsevents)
    obsevents2 = np.asarray(obsevents2)
    data1={'meanprobs':meanprobs[non_nan],'meanprobs2':meanprobs2[non_nan]}
    data2={'expevents':expevents[non_nan],'expevents2':expevents2[non_nan]}
    data3={'obsevents':obsevents[non_nan],'obsevents2':obsevents2[non_nan]}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)
    
    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus bins - 2 = 2
    tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    pvalue=1-chi2.cdf(tt, len(data3['obsevents'])-2)

    return tt, pvalue
    
def test_HosmerLemeshow(iter: int=10):
    pbar = tqdm(range(1, iter+1))
    for i in pbar:
        model = LogisticRegression()
        # generate random data
        bins = np.random.randint(low=2, high=30)
        X, y = np.random.normal(loc=0, scale=1, size=(30000, 30)), np.random.randint(low=0, high=2, size=(30000,))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        tt, pvalue = truth_HosmerLemeshow(model=model, Y=y_test, X_test=X_test, pihat=y_prob, bins=bins)
        _, _, tt2, pvalue2 = get_calibration_curve(y_test, y_prob, n_bins=bins)
        
        if np.abs(tt2 - tt) > 1e-5:
            print(f"{'*'*50} FAIL {'*'*50}")
        elif np.abs(pvalue - pvalue2) > 1e-5:
            print(f"{'*'*50} FAIL {'*'*50}")
    
    print(f"{'*'*50} ALL PASS {'*'*50}")

def test_decision_tree_split_calculation(iter: int=10000):
    for _ in tqdm(range(iter), desc='Testing Decision Tree Split Calculation...'):
        X_train, y_train = np.random.normal(size=(5000, 30)), np.random.randint(low=0, high=2, size=(5000,))
        tree = DecisionTreeClassifier(max_depth=np.random.randint(low=1, high=30))
        tree.fit(X_train, y_train)
        tree = tree.tree_
        true_splits_num = len(tree.threshold[tree.threshold != -2])
        calc_splits_num = tree.node_count - tree.n_leaves
        assert true_splits_num == calc_splits_num, 'FAIL'
    print(f"{'*'*50} ALL PASS {'*'*50}")

if __name__ == "__main__":
    test_decision_tree_split_calculation()