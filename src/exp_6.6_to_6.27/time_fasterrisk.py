import argparse
import timeit

import mimic_pipeline.utils as utils
import numpy as np
import pandas as pd
from mimic_pipeline.feature import BinBinarizer
from mimic_pipeline.model import FasterRisk

import wandb


def evaluate_fasterrisk_time(group_sparsity_list, sparsity_list, X_train, y_train, group_idx):
    timer = timeit.default_timer
    wandb.log({'starting time': timer()})
    for group_sparsity in group_sparsity_list:
        print(f"{'*'*50} GROUP SPARSITY: {group_sparsity} {'*'*50}")
        for sparsity in sparsity_list:
            print(f"{'-'*30} SPARSITY: {sparsity} {'-'*30}")
            time_arr = []
            for i in range(1, 6):
                model = FasterRisk(group_sparsity=group_sparsity, k=sparsity, featureIndex_to_groupIndex=group_idx)
                start_time = timer()
                model.fit(X_train, y_train)
                time_spent = (timer() - start_time) / 60
                time_arr.append(time_spent)
                print(f"REPETITION {i}, time elapsed: {time_spent:.2f} mins")
                
            wandb.log({
                'Mean Time Elapsed (mins)': np.asarray(time_arr).mean(),
                'Std Time Elapsed (mins)': np.asarray(time_arr).std(),
                'Group Sparsity': group_sparsity,
                'Sparsity': sparsity,
            })
                

if __name__ == '__main__':
    wandb.init(
        entity='dukeds-mimic2023',
        project='MIMIC (6.15)',
        name=f'TIME FasterRisk',
        group='FasterRisk MIMIC',
        save_code=True,
    )
    utils.seed_everything()
    group_sparsity = [5, 10, 20, 30, 40, 45]
    sparsity = [20, 40, 60, 80, 100]
    
    train = pd.read_csv('src/exp_6.6_to_6.27/data/TRAIN-union-features.csv')
    test = pd.read_csv('src/exp_6.6_to_6.27/data/TEST-union-features.csv')
    entire = pd.concat([train, test], axis=0)
    
    X_train, y_train = entire.drop('hospital_expire_flag', axis=1), entire['hospital_expire_flag']
    print(f"Shape: {X_train.shape}")
    binarizer = BinBinarizer(interval_width=1, whether_interval=False, group_sparsity=True)
    X_train, group_idx = binarizer.fit_transform(X_train)
    print(f"Group Sparsity: {group_sparsity}\nSparsity: {sparsity}")
    evaluate_fasterrisk_time(group_sparsity, sparsity, X_train, y_train, group_idx)
    
    