# Description: utility functions to make the main script more readable
# Author: Anton D. Lautrup
# Date: 02-05-2024

import os
import pandas as pd

def dataset_train_test(path, frac = 0.8):
    df_base = pd.read_csv(path + '.csv')

    df_train = df_base.sample(frac=frac, random_state=42).dropna()
    df_test = df_base.drop(df_train.index).dropna()

    df_train.to_csv(path + '_train.csv', index=False)
    df_test.to_csv(path + '_test.csv', index=False)

def prepare_directories(data_dict):
    """Function required or Synthpop and Datasynthesizers halt due to forced file printing"""
    for data_name in data_dict.keys():
        directory = 'synthesis_info_' + os.path.dirname(data_name)
        # print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

def results_formatting(bmk_df, data_path, data_size):
    """Function to format the results of the benchmarking"""

    ### Change from MultiIndex to regular form
    bmk_df.columns = ['_'.join(col) for col in bmk_df.columns.values]
    bmk_df.reset_index(inplace=True)
    bmk_df.rename(columns={'dataset':'model'}, inplace=True)

    ### Add metadata to the benchmarking results
    meta_df = pd.DataFrame([{
        'dataset': data_path.split('/')[-2],
        'size_rcds': data_size[0],
        'size_atts': data_size[1]}]*len(bmk_df))
    
    bmk_df = pd.concat([meta_df, bmk_df], axis=1)

    ### Add generation time from elapsed_times file
    #elapsed_times = pd.read_csv('elapsed_times.csv')
    #elapsed_times = elapsed_times[elapsed_times['dataset'] == data_path]
    #elapsed_times = elapsed_times[['model', 'time']]
    #bmk_df = bmk_df.merge(elapsed_times, on='model', how='left')
    return bmk_df

def rank_results_formatting(bmk_df, data_path, data_size):
    """Function to format the results of the benchmarking"""

    ### Change from MultiIndex to regular form
    # bmk_df.columns = ['_'.join(col) for col in bmk_df.columns.values]
    bmk_df.reset_index(inplace=True)
    bmk_df.rename(columns={'dataset':'model'}, inplace=True)

    ### Add metadata to the benchmarking results
    meta_df = pd.DataFrame([{
        'dataset': data_path.split('/')[-2],
        'size_rcds': data_size[0],
        'size_atts': data_size[1]}]*len(bmk_df))
    
    bmk_df = pd.concat([meta_df, bmk_df], axis=1)

    ### Add generation time from elapsed_times file
    #elapsed_times = pd.read_csv('elapsed_times.csv')
    #elapsed_times = elapsed_times[elapsed_times['dataset'] == data_path]
    #elapsed_times = elapsed_times[['model', 'time']]
    #bmk_df = bmk_df.merge(elapsed_times, on='model', how='left')
    return bmk_df