import os
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

from main_ds import correlation_cog, correlation_tau, ab_status_and_apoe4

def synthetic_analysis(data_dir):
    cog_results = {'all': [], 'CN': [], 'MCI': [], 'Dementia': []}
    tau_results = {'all': [], 'ab_pos': [], 'ab_neg': []}
    perc_results = {'CN': [], 'MCI': [], 'Dementia': []}

    for f in tqdm(os.listdir(data_dir)):

        df = pd.read_csv(os.path.join(data_dir, f))
        cog_mat = correlation_cog(df)
        tau_mat = correlation_tau(df)
        perc_mat = ab_status_and_apoe4(df)
    
        for key in cog_mat.keys():
            cog_results[key].append(cog_mat[key])
        for key in tau_mat.keys():
            tau_results[key].append(tau_mat[key])
        for key in perc_mat.keys():
            perc_results[key].append(perc_mat[key])

    for key in cog_results.keys():
        print(f'{key}: {np.round(np.mean(cog_results[key]), 3)} +- {np.round(np.std(cog_results[key]), 3)}')

    for key in tau_results.keys():
        print(f'{key}: {np.round(np.mean(tau_results[key]),3)} +- {np.round(np.std(tau_results[key]), 3)}')

    for key in perc_results.keys():
        avg = np.round(np.mean(perc_results[key], axis=0),3) * 100
        stdev = np.round(np.std(perc_results[key], axis=0),3) * 100
        print(f'{key} AB+: {avg[0]} +- {stdev[0]}')
        for i in range(1,4):
            print(f'    APOE4 = {i-1}: {avg[i]} +- {stdev[i]}')

    return cog_results, tau_results, perc_results


def save_to_xlc(cog_results, tau_results, perc_results, dataset_name=None, file_path='/home/fi5666wi/Python/WASP-DDLS/results.xlsx'):
    # Create a list of tuples with mean and std values
    results = [
        ("ADAS ~ MMSE", np.round(np.mean(cog_results['all']), 3), np.round(np.std(cog_results['all']), 3)),
        ("PTAU ~ TAU", np.round(np.mean(tau_results['all']), 3), np.round(np.std(tau_results['all']), 3)),
        ("AB+ (CN)", np.round(np.mean(perc_results['CN'], axis=0)[0], 3) * 100, np.round(np.std(perc_results['CN'], axis=0)[0], 3) * 100),
        ("APOE4 = 0 (CN)", np.round(np.mean(perc_results['CN'], axis=0)[1], 3) * 100, np.round(np.std(perc_results['CN'], axis=0)[1], 3) * 100),
        ("APOE4 = 1 (CN)", np.round(np.mean(perc_results['CN'], axis=0)[2], 3) * 100, np.round(np.std(perc_results['CN'], axis=0)[2], 3) * 100),
        ("APOE4 = 2 (CN)", np.round(np.mean(perc_results['CN'], axis=0)[3], 3) * 100, np.round(np.std(perc_results['CN'], axis=0)[3], 3) * 100),
        ("AB+ (MCI)", np.round(np.mean(perc_results['MCI'], axis=0)[0], 3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[0], 3) * 100),
        ("APOE4 = 0 (MCI)", np.round(np.mean(perc_results['MCI'], axis=0)[1], 3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[1], 3) * 100),
        ("APOE4 = 1 (MCI)", np.round(np.mean(perc_results['MCI'], axis=0)[2], 3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[2], 3) * 100),
        ("APOE4 = 2 (MCI)", np.round(np.mean(perc_results['MCI'], axis=0)[3], 3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[3], 3) * 100),
        ("AB+ (AD)", np.round(np.mean(perc_results['Dementia'], axis=0)[0], 3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[0], 3) * 100),
        ("APOE4 = 0 (AD)", np.round(np.mean(perc_results['Dementia'], axis=0)[1], 3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[1], 3) * 100),
        ("APOE4 = 1 (AD)", np.round(np.mean(perc_results['Dementia'], axis=0)[2], 3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[2], 3) * 100),
        ("APOE4 = 2 (AD)", np.round(np.mean(perc_results['Dementia'], axis=0)[3], 3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[3], 3) * 100)
    ]

    # Create a DataFrame with alternating mean and std values in one row
    columns = []
    values = []
    for label, mean, std in results:
        columns.append(f"{label} Mean")
        columns.append(f"{label} Std")
        values.append(mean)
        values.append(std)

    new_df = pd.DataFrame([values], columns=columns)
    if dataset_name is not None:
        new_df.insert(0, 'Dataset', dataset_name)

    # Check if the file exists
    if os.path.exists(file_path):
        # Append to the existing file
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Get the last row in the existing Excel file
            existing_df = pd.read_excel(file_path)
            startrow = len(existing_df) + 1

            # Write the new data
            new_df.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        # Write a new file
        new_df.to_excel(file_path, index=False)


if __name__ == '__main__':
    dataset = 'adni'
    sp_dir = f'/home/fi5666wi/Python/WASP-DDLS/synthpop/{dataset}/'
    settings = ['default', 'optim']
    ctgan_dirs = [f'/home/fi5666wi/Python/WASP-DDLS/ctgan/{dataset}/ctgan_{s}_epochs_750/' for s in settings]
    tabpfn_dirs = [f'/home/fi5666wi/Python/WASP-DDLS/tabpfn/{dataset}/tabpfn_t_{t}/' for t in [1.0, 0.75, 0.5, 0.25]]

    for idir in [sp_dir] + ctgan_dirs + tabpfn_dirs:
        print(f'Analyzing {idir}')
        cog_results, tau_results, perc_results = synthetic_analysis(idir)
        save_to_xlc(cog_results, tau_results, perc_results, dataset_name=idir.split('/')[-2], file_path='/home/fi5666wi/Python/WASP-DDLS/results_final.xlsx')

    epsilons = [5, 10, 50, 100, 200, 'zero']
    for epsilon in epsilons:
        bn_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/{dataset}/degree2_eps_{epsilon}'
        print(f'Analyzing {bn_dir}')
        cog_results, tau_results, perc_results = synthetic_analysis(bn_dir)
        save_to_xlc(cog_results, tau_results, perc_results, dataset_name=bn_dir.split('/')[-1], file_path='/home/fi5666wi/Python/WASP-DDLS/results_final.xlsx')
        