import os
import json

import pandas as pd
import numpy as np
from tqdm import tqdm


def get_correlation_matrix(data_dir, correlation_pairs):
    corr_results = {f'{pair[0]} ~ {pair[1]}': [] for pair in correlation_pairs}
    for f in tqdm(os.listdir(data_dir)):
        df = pd.read_csv(os.path.join(data_dir, f))
        for pair in correlation_pairs:
            name = f'{pair[0]} ~ {pair[1]}'
            corr_results[name].append(df[[pair[0], pair[1]]].corr()[pair[0]][pair[1]])
    return corr_results


def get_percentages(data_dir, status_col, group_col=None):
    perc_results = {}
    for f in tqdm(os.listdir(data_dir)):
        df = pd.read_csv(os.path.join(data_dir, f))
        if group_col:
            groups = df[group_col].unique()
            for group in groups:
                key = f'{group_col}={group}'
                if key not in perc_results:
                    perc_results[key] = []
                sub_df = df[df[group_col] == group]
                perc = sub_df[status_col].mean() * 100
                perc_results[key].append(perc)
        else:
            if 'all' not in perc_results:
                perc_results['all'] = []
            df[status_col] = df[status_col].map({'negative': 0, 'positive': 1})
            perc = df[status_col].mean() * 100
            perc_results['all'].append(perc)
    return perc_results

def save_to_xlc(corr, perc, dataset_name=None, file_path='/home/fi5666wi/Python/WASP-DDLS/results.xlsx'):
    # Create a list of tuples with mean and std values
    results = [  
        ("AB+", np.round(np.mean(perc['all'], axis=0), 3), np.round(np.std(perc['all'], axis=0), 3)),
    ]

    for key in corr.keys():
        mean_corr = np.round(np.mean(corr[key]), 3)
        std_corr = np.round(np.std(corr[key]), 3)
        results.append((key, mean_corr, std_corr))

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
    dataset = 'a4'
    sp_dir = f'/home/fi5666wi/Python/WASP-DDLS/synthpop/{dataset}/'
    settings = ['default', 'optim']
    ctgan_dirs = [f'/home/fi5666wi/Python/WASP-DDLS/ctgan/{dataset}/ctgan_{s}_epochs_750/' for s in settings]
    tabpfn_dirs = [f'/home/fi5666wi/Python/WASP-DDLS/tabpfn/{dataset}/tabpfn_t_{t}/' for t in [1.0]]
    ds_dirs = [f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/{dataset}/degree2_eps_{epsilon}' for epsilon in [5, 10, 50, 100, 200, 'zero']]

    corrs = [('MMSCORE', 'DIGITTOTAL'), ('MMSCORE', 'LDELTOTAL'), ('LIMMTOTAL', 'LDELTOTAL')]


    for idir in ctgan_dirs + tabpfn_dirs:
        print(f'Analyzing {idir}')
        corr = get_correlation_matrix(idir, corrs)
        ab_plus = get_percentages(idir, status_col='AB_status')

        save_to_xlc(corr, ab_plus, dataset_name=idir.split('/')[-2], 
                    file_path='/home/fi5666wi/Python/WASP-DDLS/results_a4_final.xlsx')
        
    for idir in ds_dirs:
        print(f'Analyzing {idir}')
        corr = get_correlation_matrix(idir, corrs)
        ab_plus = get_percentages(idir, status_col='AB_status')

        save_to_xlc(corr, ab_plus, dataset_name=idir.split('/')[-1], 
                    file_path='/home/fi5666wi/Python/WASP-DDLS/results_a4_final.xlsx')
        
    print(f'Analyzing {sp_dir}')
    corr = get_correlation_matrix(sp_dir, corrs)
    ab_plus = get_percentages(sp_dir, status_col='AB_status')

    save_to_xlc(corr, ab_plus, dataset_name=sp_dir.split('/')[-3], 
                file_path='/home/fi5666wi/Python/WASP-DDLS/results_a4_final.xlsx')
    
    real_data = pd.read_csv(f'/home/fi5666wi/R/data/DDLS/{dataset}_all.csv')
    corr_real = {f'{pair[0]} ~ {pair[1]}': [] for pair in corrs}
    for pair in corrs:
        name = f'{pair[0]} ~ {pair[1]}'
        corr_real[name].append(real_data[[pair[0], pair[1]]].corr()[pair[0]][pair[1]])

    perc_real = {}
    perc_real['all'] = []
    real_data['AB_status'] = real_data['AB_status'].map({'negative': 0, 'positive': 1})
    perc = real_data['AB_status'].mean() * 100
    perc_real['all'].append(perc)

    save_to_xlc(corr_real, perc_real, dataset_name="Real", 
                file_path='/home/fi5666wi/Python/WASP-DDLS/results_a4_final.xlsx')