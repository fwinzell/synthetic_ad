from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

from synthesize import synthesize
import os
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse

def load_description(file_dict):
    description_file = file_dict['description_file']
    with open(description_file, 'r') as f:
        description = json.load(f)
    return description


def generate_and_save_data(description_file, num_tuples_to_generate, output_file):
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(output_file)


def generate_synthetic_data(description_file, num_tuples_to_generate=1000):
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

    cog_mat = correlation_cog(generator.synthetic_dataset)
    tau_mat = correlation_tau(generator.synthetic_dataset)
    perc_mat = ab_status_and_apoe4(generator.synthetic_dataset)

    return cog_mat, tau_mat, perc_mat

def correlation_cog(df):
    corr_matrrix = {}
    corr_matrrix['all'] = df[['ADAS13', 'MMSE']].corr()['ADAS13']['MMSE']
    for dx in ['CN', 'MCI', 'Dementia']:
        subset = df[df['DX'] == dx]
        pcc = subset[['ADAS13', 'MMSE']].corr()['ADAS13']['MMSE']
        corr_matrrix[dx] = pcc

    return corr_matrrix

def correlation_tau(df):
    corr_matrrix = {}
    corr_matrrix['all'] = df[['TAU', 'PTAU']].corr()['PTAU']['TAU']
    ab_pos = df[df['AV45'] > 1.11]
    ab_neg = df[df['AV45'] <= 1.11]
    corr_matrrix['ab_pos'] = ab_pos[['TAU', 'PTAU']].corr()['PTAU']['TAU']
    corr_matrrix['ab_neg'] = ab_neg[['TAU', 'PTAU']].corr()['PTAU']['TAU']

    return corr_matrrix

def ab_status_and_apoe4(df):

    perc_matrix = {}
    for dx in ['CN', 'MCI', 'Dementia']:
        dx_df = df[df['DX'] == dx]
        n_tot = dx_df.shape[0]
        n_ab_pos = dx_df[dx_df['AV45'] > 1.11].shape[0]
        n_zero = dx_df[dx_df['APOE4'] == 0].shape[0]
        n_one = dx_df[dx_df['APOE4'] == 1].shape[0]
        n_two = dx_df[dx_df['APOE4'] == 2].shape[0]

        perc_matrix[dx] = [n_ab_pos/n_tot, n_zero/n_tot, n_one/n_tot, n_two/n_tot] # ab+, apoe4 = 0, apoe4 = 1, apoe4 = 2

    return perc_matrix
    

def load_real_data(train=True, dataset="adni"):
    if train:
        return pd.read_csv(f'/home/fi5666wi/R/data/DDLS/{dataset}_train.csv')
    else:
        return pd.read_csv(f'/home/fi5666wi/R/data/DDLS/{dataset}_test.csv')

        

def synthetic_analysis(bn_dir, load_data=False):
    real_data = load_real_data()
    num_tuples = real_data.shape[0]

    cog_results = {'all': [], 'CN': [], 'MCI': [], 'Dementia': []}
    tau_results = {'all': [], 'ab_pos': [], 'ab_neg': []}
    perc_results = {'CN': [], 'MCI': [], 'Dementia': []}

    for f in tqdm(os.listdir(bn_dir)):
        if load_data:
            df = pd.read_csv(os.path.join(bn_dir, f))
            cog_mat = correlation_cog(df)
            tau_mat = correlation_tau(df)
            perc_mat = ab_status_and_apoe4(df)
        else:
            cog_mat, tau_mat, perc_mat = generate_synthetic_data(os.path.join(bn_dir, f), num_tuples)
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
        #print(f'{key} AB+, APOE4 0, 1, 2: {np.round(np.mean(perc_results[key], axis=0),3)} +- {np.round(np.std(perc_results[key], axis=0),3)}')
        print(f'{key} AB+: {avg[0]} +- {stdev[0]}')
        for i in range(1,4):
            print(f'    APOE4 = {i-1}: {avg[i]} +- {stdev[i]}')

    #print("===== Real Data =====")
    #print(correlation_cog(real_data))
    #print(correlation_tau(real_data))
    #print(ab_status_and_apoe4(real_data))
    #save_to_xlc(cog_results, tau_results, perc_results)
    #save_to_xlc2(cog_results, tau_results, perc_results)
    return cog_results, tau_results, perc_results
    


def real_analysis(n_samples=100):
    tr_data = load_real_data(train=True)
    real_data = pd.concat([tr_data, load_real_data(train=False)], ignore_index=True)

    cog_results = {'all': []}
    tau_results = {'all': []}
    perc_results = {'CN': [], 'MCI': [], 'Dementia': []}

    for i in tqdm(range(n_samples)):
        sample = real_data.sample(n=tr_data.shape[0], replace=True)
        cog_mat = correlation_cog(sample)
        tau_mat = correlation_tau(sample)
        perc_mat = ab_status_and_apoe4(sample)

        cog_results['all'].append(cog_mat['all'])
        tau_results['all'].append(tau_mat['all'])

        for key in perc_results.keys():
            perc_results[key].append(perc_mat[key])


    print(f"Cog: {np.round(np.mean(cog_results['all']), 3)} +- {np.round(np.std(cog_results['all']), 3)}")
    print(f"Tau: {np.round(np.mean(tau_results['all']), 3)} +- {np.round(np.std(tau_results['all']), 3)}")
  
    for key in perc_results.keys():
        avg = np.round(np.mean(perc_results[key], axis=0),3) * 100
        stdev = np.round(np.std(perc_results[key], axis=0),3) * 100
        print(f'{key} AB+: {avg[0]} +- {stdev[0]}')
        for i in range(1,4):
            print(f'    APOE4 = {i-1}: {avg[i]} +- {stdev[i]}')

    #save_to_xlc2(cog_results, tau_results, perc_results)
    return cog_results, tau_results, perc_results


def save_to_xlc(cog_results, tau_results, perc_results):

    save_dict = {
        "ADAS ~ MMSE": [np.round(np.mean(cog_results['all']),3) , np.round(np.std(cog_results['all']), 3)],
        "PTAU ~ TAU": [np.round(np.mean(tau_results['all']),3), np.round(np.std(tau_results['all']), 3)],
        "AB+ (CN)": [np.round(np.mean(perc_results['CN'], axis=0)[0],3) * 100, np.round(np.std(perc_results['CN'], axis=0)[0],3) * 100],
        "APOE4 = 0 (CN)": [np.round(np.mean(perc_results['CN'], axis=0)[1],3) * 100, np.round(np.std(perc_results['CN'], axis=0)[1],3) * 100],
        "APOE4 = 1 (CN)": [np.round(np.mean(perc_results['CN'], axis=0)[2],3) * 100, np.round(np.std(perc_results['CN'], axis=0)[2],3) * 100],
        "APOE4 = 2 (CN)": [np.round(np.mean(perc_results['CN'], axis=0)[3],3) * 100, np.round(np.std(perc_results['CN'], axis=0)[3],3) * 100],
        "AB+ (MCI)": [np.round(np.mean(perc_results['MCI'], axis=0)[0],3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[0],3) * 100],
        "APOE4 = 0 (MCI)": [np.round(np.mean(perc_results['MCI'], axis=0)[1],3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[1],3) * 100],
        "APOE4 = 1 (MCI)": [np.round(np.mean(perc_results['MCI'], axis=0)[2],3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[2],3) * 100],
        "APOE4 = 2 (MCI)": [np.round(np.mean(perc_results['MCI'], axis=0)[3],3) * 100, np.round(np.std(perc_results['MCI'], axis=0)[3],3) * 100],
        "AB+ (AD)": [np.round(np.mean(perc_results['Dementia'], axis=0)[0],3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[0],3) * 100],
        "APOE4 = 0 (AD)": [np.round(np.mean(perc_results['Dementia'], axis=0)[1],3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[1],3) * 100],
        "APOE4 = 1 (AD)": [np.round(np.mean(perc_results['Dementia'], axis=0)[2],3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[2],3) * 100],
        "APOE4 = 2 (AD)": [np.round(np.mean(perc_results['Dementia'], axis=0)[3],3) * 100, np.round(np.std(perc_results['Dementia'], axis=0)[3],3) * 100]
    }

    save_df = pd.DataFrame(save_dict, index=['Mean', 'Std'])
    save_df.to_excel('results.xlsx', index=True)


def save_to_xlc2(cog_results, tau_results, perc_results, epsilon, file_path='/home/fi5666wi/Python/WASP-DDLS/results.xlsx'):
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
    new_df.insert(0, 'Epsilon', epsilon)

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

def main_generate(config, dir_name, dataset="adni"):
    save_dir = f'{config.base_dir}/DS-synthetic-data/{dir_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_data = pd.read_csv(f'{config.data_base_dir}/{dataset}_train.csv')
    num_tuples = real_data.shape[0]
    bn_dir = f'{config.base_dir}/DS-bayesian-networks/{dataset}/{dir_name}/'

    for f in tqdm(os.listdir(bn_dir)):
        description_file = os.path.join(bn_dir, f)
        output_file = os.path.join(save_dir, f.replace('.json', '.csv'))
        if not os.path.exists(output_file):
            generate_and_save_data(description_file, num_tuples, output_file)
        else:
            print(f'Skipping {output_file}, already exists.')

    print('done')


def parse_config():
    parser = argparse.ArgumentParser(description="Generate synthetic data using Bayesian networks.")
    parser.add_argument('--data_base_dir', type=str, help='Directory name for loading data files.', 
                        default='/home/fi5666wi/R/data/DDLS/')
    parser.add_argument('--base_dir', type=str, help='Base directory, used for loading BNs and saving generated data.', 
                        default='/home/fi5666wi/Python/WASP-DDLS')
    
    parser.add_argument('--dataset', type=str, default="adni_plus", help='Dataset name (default: adni).')
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    config = parse_config()
    #eps = 750
    
    #c_r, t_r, p_r = real_analysis(100)
    #save_to_xlc2(c_r, t_r, p_r, 'Real')
    #for eps in [5, 10, 25, 50, 100, 200]:
    #    print(f'Generating synthetic data for epsilon = {eps}')
        #main_generate(f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/degree2_eps_{eps}')
    #    c,t,p = synthetic_analysis(f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/degree2_eps_{eps}', load_data=True)
    #    save_to_xlc2(c,t,p, eps, file_path='/home/fi5666wi/Python/WASP-DDLS/results2.xlsx')
    
    #real_vs_synthetic(f'/Users/filipwinzell/WASP-DDLS/DS-bayesian-networks/degree3_deter')
    #real_analysis(100)

    for eps in [100, 'zero']:
        main_generate(config, f'degree2_eps_{eps}', dataset=config.dataset)

