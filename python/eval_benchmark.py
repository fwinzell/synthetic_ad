import pandas as pd
import numpy as np
from syntheval import SynthEval
from syntheval_benchmark_utils import results_formatting, rank_results_formatting
import os
from tqdm import tqdm
import glob

import sys
#syntheval_path = os.path.dirname(os.path.abspath(syntheval.__file__))
#utils_path = os.path.join(syntheval_path, 'utils')
#sys.path.append(utils_path)
#from utils import results_formatting, rank_results_formatting
#from syntheval import SynthEval

#from main_ds import load_real_data, load_description
from train_svm import data_loading, impute_missing_values
from DataSynthesizer.DataGenerator import DataGenerator

import argparse

import warnings
warnings.filterwarnings("ignore")
print("Warning: all warnings suppressed")

def load_real_data(data_dir, dataset="adni", drop_cat=False):
    tr_data = pd.read_csv(f'{data_dir}/{dataset}_train.csv')
    test_data = pd.read_csv(f'{data_dir}/{dataset}_test.csv')

    if 'Month' in tr_data.columns:
        tr_data = tr_data.drop("Month", axis=1)
    if 'DXN' in tr_data.columns:
        tr_data = tr_data.drop("DXN", axis=1)
    if 'VISCODE' in tr_data.columns:
        tr_data = tr_data.drop("VISCODE", axis=1)
    if 'EXAMDAY' in tr_data.columns:
        tr_data = tr_data.drop("EXAMDAY", axis=1)
    
    if 'Month' in test_data.columns:
        test_data = test_data.drop("Month", axis=1)
    if 'DXN' in test_data.columns:
        test_data = test_data.drop("DXN", axis=1)
    if 'VISCODE' in test_data.columns:
        test_data = test_data.drop("VISCODE", axis=1)
    if 'EXAMDAY' in test_data.columns:
        test_data = test_data.drop("EXAMDAY", axis=1)

    if dataset == "a4":
        id_col = 'BID'
    else:
        id_col = 'RID'

    test_data = test_data.drop(columns=[id_col], axis=1)
    tr_data = tr_data.drop(columns=[id_col], axis=1)

    if "adni" in dataset:
        categorical_attributes = ['PTGENDER', 'PTEDUCAT', 'APOE4', 'DX']
    elif dataset == "a4":
        categorical_attributes = ['AB_status', 'PTGENDER', 'PTETHNIC', 'PTEDUCAT', 'PTMARRY', 'PTNOTRT', 'PTHOME']
    test_data = impute_missing_values(test_data, seed=0, cats=categorical_attributes)
    tr_data = impute_missing_values(tr_data, seed=0, cats=categorical_attributes)

    if drop_cat:
        test_data = test_data.drop(columns=categorical_attributes, axis=1)
        tr_data = tr_data.drop(columns=categorical_attributes, axis=1)

    return tr_data, test_data

def run_syntheval_test(syn_df):
    real_df, test_df = load_real_data()

    tar = 'DX'
    categorical_attributes = ['PTGENDER', 'PTEDUCAT', 'APOE4', 'DX']

    se = SynthEval(real_df, holdout_dataframe=test_df)
    se.evaluate(syn_df, tar, "full_eval")


def load_synthetic_with_imputation(syn_file, cats, drop_cat=False):
    syn_df = pd.read_csv(syn_file)

    if 'VISCODE' in syn_df.columns:
        syn_df = syn_df.drop("VISCODE", axis=1)
    if 'EXAMDAY' in syn_df.columns:
        syn_df = syn_df.drop("EXAMDAY", axis=1)
    if 'DXN' in syn_df.columns:
        syn_df = syn_df.drop("DXN", axis=1)

    syn_df = impute_synthetic_data(syn_df, cats=cats, seed=0, drop_cats=drop_cat)

    return syn_df

def generate_synthetic_with_imputation(description_file, num_tuples_to_generate):
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

    syn_df = pd.DataFrame(generator.synthetic_dataset)
    return impute_synthetic_data(syn_df)

def impute_synthetic_data(df, cats, seed=0, drop_cats=False):
    if 'RID' in df.columns:
        df = df.drop(columns=['RID'], axis=1)
    if 'BID' in df.columns:
        df = df.drop(columns=['BID'], axis=1)
    df = impute_missing_values(df, seed=seed, cats=cats)
    if drop_cats:
        df = df.drop(columns=cats, axis=1)

    return df

def benchmark_privacy(syn_dir, generate=False):
    real_df, test_df = load_real_data()

    tar = 'DX'
    categorical_attributes = ['PTGENDER', 'PTEDUCAT', 'APOE4', 'DX']

    syn_dict = {}
    print("====== loading data ======")
    for i,f in tqdm(enumerate(os.listdir(syn_dir))):
        fname = os.path.splitext(f)[0]
        if generate:
            syn_dict[fname] = generate_synthetic_with_imputation(f'{syn_dir}/{f}', real_df.shape[0])
        else:
            syn_dict[fname] = load_synthetic_with_imputation(f'{syn_dir}/{f}')
    print("Data loaded successfully")

    metrics = {
        "nndr"      : {},
        "nnaa"      : {},
        "dcr"       : {},
        "hit_rate"  : {"thres_percent": 0.0333},
        "eps_risk"  : {},
        "mia"       : {"num_eval_iter": 5},
        "att_discl" : {}
    }

    se = SynthEval(real_df, holdout_dataframe=test_df, cat_cols=categorical_attributes)
    bmk_df, rnk_df = se.benchmark(syn_dict, tar, **metrics)

    return bmk_df, rnk_df

def benchmark_utility_and_privacy(data_dir, syn_dir, generate=False, dataset="adni"):
    real_df, test_df = load_real_data(data_dir, dataset=dataset)

    if "adni" in dataset:
        tar = 'DX'
        categorical_attributes = ['PTGENDER', 'PTEDUCAT', 'APOE4', 'DX']
    elif dataset == "a4":
        tar = 'AB_status'
        categorical_attributes = ['AB_status', 'PTGENDER', 'PTETHNIC', 'PTEDUCAT', 'PTMARRY', 'PTNOTRT', 'PTHOME']

    syn_dict = {}
    print("====== loading data ======")
    for i,f in tqdm(enumerate(os.listdir(syn_dir))):
        fname = os.path.splitext(f)[0]
        if generate:
            syn_dict[fname] = generate_synthetic_with_imputation(f'{syn_dir}/{f}', real_df.shape[0])
        else:
            syn_dict[fname] = load_synthetic_with_imputation(f'{syn_dir}/{f}', cats=categorical_attributes)
    print("Data loaded successfully")

    metrics = {
        "pca"      : {},
        "mi_diff"  : {},
        "ks_test"  : {"sig_lvl": 0.05, "n_perms": 1000},
        #"auroc_diff"  : {"model": "log_reg"}, can implement this separately in R?
        "cls_acc"  : {},
        "nndr"      : {},
        "nnaa"      : {},
        "hit_rate"  : {"thres_percent": 0.0333},
        "eps_risk"  : {},
        "mia"  : {"num_eval_iter": 1},
        "att_discl" : {}
    }

    se = SynthEval(real_df, holdout_dataframe=test_df, cat_cols=categorical_attributes, verbose=True)
    bmk_df, rnk_df = se.benchmark(syn_dict, tar, rank_strategy="linear", **metrics)

    return bmk_df, rnk_df


def test_benchmark_utility_and_privacy():
    real_df, test_df = load_real_data()

    tar = 'DX'
    categorical_attributes = ['PTGENDER', 'PTEDUCAT', 'APOE4', 'DX']

    real_df, syn_df = np.array_split(real_df.sample(frac=1), 2) 
    data_size = real_df.shape

    metrics = {
        "pca"      : {},
        "mi_diff"  : {},
        "ks_test"  : {"sig_lvl": 0.05, "n_perms": 1000},
        #"auroc_diff"  : {"model": "log_reg"}, can implement this separately in R?
        "cls_acc"  : {},
        "nndr"      : {},
        "nnaa"      : {},
        "hit_rate"  : {"thres_percent": 0.0333},
        "eps_risk"  : {},
        "mia"  : {"num_eval_iter": 1},
        "att_discl" : {}
    }
    syn_dict = {'synthetic': syn_df}

    se = SynthEval(real_df, holdout_dataframe=test_df, cat_cols=categorical_attributes, verbose=True)
    bmk_df, rnk_df = se.benchmark(syn_dict, tar, rank_strategy="linear", **metrics)

    for f in glob.glob("SE_benchmark_*.csv"):
        os.remove(f)

    dpath = '/home/fi5666wi/R/data/DDLS/'
    res_df = results_formatting(bmk_df, dpath, data_size)
    rnk_res_df = rank_results_formatting(rnk_df, dpath, data_size)

    set_name = f"test"
    res_df.to_csv(f'/home/fi5666wi/Python/WASP-DDLS/SE-benchmark/bmk_new_{set_name}.csv', index=False)
    rnk_res_df.to_csv(f'/home/fi5666wi/Python/WASP-DDLS/SE-benchmark/rnk_new_{set_name}.csv', index=False)
    print('herå')


def benchmark_mia_test(syn_dir, generate=False):
    real_df, test_df = load_real_data()

    tar = 'DX'
    categorical_attributes = ['PTGENDER', 'PTEDUCAT', 'APOE4', 'DX']

    syn_dict = {}
    print("====== loading data ======")
    for i,f in tqdm(enumerate(os.listdir(syn_dir))):
        fname = os.path.splitext(f)[0]
        if generate:
            syn_dict[fname] = generate_synthetic_with_imputation(f'{syn_dir}/{f}', real_df.shape[0])
        else:
            syn_dict[fname] = load_synthetic_with_imputation(f'{syn_dir}/{f}')
    print("Data loaded successfully")

    metrics = {
        "mia_rfc"  : {"num_eval_iter": 1}
    }

    se = SynthEval(real_df, holdout_dataframe=test_df, cat_cols=categorical_attributes)
    bmk_df, rnk_df = se.benchmark(syn_dict, tar, rank_strategy="linear", **metrics)

    return bmk_df, rnk_df


def main_ds(dataset="adni"):
    paths = parse_paths()
    for eps in [100, 'zero']:
        degree = 2
        #bn_dir = f'/Users/filipwinzell/WASP-DDLS/DS-bayesian-networks/degree{degree}_eps_{eps}/'
        #bn_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-bayesian-networks/degree{degree}_eps_{eps}/'
        syn_dir = f'{paths.base_dir}/DS-synthetic-data/{dataset}/degree{degree}_eps_{eps}/'

        real_df, _ = load_real_data(paths.data_base_dir, dataset=dataset)
        data_size = real_df.shape

        #bmk_df, rnk_df = benchmark_privacy(syn_dir)
        #res_df = results_formatting(bmk_df)
        #rnk_df = rank_results_formatting(rnk_df)
        #mean_df = bmk_df.mean()

        #bmk_df.to_excel('benchmark_results.xlsx', index=True)

        bmk_df, rnk_df = benchmark_utility_and_privacy(paths.data_base_dir, syn_dir, generate=False, dataset=dataset)
        ### Cleanup
        for f in glob.glob("SE_benchmark_*.csv"):
            os.remove(f)


        res_df = results_formatting(bmk_df, syn_dir, data_size)
        rnk_res_df = rank_results_formatting(rnk_df, syn_dir, data_size)

        set_name = f"deg{degree}_eps{eps}"
        save_dir = f'{paths.base_dir}/SE-benchmark/{dataset}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        res_df.to_csv(f'{save_dir}/bmk_new_{set_name}.csv', index=False)
        rnk_res_df.to_csv(f'{save_dir}/rnk_new_{set_name}.csv', index=False)

        #bmk_df.to_csv(f'/Users/filipwinzell/WASP-DDLS/SE-benchmark/bmk_df_{set_name}.csv', index=True)
        #rnk_df.to_csv(f'/Users/filipwinzell/WASP-DDLS/SE-benchmark/rnk_df_{set_name}.csv', index=True)

        print('herå')


def main_synthpop(dataset):
    paths = parse_paths()
    syn_dir = f'{paths.base_dir}/synthpop/{dataset}/'

    real_df, _ = load_real_data(paths.data_base_dir, dataset=dataset)
    data_size = real_df.shape

    bmk_df, rnk_df = benchmark_utility_and_privacy(paths.data_base_dir,syn_dir, generate=False, dataset=dataset)
    ### Cleanup
    for f in glob.glob("SE_benchmark_*.csv"):
        os.remove(f)


    res_df = results_formatting(bmk_df, syn_dir, data_size)
    rnk_res_df = rank_results_formatting(rnk_df, syn_dir, data_size)

    set_name = f"synthpop"
    save_dir = f'{paths.base_dir}/SE-benchmark/{dataset}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    res_df.to_csv(f'{save_dir}/bmk_new_{set_name}.csv', index=False)
    rnk_res_df.to_csv(f'{save_dir}/rnk_new_{set_name}.csv', index=False)

    print('herå')

def main_ctgan(set_name, epochs_list, dataset="adni"):
    paths = parse_paths()

    for epochs in epochs_list:
        syn_dir = f'{paths.base_dir}/ctgan/{dataset}/{set_name}_epochs_{epochs}/'

        real_df, _ = load_real_data(paths.data_base_dir, dataset=dataset)
        data_size = real_df.shape

        bmk_df, rnk_df = benchmark_utility_and_privacy(paths.data_base_dir, syn_dir, generate=False, dataset=dataset)
        ### Cleanup
        for f in glob.glob("SE_benchmark_*.csv"):
            os.remove(f)


        res_df = results_formatting(bmk_df, syn_dir, data_size)
        rnk_res_df = rank_results_formatting(rnk_df, syn_dir, data_size)

        set_name = f"{set_name}_epochs_{epochs}"
        save_dir = f'{paths.base_dir}/SE-benchmark/{dataset}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        res_df.to_csv(f'{save_dir}/bmk_new_{set_name}.csv', index=False)
        rnk_res_df.to_csv(f'{save_dir}/rnk_new_{set_name}.csv', index=False)


        print('herå')
    

def main_tabpfn(t_list, dataset="adni"):
    paths = parse_paths()

    for tmp in t_list:
        syn_dir = f'{paths.base_dir}/tabpfn/{dataset}/tabpfn_t_{tmp}/'

        real_df, _ = load_real_data(paths.data_base_dir, dataset=dataset)
        data_size = real_df.shape

        bmk_df, rnk_df = benchmark_utility_and_privacy(paths.data_base_dir, syn_dir, generate=False, dataset=dataset)
        ### Cleanup
        for f in glob.glob("SE_benchmark_*.csv"):
            os.remove(f)


        res_df = results_formatting(bmk_df, syn_dir, data_size)
        rnk_res_df = rank_results_formatting(rnk_df, syn_dir, data_size)


        set_name = f"tabpfn_t_{tmp}"
        save_dir = f'{paths.base_dir}/SE-benchmark/{dataset}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        res_df.to_csv(f'{save_dir}/bmk_new_{set_name}.csv', index=False)
        rnk_res_df.to_csv(f'{save_dir}/rnk_new_{set_name}.csv', index=False)


        print('herå')


def parse_paths():
    parser = argparse.ArgumentParser(description="Generate synthetic data using Bayesian networks.")
    parser.add_argument('--data_base_dir', type=str, help='Directory name for loading data files.', 
                        default='/home/fi5666wi/R/data/DDLS/')
    parser.add_argument('--base_dir', type=str, help='Base directory, used for loading BNs and saving generated data.', 
                        default='/home/fi5666wi/Python/WASP-DDLS')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #main_synthpop(dataset="adni_plus")
    #main_ds(dataset="adni_plus")
    #main_ctgan(set_name="ctgan_optim", epochs_list=[750], dataset="a4")
    #main_ctgan(set_name="ctgan_default", epochs_list=[750], dataset="adni_plus")
    main_tabpfn(t_list=[1.0], dataset="adni_plus")

    #test_benchmark_utility_and_privacy()



    






