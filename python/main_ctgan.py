from ctgan import CTGAN
import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from itertools import product

from ctgan import load_demo
from train_svm import data_loading, scale_data, fit_svm_classif, fit_svm_regressor
from main_ml import frame2arrays
from utils import KSMetric

from main_ds import parse_config

import argparse

def data_loading_without_label(file_name, id_col="RID"):
    df = pd.read_csv(file_name)

    X = df.drop(id_col, axis=1)
    if 'Month' in X.columns:
        X = X.drop("Month", axis=1)
    if 'VISCODE' in X.columns:
        X = X.drop("VISCODE", axis=1)
    #X = X[["ADAS13", "MMSE", "AGE"]]
    return X

def impute_missing_values(X, seed=0, cats=["PTGENDER", "PTEDUCAT", "APOE4"]):
    X = X.copy()

    conts = [col for col in X.columns if col not in cats]
    min_values = X[conts].min()
    max_values = X[conts].max()

    cont_imp = IterativeImputer(max_iter=10, random_state=seed, max_value=max_values, min_value=min_values)
    cat_imp = SimpleImputer(strategy='most_frequent')

    X[cats] = cat_imp.fit_transform(X[cats])
    X[conts] = cont_imp.fit_transform(X[conts])
    return X

def demo_ctgan():
    real_data = load_demo()

    # Names of the columns that are discrete
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]

    ctgan = CTGAN(epochs=10)
    ctgan.fit(real_data, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(1000)

    return synthetic_data


def main_ctgan(config, seed=0, epochs=10, hparams={}, run_svm=False, dataset="adni"):
    input_data_file = f'{config.data_base_dir}/{dataset}_train.csv'
    #gan_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-bayesian-networks/ctgan/'
    #if not os.path.exists(gan_dir):
    #    os.makedirs(gan_dir)

    id_col = "BID" if dataset == "a4" else "RID"
    real_data = data_loading_without_label(input_data_file, id_col=id_col)

    num_tuples = real_data.shape[0]

    if "adni" in dataset:
        discrete_columns = ['PTGENDER',
                            'PTEDUCAT',
                            'APOE4',
                            'DX']
    elif dataset == "a4":
        discrete_columns = [
            'AB_status',
            'PTGENDER',
            'PTETHNIC',
            'PTEDUCAT',
            'PTMARRY',
            'PTNOTRT',
            'PTHOME'
        ]
    real_data_imp = impute_missing_values(real_data, seed=seed, cats=discrete_columns)

    ctgan = CTGAN(epochs=epochs, enable_gpu=True, verbose=True, **hparams)
    ctgan.fit(real_data_imp, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(num_tuples)

    if run_svm:
        label = 'DX' if dataset == "adni" else 'AB_status'
        svm_results = run_ml_test(synthetic_data, label=label)
        print(f"SVM results CTGAN epochs={epochs}, seed={seed}: Acc={svm_results['acc']}, AUC={svm_results['auc']}")

    return synthetic_data


def run_ml_test(synthetic_data, test_data=None, label='DX'):
    svm_params = {
        'DX': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'AB_status': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
    }
    if test_data is not None:
        X_test, y_test, _ = frame2arrays(test_data, label=label)
    else:
        X_test, y_test, _ = data_loading('/home/fi5666wi/R/data/DDLS/adni_test.csv', label=label)
    X_test_i = impute_missing_values(X_test)

    X_test_i = scale_data(X_test_i)
    X_test = scale_data(X_test)

    svm_results = {'acc': [], 'auc': [], 'cm': []}
    X_syn, y_syn, _ = frame2arrays(synthetic_data, label=label)
    X_syn_i = impute_missing_values(X_syn)

    clf_svm = fit_svm_classif(scale_data(X_syn_i), y_syn, svm_params[label])

    y_hat_svm = clf_svm.predict(X_test_i)
    y_prob_svm = clf_svm.predict_proba(X_test_i)

    if label == 'AB':
        y_prob_svm = y_prob_svm[:,1]

    svm_results['acc'].append(np.mean(y_hat_svm == y_test))
    svm_results['auc'].append(roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted'))
    svm_results['cm'].append(confusion_matrix(y_test, y_hat_svm))

    return svm_results


def cross_val_loop(ctgan, seed, label='DX'):
    input_data_file = '/home/fi5666wi/R/data/DDLS/adni_train_ds.csv'

    real_data = data_loading_without_label(input_data_file)
    discrete_columns = ['PTGENDER',
                        'PTEDUCAT',
                        'APOE4',
                        'DX']
    real_data_imp = impute_missing_values(real_data, seed=seed, cats=discrete_columns)

    # df is your full pandas DataFrame
    real_data_imp = real_data_imp.reset_index(drop=True)

    X_real = real_data_imp.drop(columns=[label])
    y_real = real_data_imp[label]

    n_splits = 5
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    aucs = []
    accs = []
    ks_stats = []
    frac_sigs = []

    KST = KSMetric(sig_lvl=0.05, n_perms=100, cat_cols=discrete_columns, num_cols=[col for col in real_data_imp.columns if col not in discrete_columns])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_real, y_real)):
        train_df = real_data_imp.iloc[train_idx].copy()
        val_df   = real_data_imp.iloc[val_idx].copy()

        print(f"Fitting CTGAN on fold {fold}: ({len(train_df)} train, {len(val_df)} val)")
        ctgan.fit(train_df, discrete_columns)
        print("Done")

        print("Generating synthetic data")
        synthetic_data = ctgan.sample(len(train_df))
        print("Done")

        print("Running tests")
        svm_results = run_ml_test(synthetic_data, test_data=val_df, label=label)
        ks_results = KST.evaluate(real_data=val_df, synt_data=synthetic_data)
        print(f"Done: AUC={svm_results['auc'][0]}, Acc={svm_results['acc'][0]}")
        print(f"Avg KS stat={ks_results['avg stat']}, Frac sigs={ks_results['frac sigs']}")

        accs.append(svm_results['acc'][0])
        aucs.append(svm_results['auc'][0])
        ks_stats.append(ks_results['avg stat'])
        frac_sigs.append(ks_results['frac sigs'])

    return {
        'acc': accs,
        'auc': aucs,
        'ks_stat': ks_stats,
        'frac_sigs': frac_sigs
    }
    


def analyze_variables(df, df_2=None):
    # Analyze the variables in the DataFrame
    print("DataFrame shape:", df.shape)
    colnames = df.columns.tolist()
    for col in colnames:
        print(f"Column: {col}")
        range_values = df[col].min(), df[col].max()
        print(f"Range: {range_values[0]} - {range_values[1]}")
        n_missing = df[col].isnull().sum()
        print(f"Missing values: {n_missing}")

        if df_2 is not None:
            if col in df_2.columns:
                range_values_2 = df_2[col].min(), df_2[col].max()
                print(f"Range (synthetic): {range_values_2[0]} - {range_values_2[1]}")
                n_missing_2 = df_2[col].isnull().sum()
                print(f"Missing values (synthetic): {n_missing_2}")


def run_loop():
    for epochs in [10, 50, 100]:
        for seed in range(100):
            synthetic_data = main_ctgan(epochs=epochs, seed=seed)
            folder = f'/home/fi5666wi/Python/WASP-DDLS/ctgan/ctgan_epochs_{epochs}/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            synthetic_data.to_csv(os.path.join(folder, f"ctsyn_{seed}.csv"), index=False)


def run_cross_val():
    #default = CTGAN(epochs=100, enable_gpu=True, verbose=True)
    #base_results = cross_val_loop(default, seed=0)

    all_results = {}
    for epochs in [100, 250, 500, 750, 1000]:
        ctgan = CTGAN(
            generator_dim=(32, 32),
            discriminator_dim=(32, 32),
            batch_size=50,
            epochs=epochs,
            generator_lr=1e-4,
            discriminator_lr=1e-4,
            discriminator_steps=1,
            enable_gpu=True,
            verbose=True
        )
        new_results = cross_val_loop(ctgan, seed=0)
        all_results[epochs] = new_results

    for epochs, new_results in all_results.items():
        print(f"Results for CTGAN with {epochs} epochs:")
        print("-------------------------------")
        print(f"       AUC      ACC")
        print(f"New: {np.mean(new_results['auc'])} ({np.std(new_results['auc'])})      {np.mean(new_results['acc'])} ({np.std(new_results['acc'])})")
        print("-------------------------------")
        print("       KS stats      KS frac sigs")
        print(f"New: {np.mean(new_results['ks_stat'])} ({np.std(new_results['ks_stat'])})      {np.mean(new_results['frac_sigs'])} ({np.std(new_results['frac_sigs'])})")
        print("-------------------------------")


def hparam_grid_search(epochs=750):
    #default = CTGAN(epochs=100, enable_gpu=True, verbose=True)
    #base_results = cross_val_loop(default, seed=0)
    search_space = {
        "batch_size": [20, 50, 100],
        "generator_lr": [5e-5, 1e-4],
        "discriminator_lr": [5e-5, 1e-4],
        "generator_dim": [(32, 32), (64, 64)],
        "discriminator_dim": [(32, 32), (64, 64)]
    }

    keys = list(search_space.keys())
    values = list(search_space.values())

    param_grid = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

    best_auc = 0.0
    best_params = None
    best_results = None
    for i, params in enumerate(param_grid):
        ctgan = CTGAN(
            epochs=epochs,
            enable_gpu=True,
            verbose=True,
            **params
        )
        new_results = cross_val_loop(ctgan, seed=0)
        mean_auc = np.mean(new_results['auc'])
        print(f"Grid search {i+1}/{len(param_grid)}: params={params}, mean AUC={mean_auc}")
        if mean_auc > best_auc:
            print("New best results!!")
            best_auc = mean_auc
            best_params = params
            best_results = new_results
    
    print("Best hyperparameters found:")
    print(best_params)
    print("With results:")
    print(best_results)

# RESULTS FROM RUNNING THE GRID SEARCH 2025-12-19
# Best hyperparameters found:
# {'batch_size': 50, 'generator_lr': 0.0001, 'discriminator_lr': 0.0001, 'generator_dim': (64, 64), 'discriminator_dim': (32, 32)}
# With results:
# {'acc': [np.float64(0.6051948051948052), np.float64(0.6519480519480519), np.float64(0.6935064935064935), np.float64(0.6675324675324675), np.float64(0.6051948051948052)], 'auc': [0.7666010107599734, 0.7957235230964932, 0.8014202424516296, 0.8047752574429472, 0.7838913528994146], 'ks_stat': [np.float64(0.12987330447330447), np.float64(0.1378436507936508), np.float64(0.13351630591630592), np.float64(0.1435469696969697), np.float64(0.12695028860028856)], 'frac_sigs': [0.9444444444444444, 0.7777777777777778, 0.8888888888888888, 0.8333333333333334, 1.0]}

def synthesize_datasets(config, epochs=[750], save_name="ctgan", hparams={}, dataset="adni"):
    for max_epoch in epochs:
        for seed in range(100):
            folder = f'{config.base_dir}/ctgan/{dataset}/{save_name}_epochs_{max_epoch}/'
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_file = os.path.join(folder, f"ctsyn_{seed}.csv")
            if os.path.exists(save_file): 
                continue
            else:
                print(f"Synthesizing dataset with CTGAN epochs={max_epoch}, seed={seed}")
                synthetic_data = main_ctgan(config, epochs=max_epoch, seed=seed, hparams=hparams, run_svm=False, dataset=dataset)
                synthetic_data.to_csv(save_file, index=False)


if __name__ == '__main__':
    config = parse_config()
    config.dataset= "adni_plus"

    #synthetic_data = main_ctgan()
    #print(synthetic_data.head())
    #run_syntheval_test(synthetic_data)
    #analyze_variables(synthetic_data)
    #synthetic_data.to_csv('/home/fi5666wi/Python/WASP-DDLS/ctgan/adni_ctgan_synthetic_test.csv', index=False)

    #real_data = data_loading('/home/fi5666wi/R/data/DDLS/adni_train_ds.csv')
    #print(real_data.head())
    #analyze_variables(real_data)

    #real_data_imp = impute_missing_values(real_data, seed=0, cats=['PTGENDER','PTEDUCAT','APOE4','DX'])
    #print(real_data_imp.head())
    #analyze_variables(real_data, real_data_imp)


    #run_cross_val()
    #hparam_grid_search(epochs=750)

    optim_hparams = {'batch_size': 50, 'generator_lr': 0.0001, 'discriminator_lr': 0.0001, 'generator_dim': (64, 64), 'discriminator_dim': (32, 32)}
    #synthesize_datasets(config, epochs=[750], save_name="ctgan_optim", hparams=optim_hparams, dataset=config.dataset)
    synthesize_datasets(config, epochs=[750], save_name="ctgan_default", hparams={}, dataset=config.dataset)

