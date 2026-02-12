from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions import unsupervised

import pandas as pd
import numpy as np
import random
import torch
import os

from huggingface_hub import login

from main_ml import frame2arrays
from train_svm import impute_missing_values
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description="Generate synthetic data using TabPFN.")
    parser.add_argument('--data_base_dir', type=str, help='Directory name for loading data files.', 
                        default='/home/fi5666wi/R/data/DDLS/')
    parser.add_argument('--base_dir', type=str, help='Base directory, used for loading BNs and saving generated data.', 
                        default='/home/fi5666wi/Python/WASP-DDLS')
    
    parser.add_argument('--dataset', type=str, default="adni_plus", help='Dataset name (default: adni).')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hf_login(path_to_token):
    with open(path_to_token, 'r') as f:
        token = f.read().strip()
    login(token)


def data_loading_without_label(file_name, cats, id_col="RID", label_col="DX"):
    df = pd.read_csv(file_name)

    if id_col is not None:
        X = df.drop(id_col, axis=1)
    else:
        X = df.copy()
    if 'Month' in X.columns:
        X = X.drop("Month", axis=1)

    if 'VISCODE' in X.columns:
        X = X.drop("VISCODE", axis=1)
    if 'EXAMDAY' in X.columns:
        X = X.drop("EXAMDAY", axis=1)

    #X = X[["ADAS13", "MMSE", "AGE"]]
    if label_col == "DX":
        class_mapping = {"CN": 0, "MCI": 1, "Dementia": 2}
        X['DX'] = X['DX'].map(class_mapping)
    elif label_col == "AB_status":
        class_mapping = {"negative": 0, "positive": 1}
        X['AB_status'] = X['AB_status'].map(class_mapping)

    cats.remove(label_col)
    for cat in cats:
        X[cat] = X[cat].astype('category')
   
    return X

def main_generate(config, temp, seed, display=False, dataset="adni"):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_data_file = f'{config.data_base_dir}/{dataset}_train.csv'
    if "adni" in dataset:
        categorical_features = ['DX', 'PTGENDER', 'APOE4', 'PTEDUCAT']
        X_real = data_loading_without_label(input_data_file, cats=categorical_features, id_col="RID", label_col="DX")
        columns_log_transform = ['AV45', 'ABETA', 'TAU', 'PTAU', 'Ventricles', 'ICV', 'WholeBrain', 'Hippocampus']
    elif dataset == "a4":
        categorical_features = ['AB_status', 'PTGENDER', 'PTETHNIC', 'PTEDUCAT', 'PTMARRY', 'PTNOTRT', 'PTHOME']
        X_real = data_loading_without_label(input_data_file, cats=categorical_features, id_col="BID", label_col="AB_status")

    #X_real_log = X_real.copy()
    #X_real_log[columns_log_transform] = X_real_log[columns_log_transform].apply(np.log)

    #X_imputed = impute_missing_values(X_real, seed=42, cats=['PTGENDER', 'APOE4', 'DX', 'PTEDUCAT'])

    #df = pd.read_csv(input_data_file)

    #X_real, y_real, labels = frame2arrays(df, label='DX') 

    model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(device=device), tabpfn_reg=TabPFNRegressor(device=device)
    )
    model.set_categorical_features(categorical_features=categorical_features)

    model = model.fit(X_real)

    synthetic_data = model.generate_synthetic_data(n_samples=X_real.shape[0], t=temp, n_permutations=1)
    synthetic_df = pd.DataFrame(synthetic_data, columns=X_real.columns)

    #synthetic_df[columns_log_transform] = synthetic_df[columns_log_transform].apply(np.exp)

    synthetic_df = post_process(synthetic_df, categorical_columns=categorical_features, dataset=dataset)
 
    if display:
        print(synthetic_df.head())
        display_histograms(X_real, synthetic_df)

    return synthetic_df

def post_process(synthetic_df, categorical_columns, dataset="adni"):
    for col in categorical_columns:
        x = synthetic_df[col]
        max_val = x.replace([np.inf], np.nan).max()
        min_val = x.replace([-np.inf], np.nan).min()
        synthetic_df[col] = x.clip(lower=min_val, upper=max_val).round().astype(int)
        #synthetic_df[col] = synthetic_df[col].round().astype(int)

    if "adni" in dataset:
        synthetic_df['DX'] = synthetic_df['DX'].map({0: 'CN', 1: 'MCI', 2: 'Dementia'})
    elif dataset == "a4":
        synthetic_df['AB_status'] = synthetic_df['AB_status'].map({0: 'negative', 1: 'positive'})
    
    #non_negatives = ['AGE', 'AV45', 'ABETA', 'TAU', 'PTAU', 'Ventricles', 'ICV', 'WholeBrain', 'MidTemp', 'Fusiform',
    #                 'Entorhinal', 'Hippocampus']
    non_negatives = synthetic_df.select_dtypes(include=[np.number]).columns.tolist() # Check that this works
    
    for col in non_negatives:
        synthetic_df.loc[synthetic_df[col] < 0, col] = np.nan

    if "adni" in dataset:
        synthetic_df['ADAS13'] = synthetic_df['ADAS13'].clip(lower=0, upper=85)
        synthetic_df['MMSE'] = synthetic_df['MMSE'].clip(lower=0, upper=30)
    elif dataset == "a4":
        synthetic_df["MMSCORE"] = synthetic_df["MMSCORE"].clip(lower=0, upper=30)
        synthetic_df["CDSOB"] = synthetic_df["CDSOB"].clip(lower=0, upper=18)

    return synthetic_df


def display_distributions(real_df, synthetic_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for column in real_df.columns:
        plt.figure(figsize=(10, 5))
        if real_df[column].dtype in ['object', 'category']:
            # Handle categorical columns
            real_counts = real_df[column].value_counts(normalize=True)
            synthetic_counts = synthetic_df[column].value_counts(normalize=True)

            real_counts = real_counts.reindex(real_counts.index.union(synthetic_counts.index), fill_value=0)
            synthetic_counts = synthetic_counts.reindex(real_counts.index, fill_value=0)

            bar_width = 0.4
            x = range(len(real_counts.index))

            plt.bar(x, real_counts.values, width=bar_width, label='Real', color='blue', alpha=0.5)
            plt.bar([i + bar_width for i in x], synthetic_counts.values, width=bar_width, label='Synthetic', color='orange', alpha=0.5)
            plt.xticks([i + bar_width / 2 for i in x], real_counts.index, rotation=45)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Proportion')
        else:
            # Handle numerical columns
            sns.kdeplot(real_df[column], label='Real', color='blue', fill=True, alpha=0.5)
            sns.kdeplot(synthetic_df[column], label='Synthetic', color='orange', fill=True, alpha=0.5)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Density')

        plt.legend()
        plt.tight_layout()
        plt.show()

def display_histograms(real_df, synthetic_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    for column in real_df.columns:
        df = pd.DataFrame({
            'Real': real_df[column],
            'Synthetic': synthetic_df[column]
        })
        n_bins = np.min([30, len(real_df[column].unique()), len(synthetic_df[column].unique())])
        df.plot.hist(bins=n_bins, alpha=0.5, density=True)

        plt.legend()
        plt.tight_layout()
        plt.show()

def synthesize_datasets(config):
    temps = [1.0]
    for temp in temps:
        for seed in range(100):
            folder = f'{config.base_dir}/tabpfn/{config.dataset}/tabpfn_t_{temp}/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            save_file = os.path.join(folder, f"tabpfn_{seed}.csv")
            if os.path.exists(save_file):
                print(f"File {save_file} already exists. Skipping...")
            else:
                synthetic_df = main_generate(config, temp=temp, seed=seed, display=False, dataset=config.dataset)
                synthetic_df.to_csv(save_file, index=False)

if __name__ == "__main__":
    hf_login('/home/fi5666wi/hugface-token')
    config = parse_config()
    #main_generate(config, temp=1.0, seed=42, display=True, dataset=config.dataset)
    synthesize_datasets(config)



