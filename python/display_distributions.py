import os
import pandas as pd
import numpy as np

from tabpfn_generate import data_loading_without_label, display_distributions


def display_random_adni(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    random_file = np.random.choice(files)
    file_path = os.path.join(data_dir, random_file)
    print(f"Displaying distributions for file: {random_file}")

    input_data_file = f'/home/fi5666wi/R/data/DDLS/adni_train_ds.csv'
    categorical_features = ['DX', 'PTGENDER', 'APOE4', 'PTEDUCAT']
    real_data = data_loading_without_label(input_data_file, cats=categorical_features.copy(), id_col="RID", label_col="DX")

    synthetic_data = data_loading_without_label(file_path, cats=categorical_features.copy(), id_col=None, label_col="DX")

    display_distributions(real_data, synthetic_data)


if __name__ == "__main__":
    ctgan_dir = '/home/fi5666wi/Python/WASP-DDLS/ctgan/ctgan_optim_epochs_750/'
    synthpop_dir = '/home/fi5666wi/Python/WASP-DDLS/synthpop/syn_ranseq/'
    ds_dir = '/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/degree2_eps_zero/'
    tabpfn_dir = '/home/fi5666wi/Python/WASP-DDLS/tabpfn/tabpfn_t_1.0/'

    #display_random_adni(ctgan_dir)
    #display_random_adni(synthpop_dir)
    #display_random_adni(ds_dir)
    display_random_adni(tabpfn_dir)
