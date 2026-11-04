from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

from synthesize import synthesize
import os
import json
from statsmodels.stats.contingency_tables import mcnemar

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm


from train_svm import data_loading, impute_missing_values, scale_data, impute_with_zeros



# Calculate distance between real and synthetic data fro different epsilon values
# See how privancy is affected by different epsilon values

def load_dataset_from_csv(input_data_file):
    return pd.read_csv(input_data_file)


def load_real_data():
    tr_data = pd.read_csv('/Users/filipwinzell/R/data/DDLS/adni_train_ml.csv')
    test_data = pd.read_csv('/Users/filipwinzell/R/data/DDLS/adni_test.csv')

    # Add a column to indicate the source of the data
    tr_data['source'] = 1 # 1 for training data
    test_data['source'] = 0 # 0 for test data
    
    # Concatenate the DataFrames
    merged_data = pd.concat([tr_data, test_data], ignore_index=True)
    return merged_data


def nearest_neighbour_distance(real_data, synthetic_data, seed=7):
    from sklearn.neighbors import NearestNeighbors

    X_real = real_data.drop(columns=['source', 'DX', 'DXN', 'RID'], axis=1)
    X_real = impute_missing_values(X_real, seed=seed)
    #X_real = impute_with_zeros(X_real)
    X_real = scale_data(X_real)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_real)

    X_synthetic = synthetic_data.drop(columns=['DX', 'RID'], axis=1)
    X_synthetic = impute_missing_values(X_synthetic, seed=seed)
    #X_synthetic = impute_with_zeros(X_synthetic)
    X_synthetic = scale_data(X_synthetic)
    distances, indices = nbrs.kneighbors(X_synthetic)

    return distances, indices

    #

def mcnemar_test(y_hat, r, alpha=0.05, verbose=False):
    y_1 = sum(y_hat['source'])
    y_0 = len(y_hat) - y_1
    table = np.array([[y_1*r, y_1*(1-r)], [y_0*r, y_0*(1-r)]])
    result = mcnemar(table, exact=False)

    if verbose:
        if result.pvalue > alpha:
            print("Fail to reject the null hypothesis (NN is not significantly better than random guessing).")
        else:
            print("Reject the null hypothesis (NN is significantly better than random guessing).")

    return result

def percent_missing_values(data):
    total = 0
    for col in data.columns:
        missing = data[col].isnull().sum()
        total += missing
        if missing > 0:
            print(f'{col}: {missing} ({missing / len(data) * 100:.2f}%)')

    return total / (len(data) * len(data.columns))

def plot_distance_distribution(distances, title):
    # Flatten the distances array
    distances = distances.flatten()

    # Plot histogram
    plt.hist(distances, bins=50, density=True, alpha=0.6, color='g')

    # Fit a normal distribution to the data
    mu, std = norm.fit(distances)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = f"{title}\nFit results: mu = {mu:.2f},  std = {std:.2f}"
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    data = load_real_data()
    #print(data.head())
    ratio = sum(data['source']) / len(data)
    eps = 500

    main_dir = f'/Users/filipwinzell/WASP-DDLS/DS-synthetic-data/degree3_eps_{eps}/'
    #main_dir = f'/Users/filipwinzell/WASP-DDLS/DS-synthetic-data/degree3_deter/'

    nnacc = []
    mcn_stats = []
    pvals = []
    failed = []
    avg_dist = []
    verbose = False
    alpha = 0.05

    for i,f in enumerate(os.listdir(main_dir)):

        synthetic = pd.read_csv(os.path.join(main_dir, f))

        dist,idxs = nearest_neighbour_distance(data, synthetic, seed=1342)
        #plot_distance_distribution(dist, f)

        closest = data.iloc[idxs.flatten()]
        nnratio = sum(closest['source']) / len(closest)
        mcn = mcnemar_test(closest, ratio, verbose=verbose)
       
        if verbose:
            print(f'Ratio of training nearest neighbours: {nnratio}')
            print(f'McNemar test: {mcn.pvalue} (If it is less than 0.05, we have a problem)')
            print(f'Mean distance: {np.mean(dist)} (std: {np.std(dist)})')

        if mcn.pvalue < alpha:
            failed.append(f)
            if verbose:
                print(f'Failed test for {f}')

        nnacc.append(nnratio)
        mcn_stats.append(mcn.statistic)
        pvals.append(mcn.pvalue)
        avg_dist.append(np.mean(dist))

    print(f'Failed {len(failed)} out of {i+1} tests')
    print(f'Failed tests: {failed}')

    print(f'Mean NN ratio: {np.mean(nnacc)} (std: {np.std(nnacc)})')
    print(f'Mean McNemar statistic: {np.mean(mcn_stats)} (std: {np.std(mcn_stats)})')
    print(f'Mean p-value: {np.mean(pvals)} (std: {np.std(pvals)})')
    print(f'Mean distance: {np.mean(avg_dist)} (std: {np.std(avg_dist)})')

