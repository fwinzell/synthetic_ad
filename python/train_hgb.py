
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from train_svm import data_loading, data_loading_a4, scale_data, classif_plot, regress_plot

def fit_hgb(X_train, y_train, params=None):
    cats = ["PTGENDER", "PTEDUCAT", "APOE4"]
    hgbc = HistGradientBoostingClassifier(categorical_features=cats, **params)
    hgbc.fit(X_train, y_train)
    return hgbc

def fit_hgb_reg(X_train, y_train, lr=0.01, max_iter=1000, early_stopping=True, n_iter_no_change=10, tol=1e-7, random_state=0, verbose=1, max_bins=255):
    params = {
        'learning_rate': lr,
        'max_iter': max_iter,
        'early_stopping': early_stopping,
        'n_iter_no_change': n_iter_no_change,
        'tol': tol,
        'random_state': random_state,
        'verbose': verbose,
        'max_bins': max_bins
    }
    cats = ["PTGENDER", "PTEDUCAT", "APOE4"]
    hgbr = HistGradientBoostingRegressor(categorical_features=cats, **params)
    hgbr.fit(X_train, y_train)
    return hgbr
    
def hgbc_grid_search(X_train, y_train, cats=["PTGENDER", "PTEDUCAT", "APOE4"]):
    hgbc = HistGradientBoostingClassifier(categorical_features=cats)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [100, 200, 300],
        'max_leaf_nodes': [31, 63, 127],
        'max_depth': [None, 3, 5, 7],
        'min_samples_leaf': [20, 50, 100],
        'l2_regularization': [0, 1, 10],
        'max_bins': [125, 255]
    }
    grid_search = GridSearchCV(hgbc, param_grid, cv=5, scoring='roc_auc_ovr_weighted')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def hgbr_grid_search(X_train, y_train, cats=["PTGENDER", "PTEDUCAT", "APOE4"]):
    hgbr = HistGradientBoostingRegressor(categorical_features=cats)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [100, 200, 300],
        'max_leaf_nodes': [31, 63, 127],
        'max_depth': [None, 3, 5, 7],
        'min_samples_leaf': [20, 50, 100],
        'l2_regularization': [0, 1, 10],
        'max_bins': [125, 255]
    }
    grid_search = GridSearchCV(hgbr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_


def main_grid_search(dataset):
    file_real = f'/home/fi5666wi/R/data/DDLS/{dataset}_train_ml.csv'

    if dataset == 'a4':
        label = 'MMSCORE'
        X_train, y_train, y_labels = data_loading_a4(file_real, label=label)
        X_test, y_test, y_true = data_loading_a4(f'/home/fi5666wi/R/data/DDLS/{dataset}_test.csv', label=label)
        cats = ['PTGENDER', 'PTEDUCAT', 'PTETHNIC', 'PTMARRY', 'PTNOTRT', 'PTHOME']
    else:
        label = 'AB'
        X_train, y_train, y_labels = data_loading(file_real, label=label)
        X_test, y_test, y_true = data_loading(f'/home/fi5666wi/R/data/DDLS/{dataset}_test.csv', label=label)
        cats = ['PTGENDER', 'PTEDUCAT', 'APOE4']
    print('Data loaded successfully')

    #X_train = impute_missing_values(X_train)
    X_train = scale_data(X_train)

    if label == 'DX' or label == 'AB' or label == 'AB_status':
        #params = {'l2_regularization': 0, 'learning_rate': 0.2, 'max_bins': 125, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100}
        #hgb = fit_hgb(X_train, y_train, params)
        hgb = hgbc_grid_search(X_train, y_train, cats=cats)
    else:
        #hgb = fit_hgb_reg(X_train, y_train, max_iter=1000)
        hgb = hgbr_grid_search(X_train, y_train, cats=cats)

    
    y_hat = hgb.predict(scale_data(X_test))

    if label == 'DX':
        y_prob = hgb.predict_proba(scale_data(X_test))
        classif_plot(y_hat, y_test, label, y_prob)
    elif label == 'AB' or label == 'AB_status':
        y_prob = hgb.predict_proba(scale_data(X_test))[:,1]
        classif_plot(y_hat, y_test, label, y_prob)
    else:
        regress_plot(y_hat, y_test)

    """
    max_iters = [50, 100, 200]
    for mi in max_iters:
        hgbr = fit_hgbr(X_train, y_train, max_iter=mi, verbose=0, early_stopping=False)
        y_hat = hgbr.predict(scale_data(X_test))
        print(f'Accuracy for max_iter={mi}:', np.mean(y_hat == y_test))
    """
    
    #y_test[y_test == 2] = 1
    #predict_and_plot(X_test, y_test, clf)
    #print('Accuracy:', np.mean(y_hat == y_test))
    # f1 score
    #f1 = f1_score(y_test, y_hat, average='weighted')
    #print('F1 score:', f1)

    #display_confusion_matrix(y_test, y_hat)
    #visualize_with_cog(X_test, y_hat, y_test)

if __name__ == "__main__":
    main_grid_search('a4')

##### Results of grid search ######
# HGBR -> MMSE
# {'l2_regularization': 10, 'learning_rate': 0.1, 'max_bins': 125, 'max_depth': 3, 'max_iter': 200, 'max_leaf_nodes': 31, 'min_samples_leaf': 100}
# R2: 0.274
# RMSE: 2.29

# HGBR -> ADAS13
# {'l2_regularization': 0, 'learning_rate': 0.1, 'max_bins': 125, 'max_depth': None, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100}
# R2: 0.413
# RMSE: 7.42   

# HGBC -> AB
# {'l2_regularization': 0, 'learning_rate': 0.2, 'max_bins': 125, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100}
# Accuracy: 0.815
# F1: 0.788
# ROC AUC: 0.797

# HGBC -> DX
# {'l2_regularization': 1, 'learning_rate': 0.01, 'max_bins': 255, 'max_depth': 3, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 100}
# Accuracy: 0.732
# F1: 0.730
# ROC AUC: 0.873

# HGBC -> AB (A4)
# {'l2_regularization': 0, 'learning_rate': 0.01, 'max_bins': 255, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20}
# Accuracy: 0.988
# F1: 0.988
# AUC: 0.982

# HGBR -> MMSCORE (A4)
# {'l2_regularization': 10, 'learning_rate': 0.01, 'max_bins': 125, 'max_depth': 3, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 50}
# R2 score: 0.013
# RMSE: 1.18