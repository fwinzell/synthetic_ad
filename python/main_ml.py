from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

from synthesize import synthesize
import os
import json

import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from train_svm import data_loading, data_loading_a4, impute_missing_values, scale_data, fit_svm_classif, fit_svm_regressor
from utils import ConfusionVarianceMatrix

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor



def load_description(file_dict):
    description_file = file_dict['description_file']
    with open(description_file, 'r') as f:
        description = json.load(f)
    return description

def frame2arrays(df, label='DX'):
    le = LabelEncoder()
    if label == 'DX':
        class_mapping = {"CN": 0, "MCI": 1, "Dementia": 2}
        y = np.array(df['DX'].map(class_mapping))
    elif label == 'AB':
        df = df.dropna(subset=['AV45'])
        ab_status = df['AV45'] > 1.11
        #y = np.array(ab_status).astype(int)
        y = le.fit_transform(ab_status)
        df = df.drop('AV45', axis=1)
    elif label == 'ADAS13' or label == 'MMSE':
        df = df.dropna(subset=[label])
        y = np.array(df[label])
        df = df.drop(columns=['ADAS13', 'MMSE'], axis=1)
    elif label == 'AB_status':
        ab_status = df['AB_status'] == 'positive'
        y = le.fit_transform(ab_status)
        labels = df["AB_status"]
        X = df.drop(columns=['AB_status', 'PMODSUVR'] , axis=1)
    elif label == 'MMSCORE':
        df = df.dropna(subset=['MMSCORE'])
        y = df['MMSCORE']
        labels = df["AB_status"]
        X = df.drop(columns=['AB_status', 'MMSCORE', 'LIMMTOTAL', 'LDELTOTAL', 'DIGITTOTAL', 'CDSOB'] , axis=1)

    if 'DX' in df.columns:
        labels = df["DX"]
        X = df.drop("DX", axis=1)
    #X = X.drop('RID', axis=1)
    if 'RID' in X.columns:
        X = X.drop("RID", axis=1)
    if 'Month' in X.columns:
        X = X.drop("Month", axis=1)
    if 'DXN' in X.columns:
        X = X.drop("DXN", axis=1)

    if 'VISCODE' in X.columns:
        X = X.drop("VISCODE", axis=1)
    if 'EXAMDAY' in X.columns:
        X = X.drop("EXAMDAY", axis=1)
    if 'BID' in X.columns:
        X = X.drop("BID", axis=1)

    return X, y, labels


def generate_synthetic_data(description_file, num_tuples_to_generate=1000):
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)

    df = pd.DataFrame(generator.synthetic_dataset)

    return df

def fit_hgbc(X_train, y_train, params, cats=["PTGENDER", "PTEDUCAT", "APOE4"]):
    hgbc = HistGradientBoostingClassifier(categorical_features=cats, **params)
    hgbc.fit(X_train, y_train)
    return hgbc

def fit_hgbr(X_train, y_train, params, cats=["PTGENDER", "PTEDUCAT", "APOE4"]):
    hgbr = HistGradientBoostingRegressor(categorical_features=cats, **params)
    hgbr.fit(X_train, y_train)
    return hgbr

def run_svm(data_dir, label):
    # Classifier
    svm_params = {
        'DX': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'AB': {'C': 10, 'gamma': 'auto', 'kernel': 'linear'}
    }

    dict_list = []
    # LABEL
    
    X_test, y_test, _ = data_loading('/home/fi5666wi/R/data/DDLS/adni_test.csv', label=label)
    X_test_i = impute_missing_values(X_test)

    X_test_i = scale_data(X_test_i)
    X_test = scale_data(X_test)

    svm_results = {'acc': [], 'auc': [], 'cm': []}
        
    for f in tqdm(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(data_dir, f))
        X_syn, y_syn, _ = frame2arrays(df, label=label)
        X_syn_i = impute_missing_values(X_syn)

        #X_train = scale_data(X_syn)
        clf_svm = fit_svm_classif(scale_data(X_syn_i), y_syn, svm_params[label])

        y_hat_svm = clf_svm.predict(X_test_i)
        y_prob_svm = clf_svm.predict_proba(X_test_i)

        if label == 'AB':
            y_prob_svm = y_prob_svm[:,1]

        svm_results['acc'].append(np.mean(y_hat_svm == y_test))
        svm_results['auc'].append(roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted'))
        svm_results['cm'].append(confusion_matrix(y_test, y_hat_svm))


    if label == "DX":
        disp_lab = ['CN', 'MCI', 'Dementia']
    elif label == "AB":
        disp_lab = ['Negative', 'Positive']

    svm_cms = np.stack(svm_results['cm'], axis=0)
    svm_cm = ConfusionVarianceMatrix(svm_cms, display_labels=disp_lab)
    svm_cm = svm_cm.plot()
    print(f'SVM results for {label}:')
    print(f'Accuracy: {np.mean(svm_results["acc"])} +/- {np.std(svm_results["acc"])}')
    print(f'AUC score: {np.mean(svm_results["auc"])} +/- {np.std(svm_results["auc"])}')

    dict_list.append({'model': 'svm', 'label': label, 'metric': "acc", 'mean': np.mean(svm_results["acc"]), 'std': np.std(svm_results["acc"])})
    dict_list.append({'model': 'svm', 'label': label, 'metric': "auc", 'mean': np.mean(svm_results["auc"]), 'std': np.std(svm_results["auc"])})

    plt.show()



def run_hgb(bn_dir, num_tuples):
    X_test, y_test, _ = data_loading('/Users/filipwinzell/R/data/DDLS/adni_test.csv')
    hgb_results = {'acc': [], 'f1': [], 'cm': []}

    for f in tqdm(os.listdir(bn_dir)):
        X_syn, y_syn, synlabs = frame2arrays(generate_synthetic_data(os.path.join(bn_dir, f), num_tuples))

 
        clf_hgb = fit_hgb(scale_data(X_syn), y_syn, max_iter=1000, verbose=0)
        y_hat = clf_hgb.predict(scale_data(X_test))

        acc = np.mean(y_hat == y_test)
        hgb_results['acc'].append(acc)

        f1 = f1_score(y_test, y_hat, average='weighted')
        hgb_results['f1'].append(f1)

        hgb_cm = confusion_matrix(y_test, y_hat)
        hgb_results['cm'].append(hgb_cm)


    hgb_cms = np.stack(hgb_results['cm'], axis=0)
    hgb_cm = ConfusionVarianceMatrix(hgb_cms, display_labels=['CN', 'MCI', 'Dementia'])
    #hgb_cm.plot()
    

    print('HistGradBoost results:')
    print(f'Accuracy: {np.mean(hgb_results["acc"])} +/- {np.std(hgb_results["acc"])}')
    print(f'F1 score: {np.mean(hgb_results["f1"])} +/- {np.std(hgb_results["f1"])}')

    return hgb_cm



def bootstrap_svm(file_name, label, params, n=100, load_csv=False, num_tuples=1000, dataset="adni"):
    if label == 'AB' or label == 'DX' or label == 'AB_status':
        svm_results = {'acc': [], 'auc': [], 'cm': []}
    else:
        svm_results = {'r2': [], 'rmse': []}

    if dataset == "a4":
        X_test, y_test, _ = data_loading_a4(f'/home/fi5666wi/R/data/DDLS/a4_test.csv', label=label)
        cats = ['PTGENDER', 'PTEDUCAT', 'PTETHNIC', 'PTMARRY', 'PTNOTRT', 'PTHOME']
        id_col = "BID"
    elif dataset == "adni":
        X_test, y_test, _ = data_loading('/home/fi5666wi/R/data/DDLS/adni_test.csv', label=label)
        cats = ['PTGENDER', 'PTEDUCAT', 'APOE4']
        id_col = "RID"

    X_test = impute_missing_values(X_test, cats=cats)

    if load_csv:
        df = pd.read_csv(file_name)
    else:
        df = generate_synthetic_data(file_name, num_tuples)
    rids = df[id_col].unique()
    for i in tqdm(range(n)):
        sample = np.random.choice(rids, len(rids), replace=True)
        sample_df = df[df[id_col].isin(sample)]
        X_train, y_train, _ = frame2arrays(sample_df, label=label)
        X_train = impute_missing_values(X_train, cats=cats)
        X_train = scale_data(X_train)

        if label == 'DX' or label == 'AB' or label == 'AB_status':
            clf_svm = fit_svm_classif(X_train, y_train, params)
            y_hat = clf_svm.predict(scale_data(X_test))
            y_prob = clf_svm.predict_proba(scale_data(X_test))
            if label == 'AB' or label == 'AB_status':
                y_prob = y_prob[:,1]

            acc = np.mean(y_hat == y_test)
            svm_results['acc'].append(acc)

            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            svm_results['auc'].append(auc)

            svm_cm = confusion_matrix(y_test, y_hat)
            svm_results['cm'].append(svm_cm)
        else:
            reg_svm = fit_svm_regressor(X_train, y_train, params)
            y_hat = reg_svm.predict(scale_data(X_test))

            r2 = r2_score(y_test, y_hat)
            svm_results['r2'].append(r2)

            rmse = np.sqrt(mean_squared_error(y_test, y_hat))
            svm_results['rmse'].append(rmse)
        

    if label == 'DX' or label == 'AB' or label == 'AB_status':
        svm_cms = np.stack(svm_results['cm'], axis=0)
        if label == 'DX': disp_labs = ['CN', 'MCI', 'Dementia']
        elif label == 'AB': disp_labs = ['Negative', 'Positive']
        elif label == 'AB_status': disp_labs = ['Negative', 'Positive']

        svm_cm = ConfusionVarianceMatrix(svm_cms, display_labels=disp_labs)
        svm_cm = svm_cm.plot()
    

    if label == 'DX' or label == 'AB' or label == 'AB_status':
        print('SVM results:')
        print(f'Accuracy: {np.mean(svm_results["acc"])} +/- {np.std(svm_results["acc"])}')
        print(f'AUC: {np.mean(svm_results["auc"])} +/- {np.std(svm_results["auc"])}')
        plt.show()
    else:
        print('SVM results:')
        print(f'R2: {np.mean(svm_results["r2"])} +/- {np.std(svm_results["r2"])}')
        print(f'RMSE: {np.mean(svm_results["rmse"])} +/- {np.std(svm_results["rmse"])}')

    return svm_results


def bootstrap_hgb(file_name, label, params, n=100, load_csv=False, num_tuples=1000, dataset="adni"):
    if label == 'AB' or label == 'DX' or label == 'AB_status':
        hgb_results = {'acc': [], 'auc': [], 'cm': []}
    else:
        hgb_results = {'r2': [], 'rmse': []}
    
    if dataset == "a4":
        X_test, y_test, _ = data_loading_a4(f'/home/fi5666wi/R/data/DDLS/a4_test.csv', label=label)
        id_col = "BID"
        cats = ['PTGENDER', 'PTEDUCAT', 'PTETHNIC', 'PTMARRY', 'PTNOTRT', 'PTHOME']
    elif dataset == "adni":
        X_test, y_test, _ = data_loading('/home/fi5666wi/R/data/DDLS/adni_test.csv', label=label)
        id_col = "RID"
        cats = ['PTGENDER', 'PTEDUCAT', 'APOE4']

    if load_csv:
        df = pd.read_csv(file_name)
    else:
        df = generate_synthetic_data(file_name, num_tuples)
    rids = df[id_col].unique()
    for i in tqdm(range(n)):
        sample = np.random.choice(rids, len(rids), replace=True)
        sample_df = df[df[id_col].isin(sample)]
        X_train, y_train, _ = frame2arrays(sample_df, label=label)

        X_train = scale_data(X_train)

        if label == 'DX' or label == 'AB' or label == 'AB_status':
            hgbc = fit_hgbc(X_train, y_train, params, cats=cats)
            y_hat = hgbc.predict(scale_data(X_test))
            y_prob = hgbc.predict_proba(scale_data(X_test))
            if label == 'AB' or label == 'AB_status':
                y_prob = y_prob[:,1]

            acc = np.mean(y_hat == y_test)
            hgb_results['acc'].append(acc)

            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            hgb_results['auc'].append(auc)

            cm = confusion_matrix(y_test, y_hat)
            hgb_results['cm'].append(cm)
        else:
            reg_hgb = fit_hgbr(X_train, y_train, params, cats=cats)
            y_hat = reg_hgb.predict(scale_data(X_test))

            r2 = r2_score(y_test, y_hat)
            hgb_results['r2'].append(r2)

            rmse = np.sqrt(mean_squared_error(y_test, y_hat))
            hgb_results['rmse'].append(rmse)
        

    if label == 'DX' or label == 'AB' or label == 'AB_status':
        hgb_cms = np.stack(hgb_results['cm'], axis=0)
        if label == 'DX': disp_labs = ['CN', 'MCI', 'Dementia']
        elif label == 'AB': disp_labs = ['Negative', 'Positive']
        elif label == 'AB_status': disp_labs = ['Negative', 'Positive']

        hgb_cm = ConfusionVarianceMatrix(hgb_cms, display_labels=disp_labs)
        hgb_cm = hgb_cm.plot()
    

    if label == 'DX' or label == 'AB' or label == 'AB_status':
        print('HGB results:')
        print(f'Accuracy: {np.mean(hgb_results["acc"])} +/- {np.std(hgb_results["acc"])}')
        print(f'AUC: {np.mean(hgb_results["auc"])} +/- {np.std(hgb_results["auc"])}')
        plt.show()
    else:
        print('HGB results:')
        print(f'R2: {np.mean(hgb_results["r2"])} +/- {np.std(hgb_results["r2"])}')
        print(f'RMSE: {np.mean(hgb_results["rmse"])} +/- {np.std(hgb_results["rmse"])}')

    return hgb_results
    

def example():
    cm_array = np.random.randint(0, 100, (3, 3, 3))
    cm = ConfusionVarianceMatrix(cm_array, display_labels=["A", "B", "C"])
    cm.plot()


def main_ml(data_dir, verbose=True):
    # This method is used to run all experiments i.e. SVC, SVR, HGBR, HGBC on generated synthetic data
    svm_params = {
        'DX': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'AB': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'ADAS13': {'C': 100, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'},
        'MMSE': {'C': 50, 'epsilon': 1, 'gamma': 'auto', 'kernel': 'rbf'}
    }

    hgb_params = {
        'DX': {'l2_regularization': 1, 'learning_rate': 0.01, 'max_bins': 255, 'max_depth': 3, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'AB': {'l2_regularization': 0, 'learning_rate': 0.2, 'max_bins': 125, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'ADAS13': {'l2_regularization': 0, 'learning_rate': 0.1, 'max_bins': 125, 'max_depth': None, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'MMSE': {'l2_regularization': 10, 'learning_rate': 0.1, 'max_bins': 125, 'max_depth': 3, 'max_iter': 200, 'max_leaf_nodes': 31, 'min_samples_leaf': 100}
    }

    dict_list = []
    # LABEL
    for label in ['DX', 'AB']:
        X_test, y_test, _ = data_loading('/home/fi5666wi/R/data/DDLS/adni_test.csv', label=label)
        X_test_i = impute_missing_values(X_test)

        X_test_i = scale_data(X_test_i)
        X_test = scale_data(X_test)

        svm_results = {'acc': [], 'auc': [], 'cm': []}
        hgb_results = {'acc': [], 'auc': [], 'cm': []}
        for f in tqdm(os.listdir(data_dir)):
            #X_syn, y_syn, _ = data_loading(os.path.join(data_dir, f), label='DX')
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(data_dir, f))
            X_syn, y_syn, _ = frame2arrays(df, label=label)
            X_syn_i = impute_missing_values(X_syn)

            #X_train = scale_data(X_syn)
            clf_svm = fit_svm_classif(scale_data(X_syn_i), y_syn, svm_params[label])
            clf_hgb = fit_hgbc(scale_data(X_syn), y_syn, hgb_params[label])

            y_hat_svm = clf_svm.predict(X_test_i)
            y_prob_svm = clf_svm.predict_proba(X_test_i)

            y_hat_hgb = clf_hgb.predict(X_test)
            y_prob_hgb = clf_hgb.predict_proba(X_test)

            if label == 'AB':
                y_prob_hgb, y_prob_svm = y_prob_hgb[:,1], y_prob_svm[:,1]

            svm_results['acc'].append(np.mean(y_hat_svm == y_test))
            svm_results['auc'].append(roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted'))
            svm_results['cm'].append(confusion_matrix(y_test, y_hat_svm))

            hgb_results['acc'].append(np.mean(y_hat_hgb == y_test))
            hgb_results['auc'].append(roc_auc_score(y_test, y_prob_hgb, multi_class='ovr', average='weighted'))
            hgb_results['cm'].append(confusion_matrix(y_test, y_hat_hgb))


        if label == "DX":
            disp_lab = ['CN', 'MCI', 'Dementia']
        elif label == "AB":
            disp_lab = ['Negative', 'Positive']

        svm_cms = np.stack(svm_results['cm'], axis=0)
        if verbose:
            svm_cm = ConfusionVarianceMatrix(svm_cms, display_labels=disp_lab)
            svm_cm = svm_cm.plot()
        print(f'SVM results for {label}:')
        print(f'Accuracy: {np.mean(svm_results["acc"])} +/- {np.std(svm_results["acc"])}')
        print(f'AUC score: {np.mean(svm_results["auc"])} +/- {np.std(svm_results["auc"])}')

        dict_list.append({'model': 'svm', 'label': label, 'metric': "acc", 'mean': np.mean(svm_results["acc"]), 'std': np.std(svm_results["acc"])})
        dict_list.append({'model': 'svm', 'label': label, 'metric': "auc", 'mean': np.mean(svm_results["auc"]), 'std': np.std(svm_results["auc"])})


        hgb_cms = np.stack(hgb_results['cm'], axis=0)
        if verbose:
            hgb_cm = ConfusionVarianceMatrix(hgb_cms, display_labels=disp_lab)
            hgb_cm.plot()
        
        print(f'HistGradBoost results for {label}:')
        print(f'Accuracy: {np.mean(hgb_results["acc"])} +/- {np.std(hgb_results["acc"])}')
        print(f'AUC score: {np.mean(hgb_results["auc"])} +/- {np.std(hgb_results["auc"])}')

        dict_list.append({'model': 'hgb', 'label': label, 'metric': "acc", 'mean': np.mean(hgb_results["acc"]), 'std': np.std(hgb_results["acc"])})
        dict_list.append({'model': 'hgb', 'label': label, 'metric': "auc", 'mean': np.mean(hgb_results["auc"]), 'std': np.std(hgb_results["auc"])})

        if verbose:
            plt.show()

    for label in ['ADAS13', 'MMSE']:
        X_test, y_test, _ = data_loading('/home/fi5666wi/R/data/DDLS/adni_test.csv', label=label)
        X_test_i = impute_missing_values(X_test)
        svm_results = {'r2': [], 'rmse': []}
        hgb_results = {'r2': [], 'rmse': []}
        for f in tqdm(os.listdir(data_dir)):
            #X_syn, y_syn, _ = data_loading(os.path.join(data_dir, f), label='DX')
            df = pd.read_csv(os.path.join(data_dir, f))
            X_syn, y_syn, _ = frame2arrays(df, label=label)
            X_syn_i = impute_missing_values(X_syn)

            #X_train = scale_data(X_syn)
            reg_svm = fit_svm_regressor(scale_data(X_syn_i), y_syn, svm_params[label])
            reg_hgb = fit_hgbr(scale_data(X_syn), y_syn, hgb_params[label])

            y_hat_svm = reg_svm.predict(scale_data(X_test_i))
            y_hat_hgb = reg_hgb.predict(scale_data(X_test))

            svm_results['r2'].append(r2_score(y_test, y_hat_svm))
            svm_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_hat_svm)))

            hgb_results['r2'].append(r2_score(y_test, y_hat_hgb))
            hgb_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_hat_hgb)))


        print(f'SVM results for {label}:')
        print(f'R2: {np.mean(svm_results["r2"])} +/- {np.std(svm_results["r2"])}')
        print(f'RMSE: {np.mean(svm_results["rmse"])} +/- {np.std(svm_results["rmse"])}')

        dict_list.append({'model': 'svm', 'label': label, 'metric': "r2", 'mean': np.mean(svm_results["r2"]), 'std': np.std(svm_results["r2"])})
        dict_list.append({'model': 'svm', 'label': label, 'metric': "rmse", 'mean': np.mean(svm_results["rmse"]), 'std': np.std(svm_results["rmse"])})
        
        print(f'HistGradBoost results for {label}:')
        print(f'R2: {np.mean(hgb_results["r2"])} +/- {np.std(hgb_results["r2"])}')
        print(f'RMSE: {np.mean(hgb_results["rmse"])} +/- {np.std(hgb_results["rmse"])}')

        dict_list.append({'model': 'hgb', 'label': label, 'metric': "r2", 'mean': np.mean(hgb_results["r2"]), 'std': np.std(hgb_results["r2"])})
        dict_list.append({'model': 'hgb', 'label': label, 'metric': "rmse", 'mean': np.mean(hgb_results["rmse"]), 'std': np.std(hgb_results["rmse"])})


    res_df = pd.DataFrame(dict_list)
    return res_df


def main_ml_a4(data_dir, verbose=True):
    # This method is used to run all experiments i.e. SVC, SVR, HGBR, HGBC on generated synthetic data
    svm_params = {
        'AB_status': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'MMSCORE': {'C': 2.5, 'epsilon': 1, 'gamma': 'scale', 'kernel': 'rbf'}
    }

    hgb_params = {
        'AB_status': {'l2_regularization': 0, 'learning_rate': 0.01, 'max_bins': 255, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20},
        'MMSCORE': {'l2_regularization': 10, 'learning_rate': 0.01, 'max_bins': 125, 'max_depth': 3, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 50}
    }

    dict_list = []
    # LABEL
    cats=['PTGENDER', 'PTEDUCAT', 'PTETHNIC', 'PTMARRY', 'PTNOTRT', 'PTHOME']
    for label in ['AB_status']:
        X_test, y_test, _ = data_loading_a4('/home/fi5666wi/R/data/DDLS/a4_test.csv', label=label)
        X_test_i = impute_missing_values(X_test, cats=cats)

        X_test_i = scale_data(X_test_i)
        X_test = scale_data(X_test)

        svm_results = {'acc': [], 'auc': [], 'cm': []}
        hgb_results = {'acc': [], 'auc': [], 'cm': []}
        for f in tqdm(os.listdir(data_dir)):
            #X_syn, y_syn, _ = data_loading(os.path.join(data_dir, f), label='DX')
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(data_dir, f))
            X_syn, y_syn, _ = frame2arrays(df, label=label)
            X_syn_i = impute_missing_values(X_syn, cats=cats)

            #X_train = scale_data(X_syn)
            clf_svm = fit_svm_classif(scale_data(X_syn_i), y_syn, svm_params[label])
            clf_hgb = fit_hgbc(scale_data(X_syn), y_syn, hgb_params[label], cats=cats)

            y_hat_svm = clf_svm.predict(X_test_i)
            y_prob_svm = clf_svm.predict_proba(X_test_i)

            y_hat_hgb = clf_hgb.predict(X_test)
            y_prob_hgb = clf_hgb.predict_proba(X_test)
        
            y_prob_hgb, y_prob_svm = y_prob_hgb[:,1], y_prob_svm[:,1]

            svm_results['acc'].append(np.mean(y_hat_svm == y_test))
            svm_results['auc'].append(roc_auc_score(y_test, y_prob_svm, multi_class='ovr', average='weighted'))
            svm_results['cm'].append(confusion_matrix(y_test, y_hat_svm))

            hgb_results['acc'].append(np.mean(y_hat_hgb == y_test))
            hgb_results['auc'].append(roc_auc_score(y_test, y_prob_hgb, multi_class='ovr', average='weighted'))
            hgb_results['cm'].append(confusion_matrix(y_test, y_hat_hgb))


        if label == "AB_status":
             disp_lab = ['Negative', 'Positive']

        svm_cms = np.stack(svm_results['cm'], axis=0)
        if verbose:
            svm_cm = ConfusionVarianceMatrix(svm_cms, display_labels=disp_lab)
            svm_cm = svm_cm.plot()
        print(f'SVM results for {label}:')
        print(f'Accuracy: {np.mean(svm_results["acc"])} +/- {np.std(svm_results["acc"])}')
        print(f'AUC score: {np.mean(svm_results["auc"])} +/- {np.std(svm_results["auc"])}')

        dict_list.append({'model': 'svm', 'label': label, 'metric': "acc", 'mean': np.mean(svm_results["acc"]), 'std': np.std(svm_results["acc"])})
        dict_list.append({'model': 'svm', 'label': label, 'metric': "auc", 'mean': np.mean(svm_results["auc"]), 'std': np.std(svm_results["auc"])})


        hgb_cms = np.stack(hgb_results['cm'], axis=0)
        if verbose:
            hgb_cm = ConfusionVarianceMatrix(hgb_cms, display_labels=disp_lab)
            hgb_cm.plot()
        
        print(f'HistGradBoost results for {label}:')
        print(f'Accuracy: {np.mean(hgb_results["acc"])} +/- {np.std(hgb_results["acc"])}')
        print(f'AUC score: {np.mean(hgb_results["auc"])} +/- {np.std(hgb_results["auc"])}')

        dict_list.append({'model': 'hgb', 'label': label, 'metric': "acc", 'mean': np.mean(hgb_results["acc"]), 'std': np.std(hgb_results["acc"])})
        dict_list.append({'model': 'hgb', 'label': label, 'metric': "auc", 'mean': np.mean(hgb_results["auc"]), 'std': np.std(hgb_results["auc"])})

        if verbose:
            plt.show()

    for label in ['MMSCORE']:
        X_test, y_test, _ = data_loading_a4('/home/fi5666wi/R/data/DDLS/a4_test.csv', label=label)
        X_test_i = impute_missing_values(X_test, cats=cats)
        svm_results = {'r2': [], 'rmse': []}
        hgb_results = {'r2': [], 'rmse': []}
        for f in tqdm(os.listdir(data_dir)):
            #X_syn, y_syn, _ = data_loading(os.path.join(data_dir, f), label='DX')
            df = pd.read_csv(os.path.join(data_dir, f))
            X_syn, y_syn, _ = frame2arrays(df, label=label)
            X_syn_i = impute_missing_values(X_syn, cats=cats)

            #X_train = scale_data(X_syn)
            reg_svm = fit_svm_regressor(scale_data(X_syn_i), y_syn, svm_params[label])
            reg_hgb = fit_hgbr(scale_data(X_syn), y_syn, hgb_params[label], cats=cats)

            y_hat_svm = reg_svm.predict(scale_data(X_test_i))
            y_hat_hgb = reg_hgb.predict(scale_data(X_test))

            svm_results['r2'].append(r2_score(y_test, y_hat_svm))
            svm_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_hat_svm)))

            hgb_results['r2'].append(r2_score(y_test, y_hat_hgb))
            hgb_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_hat_hgb)))


        print(f'SVM results for {label}:')
        print(f'R2: {np.mean(svm_results["r2"])} +/- {np.std(svm_results["r2"])}')
        print(f'RMSE: {np.mean(svm_results["rmse"])} +/- {np.std(svm_results["rmse"])}')

        dict_list.append({'model': 'svm', 'label': label, 'metric': "r2", 'mean': np.mean(svm_results["r2"]), 'std': np.std(svm_results["r2"])})
        dict_list.append({'model': 'svm', 'label': label, 'metric': "rmse", 'mean': np.mean(svm_results["rmse"]), 'std': np.std(svm_results["rmse"])})
        
        print(f'HistGradBoost results for {label}:')
        print(f'R2: {np.mean(hgb_results["r2"])} +/- {np.std(hgb_results["r2"])}')
        print(f'RMSE: {np.mean(hgb_results["rmse"])} +/- {np.std(hgb_results["rmse"])}')

        dict_list.append({'model': 'hgb', 'label': label, 'metric': "r2", 'mean': np.mean(hgb_results["r2"]), 'std': np.std(hgb_results["r2"])})
        dict_list.append({'model': 'hgb', 'label': label, 'metric': "rmse", 'mean': np.mean(hgb_results["rmse"]), 'std': np.std(hgb_results["rmse"])})


    res_df = pd.DataFrame(dict_list)
    return res_df


def bootstrap_all_on_real(dataset):
    file_real = f'/home/fi5666wi/R/data/DDLS/{dataset}_train_ml.csv'
    #bn_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-bayesian-networks/degree3_eps_100/'
    svm_params = {
        'DX': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'AB': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
        'ADAS13': {'C': 100, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'},
        'MMSE': {'C': 50, 'epsilon': 1, 'gamma': 'auto', 'kernel': 'rbf'},
        'AB_status': {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
        'MMSCORE': {'C': 2.5, 'epsilon': 1, 'gamma': 'scale', 'kernel': 'rbf'}
    }

    hgb_params = {
        'DX': {'l2_regularization': 1, 'learning_rate': 0.01, 'max_bins': 255, 'max_depth': 3, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'AB': {'l2_regularization': 0, 'learning_rate': 0.2, 'max_bins': 125, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'ADAS13': {'l2_regularization': 0, 'learning_rate': 0.1, 'max_bins': 125, 'max_depth': None, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'MMSE': {'l2_regularization': 10, 'learning_rate': 0.1, 'max_bins': 125, 'max_depth': 3, 'max_iter': 200, 'max_leaf_nodes': 31, 'min_samples_leaf': 100},
        'AB_status': {'l2_regularization': 0, 'learning_rate': 0.01, 'max_bins': 255, 'max_depth': 3, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20},
        'MMSCORE': {'l2_regularization': 10, 'learning_rate': 0.01, 'max_bins': 125, 'max_depth': 3, 'max_iter': 300, 'max_leaf_nodes': 31, 'min_samples_leaf': 50}
    }


    dict_list = []
    if dataset == "adni":
        for label in ['DX', 'AB', 'ADAS13', 'MMSE']:

            svm_dict = bootstrap_svm(file_real, label, svm_params[label], load_csv=True, n=100)
            hgb_dict = bootstrap_hgb(file_real, label, hgb_params[label], load_csv=True, n=100)

            if label == 'DX' or label == 'AB':
                metrics = ['acc', 'auc']
            else:
                metrics = ['r2', 'rmse']

            dict_list.append({'model': 'svm', 'label': label, 'metric': metrics[0], 'mean': np.mean(svm_dict[metrics[0]]), 'std': np.std(svm_dict[metrics[0]])})
            dict_list.append({'model': 'svm', 'label': label, 'metric': metrics[1], 'mean': np.mean(svm_dict[metrics[1]]), 'std': np.std(svm_dict[metrics[1]])})
            dict_list.append({'model': 'hgb', 'label': label, 'metric': metrics[0], 'mean': np.mean(hgb_dict[metrics[0]]), 'std': np.std(hgb_dict[metrics[0]])})
            dict_list.append({'model': 'hgb', 'label': label, 'metric': metrics[1], 'mean': np.mean(hgb_dict[metrics[1]]), 'std': np.std(hgb_dict[metrics[1]])})
            
        res_df = pd.DataFrame(dict_list)
    elif dataset == "a4":
        for label in ['AB_status', 'MMSCORE']:

            svm_dict = bootstrap_svm(file_real, label, svm_params[label], load_csv=True, n=100, dataset="a4")
            hgb_dict = bootstrap_hgb(file_real, label, hgb_params[label], load_csv=True, n=100, dataset="a4")

            if label == 'AB_status':
                metrics = ['acc', 'auc']
            else:
                metrics = ['r2', 'rmse']

            dict_list.append({'model': 'svm', 'label': label, 'metric': metrics[0], 'mean': np.mean(svm_dict[metrics[0]]), 'std': np.std(svm_dict[metrics[0]])})
            dict_list.append({'model': 'svm', 'label': label, 'metric': metrics[1], 'mean': np.mean(svm_dict[metrics[1]]), 'std': np.std(svm_dict[metrics[1]])})
            dict_list.append({'model': 'hgb', 'label': label, 'metric': metrics[0], 'mean': np.mean(hgb_dict[metrics[0]]), 'std': np.std(hgb_dict[metrics[0]])})
            dict_list.append({'model': 'hgb', 'label': label, 'metric': metrics[1], 'mean': np.mean(hgb_dict[metrics[1]]), 'std': np.std(hgb_dict[metrics[1]])})
            
        res_df = pd.DataFrame(dict_list)
    return res_df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data using Bayesian networks.")
    parser.add_argument('--data_base_dir', type=str, help='Directory name for loading data files.', 
                        default='/home/fi5666wi/R/data/DDLS/')
    parser.add_argument('--base_dir', type=str, help='Base directory, used for loading BNs and saving generated data.', 
                        default='/home/fi5666wi/Python/WASP-DDLS')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Main method
    # Run bootstrap for estimating variance on one dataset (e.g. real training data)
    # Run run methods for comparing different synthetic datasets
    # All methods evaluated on real test data
    args = parse_args()

    #file_real = f'{args.data_base_dir}/adni_train_ml.csv'
    #X_real, _, _ = data_loading(file_real)
    #num_tuples = X_real.shape[0]
    #syn_dir = f'/home/fi5666wi/Python/WASP-DDLS/tabpfn/adni/tabpfn_t_1.0/'
    #params = {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
    #label = 'AB'
    
    #svm_dict = bootstrap_svm(file_real, label, params, load_csv=True, n=100)
    #hgb_dict = bootstrap_hgb(file_real, label, params, load_csv=True, n=100)

    #run_svm(syn_dir, label)

    #for epsilon in [5, 10, 50, 100, 200, 'zero']:
    #epsilon = 'zero'
    #    data_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/a4/degree2_eps_{epsilon}/'
    #    res_df = main_ml_a4(data_dir, verbose=False)
    #    res_df.to_csv(f'/home/fi5666wi/Python/WASP-DDLS/ML-results/a4/deg2_eps_{epsilon}.csv', index=False)

    #for temp in [1.0]:
    #    data_dir = f'{args.base_dir}/tabpfn/a4/tabpfn_t_{temp}/'
    #    res_df = main_ml_a4(data_dir, verbose=False)
    #    res_df.to_csv(f'{args.base_dir}/ML-results/a4/tabpfn_t_{temp}.csv', index=False)

    #for degree in [1, 2, 3, 4]:
    #    epsilon = 'zero'
    #    data_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/degree{degree}_eps_{epsilon}/'
    #    res_df = main_ml(data_dir, verbose=False)
    #    res_df.to_csv(f'/home/fi5666wi/Python/WASP-DDLS/ML-results/deg_{degree}.csv', index=False)

    #synthpop_dir = '/home/fi5666wi/Python/WASP-DDLS/synthpop/a4/'
    #synp_df = main_ml_a4(synthpop_dir, verbose=False)
    #synp_df.to_csv('/home/fi5666wi/Python/WASP-DDLS/ML-results/a4/synthpop.csv', index=False)

    #for epochs in [750]:
    #    for setting  in ['default', 'optim']:
    #        ctgan_dir = f'{args.base_dir}/ctgan/a4/ctgan_{setting}_epochs_{epochs}/'
    #        res_df = main_ml_a4(ctgan_dir, verbose=False)
    #        res_df.to_csv(f'{args.base_dir}/ML-results/a4/ctgan_{setting}_epochs_{epochs}.csv', index=False)

    res_df = bootstrap_all_on_real(dataset='a4')
    res_df.to_csv('/home/fi5666wi/Python/WASP-DDLS/ML-results/a4/real_data.csv', index=False)
 


##### Results of grid search ######
# These results are from running the grid search methods in train_svm.py and train_hgb.py
# Enjoy!
#
# SVR -> MMSE
# {'C': 50, 'epsilon': 1, 'gamma': 'auto', 'kernel': 'rbf'}
# R2: 0.308
# RMSE: 2.24

# SVR -> ADAS13
# {'C': 100, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}
# R2: 0.361
# RMSE: 7.74    

# SVC -> AB
# {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
# Accuracy: 0.776
# F1: 0.775
# AUC (1 vs all): 0.873

# SVC -> DX
# {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
# Accuracy: 0.745
# F1: 0.743
# AUC (1 vs all): 0.891

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

