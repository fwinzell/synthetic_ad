from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, r2_score, mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def data_loading(file_name, label='DX'):
    df = pd.read_csv(file_name)
    le = LabelEncoder()
    #le.fit(["CN", "MCI", "Dementia"])
    if label == 'DX':
        y = le.fit_transform(df['DXN'])
        labels = df["DX"]
        X = df.drop(columns=['DX', 'DXN'], axis=1)
    elif label == 'AB':
        df = df.dropna(subset=['AV45'])
        ab_status = df['AV45'] > 1.11
        y = le.fit_transform(ab_status)
        labels = df["DX"]
        X = df.drop(columns=['DX', 'AV45'] , axis=1)
        if 'DXN' in X.columns:
            X = X.drop('DXN', axis=1)
    elif label == 'ADAS13':
        df = df.dropna(subset=['ADAS13'])
        y = df['ADAS13']
        labels = df["DX"]
        X = df.drop(columns=['DX', 'DXN', 'ADAS13', 'MMSE'] , axis=1)
    elif label == 'MMSE':
        df = df.dropna(subset=['MMSE'])
        y = df['MMSE']
        labels = df["DX"]
        X = df.drop(columns=['DX', 'DXN', 'ADAS13', 'MMSE'] , axis=1)
    else:
        # throw error
        raise NotImplementedError

    if 'RID' in X.columns:
        X = X.drop('RID', axis=1)
    if 'Month' in X.columns:
        X = X.drop("Month", axis=1)
    #X = X[["ADAS13", "MMSE", "AGE"]]
    return X, y, labels

def data_loading_a4(file_name, label='AB_status'):
    df = pd.read_csv(file_name)
    le = LabelEncoder()

    if label == 'AB_status':
        ab_status = df['AB_status'] == 'positive'
        y = le.fit_transform(ab_status)
        labels = df["AB_status"]
        X = df.drop(columns=['AB_status', 'PMODSUVR'] , axis=1)
    elif label == 'MMSCORE':
        df = df.dropna(subset=['MMSCORE'])
        y = df['MMSCORE']
        labels = df["AB_status"]
        X = df.drop(columns=['AB_status', 'MMSCORE', 'LIMMTOTAL', 'LDELTOTAL', 'DIGITTOTAL', 'CDSOB'] , axis=1)
    else:
        # throw error
        raise NotImplementedError

    X = X.drop('BID', axis=1)
    if 'VISCODE' in X.columns:
        X = X.drop("VISCODE", axis=1)
    if 'EXAMDAY' in X.columns:
        X = X.drop("EXAMDAY", axis=1)

    return X, y, labels


def frame2arrays(df, label='DX'):
    if label == 'DX':
        class_mapping = {"CN": 0, "MCI": 1, "Dementia": 2}
        y = np.array(df['DX'].map(class_mapping))
    elif label == 'AB':
        df = df.dropna(subset=['AV45'])
        ab_status = df['AV45'] > 1.11
        y = np.array(ab_status).astype(int)
        df = df.drop('AV45', axis=1)
    elif label == 'ADAS13' or label == 'MMSE':
        df = df.dropna(subset=[label])
        y = np.array(df[label])
        df = df.drop(columns=['ADAS13', 'MMSE'], axis=1)

    labels = df["DX"]
    X = df.drop(columns=['DX', 'RID'], axis=1)
    #X = X.drop('RID', axis=1)
    if 'Month' in X.columns:
        X = X.drop("Month", axis=1)
    if 'DXN' in X.columns:
        X = X.drop("DXN", axis=1)

    return X, y, labels

def impute_missing_values(X, seed=0, cats=["PTGENDER", "PTEDUCAT", "APOE4"]):
    cont_imp = IterativeImputer(max_iter=10, random_state=seed)
    cat_imp = SimpleImputer(strategy='most_frequent')
    conts = [col for col in X.columns if col not in cats]
    X[cats] = cat_imp.fit_transform(X[cats])
    while np.any(np.isinf(X[conts])):
        X[conts] = np.where(np.isinf(X[conts]), np.nan, X[conts])
    X[conts] = cont_imp.fit_transform(X[conts])
    return X

def impute_with_zeros(X):
    cat_imp = SimpleImputer(strategy='most_frequent')
    cats = ["PTGENDER", "PTEDUCAT", "APOE4"]
    conts = [col for col in X.columns if col not in cats]
    X[cats] = cat_imp.fit_transform(X[cats])
    X[conts] = X[conts].fillna(0)
    return X

def scale_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return X

def fit_svm_classif(X_train, y_train, params=None):
    clf = SVC(probability=True, **params)
    clf.fit(X_train, y_train)
    return clf


def fit_svm_regressor(X_train, y_train, params=None):
    clf = SVR(**params)
    clf.fit(X_train, y_train)
    return clf


def visualize_with_pca(X, y_hat, y_true):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(1)
    cmap = plt.cm.colors.ListedColormap(['yellow', 'orange', 'red'])
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_hat, s=25, cmap=cmap, marker='o')

    # Create an array for edge colors
    edge_colors = ['g' if yh == yt else 'r' for yh, yt in zip(y_hat, y_true)]
    # Plot each data point again, this time with no fill color and edge color green if correctly classified, red otherwise
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=25, edgecolors=edge_colors, facecolors='none', linewidth=1)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA plot of the data')

    plt.show()


def visualize_with_cog_old(X, y_hat, y_true):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cmap = plt.cm.colors.ListedColormap(['yellow', 'orange', 'red'])
    ax1.scatter(X["ADAS13"], X["MMSE"], c=y_hat, s=10, cmap=cmap, marker='o')
    ax1.set_title('Predicted labels')

    # Create an array for edge colors
    #edge_colors = ['g' if yh == yt else 'r' for yh, yt in zip(y_hat, y_true)]
    # Plot each data point again, this time with no fill color and edge color green if correctly classified, red otherwise
    #plt.scatter(X["ADAS13"], X["MMSE"], s=10, edgecolors=edge_colors, facecolors='none', linewidth=1)

    ax1.set(xlabel = 'ADAS13', ylabel = 'MMSE')
    ax2.set(xlabel = 'ADAS13')

    ax2.scatter(X["ADAS13"], X["MMSE"], c=y_true, s=10, cmap=cmap, marker='o')
    ax2.set_title('True labels')

    ax2.legend(labels=["CN", "MCI", "Dementia"])

    #plt.xlabel('ADAS13')
    #plt.ylabel('MMSE')

    plt.show()

def visualize_with_cog(X, y_hat, y_true):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cmap = plt.cm.colors.ListedColormap(['yellow', 'orange', 'red'])
    
    for i, label in enumerate(["CN", "MCI", "Dementia"]):
        ax1.scatter(X["ADAS13"][y_hat == i], X["MMSE"][y_hat == i], color=cmap.colors[i], s=10, marker='o', label=label)
    ax1.set_title('Predicted labels')
    ax1.set(xlabel = 'ADAS13', ylabel = 'MMSE')
    ax1.legend()

    for i, label in enumerate(["CN", "MCI", "Dementia"]):
        ax2.scatter(X["ADAS13"][y_true == i], X["MMSE"][y_true == i], color=cmap.colors[i], s=10, marker='o', label=label)
    ax2.set_title('True labels')
    ax2.set(xlabel = 'ADAS13')
    ax2.legend()

    plt.show()


def display_confusion_matrix(y_true, y_hat, labels=['CN', 'MCI', 'Dementia']):
    cm = confusion_matrix(y_true, y_hat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


def predict_with_pca_plot(X_test, y_test, clf):
    y_hat = clf.predict(scale_data(X_test))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    supvec = clf.support_vectors_
    plt.figure(1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_hat, s=10, cmap='bwr', marker='o')
    plt.scatter(supvec[:, 0], supvec[:, 1], c='black', s=10, marker='x')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA plot of the data')

    display_confusion_matrix(y_test, y_hat)
    plt.show()

def classif_plot(y_hat, y_test, label, y_prob=None):
    #y_test[y_test == 2] = 1
    #predict_and_plot(X_test, y_test, clf)

    print('Accuracy:', np.mean(y_hat == y_test))
    print(classification_report(y_test, y_hat))

    # f1 score
    f1 = f1_score(y_test, y_hat, average='weighted')
    print('F1 score:', f1)

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        print('AUC:', auc)

    if label == 'DX':
        display_confusion_matrix(y_test, y_hat, labels=['CN', 'MCI', 'Dementia'])
    elif label == 'AB':
        display_confusion_matrix(y_test, y_hat, labels=['Negative', 'Positive'])

    #visualize_with_pca(X_test, y_hat, y_test)
    #visualize_with_cog(X_test, y_hat, y_test)

def regress_plot(y_hat, y_test):
    r2score = r2_score(y_test, y_hat)
    mse = mean_squared_error(y_test, y_hat)
    print('R2 score:', r2score)
    print('MSE:', mse)
    print('RMSE:', np.sqrt(mse))

    plt.scatter(y_test, y_hat)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Predicted vs True values')
    plt.show()


def svr_grid_search(X_train, y_train):
    param_grid = {
        'kernel': ['rbf'],
        'C': [2.5, 5, 10, 50, 100],
        'epsilon': [0.01, 0.1, 1],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }

    clf = SVR()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_


def svc_grid_search(X_train, y_train):
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 2.5, 5, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }

    clf = SVC(probability=True)
    grid_search = GridSearchCV(clf, param_grid, cv=6, scoring='roc_auc_ovr_weighted')
    grid_search.fit(X_train, y_train)
    #cv_res = grid_search.cv_results_
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def weight_analysis(clf, feature_names):
    w = clf.coef_.ravel()
    coef = pd.Series(w, index=feature_names).sort_values(key=np.abs, ascending=False)
    norm = np.linalg.norm(w)
    cumulative = np.cumsum(np.abs(coef.values)) / np.sum(np.abs(coef.values))

    return norm, cumulative, coef


def evaluate_svm():
    import os

    dataset = "adni"
    label = 'AB'
    file_real = f'/home/fi5666wi/R/data/DDLS/{dataset}_train_ml.csv'

    syn_dir = f'/home/fi5666wi/Python/WASP-DDLS/tabpfn/adni/tabpfn_t_1.0/'
    #syn_dir = '/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/adni/degree2_eps_100/'

    X_train, y_train, y_labels = data_loading(file_real, label=label)
    X_test, y_test, y_true = data_loading(f'/home/fi5666wi/R/data/DDLS/{dataset}_test.csv', label=label)
    print('Data loaded successfully')

    X_train = impute_missing_values(X_train)
    X_train = scale_data(X_train)
    X_test = impute_missing_values(X_test)
    X_test = scale_data(X_test)

    params = {'C': 10, 'gamma': 0.1, 'kernel': 'linear'}
    clf = fit_svm_classif(X_train, y_train, params)

    y_hat = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    feature_names = X_train.columns
    
    norm_real, cum_real, coef_real = weight_analysis(clf, feature_names)

    coef_df = coef_real


    plt.figure()
    plt.plot(cum_real, 'b-', label='Real data')

    syn_files = os.listdir(syn_dir)
    for i in range(5):
        X_syn, y_syn, y_labels_syn = data_loading(os.path.join(syn_dir, syn_files[i]), label=label)
        X_syn = impute_missing_values(X_syn)
        X_syn = scale_data(X_syn)

        clf_syn = fit_svm_classif(X_syn, y_syn, params)
        y_hat_syn = clf_syn.predict(X_test)
        y_prob_syn = clf_syn.predict_proba(X_test)
        norm_syn, cum_syn, coef_syn = weight_analysis(clf_syn, feature_names)
        
        coef_df = pd.concat([coef_df, coef_syn], axis=1)

        plt.plot(cum_syn, 'r--', label=f'Synthetic data {i+1}')


    plt.xlabel('Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Feature Importance Analysis')
    plt.legend()
    plt.show()

    print('Coefficeients from real data SVM:')
    print(coef_df)

    classif_plot(y_hat, y_test, label, y_prob[:,1])

def main_grid_search(dataset):
    file_real = f'/home/fi5666wi/R/data/DDLS/{dataset}_train_ml.csv'
    #file_long = '/Users/filipwinzell/R/data/DDLS/adni_train_long.csv'
    #file_real_2 = '/Users/filipwinzell/R/data/DDLS/adni_train_2.csv'

    #file_syn = f'/home/fi5666wi/Python/WASP-DDLS/DS-synthetic-data/degree3_eps_100/bn_{dataset}_0.csv'

    if dataset == 'a4':
        label = 'MMSCORE'
        X_train, y_train, y_labels = data_loading_a4(file_real, label=label)
        X_test, y_test, y_true = data_loading_a4(f'/home/fi5666wi/R/data/DDLS/{dataset}_test.csv', label=label)
        cats = ['PTGENDER', 'PTEDUCAT', 'PTETHNIC', 'PTMARRY', 'PTNOTRT', 'PTHOME']
    else:
        label = 'AB'
        #X_train, y_train, y_labels = frame2arrays(pd.read_csv(file_syn), label=label) 
        X_train, y_train, y_labels = data_loading(file_real, label=label)
        #X_train, y_train, y_labels = frame2arrays(pd.read_csv(file_real), label=label) 
        X_test, y_test, y_true = data_loading(f'/home/fi5666wi/R/data/DDLS/{dataset}_test.csv', label=label)
        cats = ['PTGENDER', 'PTEDUCAT', 'APOE4']
    print('Data loaded successfully')

    X_train = impute_missing_values(X_train, cats=cats)
    X_train = scale_data(X_train)
    X_test = impute_missing_values(X_test, cats=cats)
    X_test = scale_data(X_test)

    if label == 'DX' or label == 'AB' or label == 'AB_status':
        #params = {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
        #clf = fit_svm_classif(X_train, y_train, params)
        clf = svc_grid_search(X_train, y_train)
        y_prob = clf.predict_proba(scale_data(X_test))  
    else:
        #params = {'C': 100, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}
        #clf = fit_svm_regressor(X_train, y_train, params)
        clf = svr_grid_search(X_train, y_train)

    #clf = fit_svm_multi(X_train, y_train)
    y_hat = clf.predict(scale_data(X_test))

    #sq_err = (y_hat - y_test)**2
    #print('Mean squared error:', np.mean(sq_err))
    #print(f'Max { np.max(sq_err)} Min {np.min(sq_err)}')

    if label == 'DX':
        classif_plot(y_hat, y_test, label, y_prob)
    elif label == 'AB' or label == 'AB_status':
        classif_plot(y_hat, y_test, label, y_prob[:,1])
    else:
        regress_plot(y_hat, y_test)

if __name__ == "__main__":
    evaluate_svm()
    #main_grid_search('a4')

##### Results of grid search ######
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

# SVC -> AB (A4)
# {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
# Accuracy: 0.893
# F1: 0.896
# AUC : 0.945

# SVR -> MMSCORE (A4)
# {'C': 2.5, 'epsilon': 1, 'gamma': 'scale', 'kernel': 'rbf'}
# R2 score: 0.020
# RMSE: 1.180
