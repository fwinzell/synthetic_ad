from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import networkx as nx
import os

from main_ds import parse_config


def generate_bn(input_data_file,
                description_file,
               categorical_attributes,
               candidate_keys,
               epsilon: int = 0,
               eta: int = 0,
               degree_of_bayesian_network: int = 2,
               threshold_value: int = 20,
               seed: int = 0,
               save_bn: bool = True,
               fixed_root: str = None
               ):
    """
    args:
    file_dict: The input dataset file names.
    categorical_attributes: A dictionary of which attributes are categorical. The key is the attribute name and the value
    candidate_keys: A dictionary of which attributes are candidate keys. The key is the attribute name and the value is

    epsilon: A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not
    change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
    Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.

    eta: epsilon for constructing the Bayesian network. Set them to the same value to keep the original differential privacy set up

    degree_of_bayesian_network: The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.

    threshold_value: An attribute is categorical if its domain size is less than this threshold.

    """

    #### STEP 3: DATA DESCRIPTION ####
    describer = DataDescriber(category_threshold=threshold_value, fixed_root=fixed_root)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data_file,
                                                            epsilon=epsilon,
                                                            eta=eta,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys,
                                                            seed=seed)
    if save_bn:
        describer.save_dataset_description_to_file(description_file)

    return describer


def privacy_mode(config, epsilon, degree, dataset="adni"):
    input_data_file = f'{config.data_base_dir}/{dataset}_train.csv'
    degree_of_bayesian_network = degree
    bn_dir = f'{config.base_dir}/DS-bayesian-networks/{dataset}/degree{degree_of_bayesian_network}_eps_{epsilon}/'
    if not os.path.exists(bn_dir):
        os.makedirs(bn_dir)

    if dataset == "adni" or dataset == "adni_plus":
        categorical_attributes = {'PTGENDER': True,
                                'PTEDUCAT': True,
                                'APOE4': True,
                                'DX': True}
        
        # specify which attributes are candidate keys of input dataset.
        candidate_keys = {'RID': True}

    elif dataset == "a4":
        categorical_attributes = {
            'AB_status': True,
            'PTGENDER': True,
            'PTETHNIC': True,
            'PTEDUCAT': True,
            'PTMARRY': True,
            'PTNOTRT': True,
            'PTHOME': True
        }

        candidate_keys = {'BID': True}


    for seed in range(100):
        description_file = f'{bn_dir}/bn_{dataset}_{seed}.json'
        _ = generate_bn(input_data_file, description_file, categorical_attributes, candidate_keys, 
                        degree_of_bayesian_network=degree_of_bayesian_network,
                        epsilon=epsilon, eta=epsilon, seed=seed, save_bn=True)
        print(f'Generated BN for seed {seed}')

def deterministic_bn(degree_of_bayesian_network=3, dataset="adni"):
    input_data_file = f'/home/fi5666wi/R/data/DDLS/{dataset}_train.csv'
    #degree_of_bayesian_network = degree
    epsilon = 0
    bn_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-bayesian-networks/{dataset}/degree{degree_of_bayesian_network}_eps_zero/'
    if not os.path.exists(bn_dir):
        os.makedirs(bn_dir)

    if dataset == "adni" or dataset == "adni_plus":
        categorical_attributes = {'PTGENDER': True,
                                'PTEDUCAT': True,
                                'APOE4': True,
                                'DX': True}
        # specify which attributes are candidate keys of input dataset.
        candidate_keys = {'RID': True}
    elif dataset == "a4":
        categorical_attributes = {
            'AB_status': True,
            'PTGENDER': True,
            'PTETHNIC': True,
            'PTEDUCAT': True,
            'PTMARRY': True,
            'PTNOTRT': True,
            'PTHOME': True
        }
        candidate_keys = {'BID': True}

    # Read the column names of the CSV file
    df = pd.read_csv(input_data_file)
    column_names = df.columns.tolist()
    print("Column names:", column_names)

    # Remove candidate keys from the list of column names
    for key in candidate_keys.keys():
        column_names.remove(key)

    for attribute in column_names:
        description_file = f'{bn_dir}/bn_{dataset}_{attribute}.json'
        describer = DataDescriber(category_threshold=20, fixed_root=attribute)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data_file,
                                                            epsilon=epsilon,
                                                            eta=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys,
                                                            seed=0)
        
        describer.save_dataset_description_to_file(description_file)
        print(f'Generated BN for attribute {attribute}')

def semi_privacy_mode(epsilon, degree, root):
    input_data_file = '/Users/filipwinzell/R/data/DDLS/adni_train_ds.csv'
    degree_of_bayesian_network = degree
    bn_dir = f'/Users/filipwinzell/WASP-DDLS/DS-bayesian-networks/degree{degree_of_bayesian_network}_root_{root}/'
    if not os.path.exists(bn_dir):
        os.makedirs(bn_dir)

    categorical_attributes = {'PTGENDER': True,
                              'PTEDUCAT': True,
                              'APOE4': True,
                              'DX': True}

    # specify which attributes are candidate keys of input dataset.
    candidate_keys = {'RID': True}

    description_file = f'{bn_dir}/bn_eps_{epsilon}.json'
    _ = generate_bn(input_data_file, description_file, categorical_attributes, candidate_keys, epsilon=epsilon, eta=0, save_bn=True, fixed_root=root)


if __name__ == '__main__':
    config = parse_config()

    for deg in [2]:
        deterministic_bn(degree_of_bayesian_network=deg, dataset="adni_plus")
        print(f'########## Generated BNs for degree {deg} ###########')

    for eps in [100]:
        privacy_mode(config, epsilon=eps, degree=2, dataset="adni_plus")
        print(f'########## Generated BNs for epsilon {eps} ###########')
    #deterministic_bn()

    #root = 'AGE'
    #for eps in [0.1, 0.5, 1, 2, 10, 15, 25, 100, 200, 300, 500]:
    #    semi_privacy_mode(epsilon=eps, degree=3, root=root)
    #    print(f'Generated BN for epsilon {eps}')


