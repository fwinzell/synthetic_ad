from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx


def synthesize(file_dict,
               categorical_attributes,
               candidate_keys,
               num_tuples_to_generate,
               epsilon: int = 0,
               degree_of_bayesian_network: int = 2,
               threshold_value: int = 20,
               seed: int = 0
               ):
    """
    args:
    file_dict: The input dataset file names.
    categorical_attributes: A dictionary of which attributes are categorical. The key is the attribute name and the value
    candidate_keys: A dictionary of which attributes are candidate keys. The key is the attribute name and the value is

    num_tuples_to_generate: Number of tuples generated in synthetic dataset.

    epsilon: A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not
    change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
    Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.

    degree_of_bayesian_network: The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.

    threshold_value: An attribute is categorical if its domain size is less than this threshold.

    """

    input_data = file_dict['input_data']
    description_file = file_dict['description_file']
    synthetic_data = file_dict['synthetic_data']

    #### STEP 3: DATA DESCRIPTION ####
    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys,
                                                            seed=seed)
    describer.save_dataset_description_to_file(description_file)

    display_bayesian_network(describer.bayesian_network)

    #### STEP 4: DATA GENERATION ####
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

    return describer


def eval(file_dict):
    input_data = file_dict['input_data']
    description_file = file_dict['description_file']
    synthetic_data = file_dict['synthetic_data']

    #### STEP 5: DATA INSPECTION ####
    # Read both datasets using Pandas.
    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)
    # Read attribute description from the dataset description file.
    attribute_description = read_json_file(description_file)['attribute_description']

    inspector = ModelInspector(input_df, synthetic_df, attribute_description)

    for attribute in synthetic_df.columns:
        inspector.compare_histograms(attribute)

    inspector.mutual_information_heatmap()

    plt.show()


def create_graph(bn):
    nodes = [k[0] for k in bn]
    edge_list = [(v, k) for k, vs in bn for v in vs]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    pos = nx.circular_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = list(pos[node])
    return G



if __name__ == '__main__':
    files = {}
    # input dataset
    files['input_data'] = '/Users/filipwinzell/R/data/DDLS/adni_train_ds.csv'
    # location of two output files
    mode = 'correlated_attribute_mode'
    num_tuples = 2000
    files['description_file'] = f'/Users/filipwinzell/WASP-DDLS/datasynth/DataSynthesizer/notebooks/out/{mode}/description_adni_train.json'
    files['synthetic_data'] = f'/Users/filipwinzell/WASP-DDLS/datasynth/DataSynthesizer/notebooks/out/{mode}/synthetic_adni_{num_tuples}.csv'

    categorical_attributes = {'PTGENDER': True,
                              'PTEDUCAT': True,
                              'APOE4': True,
                              'DX': True}
                              #'DX.bl': True,
                              #'DX.m06': True,
                              #'DX.m12': True}


    # specify which attributes are candidate keys of input dataset.
    candidate_keys = {'RID': True}

    d = synthesize(file_dict=files, categorical_attributes=categorical_attributes,
                   candidate_keys=candidate_keys, num_tuples_to_generate=num_tuples, degree_of_bayesian_network=3, seed=7, epsilon=100)

    G = create_graph(d.bayesian_network)
    #display_network_graph(G, mode="color")
    nx.draw_circular(G, with_labels=True)
    plt.show()

    eval(files)
