import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from graph import get_graph, plot_graph

def read_json_file(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    epsilon = 10
    bn_dir = f'/home/fi5666wi/Python/WASP-DDLS/DS-bayesian-networks/degree2_eps_{epsilon}/'
    #bn_dir = f'/Users/filipwinzell/WASP-DDLS/DS-bayesian-networks/degree3_root_AGE/'
    input_data_file = '/home/fi5666wi/R/data/DDLS/adni_train_ds.csv'
    adni_data = pd.read_csv(input_data_file)
    vars = adni_data.keys().tolist()
    vars.remove('RID')

    bn_df = pd.DataFrame(0, index = vars, columns = vars)
    roots = []

    json_files = [f for f in os.listdir(bn_dir) if f.endswith('.json')]

    for i,f in enumerate(json_files):
        print(f"Processing file {f}")
        desc = read_json_file(os.path.join(bn_dir, f))
        bn = dict(desc['bayesian_network'])

        #G, edges = get_graph(bn, weight=1, color='gray')
        #plot_graph(G, node_color='lightblue')

        vars = list(bn.keys())
        edge_list = [(v, k) for k, vs in bn.items() for v in vs]
        roots.append(edge_list[0][0])
        #max_length = max(len(vs) for k, vs in bn.items())
        #print(f"Degree of Bayesian Network: {max_length}")

        for edge in edge_list:
            bn_df.loc[edge[0], edge[1]] += 1

    #print(bn_df)
    print(Counter(roots))

    # divide bn_df by the number of files
    bn_df = (bn_df / i)*100

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(bn_df, annot=True, fmt=".0f", cmap="YlGnBu")
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    plt.title(f"Bayesian Network Edge Frequency Heatmap (epsilon = {epsilon})")
    plt.tight_layout()
    
    #G, edges = get_graph(bn, weight=1, color='gray')
    #plot_graph(G, node_color='lightblue')

    #plt.savefig(f'/home/fi5666wi/Python/WASP-DDLS/bn_plots/bn_plot_{epsilon}.png', dpi=1000)
    
    plt.show()
