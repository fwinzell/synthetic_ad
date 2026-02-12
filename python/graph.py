import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

def get_traces(G, adj_mode="color"):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    if adj_mode == "color":
        node_trace.marker.color = node_adjacencies
    elif adj_mode == "size":
        node_trace.marker.size = node_adjacencies
    else:
        assert False, "adj_mode must be either 'color' or 'size'."

    node_trace.text = node_text

    return edge_trace, node_trace


def display_graph(edge_trace, node_trace):
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Network graph made with Python',
                    titlefont_size=16,
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="FU",
                        showarrow=True,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                    )
    fig.show()


def display_network_graph(G, mode="color"):
    edge_trace, node_trace = get_traces(G, mode)
    display_graph(edge_trace, node_trace)

def read_json_file(filename, mac=False):
    # Opening JSON file
    linux_path = '/home/fi5666wi/Python/WASP-DDLS/DataSynthesizer/out/correlated_attribute_mode/'
    #mac_path = '/Users/filipwinzell/WASP-DDLS/datasynth/DataSynthesizer/notebooks/out/correlated_attribute_mode/'
    mac_path = '/Users/filipwinzell/WASP-DDLS/DS-bayesian-networks'
    if mac:
        f = open(os.path.join(mac_path, filename))
    else:
        f = open(os.path.join(linux_path, filename))
    # returns JSON object as a dictionary
    data = json.load(f)

    # Closing file
    f.close()

    return data


def get_graph_example():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                      (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                      (5, 6), (5, 7), (6, 7)])
    pos = nx.circular_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = list(pos[node])
    return G

def get_simple_graph():
    G = nx.DiGraph()
    vars = ["ICV", "WholeBrain", "MidTemp", "Fusiform", "Entorhinal", "Hippocampus", "AGE", "ADAS13", "Ventricles",
            "MMSE", "PTEDUCAT", "AV45", "DX", "APOE4", "PTGENDER"]
    G.add_nodes_from(vars)
    edges = {
        #"WholeBrain": Root
        "MidTemp": ["WholeBrain"],
        "Fusiform": ["WholeBrain", "MidTemp"],
        "Entorhinal": ["WholeBrain", "MidTemp", "Fusiform"],
        "Hippocampus": ["FusiForm", "Entorhinal", "MidTemp"],
        "AGE": ["Entorhinal", "Hippocampus", "WholeBrain"],
        "ADAS13": ["AGE", "Entorhinal", "WholeBrain"],
        "Ventricles": ["AGE", "ADAS13", "Entorhinal"],
        "MMSE": ["AGE", "ADAS13", "Entorhinal"],
        "PTEDUCAT": ["AGE", "ADAS13", "Entorhinal"],
        "AV45": ["AGE", "ABETA", "Entorhinal"],
        "ICV": ["AGE", "ADAS13", "WholeBrain"],
        "DX": ["AGE", "ADAS13", "Entorhinal"],
        "APOE4": ["AGE", "ADAS13", "Entorhinal"],
        "PTGENDER": ["AGE", "ADAS13", "WholeBrain"]
    }

    edges_deg4 = {
        # "WholeBrain": Root
        "MidTemp": ["WholeBrain"],
        "Fusiform": ['MidTemp', 'WholeBrain'],
        "Entorhinal": ['MidTemp', 'Fusiform', 'WholeBrain'],
        "Hippocampus": ['MidTemp', 'Fusiform', 'Entorhinal', 'WholeBrain'],
        "AGE": ['Fusiform', 'Entorhinal', 'Hippocampus', 'WholeBrain'],
        "ADAS13": ['Entorhinal', 'Hippocampus', 'AGE', 'WholeBrain'],
        "Ventricles": ['Entorhinal', 'AGE', 'ADAS13', 'WholeBrain'],
        "MMSE": ['Entorhinal', 'AGE', 'ADAS13', 'WholeBrain'],
        "PTEDUCAT": ['Entorhinal', 'AGE', 'ADAS13', 'WholeBrain'],
        "ABETA": ['AGE', 'ADAS13', 'Ventricles', 'Entorhinal'],
        "AV45": ['Hippocampus', 'AGE', 'ADAS13', 'Fusiform'],
        "PTAU": ['AGE', 'Ventricles', 'ABETA', 'Entorhinal'],
        "ICV": ['Entorhinal', 'AGE', 'ADAS13', 'WholeBrain'],
        "TAU": ['AGE', 'ADAS13', 'PTAU', 'Fusiform'],
        "DX": ['AGE', 'ADAS13', 'PTEDUCAT', 'MidTemp'],
        "APOE4": ['Entorhinal', 'AGE', 'ADAS13', 'WholeBrain'],
        "PTGENDER": ['AGE', 'ADAS13', 'Ventricles', 'Fusiform']
    }
    edge_list = [(v, k) for k, vs in edges.items() for v in vs]

    G.add_edges_from(edge_list)

    pos = nx.circular_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = list(pos[node])
    return G, edges


def get_graph(edge_dict, weight=1, color='black'):
    G = nx.DiGraph()
    vars = list(edge_dict.keys())
    G.add_nodes_from(vars)
    edge_list = [(v, k) for k, vs in edge_dict.items() for v in vs]
    G.add_edges_from(edge_list, weight=weight, color=color)

    pos = nx.circular_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = list(pos[node])
    return G, edge_dict

def plot_graph(G, node_color='lightblue'):
    shells = None
    colors = nx.get_edge_attributes(G, 'color').values()
    weights = nx.get_edge_attributes(G, 'weight').values()

    fig = nx.draw_shell(G, nlist=shells, with_labels=True,
                        edge_color=colors,
                        width=list(weights),
                        node_color=node_color,
                        node_size=500,
                        font_size=9)

    plt.show()

def custom_bn():
    # This needs to acyclic, how to ensure that?
    bn = [
        ['Ventricles', ['AGE', 'Entorhinal', 'ICV']], 
        ['ICV', ['PTGENDER']], 
        ['WholeBrain', ['AGE', 'ICV']], 
        ['Entorhinal', ['AGE', 'ICV', 'ADAS13']], 
        ['Hippocampus', ['ICV']], 
        ['MidTemp', []], 
        ['Fusiform', []], 
        ['ADAS13', []], 
        ['MMSE', []], 
        ['PTEDUCAT', []], 
        ['TAU', []], 
        ['PTAU', []], 
        ['ABETA', []], 
        ['AV45', []], 
        ['APOE4', []], 
        ['DX', []], 
        ['PTGENDER', []]
    ]

def compare_bn():
    desc1 = read_json_file('degree3_deter/bn_adni_AGE.json', mac=True)
    desc2 = read_json_file('degree3_deter/bn_adni_Ventricles.json', mac=True)

    e1 = dict(desc1['bayesian_network'])
    e2 = dict(desc2['bayesian_network'])

    # init empty dict with same keys as e2
    added = {k: [] for k in e2.keys()}
    # for each key in e2, remove all values that are in the same key in e1
    for k, v in e2.items():
        if not k in e1.keys():
            added[k] = v
        else:
            added[k] = [x for x in v if x not in e1[k]]

    # do the same for removed edges
    removed = {k: [] for k in e1.keys()}
    for k, v in e1.items():
        if not k in e2.keys():
            removed[k] = v
        else:
            removed[k] = [x for x in v if x not in e2[k]]

    G, _ = get_graph(e1, color='gray')

    G.add_edges_from([(v, k) for k, vs in added.items() for v in vs], color='green')
    G.add_edges_from([(v, k) for k, vs in removed.items() for v in vs], color='red')

    #keep = ['TAU', 'PTAU'] + e2['TAU'] + e2['PTAU']
    #G.remove_nodes_from([n for n in G.nodes() if n not in keep])

    plot_graph(G, node_color='lightblue')


if __name__ == '__main__':
    #compare_bn()

    description = read_json_file('degree3_deter/bn_adni_AGE.json', mac=False)
    edge_dict = dict(description['bayesian_network'])

    #G = nx.random_geometric_graph(200, 0.125)
    G, edges = get_graph(edge_dict, weight=1, color='gray')

    plot_graph(G, node_color='lightblue')

    #edge_trace, node_trace = get_traces(G, "color")
    #display_graph(edge_trace, node_trace)
