import numpy as np


def build_node_features(graph_data):

    nodes = [n["id"] for n in graph_data["nodes"]]
    node_map = {n: i for i, n in enumerate(nodes)}

    degree = np.zeros(len(nodes))
    total_value = np.zeros(len(nodes))

    for e in graph_data["edges"]:

        src = node_map[e["source"]]
        dst = node_map[e["target"]]

        value = float(e.get("weight",0))

        degree[src] += 1
        degree[dst] += 1

        total_value[src] += value
        total_value[dst] += value

    features = np.vstack([
        degree,
        total_value
    ]).T

    return features