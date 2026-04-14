import torch


def convert_to_pyg(graph_data, feature_dim=3):

    node_map = {}
    nodes = graph_data["nodes"]

    for i, n in enumerate(nodes):
        node_map[n["id"]] = i

    # edges
    edge_index = []

    in_deg = torch.zeros(len(nodes), dtype=torch.float32)
    out_deg = torch.zeros(len(nodes), dtype=torch.float32)

    for e in graph_data["edges"]:
        src = node_map[e["source"]]
        dst = node_map[e["target"]]

        edge_index.append([src, dst])
        out_deg[src] += 1.0
        in_deg[dst] += 1.0

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Simple node features (degree stats)
    deg = in_deg + out_deg
    base = torch.stack([deg, in_deg, out_deg], dim=1)

    if feature_dim <= 3:
        x = base[:, :max(1, feature_dim)]
    else:
        x = torch.zeros((len(nodes), feature_dim), dtype=torch.float32)
        x[:, :3] = base

    return x, edge_index, node_map
