import networkx as nx
from collections import defaultdict

try:
    import community as community_louvain
except:
    community_louvain = None


# -----------------------------------------
# BUILD NETWORKX GRAPH
# -----------------------------------------
def build_nx_graph(graph_data):
    G = nx.Graph()

    for edge in graph_data["edges"]:
        u = edge["source"]
        v = edge["target"]
        w = float(edge.get("weight", 0))

        G.add_edge(u, v, weight=w)

    return G


# -----------------------------------------
# DETECT COMMUNITIES (LOUVAIN)
# -----------------------------------------
def detect_clusters(graph_data):

    if community_louvain is None:
        return {}

    G = build_nx_graph(graph_data)

    partition = community_louvain.best_partition(G)

    return partition


# -----------------------------------------
# ANALYZE CLUSTERS
# -----------------------------------------
def analyze_clusters(partition, risk_scores):

    clusters = defaultdict(list)

    for node, cluster_id in partition.items():
        clusters[cluster_id].append(node)

    cluster_info = []

    for cid, nodes in clusters.items():

        risks = [risk_scores.get(n, 0) for n in nodes]

        avg_risk = sum(risks) / len(risks) if risks else 0

        cluster_info.append({
            "cluster_id": cid,
            "size": len(nodes),
            "avg_risk": round(avg_risk, 4),
            "nodes": nodes[:10]  # limit
        })

    # sort by risk
    cluster_info = sorted(cluster_info, key=lambda x: x["avg_risk"], reverse=True)

    return cluster_info