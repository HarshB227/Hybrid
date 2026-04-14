import networkx as nx
from typing import Dict
import numpy as np


def compute_graph_features(G: nx.DiGraph, target_wallet: str) -> Dict:

    features = {}

    if target_wallet not in G:
        return {}

    # -----------------------------
    # BASIC DEGREE FEATURES
    # -----------------------------
    in_deg = G.in_degree(target_wallet)
    out_deg = G.out_degree(target_wallet)

    features["in_degree"] = in_deg
    features["out_degree"] = out_deg
    features["degree_ratio"] = out_deg / (in_deg + 1)

    # -----------------------------
    # VALUE FLOW FEATURES
    # -----------------------------
    in_edges = list(G.in_edges(target_wallet, data=True))
    out_edges = list(G.out_edges(target_wallet, data=True))

    in_values = [d["weight"] for _, _, d in in_edges]
    out_values = [d["weight"] for _, _, d in out_edges]

    features["in_value"] = sum(in_values)
    features["out_value"] = sum(out_values)

    features["avg_in_value"] = np.mean(in_values) if in_values else 0
    features["avg_out_value"] = np.mean(out_values) if out_values else 0

    features["value_ratio"] = features["out_value"] / (features["in_value"] + 1)

    # -----------------------------
    # TRANSACTION FREQUENCY FEATURES
    # -----------------------------
    in_counts = [d["count"] for _, _, d in in_edges]
    out_counts = [d["count"] for _, _, d in out_edges]

    features["tx_in_count"] = sum(in_counts)
    features["tx_out_count"] = sum(out_counts)

    features["avg_tx_in"] = np.mean(in_counts) if in_counts else 0
    features["avg_tx_out"] = np.mean(out_counts) if out_counts else 0

    # -----------------------------
    # CONCENTRATION RISK (KEY SIGNAL)
    # -----------------------------
    def concentration(values):
        total = sum(values)
        if total == 0:
            return 0
        return max(values) / total

    features["in_concentration"] = concentration(in_values)
    features["out_concentration"] = concentration(out_values)

    # -----------------------------
    # COUNTERPARTY DIVERSITY
    # -----------------------------
    unique_in = len(set(u for u, _, _ in in_edges))
    unique_out = len(set(v for _, v, _ in out_edges))

    features["unique_senders"] = unique_in
    features["unique_receivers"] = unique_out

    features["diversity_ratio"] = unique_out / (unique_in + 1)

    # -----------------------------
    # CENTRALITY (KEEP BUT NOT PRIMARY)
    # -----------------------------
    try:
        pr = nx.pagerank(G)
        features["pagerank"] = pr.get(target_wallet, 0)
    except:
        features["pagerank"] = 0

    # -----------------------------
    # CLUSTERING (LOW VALUE BUT OK)
    # -----------------------------
    try:
        features["clustering"] = nx.clustering(G.to_undirected(), target_wallet)
    except:
        features["clustering"] = 0

    # -----------------------------
    # SUSPICIOUS PATTERNS
    # -----------------------------

    # 1. Fan-out (scam dispersal pattern)
    features["is_fan_out"] = int(out_deg > 10 and features["out_concentration"] > 0.5)

    # 2. Fan-in (collector wallet)
    features["is_fan_in"] = int(in_deg > 10 and features["in_concentration"] > 0.5)

    # 3. Pass-through wallet (laundering behavior)
    features["is_pass_through"] = int(
        features["in_value"] > 0 and
        abs(features["in_value"] - features["out_value"]) / (features["in_value"] + 1) < 0.2
    )

    # 4. Low diversity high volume (bot / scam)
    features["low_diversity_high_value"] = int(
        features["unique_receivers"] < 3 and features["out_value"] > 100
    )

    return features