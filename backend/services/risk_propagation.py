import math
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    PROPAGATION_TARGET_RISK, PROPAGATION_NODE_RISK,
    PROPAGATION_ALPHA, PROPAGATION_ITERATIONS,
)


# -----------------------------------------
# BUILD ADJACENCY LIST (NORMALIZED)
# -----------------------------------------
def build_adjacency(graph_data):
    adj = defaultdict(list)

    for edge in graph_data["edges"]:
        src = edge["source"]
        dst = edge["target"]
        weight = float(edge.get("weight", 0))

        # 🔥 normalized weight (0–1)
        w = min(math.log1p(weight) / 10, 1.0)

        adj[src].append((dst, w))

    return adj


# -----------------------------------------
# INITIALIZE BASE RISK
# -----------------------------------------
def initialize_base_risk(graph_data, target_wallet):

    base_risk = {}

    for node in graph_data["nodes"]:
        addr = node["id"]

        if addr == target_wallet:
            base_risk[addr] = PROPAGATION_TARGET_RISK
        else:
            base_risk[addr] = PROPAGATION_NODE_RISK

    return base_risk


# -----------------------------------------
# RISK PROPAGATION ENGINE (STABLE)
# -----------------------------------------
def propagate_risk(graph_data, target_wallet, alpha=PROPAGATION_ALPHA, iterations=PROPAGATION_ITERATIONS):

    adj = build_adjacency(graph_data)
    base_risk = initialize_base_risk(graph_data, target_wallet)

    risk = base_risk.copy()

    for _ in range(iterations):
        new_risk = {}

        for node in base_risk:
            neighbor_sum = 0

            for neighbor, weight in adj.get(node, []):
                neighbor_sum += weight * risk.get(neighbor, 0)

            # 🔥 combine + clamp
            updated = base_risk[node] + alpha * neighbor_sum

            # 🔥 HARD CLAMP (CRITICAL FIX)
            new_risk[node] = min(updated, 1.0)

        risk = new_risk

    return risk


# -----------------------------------------
# EXTRACT TOP RISKY NODES (NORMALIZED)
# -----------------------------------------
def get_top_risky_nodes(risk_scores, top_n=10):

    cleaned = []

    for k, v in risk_scores.items():
        cleaned.append({
            "wallet": k,
            "risk": round(min(v, 1.0), 4)  # 🔥 enforce 0–1
        })

    return sorted(cleaned, key=lambda x: x["risk"], reverse=True)[:top_n]