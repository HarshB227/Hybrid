import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    BEHAVIORAL_TX_HIGH, BEHAVIORAL_TX_MED,
    BEHAVIORAL_VALUE_HIGH, BEHAVIORAL_VALUE_MED,
)


# -----------------------------------------
# BEHAVIORAL RISK
# -----------------------------------------

def compute_behavioral_score(tx_count, avg_value):

    score = 0.0

    # activity signal
    if tx_count > BEHAVIORAL_TX_HIGH:
        score += 0.4
    elif tx_count > BEHAVIORAL_TX_MED:
        score += 0.3
    else:
        score += 0.1

    # transaction value signal
    if avg_value > BEHAVIORAL_VALUE_HIGH:
        score += 0.4
    elif avg_value > BEHAVIORAL_VALUE_MED:
        score += 0.3
    else:
        score += 0.1

    return min(score, 1.0)


# -----------------------------------------
# GRAPH RISK
# -----------------------------------------

def compute_graph_score(risk_scores, wallet):

    return min(
        risk_scores.get(wallet, 0.0),
        1.0
    )


# -----------------------------------------
# CLUSTER RISK
# -----------------------------------------

def compute_cluster_score(clusters, wallet):

    for c in clusters:

        if wallet in c.get("nodes", []):

            return min(
                c.get("avg_risk", 0.0),
                1.0
            )

    return 0.1


# -----------------------------------------
# GNN RISK
# -----------------------------------------

def compute_gnn_score(gnn_score, fallback_graph_score):

    if gnn_score is None:
        return fallback_graph_score

    return min(
        float(gnn_score),
        1.0
    )


# -----------------------------------------
# EXPLANATION TEXT
# -----------------------------------------

def generate_reasons(
    behavioral,
    graph,
    cluster,
    gnn
):

    reasons = []

    if behavioral > 0.6:
        reasons.append(
            "High transaction activity"
        )

    if graph > 0.6:
        reasons.append(
            "Connected to risky wallets"
        )

    if cluster > 0.6:
        reasons.append(
            "Part of suspicious cluster"
        )

    if gnn > 0.7:
        reasons.append(
            "GNN detected fraud pattern"
        )

    if not reasons:

        reasons.append(
            "No strong fraud indicators"
        )

    return reasons


# -----------------------------------------
# FINAL FUSION MODEL
# -----------------------------------------

def compute_final_score(
    tx_count,
    avg_value,
    risk_scores,
    clusters,
    wallet,
    gnn_score=None
):

    # 1️⃣ behavioral risk
    behavioral = compute_behavioral_score(
        tx_count,
        avg_value
    )

    # 2️⃣ graph propagation risk
    graph = compute_graph_score(
        risk_scores,
        wallet
    )

    # 3️⃣ cluster-level risk
    cluster = compute_cluster_score(
        clusters,
        wallet
    )

    # 4️⃣ GNN prediction
    gnn = compute_gnn_score(
        gnn_score,
        graph
    )

    # ---------------------------------
    # FINAL WEIGHTED FUSION
    # ---------------------------------

    final_score = (
        0.25 * behavioral +
        0.25 * graph +
        0.20 * cluster +
        0.30 * gnn
    )

    final_score = min(
        final_score,
        1.0
    )

    # ---------------------------------
    # CONFIDENCE
    # ---------------------------------

    confidence = (
        behavioral +
        graph +
        cluster +
        gnn
    ) / 4

    confidence = min(
        confidence,
        1.0
    )

    # ---------------------------------
    # EXPLANATION
    # ---------------------------------

    reasons = generate_reasons(
        behavioral,
        graph,
        cluster,
        gnn
    )

    # ---------------------------------

    return {

        "risk_score": round(
            final_score,
            4
        ),

        "confidence": round(
            confidence,
            4
        ),

        "breakdown": {

            "behavioral": round(
                behavioral,
                4
            ),

            "graph": round(
                graph,
                4
            ),

            "cluster": round(
                cluster,
                4
            ),

            "gnn": round(
                gnn,
                4
            )
        },

        "reasons": reasons
    }