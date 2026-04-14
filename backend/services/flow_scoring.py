import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    FLOW_TIME_GAP_FAST, FLOW_TIME_GAP_MED, FLOW_TIME_GAP_SLOW,
    FLOW_HIGH_THRESHOLD, FLOW_MED_THRESHOLD,
)


# -----------------------------------------
# NORMALIZATION HELPERS
# -----------------------------------------
def normalize_flow(flow):
    return min(math.log1p(flow) / 10, 1.0)


def normalize_risk(risk):
    return min(max(risk, 0), 1.0)


# -----------------------------------------
# DIRECTION SCORE
# -----------------------------------------
def compute_direction_score(path, target_wallet):
    nodes = path.nodes

    if not nodes:
        return 0.0

    if nodes[-1] == target_wallet:
        return 1.0  # incoming risk
    elif nodes[0] == target_wallet:
        return 0.6  # outgoing
    else:
        return 0.3  # indirect


# -----------------------------------------
# TIME SCORE
# -----------------------------------------
def compute_time_score(path):
    timestamps = []

    for e in path.edges:
        ts = getattr(e, "timestamp", None)
        if ts:
            timestamps.append(ts)

    if len(timestamps) < 2:
        return 0.3

    timestamps.sort()

    gaps = [
        timestamps[i+1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    if not gaps:
        return 0.3

    avg_gap = sum(gaps) / len(gaps)

    if avg_gap < FLOW_TIME_GAP_FAST:
        return 1.0
    elif avg_gap < FLOW_TIME_GAP_MED:
        return 0.8
    elif avg_gap < FLOW_TIME_GAP_SLOW:
        return 0.6
    else:
        return 0.3


# -----------------------------------------
# SCORE SINGLE FLOW PATH
# -----------------------------------------
def score_flow_path(path, risk_scores, target_wallet):

    nodes = path.nodes
    flow = float(path.flow_value)
    steps = len(path.edges)

    flow_score = normalize_flow(flow)
    length_score = min(steps / 5, 1.0)

    risks = [normalize_risk(risk_scores.get(n, 0)) for n in nodes]
    avg_risk = sum(risks) / len(risks) if risks else 0

    direction_score = compute_direction_score(path, target_wallet)
    time_score = compute_time_score(path)

    final_score = (
        0.25 * flow_score +
        0.20 * length_score +
        0.20 * avg_risk +
        0.20 * direction_score +
        0.15 * time_score
    )

    final_score = min(final_score, 1.0)

    if final_score > FLOW_HIGH_THRESHOLD:
        label = "laundering_pattern"
    elif final_score > FLOW_MED_THRESHOLD:
        label = "suspicious"
    else:
        label = "low_risk"

    return {
        "path": nodes,
        "flow": round(flow, 4),
        "steps": steps,
        "avg_risk": round(avg_risk, 4),
        "score": round(final_score, 4),
        "direction_score": round(direction_score, 3),
        "time_score": round(time_score, 3),
        "label": label
    }


# -----------------------------------------
# SCORE ALL PATHS (DEDUP + CLEAN)
# -----------------------------------------
def score_all_flows(paths, risk_scores, target_wallet, top_n=20):

    if not paths:
        return []

    seen = set()
    scored = []

    for p in paths:
        try:
            key = tuple(p.nodes)

            # 🔥 REMOVE DUPLICATES
            if key in seen:
                continue

            seen.add(key)

            scored.append(score_flow_path(p, risk_scores, target_wallet))

        except Exception:
            continue

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    return scored[:top_n]