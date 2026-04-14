from src.blockchain.alchemy_fetch import fetch_full_wallet_data
from src.blockchain.normalize import normalize_transactions
from src.blockchain.build_graph import build_transaction_graph
from src.blockchain.graph_features import compute_graph_features


# -----------------------------------------
# GRAPH SERIALIZATION (WITH TIMESTAMP)
# -----------------------------------------
def serialize_graph(G):
    edges = []

    for u, v, d in G.edges(data=True):
        edges.append({
            "source": str(u),
            "target": str(v),
            "weight": float(d.get("weight", 0)),
            "count": int(d.get("count", 1)),
            "timestamp": int(d.get("timestamp", 0))  # critical
        })

    return {
        "nodes": [{"id": str(n)} for n in G.nodes()],
        "edges": edges,
    }


# -----------------------------------------
# SUSPICIOUS PATH DETECTION (IMPROVED)
# -----------------------------------------
def detect_suspicious_paths(G, threshold=10):
    suspicious = []

    for u, v, d in G.edges(data=True):
        value = float(d.get("weight", 0))
        count = int(d.get("count", 1))

        if value > threshold or count > 5:
            suspicious.append({
                "from": str(u),
                "to": str(v),
                "value": value,
                "tx_count": count
            })

    return suspicious


# -----------------------------------------
# 🔥 SAFE MULTI-HOP EXPANSION
# -----------------------------------------
def expand_transactions(wallet, base_txs, max_neighbors=5, max_count=1000, max_pages=5):
    expanded = list(base_txs)
    if max_neighbors <= 0:
        print("[DEBUG] Neighbor expansion skipped")
        return expanded

    neighbors = set()

    # ---------------------------------
    # SAFE ADDRESS EXTRACTION
    # ---------------------------------
    for tx in base_txs:
        from_addr = tx.get("from") or tx.get("from_address")
        to_addr = tx.get("to") or tx.get("to_address")

        if isinstance(from_addr, str) and len(from_addr) > 10:
            neighbors.add(from_addr)

        if isinstance(to_addr, str) and len(to_addr) > 10:
            neighbors.add(to_addr)

    neighbors.discard(wallet)

    neighbors = list(neighbors)[:max_neighbors]

    print(f"[DEBUG] Expanding to {len(neighbors)} neighbors")

    # ---------------------------------
    # SAFE FETCH LOOP
    # ---------------------------------
    for n in neighbors:

        if not n or not isinstance(n, str):
            continue

        print(f"[DEBUG] Fetching neighbor: {n}")

        try:
            raw_n = fetch_full_wallet_data(n, max_count=max_count, max_pages=max_pages)

            if not raw_n:
                continue

            if raw_n.get("n_transactions", 0) == 0:
                continue

            txs_n = normalize_transactions(raw_n)

            if txs_n:
                expanded.extend(txs_n)

        except Exception as e:
            print(f"[WARN] Neighbor fetch failed: {n} | {e}")

    return expanded


# -----------------------------------------
# MAIN PIPELINE (STABLE VERSION)
# -----------------------------------------
def run_wallet_pipeline(wallet: str, fast_mode: bool = False):

    print(f"\n[PIPELINE] Running for wallet: {wallet}")

    max_neighbors = 5
    max_pages = 5
    max_count = 1000

    if fast_mode:
        max_neighbors = 0
        max_pages = 2
        max_count = 300
        print(
            "[DEBUG] Fast mode enabled "
            f"(max_neighbors={max_neighbors}, max_pages={max_pages}, max_count={max_count})"
        )

    # -------------------------------------
    # 1. FETCH BASE DATA
    # -------------------------------------
    raw = fetch_full_wallet_data(wallet, max_count=max_count, max_pages=max_pages)

    if not raw:
        return _empty_response(wallet, "Fetch failed")

    tx_count = raw.get("n_transactions", 0)

    if tx_count == 0:
        return _empty_response(wallet, "No transactions fetched")

    # -------------------------------------
    # 2. NORMALIZE
    # -------------------------------------
    txs = normalize_transactions(raw)

    if not txs:
        return _empty_response(wallet, "Normalization failed")

    print(f"[DEBUG] Base transactions: {len(txs)}")

    # -------------------------------------
    # 3. MULTI-HOP EXPANSION (SAFE)
    # -------------------------------------
    try:
        txs = expand_transactions(
            wallet,
            txs,
            max_neighbors=max_neighbors,
            max_count=max_count,
            max_pages=max_pages,
        )
    except Exception as e:
        print(f"[WARN] Expansion failed, using base txs only: {e}")

    print(f"[DEBUG] Final transactions: {len(txs)}")

    # -------------------------------------
    # 4. BUILD GRAPH
    # -------------------------------------
    G = build_transaction_graph(wallet, txs)

    print(f"[DEBUG] Graph nodes: {G.number_of_nodes()}")
    print(f"[DEBUG] Graph edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        return _empty_response(wallet, "Graph empty")

    # -------------------------------------
    # 5. FEATURES
    # -------------------------------------
    features = compute_graph_features(G, wallet)

    # -------------------------------------
    # 6. SERIALIZE GRAPH
    # -------------------------------------
    graph_data = serialize_graph(G)

    # -------------------------------------
    # 7. SUSPICIOUS PATHS
    # -------------------------------------
    suspicious_paths = detect_suspicious_paths(G)

    print(f"[DEBUG] Suspicious paths: {len(suspicious_paths)}")

    # -------------------------------------
    # FINAL OUTPUT
    # -------------------------------------
    return {
        "wallet": wallet,
        "n_transactions": len(txs),
        "features": features,
        "graph": graph_data,
        "suspicious_paths": suspicious_paths,
        "status": "success"
    }


# -----------------------------------------
# EMPTY RESPONSE HANDLER
# -----------------------------------------
def _empty_response(wallet, error_msg):
    print(f"[PIPELINE ERROR] {error_msg}")

    return {
        "wallet": wallet,
        "error": error_msg,
        "n_transactions": 0,
        "features": {},
        "graph": {"nodes": [], "edges": []},
        "suspicious_paths": [],
        "status": "failed"
    }
