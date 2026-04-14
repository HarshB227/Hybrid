import networkx as nx
from typing import List, Dict


def build_transaction_graph(wallet: str, transactions: List[Dict]) -> nx.DiGraph:
    """
    Build a directed transaction graph with weighted edges.

    Nodes = wallets
    Edges = transactions
    Edge attributes:
        - weight: total transferred value
        - count: number of transactions
        - timestamps: list of timestamps (for future temporal analysis)
    """

    G = nx.DiGraph()

    # Ensure target wallet exists
    G.add_node(wallet)

    for tx in transactions:
        sender = tx.get("from")
        receiver = tx.get("to")

        if not sender or not receiver:
            continue

        # Safe parsing
        value = tx.get("value") or 0.0
        try:
            value = float(value)
        except:
            value = 0.0

        timestamp = tx.get("timestamp")

        # Add / update edge
        if G.has_edge(sender, receiver):
            G[sender][receiver]["weight"] += value
            G[sender][receiver]["count"] += 1

            if timestamp:
                G[sender][receiver]["timestamps"].append(timestamp)
        else:
            G.add_edge(
                sender,
                receiver,
                weight=value,
                count=1,
                timestamps=[timestamp] if timestamp else [],
            )

    return G