from collections import defaultdict

class FlowPath:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.flow_value = float("inf")
        self.start_time = None
        self.end_time = None

    def add_step(self, tx):
        self.nodes.append(tx["to"])
        self.edges.append(tx)

        # critical: flow conservation
        self.flow_value = min(self.flow_value, tx["value"])

        if self.start_time is None:
            self.start_time = tx["timestamp"]

        self.end_time = tx["timestamp"]


# -------------------------------
# GRAPH BUILDER
# -------------------------------
def build_graph(transactions):
    graph = defaultdict(list)

    # enforce time order
    transactions = sorted(transactions, key=lambda x: x["timestamp"])

    for tx in transactions:
        graph[tx["from"]].append(tx)

    return graph


# -------------------------------
# MULTI-HOP TRACE ENGINE
# -------------------------------
def trace_funds(graph, start_address, max_depth=3, min_value=0.1):
    results = []

    def dfs(current, path, visited, depth):
        if depth > max_depth:
            return

        for tx in graph.get(current, []):

            # 1. filter noise
            if tx["value"] < min_value:
                continue

            next_addr = tx["to"]

            # 2. prevent cycles
            if next_addr in visited:
                continue

            # 3. enforce time consistency
            if path.end_time and tx["timestamp"] < path.end_time:
                continue

            # clone path
            new_path = FlowPath()
            new_path.nodes = path.nodes.copy()
            new_path.edges = path.edges.copy()
            new_path.flow_value = path.flow_value
            new_path.start_time = path.start_time
            new_path.end_time = path.end_time

            new_path.add_step(tx)

            results.append(new_path)

            dfs(
                next_addr,
                new_path,
                visited | {next_addr},
                depth + 1
            )

    # initialize
    start_path = FlowPath()
    start_path.nodes = [start_address]

    dfs(start_address, start_path, {start_address}, 1)

    return results