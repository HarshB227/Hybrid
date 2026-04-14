import sys
import os
import tempfile  # FIX 1: for unique temp graph files
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
from streamlit.components.v1 import html

API_URL = "http://127.0.0.1:8000/analyze_wallet"
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "120"))
ETHEREUM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
BTC_BASE58_RE = re.compile(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$")
BTC_BECH32_RE = re.compile(r"^(bc1|BC1)[0-9ac-hj-np-zAC-HJ-NP-Z]{11,71}$")


def detect_chain(address: str):
    if ETHEREUM_ADDRESS_RE.match(address):
        return "eth"
    if BTC_BASE58_RE.match(address) or BTC_BECH32_RE.match(address):
        return "btc"
    return None

st.set_page_config(page_title="Crypto Intelligence Dashboard", layout="wide")
st.title("Crypto Wallet Investigation Dashboard")


# -----------------------------
# INPUT
# -----------------------------
st.sidebar.header("Wallet Analysis")
wallet_address = st.sidebar.text_input("Enter Wallet Address")
analyze = st.sidebar.button("Analyze Wallet")
fast_mode = st.sidebar.checkbox("Fast mode (recommended)", value=True)


# -----------------------------
# FIX 1: SAFE GRAPH RENDER
# Uses a unique temp file per request so concurrent users
# never overwrite each other's graph.html
# -----------------------------
def render_graph(graph):
    net = Network(height="650px", width="100%", directed=True)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    for node in nodes:
        net.add_node(node["id"], label=node["id"][:6])

    for edge in edges:
        net.add_edge(
            edge["source"],
            edge["target"],
            value=edge.get("weight", 1),
        )

    # Write to a unique temp file instead of a shared "graph.html"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        net.save_graph(tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as f:
        html(f.read(), height=650)

    os.unlink(tmp_path)  # clean up after reading


# -----------------------------
# FIX 2: CACHED API CALL
# Same wallet address won't re-hit the API for 5 minutes.
# -----------------------------
@st.cache_data(ttl=300)
def fetch_wallet_data(wallet: str, fast: bool) -> dict:
    response = requests.post(
        API_URL,
        json={"wallet_address": wallet, "fast_mode": fast},
        timeout=API_TIMEOUT_SECONDS,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {detail}",
            response=response,
        )
    return response.json()


# -----------------------------
# MAIN ANALYSIS
# -----------------------------
if analyze and wallet_address:

    # FIX 3: Spinner so user knows the 8–15 s request is running
    with st.spinner("Analysing wallet — this may take up to 15 seconds..."):
        try:
            wallet_value = wallet_address.strip()
            chain = detect_chain(wallet_value)
            if chain is None:
                st.error(
                    "Invalid address. Expected Ethereum (0x + 40 hex) or "
                    "Bitcoin (1/3 legacy or bc1 bech32)."
                )
                st.stop()
            result = fetch_wallet_data(wallet_value, fast_mode)
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            st.stop()
        except ValueError:
            st.error("API returned a non-JSON response.")
            st.stop()

    if not isinstance(result, dict):
        st.error("API returned an empty response.")
        st.stop()

    if "error" in result:
        st.error(result["error"])
        st.stop()

    prob = float(result.get("scam_probability", 0)) * 100
    risk = result.get("risk_level", "UNKNOWN")
    if risk == "UNKNOWN":
        if prob > 70:
            risk = "HIGH"
        elif prob > 40:
            risk = "MEDIUM"
        else:
            risk = "LOW"
    display_risk = "MIDDLE" if risk == "MEDIUM" else risk

    # -----------------------------
    # FIX 4: IMPROVED GAUGE
    # Adds a black needle + threshold line so the reading is obvious.
    # The filled bar is slimmed down so zone colors are visible.
    # -----------------------------
    st.header("Wallet Risk Score")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"Scam risk score ({display_risk})"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {
                "color": "rgba(30,30,30,0.85)",
                "thickness": 0.04,           # slim bar = visible as a needle
            },
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "#2ecc71"},
                {"range": [40, 70],  "color": "#f1c40f"},
                {"range": [70, 100], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": prob,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"{prob:.1f}% is {display_risk}")

    # -----------------------------
    # ALERT BANNER
    # -----------------------------
    if display_risk == "HIGH":
        st.error(f"HIGH RISK — {prob:.1f}%")
    elif display_risk == "MIDDLE":
        st.warning(f"MIDDLE RISK — {prob:.1f}%")
    else:
        st.success(f"LOW RISK — {prob:.1f}%")

    # -----------------------------
    # FIX 5: METRICS — now includes ALL fields the API returns
    # Previously gnn_fraud_score, propagated_risk, avg_tx_value
    # and runtime_sec were returned but never displayed.
    # -----------------------------
    st.header("Wallet Intelligence")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions",    result.get("transaction_count", 0))
    col2.metric("Neighbours",      result.get("neighbors", 0))
    col3.metric("Avg Tx (ETH)",    f"{result.get('avg_tx_value', 0):.4f}")
    col4.metric("Runtime (s)",     result.get("runtime_sec", "—"))

    col5, col6, col7 = st.columns(3)
    col5.metric("GNN Fraud Score",  f"{result.get('gnn_fraud_score', 0):.4f}")
    col6.metric("Propagated Risk",  f"{result.get('propagated_risk', 0):.4f}")
    col7.metric("Cluster Risk",     f"{result.get('cluster_risk', 0):.4f}")

    # -----------------------------
    # FIX 6: BREAKDOWN BAR CHART replaces redundant pie chart
    # The pie showed prob vs (100-prob) — identical info to the gauge.
    # The breakdown chart shows HOW the score was computed.
    # -----------------------------
    explanation = result.get("explanation", {})
    breakdown = explanation.get("breakdown", {})

    if breakdown:
        st.header("Risk Score Breakdown")

        df_breakdown = pd.DataFrame(
            list(breakdown.items()),
            columns=["Component", "Score"],
        )
        df_breakdown["Score"] = df_breakdown["Score"].apply(
            lambda x: round(float(x), 4)
        )
        df_breakdown = df_breakdown.sort_values("Score", ascending=True)

        fig_breakdown = px.bar(
            df_breakdown,
            x="Score",
            y="Component",
            orientation="h",
            color="Score",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
            range_color=[0, 1],
            labels={"Score": "Risk score (0–1)", "Component": ""},
        )
        fig_breakdown.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

    # -----------------------------
    # INTELLIGENT FLOW PATHS
    # -----------------------------
    st.header("Intelligent Money Flow Paths")
    flows = result.get("money_flows", [])

    if not flows:
        st.info("No flow intelligence available.")
    else:
        st.success(f"{len(flows)} high-value path(s) detected")
        for f in flows[:10]:
            path_str = " → ".join(f["path"])
            text = (
                f"{path_str} | "
                f"Flow: {f['flow']} | "
                f"Score: {f['score']} | "
                f"Label: {f['label']}"
            )
            if f["label"] == "highly_suspicious":
                st.error(text)
            elif f["label"] == "suspicious":
                st.warning(text)
            else:
                st.success(text)

    # -----------------------------
    # TOP RISKY WALLETS
    # -----------------------------
    st.header("Top Risky Wallets")
    risky = result.get("top_risky_wallets", [])
    if risky:
        df_risky = pd.DataFrame(risky[:10])
        st.dataframe(df_risky, use_container_width=True, hide_index=True)
    else:
        st.info("No risky wallets identified in neighbourhood.")

    # -----------------------------
    # RISK EXPLANATION
    # -----------------------------
    st.header("Risk Explanation")
    if explanation:
        st.write("**Risk score:**", explanation.get("risk_score"))
        st.write("**Confidence:**", explanation.get("confidence"))

        reasons = explanation.get("reasons", [])
        if reasons:
            st.subheader("Reasons")
            for r in reasons:
                st.write(f"- {r}")

    # -----------------------------
    # TRANSACTION GRAPH
    # -----------------------------
    st.header("Intelligence Graph")
    graph = result.get("graph", {})
    nodes_list = graph.get("nodes", [])
    edges_list = graph.get("edges", [])

    if not nodes_list:
        st.warning("No graph data available.")
    else:
        st.success(
            f"Graph loaded: {len(nodes_list)} nodes, {len(edges_list)} edges"
        )
        render_graph(graph)

    # -------------------------
    # TRANSACTION SUMMARY
    # -------------------------

    st.subheader("Transaction Summary")

    tx_summary = result.get("transaction_summary", {})

    if tx_summary:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            tx_summary.get("total_transactions", 0)
        )

        total_amt = tx_summary.get("total_amount_transferred", {})

        col2.metric(
            "Total ETH/BTC",
            round(total_amt.get("ETH", total_amt.get("BTC", 0)), 4)
        )

        col3.metric(
            "USD Value",
            round(total_amt.get("USD", 0), 2)
        )

        st.write("### Currency Conversion")

        currency_cols = st.columns(5)

        currency_cols[0].metric("USD", round(total_amt.get("USD", 0), 2))
        currency_cols[1].metric("GBP", round(total_amt.get("GBP", 0), 2))
        currency_cols[2].metric("EUR", round(total_amt.get("EUR", 0), 2))
        currency_cols[3].metric("CAD", round(total_amt.get("CAD", 0), 2))
        currency_cols[4].metric("AUD", round(total_amt.get("AUD", 0), 2))

    # -------------------------
    # SCAM TYPE
    # -------------------------

    st.subheader("Detected Pattern")

    st.info(
        result.get(
            "suspected_attack_type",
            "Unknown"
        )
    )

