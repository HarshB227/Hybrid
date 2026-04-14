import os
import logging
import math
import statistics
import requests
import pandas as pd
from collections import Counter
from datetime import datetime

from src.config import CLEANED_DATA_FILE, FEATURE_DATA_FILE

logger = logging.getLogger(__name__)


# --------------------------------------------------
# PRICE FETCH (for ETH/BTC conversion to fiat)
# --------------------------------------------------

def get_crypto_price(symbol="bitcoin"):

    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd,gbp,eur,cad,aud"

    try:
        res = requests.get(url, timeout=10)
        data = res.json()

        return data.get(symbol, {
            "usd": 0,
            "gbp": 0,
            "eur": 0,
            "cad": 0,
            "aud": 0
        })

    except Exception as e:

        logger.warning(f"price fetch failed: {e}")

        return {
            "usd": 0,
            "gbp": 0,
            "eur": 0,
            "cad": 0,
            "aud": 0
        }


# --------------------------------------------------
# ENTROPY
# --------------------------------------------------

def calculate_entropy(values):

    if not values:
        return 0

    counter = Counter(values)

    total = sum(counter.values())

    entropy = 0

    for count in counter.values():

        p = count / total

        entropy -= p * math.log(p + 1e-9)

    return entropy


# --------------------------------------------------
# TRANSACTION SUMMARY
# --------------------------------------------------

def calculate_transaction_summary(df):

    if "income" not in df.columns:

        df["income"] = 0

    total_transactions = len(df)

    total_amount_btc = df["income"].sum()

    price = get_crypto_price("bitcoin")

    fiat_values = {

        "USD": total_amount_btc * price["usd"],
        "GBP": total_amount_btc * price["gbp"],
        "EUR": total_amount_btc * price["eur"],
        "CAD": total_amount_btc * price["cad"],
        "AUD": total_amount_btc * price["aud"],
    }

    return {

        "total_transactions": total_transactions,

        "total_amount_btc": total_amount_btc,

        "total_amount_fiat": fiat_values
    }


# --------------------------------------------------
# MAIN FEATURE BUILDER
# --------------------------------------------------

def build_features(data) -> pd.DataFrame:

    """
    Build model features from BitcoinHeist style dataset
    """

    if isinstance(data, dict):

        df = pd.DataFrame([data])

    elif isinstance(data, pd.DataFrame):

        df = data.copy()

    else:

        raise TypeError("build_features expects dict or DataFrame")


    # ---------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------

    required = [

        "year",
        "day",
        "length",
        "weight",
        "count",
        "looped",
        "neighbors",
        "income"

    ]

    for col in required:

        if col not in df.columns:

            logger.warning(f"Missing column {col}")

            df[col] = 0


    for col in required:

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


    # ---------------------------------
    # ORIGINAL FEATURES
    # ---------------------------------

    df["activity_intensity"] = df["count"] * df["neighbors"]

    df["tx_density"] = df["count"] / (df["neighbors"] + 1)

    df["loop_ratio"] = df["looped"] / (df["count"] + 1)

    df["income_per_tx"] = df["income"] / (df["count"] + 1)

    df["temporal_index"] = df["year"] * 365 + df["day"]

    df["activity_score"] = df["length"] * df["weight"]

    df["network_score"] = df["neighbors"] * df["weight"]


    # ---------------------------------
    # NEW FEATURES (transaction behavior)
    # ---------------------------------

    df["avg_transaction_value"] = df["income"] / (df["count"] + 1)

    df["transaction_velocity"] = df["count"] / (df["day"] + 1)

    df["interaction_ratio"] = df["neighbors"] / (df["count"] + 1)

    df["risk_pattern_score"] = (

        df["loop_ratio"]
        + df["tx_density"]
        + df["interaction_ratio"]

    )


    # ---------------------------------
    # ENTROPY BASED FEATURE
    # ---------------------------------

    df["entropy_like_score"] = df["neighbors"].apply(

        lambda x: math.log(x + 1)

    )


    # ---------------------------------
    # SCAM LABEL
    # ---------------------------------

    if "label" in df.columns:

        labels = df["label"].astype(str).str.lower()

        df["scam_label"] = (labels != "white").astype(int)


    # ---------------------------------
    # TRANSACTION SUMMARY
    # ---------------------------------

    tx_summary = calculate_transaction_summary(df)

    df["total_transactions"] = tx_summary["total_transactions"]

    df["total_amount_btc"] = tx_summary["total_amount_btc"]

    df["total_amount_usd"] = tx_summary["total_amount_fiat"]["USD"]

    df["total_amount_gbp"] = tx_summary["total_amount_fiat"]["GBP"]

    df["total_amount_eur"] = tx_summary["total_amount_fiat"]["EUR"]

    df["total_amount_cad"] = tx_summary["total_amount_fiat"]["CAD"]

    df["total_amount_aud"] = tx_summary["total_amount_fiat"]["AUD"]


    # ---------------------------------
    # SCAM TYPE HEURISTIC
    # ---------------------------------

    def classify_pattern(row):

        if row["loop_ratio"] > 0.6:

            return "Mixing Behaviour"

        if row["tx_density"] > 5:

            return "High Frequency Transfers"

        if row["income_per_tx"] > 10:

            return "High Value Movement"

        return "Normal"


    df["suspected_scam_type"] = df.apply(classify_pattern, axis=1)


    return df


# --------------------------------------------------
# PIPELINE RUNNER
# --------------------------------------------------

def run_feature_engineering():

    logger.info("loading cleaned dataset")

    df = pd.read_parquet(CLEANED_DATA_FILE)

    logger.info("building features")

    df = build_features(df)

    os.makedirs(os.path.dirname(FEATURE_DATA_FILE), exist_ok=True)

    df.to_parquet(FEATURE_DATA_FILE, index=False)

    logger.info("feature engineering complete")

    return df


# --------------------------------------------------
# API FEATURE VECTOR (wallet + transactions)
# --------------------------------------------------
def build_feature_vector(graph_data, wallet, transactions, chain=None):

    """
    Build a single feature vector + summary for API responses.

    Parameters
    ----------
    graph_data : dict | None
        Optional graph structure (unused for now, kept for compatibility).
    wallet : str
        Target wallet address.
    transactions : list[dict]
        List of tx-like dicts with keys such as from/to/value/weight/timestamp.
    chain : str | None
        "btc" or "eth" when known; used for price conversion.
    """

    # Normalize input
    if not transactions:
        transactions = []

    def _tx_value(tx):
        for key in ("value_eth", "value_btc", "value", "weight", "amount"):
            if key in tx and tx.get(key) is not None:
                try:
                    return float(tx.get(key, 0))
                except Exception:
                    return 0.0
        return 0.0

    def _tx_count(tx):
        try:
            return int(tx.get("count", 1))
        except Exception:
            return 1

    # Focus on txs involving the target wallet
    wallet_txs = [
        tx for tx in transactions
        if tx.get("from") == wallet or tx.get("to") == wallet
    ]

    # Default empty response
    if not wallet_txs:
        empty_summary = {
            "total_transactions": 0,
            "total_amount_btc": 0,
            "total_amount_fiat": {
                "USD": 0,
                "GBP": 0,
                "EUR": 0,
                "CAD": 0,
                "AUD": 0,
            },
            "asset": (chain or "unknown"),
        }
        return {
            "transaction_summary": empty_summary,
            "scam_type": "Unknown",
            "features": {},
        }

    # Timestamps -> year + day-of-year
    times = []
    for tx in wallet_txs:
        ts = tx.get("timestamp", 0)
        if isinstance(ts, (int, float)) and ts > 0:
            times.append(ts)

    if times:
        dt = datetime.utcfromtimestamp(min(times))
        year = int(dt.year)
        day = int(dt.timetuple().tm_yday)
    else:
        year = 0
        day = 0

    length = int(len(wallet_txs))

    # Aggregate counts where available (edge-level graphs may include "count")
    count = int(sum(_tx_count(tx) for tx in wallet_txs))

    # Self-loop transfers
    looped = int(
        sum(_tx_count(tx) for tx in wallet_txs
            if tx.get("from") == wallet and tx.get("to") == wallet)
    )

    # Neighbor count
    neighbors = set()
    for tx in wallet_txs:
        from_addr = tx.get("from")
        to_addr = tx.get("to")
        if isinstance(from_addr, str):
            neighbors.add(from_addr)
        if isinstance(to_addr, str):
            neighbors.add(to_addr)
    neighbors.discard(wallet)
    neighbors = int(len(neighbors))

    # Values
    values = [_tx_value(tx) for tx in wallet_txs]
    weight = float(statistics.mean(values)) if values else 0.0

    # Incoming value for summary/features
    income = float(
        sum(_tx_value(tx) for tx in wallet_txs if tx.get("to") == wallet)
    )

    base = {
        "year": year,
        "day": day,
        "length": length,
        "weight": weight,
        "count": count,
        "looped": looped,
        "neighbors": neighbors,
        "income": income,
    }

    feature_df = build_features(base)
    row = feature_df.iloc[0].to_dict()
    # Convert numpy/pandas scalars to native Python types for JSON safety
    row = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in row.items()
    }

    scam_type = row.get("suspected_scam_type", "Normal")

    # Price conversion
    if chain is None:
        if isinstance(wallet, str) and wallet.startswith("0x"):
            chain = "eth"
        else:
            chain = "btc"

    symbol = "bitcoin" if chain == "btc" else "ethereum"
    price = get_crypto_price(symbol)

    fiat_values = {
        "USD": income * price.get("usd", 0),
        "GBP": income * price.get("gbp", 0),
        "EUR": income * price.get("eur", 0),
        "CAD": income * price.get("cad", 0),
        "AUD": income * price.get("aud", 0),
    }

    tx_summary = {
        "total_transactions": length,
        "total_amount_btc": income,
        "total_amount_fiat": fiat_values,
        "asset": chain,
    }

    return {
        "transaction_summary": tx_summary,
        "scam_type": scam_type,
        "features": row,
    }


if __name__ == "__main__":

    run_feature_engineering()
