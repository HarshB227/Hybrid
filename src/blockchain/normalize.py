from typing import Dict, List


def normalize_transactions(raw_data: Dict) -> List[Dict]:
    normalized = []

    for item in raw_data["data"]:
        tx = item["transfer"]
        value_raw = tx.get("value", 0)
        try:
            value = float(value_raw) if value_raw not in (None, "") else 0.0
        except (TypeError, ValueError):
            value = 0.0

        normalized.append({
            "tx_hash": tx.get("hash"),
            "from": tx.get("from"),
            "to": tx.get("to"),
            "value": value,
            "asset": tx.get("asset"),
            "timestamp": tx.get("metadata", {}).get("blockTimestamp"),
            "category": tx.get("category"),
        })

    return normalized
