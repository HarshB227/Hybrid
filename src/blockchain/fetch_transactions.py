import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional

import requests

from src.blockchain.normalize import normalize_transactions
logger = logging.getLogger(__name__)

_ENV_CACHE: Optional[Dict[str, str]] = None
_MISSING_KEY_LOGGED = set()


def _load_env_file() -> Dict[str, str]:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    values: Dict[str, str] = {}
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            values[key] = value
    except FileNotFoundError:
        return values
    except Exception:
        logger.exception("[Config] Failed to read .env file")
    return values


def _get_env_value(key: str) -> str:
    value = os.getenv(key)
    if value is not None and value.strip() != "":
        return value.strip()
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env_file()
    return _ENV_CACHE.get(key, "").strip()


def _normalize_key(value: str) -> str:
    cleaned = value.strip()
    if cleaned == "":
        return ""
    if cleaned.upper() in {"YOUR_API_KEY", "CHANGEME", "REPLACE_ME"}:
        return ""
    return cleaned


def _log_missing_key(source: str, message: str) -> None:
    if source in _MISSING_KEY_LOGGED:
        return
    _MISSING_KEY_LOGGED.add(source)
    logger.warning(message)


ALCHEMY_API_KEY = _normalize_key(_get_env_value("ALCHEMY_API_KEY"))
BASE_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}" if ALCHEMY_API_KEY else None
DEFAULT_HEADERS = {"User-Agent": "hybrid-crypto-dashboard/1.0"}

MAX_RETRIES = 3
RETRY_DELAY = 1.5


# -----------------------------------------
# VALIDATION
# -----------------------------------------
def is_valid_wallet(address: str) -> bool:
    return isinstance(address, str) and address.startswith("0x") and len(address) == 42


# -----------------------------------------
# CORE REQUEST HANDLER (ROBUST)
# -----------------------------------------
def _post(payload: Dict) -> Optional[Dict]:
    if not BASE_URL:
        _log_missing_key("alchemy", "[Alchemy] API key not set (set ALCHEMY_API_KEY)")
        return None
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.post(
                BASE_URL,
                json=payload,
                timeout=20,
                headers=DEFAULT_HEADERS
            )

            if res.status_code == 200:
                data = res.json()

                if "error" in data:
                    logger.error(f"Alchemy API error: {data['error']}")
                    return None

                return data

            logger.warning(f"[Alchemy] HTTP {res.status_code}: {res.text[:200]}")

        except Exception as e:
            logger.error(f"[Alchemy] Request failed: {e}")

        time.sleep(RETRY_DELAY * (attempt + 1))

    logger.error("[Alchemy] Max retries reached")
    return None


# -----------------------------------------
# ASSET TRANSFERS (MAIN FUNCTION)
# -----------------------------------------
def fetch_asset_transfers(
    address: str,
    category: Optional[List[str]] = None,
    max_count: int = 1000,
    max_pages: int = 5,   # prevent infinite loops
) -> List[Dict]:

    if not is_valid_wallet(address):
        logger.error(f"[Fetch] Invalid wallet address: {address}")
        return []

    logger.info(f"[Fetch] Fetching transfers for: {address}")

    if category is None:
        category = ["external", "internal", "erc20", "erc721"]

    all_transfers = []
    page_key = None
    page = 0

    while True:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "0x0",
                    "toBlock": "latest",
                    "fromAddress": address,
                    "category": category,
                    "withMetadata": True,
                    "excludeZeroValue": False,
                    "maxCount": hex(max_count),
                    **({"pageKey": page_key} if page_key else {}),
                }
            ],
        }

        data = _post(payload)

        if not data or "result" not in data:
            logger.warning("[Fetch] No valid response from Alchemy")
            break

        result = data["result"]
        transfers = result.get("transfers", [])

        logger.info(f"[Fetch] Page {page} -> {len(transfers)} transfers")

        if not transfers:
            break

        all_transfers.extend(transfers)

        page_key = result.get("pageKey")
        page += 1

        if not page_key or page >= max_pages:
            break

        time.sleep(0.25)  # rate control

    logger.info(f"[Fetch] Total transfers fetched: {len(all_transfers)}")

    return all_transfers


# -----------------------------------------
# TRANSACTION RECEIPT (OPTIONAL, HEAVY)
# -----------------------------------------
def fetch_transaction_receipt(tx_hash: str) -> Optional[Dict]:

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_getTransactionReceipt",
        "params": [tx_hash],
    }

    data = _post(payload)

    if data and "result" in data:
        return data["result"]

    return None


# -----------------------------------------
# FULL WALLET INGESTION
# -----------------------------------------
def fetch_full_wallet_data(address: str) -> Dict:

    if not is_valid_wallet(address):
        return {
            "address": address, 
            "error": "Invalid wallet address",
            "n_transactions": 0,
            "data": [],
        }

    transfers = fetch_asset_transfers(address)

    if not transfers:
        logger.warning("[Fetch] No transaction data available")
        return {
            "address": address,
            "n_transactions": 0,
            "data": [],
        }

    #  DO NOT fetch receipts for all (too slow)
    # Only attach basic transfer data for now

    enriched = []
    seen = set()

    for tx in transfers:
        tx_hash = tx.get("hash")
        log_index = tx.get("logIndex")

        # Deduplication (CRITICAL)
        key = f"{tx_hash}_{log_index}"

        if key in seen:
            continue

        seen.add(key)

        enriched.append({
            "transfer": tx,
            "receipt": None  # skip heavy calls
        })

    return {
        "address": address,
        "n_transactions": len(enriched),
        "data": enriched,
    }


# -----------------------------------------
# API COMPAT WRAPPER (USED BY api/main.py)
# -----------------------------------------
def fetch_transactions(address: str) -> List[Dict]:
    raw = fetch_full_wallet_data(address)

    if not raw or raw.get("n_transactions", 0) == 0:
        return []

    try:
        txs = normalize_transactions(raw)
    except Exception as exc:
        logger.error(f"[Fetch] Normalize failed: {exc}")
        return []

    formatted: List[Dict] = []

    for tx in txs:
        formatted.append({
            "hash": tx.get("tx_hash") or tx.get("hash"),
            "from": tx.get("from"),
            "to": tx.get("to"),
            "value": tx.get("value", 0.0),
            "timestamp": tx.get("timestamp"),
            "source": "alchemy",
        })

    return formatted
