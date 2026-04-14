import os
import time
import logging
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# -----------------------------------------
# LOAD ENV
# -----------------------------------------
load_dotenv()

logger = logging.getLogger(__name__)

ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise ValueError("❌ ALCHEMY_API_KEY not found in environment")

BASE_URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

MAX_RETRIES = 3
RETRY_DELAY = 1.5


# -----------------------------------------
# VALIDATION
# -----------------------------------------
def is_valid_wallet(address: str) -> bool:
    return isinstance(address, str) and address.startswith("0x") and len(address) == 42


# -----------------------------------------
# CORE REQUEST
# -----------------------------------------
def _post(payload: Dict) -> Optional[Dict]:
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.post(BASE_URL, json=payload, timeout=20)

            if res.status_code == 200:
                try:
                    data = res.json()
                except ValueError as e:
                    logger.error(f"[Alchemy] Invalid JSON response: {e}")
                    return None
                if "error" in data:
                    logger.error(f"[Alchemy] API error: {data['error']}")
                    return None
                return data

            if res.status_code == 429:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"[Alchemy] Rate limited (429). Waiting {wait:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait)
                continue

            logger.error(f"[Alchemy] HTTP {res.status_code}: {res.text[:200]}")

        except requests.exceptions.Timeout:
            logger.error(f"[Alchemy] Request timed out (attempt {attempt + 1}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[Alchemy] Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[Alchemy] Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

        time.sleep(RETRY_DELAY * (attempt + 1))

    logger.error(f"[Alchemy] All {MAX_RETRIES} attempts failed")
    return None


# -----------------------------------------
# FETCH TRANSFERS (FIXED: IN + OUT)
# -----------------------------------------
def fetch_asset_transfers(address: str, max_count=1000, max_pages=5) -> List[Dict]:

    if not is_valid_wallet(address):
        logger.error(f"Invalid address: {address}")
        return []

    def fetch(direction_key):
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
                        direction_key: address,
                        "category": ["external", "internal", "erc20", "erc721"],
                        "withMetadata": True,
                        "excludeZeroValue": False,
                        "maxCount": hex(max_count),
                        **({"pageKey": page_key} if page_key else {}),
                    }
                ],
            }

            data = _post(payload)

            if not data or "result" not in data:
                break

            result = data["result"]
            transfers = result.get("transfers", [])

            if not transfers:
                break

            all_transfers.extend(transfers)

            page_key = result.get("pageKey")
            page += 1

            if not page_key or page >= max_pages:
                break

            time.sleep(0.2)

        return all_transfers

    # 🔥 KEY FIX: BOTH DIRECTIONS
    outgoing = fetch("fromAddress")
    incoming = fetch("toAddress")

    logger.info(f"[Fetch] OUT: {len(outgoing)} | IN: {len(incoming)}")

    return outgoing + incoming


# -----------------------------------------
# MAIN FUNCTION
# -----------------------------------------
def fetch_full_wallet_data(address: str, max_count=1000, max_pages=5) -> Dict:

    if not is_valid_wallet(address):
        return {
            "address": address,
            "error": "Invalid wallet address",
            "n_transactions": 0,
            "data": [],
        }

    transfers = fetch_asset_transfers(address, max_count=max_count, max_pages=max_pages)

    if not transfers:
        logger.warning("[Fetch] No transaction data available")
        return {
            "address": address,
            "n_transactions": 0,
            "data": [],
        }

    seen = set()
    enriched = []

    for tx in transfers:
        tx_hash = tx.get("hash")
        log_index = tx.get("logIndex")

        key = f"{tx_hash}_{log_index}"

        if key in seen:
            continue

        seen.add(key)

        enriched.append({
            "transfer": tx,
            "receipt": None
        })

    return {
        "address": address,
        "n_transactions": len(enriched),
        "data": enriched,
    }
