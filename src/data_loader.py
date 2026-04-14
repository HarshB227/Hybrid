import pandas as pd
import logging

from src.config import ( 
    BITCOIN_HEIST_FILE, 
    ETH_FIRST_ORDER_FILE,
    EXP1_BITCOIN_FILE,
    EXP2_ETHEREUM_FILE
)

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Load Bitcoin Heist Dataset
# -------------------------------------------------

def load_bitcoin_heist_data():

    logger.info("Loading Bitcoin Heist dataset...")

    df = pd.read_csv(BITCOIN_HEIST_FILE)

    logger.info(f"Dataset shape: {df.shape}")

    return df


# -------------------------------------------------
# Load Ethereum Dataset
# -------------------------------------------------

def load_ethereum_data():

    logger.info("Loading Ethereum dataset...")

    df = pd.read_csv(ETH_FIRST_ORDER_FILE)

    logger.info(f"Dataset shape: {df.shape}")

    return df


# -------------------------------------------------
# Load Exp1 Bitcoin Dataset
# -------------------------------------------------

def load_exp1_bitcoin_data():

    logger.info("Loading Exp1 Bitcoin dataset...")

    df = pd.read_csv(EXP1_BITCOIN_FILE)

    logger.info(f"Dataset shape: {df.shape}")

    return df


# -------------------------------------------------
# Load Exp2 Ethereum Dataset
# -------------------------------------------------

def load_exp2_ethereum_data():

    logger.info("Loading Exp2 Ethereum dataset...")

    df = pd.read_csv(EXP2_ETHEREUM_FILE)

    logger.info(f"Dataset shape: {df.shape}")

    return df


# -------------------------------------------------
# Basic Schema Validation
# -------------------------------------------------

def validate_bitcoin_schema(df):

    required_columns = [
        "address",
        "year",
        "day",
        "length",
        "weight",
        "count",
        "looped",
        "neighbors",
        "income",
        "label",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    logger.info("Schema validation passed.")

    return True


# -------------------------------------------------
# Load + Validate Combined Function
# -------------------------------------------------

def load_and_validate_bitcoin_data():

    df = load_bitcoin_heist_data()

    validate_bitcoin_schema(df)

    return df