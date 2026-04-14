import os
import logging

# -------------------------------------------------
# Project Root
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------------------------------------
# Data Paths
# -------------------------------------------------

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Raw datasets
BITCOIN_HEIST_FILE = os.path.join(RAW_DATA_DIR, "BitcoinHeistData.csv")
ETH_FIRST_ORDER_FILE = os.path.join(RAW_DATA_DIR, "first_order_df.csv")
EXP1_BITCOIN_FILE = os.path.join(RAW_DATA_DIR, "exp1_bitcoin_sample_test_dd.csv")
EXP2_ETHEREUM_FILE = os.path.join(RAW_DATA_DIR, "exp2_ethereum_sample_test_mbal_dd.csv")

# Output datasets
CLEANED_DATA_FILE = os.path.join(INTERIM_DATA_DIR, "cleaned_data.parquet")
FEATURE_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "feature_dataset.parquet")

# Risk log
RISK_LOG_FILE = os.path.join(DATA_DIR, "risk_log.csv")

# -------------------------------------------------
# Model Paths
# -------------------------------------------------

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")
BASELINE_SCALER_PATH = os.path.join(MODEL_DIR, "baseline_scaler.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

# -------------------------------------------------
# Reports
# -------------------------------------------------

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# -------------------------------------------------
# Training Parameters
# -------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.2

# -------------------------------------------------
# Risk Thresholds (single source of truth)
# -------------------------------------------------

RISK_HIGH_THRESHOLD = 0.7
RISK_MEDIUM_THRESHOLD = 0.4

assert RISK_HIGH_THRESHOLD > RISK_MEDIUM_THRESHOLD, "RISK_HIGH_THRESHOLD must be greater than RISK_MEDIUM_THRESHOLD"

# -------------------------------------------------
# Behavioral Scoring Thresholds
# Used by: backend/services/explainable_scoring.py
# -------------------------------------------------

BEHAVIORAL_TX_HIGH = 50    # tx_count above this → high activity signal
BEHAVIORAL_TX_MED = 20     # tx_count above this → medium activity signal
BEHAVIORAL_VALUE_HIGH = 5  # avg_value above this → high value signal
BEHAVIORAL_VALUE_MED = 1   # avg_value above this → medium value signal

# -------------------------------------------------
# Flow Scoring Thresholds
# Used by: backend/services/flow_scoring.py
# -------------------------------------------------

FLOW_TIME_GAP_FAST = 60     # seconds — rapid successive transfers
FLOW_TIME_GAP_MED = 300     # seconds — moderate transfer speed
FLOW_TIME_GAP_SLOW = 3600   # seconds — slow/normal transfer speed

FLOW_HIGH_THRESHOLD = 0.7   # score above this → laundering_pattern
FLOW_MED_THRESHOLD = 0.4    # score above this → suspicious

# -------------------------------------------------
# Risk Propagation Parameters
# Used by: backend/services/risk_propagation.py
# -------------------------------------------------

PROPAGATION_TARGET_RISK = 0.7   # initial risk assigned to the wallet under analysis
PROPAGATION_NODE_RISK = 0.1     # initial risk assigned to all other nodes
PROPAGATION_ALPHA = 0.3         # neighbour influence weight per iteration
PROPAGATION_ITERATIONS = 5      # number of propagation rounds

# -------------------------------------------------
# Centralised Logging (called once here)
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

LOG_LEVEL = "INFO"