# Hybrid — Crypto Scam Detection System

> A machine-learning system for detecting fraudulent blockchain wallets on **Bitcoin** and **Ethereum** networks. It combines XGBoost, Graph Neural Networks, risk propagation, and money-flow tracing to produce **explainable** risk scores via a REST API and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB6E2C?logo=xgboost&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Code Flow](#code-flow)
4. [Risk Scoring Engine](#risk-scoring-engine)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Tech Stack](#tech-stack)
8. [Installation](#installation)
9. [Data Setup](#data-setup)
10. [Training Pipeline](#training-pipeline)
11. [Running the API](#running-the-api)
12. [Running the Dashboard](#running-the-dashboard)
13. [Environment Variables](#environment-variables)

---

## Features

- **XGBoost Classifier** — Trained on the Bitcoin Heist ransomware dataset with 30+ engineered features
- **Graph Risk Propagation** — Spreads fraud signals through transaction neighbourhoods (5 iterations)
- **Money Flow Tracing** — DFS-based multi-hop laundering path detection
- **Louvain Community Detection** — Identifies high-risk wallet clusters
- **Graph Neural Network (GCN)** — 2-layer FraudGNN trained on the Elliptic Bitcoin dataset
- **Explainable Scoring** — Returns risk breakdown across behavioural, graph, flow, and cluster dimensions
- **FastAPI Backend** — Address validation (ETH + BTC) with a fast-mode toggle
- **Streamlit Dashboard** — Risk gauge, flow paths, graph visualisation, and multi-currency conversion
- **Multi-chain Support** — Bitcoin and Ethereum via Alchemy, Etherscan, and Blockchair APIs

---

## System Architecture

```
Blockchain APIs (Alchemy / Etherscan / Blockchair)
        ↓
Fetch & Normalise Transactions
        ↓
Build Transaction Graph (NetworkX DiGraph)
        ↓
┌──────────────────────────────────────────────────────────┐
│                   Parallel Risk Scoring                  │
│  1. XGBoost ML prediction       (30% weight)             │
│  2. Graph risk propagation      (25% weight)             │
│  3. Money flow tracing & scoring                         │
│  4. Louvain cluster detection   (20% weight)             │
│  5. Behavioural thresholds      (25% weight)             │
└──────────────────────────────────────────────────────────┘
        ↓
Explainable Score Fusion
        ↓
FastAPI → Streamlit Dashboard
```

---

## Code Flow

End-to-end execution path from user input to dashboard visualisation.

### 1. Frontend — `dashboard/app.py`

```
User (Browser)
     │  wallet address
     ▼
Streamlit Dashboard          ── dashboard/app.py
     │
     ▼
fetch_wallet_data()          ── dashboard/app.py
     │  POST /analyze_wallet (HTTP → :8000)
     ▼
```

### 2. API Layer — `api/main.py`

```
analyze_wallet()             ── api/main.py
     │
     ├── INVALID ADDRESS  →  400 Error (HTTPException)
     │
     └── VALID ADDRESS    →  Orchestrate Analysis
                              │  run pipeline
                              ▼
```

### 3. Blockchain Pipeline — `src/blockchain/pipeline.py`

```
run_wallet_pipeline()                    ── src/blockchain/pipeline.py
     │
     ├── fetch_full_wallet_data()        ── alchemy_fetch.py
     ├── normalize_transactions()        ── normalize.py
     ├── expand_transactions()           ── pipeline.py
     ├── build_transaction_graph()       ── build_graph.py
     ├── compute_graph_features()        ── graph_features.py
     └── serialize_graph()               ── pipeline.py
                                          │  graph + features
                                          ▼
```

### 4. Risk Analysis Services — `backend/services/`

Three services run in parallel on the transaction graph:

| Branch | Function | Module | Output |
|---|---|---|---|
| **A — Risk Propagation** | `propagate_risk()` | `risk_propagation.py` | `risk_scores` dict `{address → score}` |
| **B — Flow Tracing** | `trace_funds()` → `score_all_flows()` | `flow_tracing.py`, `flow_scoring.py` | scored money-flow paths |
| **C — Cluster Detection** | `detect_clusters()` → `analyze_clusters()` | `cluster_detection.py` | high-risk wallet clusters |

```
                    ┌─────────────────────┐
                    │  graph + features   │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 propagate_risk()        trace_funds()         detect_clusters()
        │                      │                      │
        ▼                      ▼                      ▼
   risk_scores            score_all_flows()    analyze_clusters()
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                  risk scores + flows + clusters
                               ▼
```

### 5. Fusion Scoring — `backend/services/explainable_scoring.py`

```
compute_final_score()                ── explainable_scoring.py
     │
     ├── Behavioral Score   (25% weight)
     ├── Graph Score        (25% weight)
     ├── Cluster Score      (20% weight)
     └── GNN / ML Score     (30% weight)
                │
                │  final_score + confidence + reasons
                ▼
```

### 6. Risk Classification — `api/main.py`

| Score Range | Classification |
|---|---|
| `score > 0.70` | 🔴 **HIGH** |
| `score > 0.40` | 🟡 **MEDIUM** |
| otherwise | 🟢 **LOW** |

```
                 ┌──── JSON response ────┐
                 ▼                       │
```

### 7. Dashboard Visualisation — `dashboard/app.py`

The dashboard renders the JSON response into six panels:

| Panel | Description |
|---|---|
| **Risk Gauge** | 0–100%, colour-coded |
| **7 KPI Metrics** | Transaction count, risk scores, etc. |
| **Score Breakdown** | Bar chart of the 4 fusion components |
| **Money Flow Paths** | Labelled + colour-coded flow visualisation |
| **Network Graph** | Interactive PyVis graph |
| **Tx Summary** | ETH / BTC totals with FX rates |

---

### Offline — ML Training Pipeline (`src/train_model.py`)

```
Load Dataset       →   Prepare Features    →   Train 3 Models           →   Save Best Model
data/processed/        train/test split        LR + RF + XGBoost            models/best_model.pkl
```

---

## Risk Scoring Engine

### Fusion Score Weights

These weights are applied in `explainable_scoring.py → compute_final_score()`.

| Component | Weight |
|---|---|
| GNN / ML Score | **30%** |
| Behavioral Score | **25%** |
| Graph (Propagation) Score | **25%** |
| Cluster Score | **20%** |

### Flow Score Weights

Applied in `flow_scoring.py → score_all_flows()`.

| Component | Weight |
|---|---|
| Flow Value (normalised) | **25%** |
| Path Length | **20%** |
| Avg Node Risk | **20%** |
| Flow Direction | **20%** |
| Time Gap Speed | **15%** |

### Pipeline Steps

1. Fetch via Alchemy API
2. Normalise raw transfers
3. Expand to neighbours *(non-fast mode only)*
4. Build NetworkX DiGraph
5. Compute 15+ graph features
6. Serialise to JSON

### Key Thresholds

| Threshold | Value |
|---|---|
| Risk HIGH | `> 0.70` |
| Risk MEDIUM | `> 0.40` |
| Flow: laundering | `> 0.70` |
| Flow: suspicious | `> 0.40` |
| Propagation α | `0.30` |
| Propagation rounds | `5` |

---

## Configuration

All tunable parameters live in `src/config.py`.

```python
# ── Risk classification thresholds ─────────────────────────────
RISK_HIGH_THRESHOLD     = 0.7    # final score → HIGH risk
RISK_MEDIUM_THRESHOLD   = 0.4    # final score → MEDIUM risk

# ── Behavioural thresholds ─────────────────────────────────────
BEHAVIORAL_TX_HIGH      = 50     # tx count → high behavioural score
BEHAVIORAL_TX_MED       = 20
BEHAVIORAL_VALUE_HIGH   = 5      # avg ETH value → high
BEHAVIORAL_VALUE_MED    = 1

# ── Flow timing (seconds) ──────────────────────────────────────
FLOW_TIME_GAP_FAST      = 60     # rapid money movement
FLOW_TIME_GAP_MED       = 300
FLOW_TIME_GAP_SLOW      = 3600

# ── Risk-propagation parameters ────────────────────────────────
PROPAGATION_TARGET_RISK = 0.7    # target wallet initial risk
PROPAGATION_NODE_RISK   = 0.1    # neighbour initial risk
PROPAGATION_ALPHA       = 0.3    # influence weight per round
PROPAGATION_ITERATIONS  = 5
```

---

## Project Structure

```
hybrid/
├── api/
│   └── main.py                    # FastAPI /analyze_wallet endpoint
├── backend/
│   └── services/
│       ├── risk_propagation.py    # Graph-based risk propagation
│       ├── flow_tracing.py        # Multi-hop money flow detection
│       ├── flow_scoring.py        # Score flows by timing & direction
│       ├── cluster_detection.py   # Louvain community detection
│       └── explainable_scoring.py # Fuse all risk signals
├── dashboard/
│   └── app.py                     # Streamlit UI
├── src/
│   ├── config.py                  # Centralised thresholds & paths
│   ├── data_loader.py             # Load & validate datasets
│   ├── preprocessing.py           # Clean & filter data
│   ├── feature_engineering.py     # Build 30+ features
│   ├── train_model.py             # Train & select best model
│   ├── evaluation.py              # Generate evaluation plots
│   ├── fusion.py                  # Hybrid ML + market risk engine
│   └── blockchain/
│       ├── alchemy_fetch.py
│       ├── fetch_transactions.py
│       ├── normalize.py
│       ├── build_graph.py
│       ├── graph_features.py
│       └── pipeline.py
├── model/
│   ├── gnn_model.py               # FraudGNN (2-layer GCN)
│   ├── graph_to_gnn.py
│   └── gnn_features.py
├── training/
│   └── train_gnn.py
├── data/
│   ├── raw/                       # Raw CSV datasets
│   ├── interim/                   # Cleaned parquet
│   ├── processed/                 # Feature-engineered parquet
│   └── elliptic/                  # Elliptic dataset for GNN
├── models/                        # Saved .pkl model files
├── reports/                       # Evaluation plots (.png)
├── requirements.txt
└── .env                           # API keys (not committed)
```

---

## Tech Stack

| Category | Libraries / Services |
|---|---|
| ML / Data Science | XGBoost, scikit-learn, pandas, NumPy |
| Deep Learning | PyTorch, PyTorch Geometric (GCN) |
| Graph Analysis | NetworkX, python-louvain |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly, PyVis |
| Blockchain APIs | Alchemy, Etherscan, Blockchair |
| Price Data | CoinGecko (no auth required) |
| Serialisation | joblib, pyarrow (parquet) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hybrid.git
cd hybrid
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the project root

```env
PYTHONPATH=./
ALCHEMY_API_KEY=your_key
ETHERSCAN_API_KEY=your_key
BLOCKCHAIR_API_KEY=your_key
```

---

## Data Setup

Place the following files in `data/raw/`:

| File | Source |
|---|---|
| `BitcoinHeistData.csv` | Kaggle — Bitcoin Heist Ransomware |
| `first_order_df.csv` | Ethereum first-order features |
| `exp1_bitcoin_sample_test_dd.csv` | *(optional)* |
| `exp2_ethereum_sample_test_mbal_dd.csv` | *(optional)* |

---

## Training Pipeline

Run each step in order:

```bash
python -m src.data_loader
python -m src.preprocessing
python -m src.feature_engineering
python -m src.train_model
python -m src.evaluation
```

### Optional — Train the GNN

```bash
python -m src.training.train_gnn
```

The training pipeline produces:

```
Load Dataset  →  Prepare Features  →  Train 3 Models       →  Save Best Model
                                       (LR + RF + XGBoost)     models/best_model.pkl
```

---

## Running the API

```bash
uvicorn api.main:app --reload
```

Interactive docs available at: <http://127.0.0.1:8000/docs>

### Example request

```bash
curl -X POST http://127.0.0.1:8000/analyze_wallet \
  -H "Content-Type: application/json" \
  -d '{"address": "0xabc...123", "fast_mode": false}'
```

### Example response

```json
{
  "address": "0xabc...123",
  "final_score": 0.82,
  "risk_level": "HIGH",
  "confidence": 0.91,
  "breakdown": {
    "behavioral_score": 0.75,
    "graph_score": 0.80,
    "cluster_score": 0.65,
    "gnn_ml_score": 0.95
  },
  "reasons": ["High tx volume", "Connected to flagged cluster", "Rapid fund movement"]
}
```

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Then open <http://localhost:8501> in your browser.

---

## Environment Variables

| Variable | Description |
|---|---|
| `ALCHEMY_API_KEY` | Alchemy API key for Ethereum data |
| `ETHERSCAN_API_KEY` | Etherscan API key |
| `BLOCKCHAIR_API_KEY` | Blockchair API key for Bitcoin data |
| `PYTHONPATH` | Set to `./` to allow module-style imports |

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore` by default.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## Acknowledgements

- **Bitcoin Heist Ransomware Dataset** — Kaggle
- **Elliptic Bitcoin Dataset** — Elliptic / MIT
- **Alchemy, Etherscan, Blockchair** — blockchain data providers
- **CoinGecko** — price data
