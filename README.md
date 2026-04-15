# Hybrid — Crypto Scam Detection System

A machine learning system for detecting fraudulent blockchain wallets on Bitcoin and Ethereum networks. Combines XGBoost classifiers, Graph Neural Networks, risk propagation, and money-flow tracing to produce explainable risk scores via a REST API and an interactive Streamlit dashboard.

---

## Features

- **XGBoost Classifier** — Trained on the Bitcoin Heist ransomware dataset with 30+ engineered features
- **Graph Risk Propagation** — Spreads fraud signals through transaction neighbourhoods (5 iterations)
- **Money Flow Tracing** — DFS-based multi-hop laundering path detection
- **Louvain Community Detection** — Identifies high-risk wallet clusters
- **Graph Neural Network (GCN)** — 2-layer FraudGNN trained on the Elliptic Bitcoin dataset
- **Explainable Scoring** — Returns risk breakdown across behavioural, graph, flow, and cluster dimensions
- **FastAPI Backend** — Address validation (ETH + BTC) and fast-mode toggle
- **Streamlit Dashboard** — Risk gauge, flow paths, graph visualisation, and multi-currency conversion
- **Multi-chain Support** — Bitcoin and Ethereum via Alchemy, Etherscan, and Blockchair APIs

---

## Architecture

```
Blockchain APIs (Alchemy / Etherscan / Blockchair)
        ↓
Fetch & Normalise Transactions
        ↓
Build Transaction Graph (NetworkX DiGraph)
        ↓
┌──────────────────────────────────────────────────────────┐
│                   Parallel Risk Scoring                  │
│  1. XGBoost ML prediction      (70% weight)              │
│  2. Graph risk propagation                               │
│  3. Money flow tracing & scoring                         │
│  4. Louvain cluster detection                            │
│  5. Behavioural thresholds                               │
└──────────────────────────────────────────────────────────┘
        ↓
Explainable Score Fusion
        ↓
FastAPI → Streamlit Dashboard
```

---

## Project Structure

```
hybrid/
├── api/
│   └── main.py                  # FastAPI /analyze_wallet endpoint
├── backend/
│   └── services/
│       ├── risk_propagation.py  # Graph-based risk propagation
│       ├── flow_tracing.py      # Multi-hop money flow detection
│       ├── flow_scoring.py      # Score flows by timing & direction
│       ├── cluster_detection.py # Louvain community detection
│       └── explainable_scoring.py # Fuse all risk signals
├── dashboard/
│   └── app.py                   # Streamlit UI
├── src/
│   ├── config.py                # Centralised thresholds & paths
│   ├── data_loader.py           # Load & validate datasets
│   ├── preprocessing.py         # Clean & filter data
│   ├── feature_engineering.py   # Build 30+ features
│   ├── train_model.py           # Train & select best model
│   ├── evaluation.py            # Generate evaluation plots
│   ├── fusion.py                # Hybrid ML + market risk engine
│   └── blockchain/
│       ├── alchemy_fetch.py
│       ├── fetch_transactions.py
│       ├── normalize.py
│       ├── build_graph.py
│       ├── graph_features.py
│       └── pipeline.py
├── model/
│   ├── gnn_model.py             # FraudGNN (2-layer GCN)
│   ├── graph_to_gnn.py
│   └── gnn_features.py
├── training/
│   └── train_gnn.py
├── data/
│   ├── raw/                     # Raw CSV datasets
│   ├── interim/                 # Cleaned parquet
│   ├── processed/               # Feature-engineered parquet
│   └── elliptic/                # Elliptic dataset for GNN
├── models/                      # Saved .pkl model files
├── reports/                     # Evaluation plots (.png)
├── requirements.txt
└── .env                         # API keys (not committed)
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
git clone https://github.com/hybrid.git
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

### 4. Create a `.env` file

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

---

## Running the API

```bash
uvicorn api.main:app --reload
```

API docs available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `ALCHEMY_API_KEY` | Alchemy API key for Ethereum data |
| `ETHERSCAN_API_KEY` | Etherscan API key |
| `BLOCKCHAIR_API_KEY` | Blockchair API key for Bitcoin data |

> **Note:** Never commit your `.env` file. It is listed in `.gitignore` by default.
