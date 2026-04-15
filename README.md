Hybrid — Crypto Scam Detection System
A machine learning system for detecting fraudulent blockchain wallets on Bitcoin and Ethereum networks. Combines XGBoost classifiers, Graph Neural Networks, risk propagation, and money-flow tracing to produce explainable risk scores via a REST API and an interactive Streamlit dashboard.

Features
XGBoost Classifier trained on the Bitcoin Heist ransomware dataset (30+ engineered features)
Graph Risk Propagation — spreads fraud signals through transaction neighborhoods (5 iterations)
Money Flow Tracing — DFS-based multi-hop laundering path detection
Louvain Community Detection — identifies high-risk wallet clusters
Graph Neural Network (GCN) — 2-layer FraudGNN trained on the Elliptic Bitcoin dataset
Explainable Scoring — returns risk breakdown across behavioral, graph, flow, and cluster dimensions
FastAPI backend with address validation (ETH + BTC) and fast-mode toggle
Streamlit dashboard with risk gauge, flow paths, graph visualization, and multi-currency conversion
Multi-chain support — Bitcoin and Ethereum via Alchemy, Etherscan, and Blockchair APIs
Architecture

Blockchain APIs (Alchemy / Etherscan / Blockchair)
        ↓
  Fetch & Normalize transactions
        ↓
  Build Transaction Graph (NetworkX DiGraph)
        ↓
  ┌──────────────────────────────────────────┐
  │  Parallel Risk Scoring                   │
  │  1. XGBoost ML prediction (70% weight)   │
  │  2. Graph risk propagation               │
  │  3. Money flow tracing & scoring         │
  │  4. Louvain cluster detection            │
  │  5. Behavioral thresholds                │
  └──────────────────────────────────────────┘
        ↓
  Explainable score fusion
        ↓
  FastAPI  →  Streamlit Dashboard
Project Structure

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
│   ├── config.py                  # Centralized thresholds & paths
│   ├── data_loader.py             # Load & validate datasets
│   ├── preprocessing.py           # Clean & filter data
│   ├── feature_engineering.py     # Build 30+ features
│   ├── train_model.py             # Train & select best model
│   ├── evaluation.py              # Generate evaluation plots
│   ├── fusion.py                  # Hybrid ML + market risk engine
│   ├── blockchain/
│   │   ├── alchemy_fetch.py
│   │   ├── fetch_transactions.py
│   │   ├── normalize.py
│   │   ├── build_graph.py
│   │   ├── graph_features.py
│   │   └── pipeline.py
│   ├── model/
│   │   ├── gnn_model.py           # FraudGNN (2-layer GCN)
│   │   ├── graph_to_gnn.py
│   │   └── gnn_features.py
│   └── training/
│       └── train_gnn.py
├── data/
│   ├── raw/                       # Raw CSV datasets
│   ├── interim/                   # Cleaned parquet
│   ├── processed/                 # Feature-engineered parquet
│   └── elliptic/                  # Elliptic dataset for GNN
├── models/                        # Saved .pkl model files
├── reports/                       # Evaluation plots (.png)
├── requirements.txt
└── .env                           # API keys (not committed)
Tech Stack
Category	Libraries / Services
ML / Data Science	XGBoost, scikit-learn, pandas, NumPy
Deep Learning	PyTorch, PyTorch Geometric (GCN)
Graph Analysis	NetworkX, python-louvain
API	FastAPI, Uvicorn, Pydantic
Dashboard	Streamlit, Plotly, PyVis
Blockchain APIs	Alchemy, Etherscan, Blockchair
Price Data	CoinGecko (no auth required)
Serialization	joblib, pyarrow (parquet)
Installation

git clone https://github.com/<your-username>/hybrid.git
cd hybrid
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
Create a .env file:


PYTHONPATH=./
ALCHEMY_API_KEY=your_key
ETHERSCAN_API_KEY=your_key
BLOCKCHAIR_API_KEY=your_key
Data Setup
Place in data/raw/:

File	Source
BitcoinHeistData.csv	Kaggle — Bitcoin Heist Ransomware
first_order_df.csv	Ethereum first-order features
exp1_bitcoin_sample_test_dd.csv	(optional)
exp2_ethereum_sample_test_mbal_dd.csv	(optional)
Training Pipeline

python -m src.data_loader
python -m src.preprocessing
python -m src.feature_engineering
python -m src.train_model
python -m src.evaluation
# Optional GNN:
python -m src.training.train_gnn
Run API

uvicorn api.main:app --reload
# http://127.0.0.1:8000/docs
Run Dashboard

streamlit run dashboard/app.py
# http://localhost:8501
Risk Levels
Level	Probability
LOW	< 0.40
MEDIUM	0.40 – 0.69
HIGH	≥ 0.70
Disclaimer
For research and educational purposes only. Not financial or legal advice.
