import os
import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from src.config import FEATURE_DATA_FILE, MODEL_DIR, TEST_SIZE, RANDOM_SEED

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
def load_dataset():
    logger.info("Loading feature dataset...")
    df = pd.read_parquet(FEATURE_DATA_FILE)
    logger.info(f"Dataset shape: {df.shape}")
    return df


# -------------------------------------------------
# PREPARE FEATURES
# -------------------------------------------------
def prepare_features(df):
    if "scam_label" not in df.columns:
        raise ValueError("Missing 'scam_label' column")

    y = df["scam_label"]

    X = df.drop(
        columns=["label", "scam_label", "address"],
        errors="ignore"
    )

    return X, y


# -------------------------------------------------
# MODELS
# -------------------------------------------------
def train_logistic(X_train, y_train):
    logger.info("Training Logistic Regression...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ))
    ])

    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    logger.info("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    logger.info("Training XGBoost...")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    return model


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
def evaluate(model, X_test, y_test, name):
    logger.info(f"Evaluating {name}...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    return auc


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def run_pipeline():

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = load_dataset()

    # sanity check
    logger.info("Checking label distribution:")
    print(df["scam_label"].value_counts())

    # -----------------------------
    # PREPARE FEATURES
    # -----------------------------
    X, y = prepare_features(df)

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    models = {
        "logistic": train_logistic(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train)
    }

    # -----------------------------
    # SELECT BEST MODEL
    # -----------------------------
    best_model = None
    best_score = 0
    best_name = None

    for name, model in models.items():

        auc = evaluate(model, X_test, y_test, name)

        if auc > best_score:
            best_score = auc
            best_model = model
            best_name = name

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")

    joblib.dump(best_model, model_path)

    logger.info(f"Best model: {best_name} (ROC-AUC={best_score:.4f})")
    logger.info(f"Model saved at: {model_path}")

    return best_model


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    run_pipeline()


def run_training_pipeline():
    return run_pipeline()