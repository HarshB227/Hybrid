import os
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_DATA_FILE,
    XGB_MODEL_PATH,
    REPORTS_DIR,
    TEST_SIZE,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

def load_feature_dataset():

    logger.info("Loading feature dataset")

    return pd.read_parquet(FEATURE_DATA_FILE)


# -------------------------------------------------
# Prepare features
# -------------------------------------------------

def prepare_data(df):

    y = df["scam_label"]
    X = df.drop(columns=["label", "scam_label", "address"])

    return X, y


# -------------------------------------------------
# Split dataset
# -------------------------------------------------

def split_data(X, y):

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

def load_model():

    logger.info("Loading trained model")

    return joblib.load(XGB_MODEL_PATH)


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test):

    logger.info("Generating confusion matrix")

    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    disp  = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    plt.close()


# -------------------------------------------------
# ROC Curve
# -------------------------------------------------

def plot_roc_curve(model, X_test, y_test):

    logger.info("Generating ROC curve")

    probs              = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _        = roc_curve(y_test, probs)
    roc_auc            = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))
    plt.close()


# -------------------------------------------------
# Precision-Recall Curve
# -------------------------------------------------

def plot_pr_curve(model, X_test, y_test):

    logger.info("Generating Precision-Recall curve")

    probs              = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "pr_curve.png"))
    plt.close()


# -------------------------------------------------
# Feature Importance
# -------------------------------------------------

def plot_feature_importance(model, X):

    logger.info("Generating feature importance")

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature":    X.columns,
        "importance": importance,
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["feature"][:15], df_imp["importance"][:15])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"))
    plt.close()


# -------------------------------------------------
# Run evaluation
# -------------------------------------------------

def run_evaluation():

    df = load_feature_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = load_model()

    plot_confusion_matrix(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    plot_pr_curve(model, X_test, y_test)
    plot_feature_importance(model, X)

    logger.info(f"Evaluation complete. Reports saved to {REPORTS_DIR}")


if __name__ == "__main__":

    run_evaluation()