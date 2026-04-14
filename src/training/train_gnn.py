import torch
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.model.gnn_model import FraudGNN
from src.training.load_elliptic import load_elliptic_dataset


DATA_PATH = r"D:\hybrid\data\elliptic"


def train():

    # -----------------------------
    # LOAD DATASET
    # -----------------------------

    x, edge_index, y = load_elliptic_dataset(DATA_PATH)

    print("\nDataset loaded")
    print("Nodes:", x.shape[0])
    print("Features:", x.shape[1])
    print("Edges:", edge_index.shape[1])

    y_np = y.numpy()

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------

    idx = np.arange(len(y_np))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y_np
    )

    train_mask = torch.zeros(len(y_np), dtype=torch.bool)
    test_mask = torch.zeros(len(y_np), dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True


    # -----------------------------
    # MODEL
    # -----------------------------

    model = FraudGNN(
        in_channels=x.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )


    # -----------------------------
    # TRAIN LOOP
    # -----------------------------

    for epoch in range(80):

        model.train()

        optimizer.zero_grad()

        preds = model(x, edge_index).squeeze()

        loss = torch.nn.functional.binary_cross_entropy(
            preds[train_mask],
            y[train_mask]
        )

        loss.backward()

        optimizer.step()


        # -----------------------------
        # EVALUATION
        # -----------------------------

        if epoch % 10 == 0:

            model.eval()

            with torch.no_grad():

                pred_test = preds[test_mask].numpy()
                y_test = y[test_mask].numpy()

                auc = roc_auc_score(
                    y_test,
                    pred_test
                )

                f1 = f1_score(
                    y_test,
                    (pred_test > 0.5).astype(int)
                )

                acc = accuracy_score(
                    y_test,
                    (pred_test > 0.5).astype(int)
                )

            print(
                f"Epoch {epoch}",
                f"Loss {loss.item():.4f}",
                f"AUC {auc:.4f}",
                f"F1 {f1:.4f}",
                f"ACC {acc:.4f}"
            )


    # -----------------------------
    # FINAL METRICS
    # -----------------------------

    model.eval()

    with torch.no_grad():

        final_preds = model(x, edge_index).squeeze()

        pred_test = final_preds[test_mask].numpy()
        y_test = y[test_mask].numpy()

        final_auc = roc_auc_score(y_test, pred_test)
        final_f1 = f1_score(y_test, (pred_test > 0.5).astype(int))
        final_acc = accuracy_score(y_test, (pred_test > 0.5).astype(int))


    print("\nFINAL RESULTS")
    print("AUC:", round(final_auc, 4))
    print("F1:", round(final_f1, 4))
    print("Accuracy:", round(final_acc, 4))


    # -----------------------------
    # SAVE MODEL
    # -----------------------------

    torch.save(
        model.state_dict(),
        "models/fraud_gnn.pt"
    )

    print("\nModel saved to models/fraud_gnn.pt")


if __name__ == "__main__":

    train()