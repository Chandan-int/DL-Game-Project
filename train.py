import argparse
import csv
import json
import math
import os
import random

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_FEATURES  = "data/features.csv"
DEFAULT_MODEL_DIR = "models"
EXPERIMENT_NAME   = "difficulty-adaptation"

FEATURE_COLS = [
    "survival_norm",
    "reaction_norm",
    "score_norm",
    "roll5_survival_norm",
    "roll5_reaction_norm",
    "roll5_score_norm",
    "sessions_log",
]
TARGET_COL = "target"


# ── Model definition

class DifficultyMLP(nn.Module):
    """
    Tiny MLP: 7 → 32 → 16 → 1
    Sigmoid on output keeps predictions in [0, 1].
    """
    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Data loading

def load_csv(path: str) -> tuple[list[list[float]], list[float]]:
    X, y = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                X.append([float(row[c]) for c in FEATURE_COLS])
                y.append(float(row[TARGET_COL]))
            except (KeyError, ValueError):
                continue
    return X, y


def train_val_split(
    X: list, y: list, val_frac: float = 0.2, seed: int = 42
) -> tuple:
    indices = list(range(len(X)))
    random.seed(seed)
    random.shuffle(indices)
    split = int(len(indices) * (1 - val_frac))
    train_idx, val_idx = indices[:split], indices[split:]
    return (
        [X[i] for i in train_idx], [y[i] for i in train_idx],
        [X[i] for i in val_idx],   [y[i] for i in val_idx],
    )


#  Standardisation

def compute_scaler(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n    = len(X)
    dims = len(X[0])
    means = [sum(row[d] for row in X) / n for d in range(dims)]
    stds  = [
        max(math.sqrt(sum((row[d] - means[d]) ** 2 for row in X) / n), 1e-8)
        for d in range(dims)
    ]
    return means, stds


def apply_scaler(
    X: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    return [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in X]


def to_tensors(X, y):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


#  Training loop

def train(
    features_path: str = DEFAULT_FEATURES,
    model_dir: str     = DEFAULT_MODEL_DIR,
    epochs: int        = 300,
    lr: float          = 0.001,
    batch_size: int    = 16,
    val_frac: float    = 0.2,
) -> None:

    # ① Load data
    if not os.path.isfile(features_path):
        print(f"[train] features not found: {features_path}. Run features.py first.")
        return

    X, y = load_csv(features_path)
    print(f"[train] loaded {len(X)} samples, {len(FEATURE_COLS)} features each")

    if len(X) < 10:
        print("[train] not enough data — need at least 10 samples. Run simulate.py.")
        return

    # ② Split
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac)
    print(f"[train] train={len(X_train)}, val={len(X_val)}")

    # ③ Standardise
    means, stds = compute_scaler(X_train)
    X_train = apply_scaler(X_train, means, stds)
    X_val   = apply_scaler(X_val,   means, stds)

    # ④ Tensors & DataLoader
    Xt, yt = to_tensors(X_train, y_train)
    Xv, yv = to_tensors(X_val,   y_val)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    # ⑤ Model, loss, optimiser
    model     = DifficultyMLP(input_dim=len(FEATURE_COLS))
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=20, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    print_every   = max(1, epochs // 10)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ◆ MLflow — everything inside this block is ONE run
    with mlflow.start_run() as run:

        # ◆ MLflow  log all hyperparameters in one call
        mlflow.log_params({
            "epochs":      epochs,
            "lr":          lr,
            "batch_size":  batch_size,
            "val_frac":    val_frac,
            "input_dim":   len(FEATURE_COLS),
            "hidden_1":    32,
            "hidden_2":    16,
            "dropout":     0.1,
            "optimizer":   "Adam",
            "loss_fn":     "MSELoss",
            "n_train":     len(X_train),
            "n_val":       len(X_val),
        })

        # ◆ MLflow - tag the run with human-readable labels
        mlflow.set_tag("model_type",   "DifficultyMLP")
        mlflow.set_tag("feature_set",  "v1_rolling5")
        mlflow.set_tag("data_source",  features_path)

        print(f"\n{'Epoch':>6}  {'Train loss':>11}  {'Val loss':>10}  {'Val MAE':>9}")
        print("─" * 44)

        # ⑥ Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = criterion(val_pred, yv).item()
                val_mae  = (val_pred - yv).abs().mean().item()

            scheduler.step(val_loss)

            # ◆ MLflow - log metrics every epoch (step= gives you the curve)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_mae":    val_mae,
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % print_every == 0 or epoch == 1:
                print(f"{epoch:>6}  {train_loss:>11.6f}  {val_loss:>10.6f}  {val_mae:>9.4f}")

        #  Restore best weights
        model.load_state_dict(best_state)
        os.makedirs(model_dir, exist_ok=True)

        # — Still save .pt file locally (game.py loads this directly)
        model_path  = os.path.join(model_dir, "difficulty_model.pt")
        scaler_path = os.path.join(model_dir, "scaler.json")

        torch.save(model.state_dict(), model_path)
        with open(scaler_path, "w") as f:
            json.dump(
                {
                    "mean": means,
                    "scale": stds,
                    "feature_cols": FEATURE_COLS,
                },
                f,
                indent=2,
            )

        # ◆ MLflow - log best summary metrics
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_mae":  (model(Xv) - yv).abs().mean().item(),
        })

        # ◆ MLflow — log scaler.json as artifact (stored inside the run)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        # ◆ MLflow — log the PyTorch model (enables model registry later)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="difficulty_mlp",
            # Registers model by name — visible in MLflow Model Registry UI
            registered_model_name="DifficultyMLP",
        )

        print(f"\n[train] best val loss  : {best_val_loss:.6f}")
        print(f"[train] model          → {model_path}")
        print(f"[train] scaler         → {scaler_path}")
        print(f"[train] MLflow run id  → {run.info.run_id}")
        print(f"[train] MLflow UI      → http://localhost:5000")
        print(f"\n[train] next → run: mlflow ui")


# ─  Inference helpers (used by game.py)

def load_model(model_dir: str = DEFAULT_MODEL_DIR):
    """
    Load trained model + scaler from local files.
    Called once at game startup — unchanged from before.
    game.py does NOT need to know about MLflow.
    """
    model_path  = os.path.join(model_dir, "difficulty_model.pt")
    scaler_path = os.path.join(model_dir, "scaler.json")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    model = DifficultyMLP(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with open(scaler_path) as f:
        scaler = json.load(f)

    return model, scaler["means"], scaler["stds"], scaler["feature_cols"]


def predict(model, means, stds, feature_cols, features: dict) -> float:
    """
    Predict difficulty for one player session.
    Unchanged — game.py calls this exactly as before.
    """
    row = [(features[c] - m) / s for c, m, s in zip(feature_cols, means, stds)]
    x   = torch.tensor([row], dtype=torch.float32)
    with torch.no_grad():
        return float(model(x).item())


# ── CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the difficulty MLP.")
    parser.add_argument("--features",   default=DEFAULT_FEATURES)
    parser.add_argument("--model-dir",  default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--batch-size", type=int,   default=16)
    args = parser.parse_args()

    train(
        features_path=args.features,
        model_dir=args.model_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )