# train_azure.py
"""
Week 5 — train.py pointed at Azure ML for experiment tracking.
Only difference from train.py: mlflow.set_tracking_uri uses Azure ML.
Everything else (model, data, features) is identical.
"""

import json
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


# ── Azure ML tracking URI ─────────────────────────────────────────────────────

def get_azure_tracking_uri() -> str:
    """Read the MLflow URI saved by verify_azure_setup.py."""
    uri_file = Path(".mlflow_uri")
    if not uri_file.exists():
        raise FileNotFoundError(
            ".mlflow_uri not found — run verify_azure_setup.py first"
        )
    return uri_file.read_text().strip()


# ── Config — identical to train.py ───────────────────────────────────────────

FEATURE_COLS = [
    "level", "deaths", "accuracy",
    "avg_reaction_time_ms", "completion_time_sec",
    "score", "difficulty_level",
]
TARGET_COL   = "difficulty_level"
DATA_PATH    = Path("data/sessions.csv")
MODEL_DIR    = Path("models")

HPARAMS = {
    "hidden_1":   32,
    "hidden_2":   16,
    "dropout":    0.1,
    "lr":         1e-3,
    "batch_size": 16,
    "epochs":     500,
    "val_frac":   0.2,
    "loss_fn":    "MSELoss",
    "optimizer":  "Adam",
}


# ── Model — identical to game.py and train.py ─────────────────────────────────

class DifficultyMLP(nn.Module):
    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HPARAMS["hidden_1"]),
            nn.ReLU(),
            nn.Dropout(HPARAMS["dropout"]),
            nn.Linear(HPARAMS["hidden_1"], HPARAMS["hidden_2"]),
            nn.ReLU(),
            nn.Linear(HPARAMS["hidden_2"], 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Feature ranges for min-max normalization ──────────────────────────────────

FEATURE_RANGES = {
    "level":                (1,    10),
    "deaths":               (0,    10),
    "accuracy":             (0.0,  1.0),
    "avg_reaction_time_ms": (100,  2000),
    "completion_time_sec":  (10,   180),
    "score":                (0,    10000),
    "difficulty_level":     (0.0,  1.0),
}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each feature column using known game ranges."""
    out = df.copy()
    for col in FEATURE_COLS:
        lo, hi = FEATURE_RANGES[col]
        out[col] = (df[col] - lo) / (hi - lo + 1e-8)
        out[col] = out[col].clip(0.0, 1.0)
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    # ① Point MLflow at Azure ML
    tracking_uri = get_azure_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("difficulty-adaptation")
    print(f"[train] MLflow → Azure ML ({tracking_uri[:60]}...)")

    # ② Load + normalize data
    df = pd.read_csv(DATA_PATH)
    # Drop rows where difficulty_level is the target itself
    # (use previous session's difficulty as a feature, predict next)
    df = df[FEATURE_COLS].dropna()
    print(f"[train] loaded {len(df)} samples")

    df_norm = normalize_df(df)
    X = df_norm[FEATURE_COLS].values.astype(np.float32)  # 7 features
    y = df_norm[TARGET_COL].values.astype(np.float32)

    # Train/val split
    n_val   = max(1, int(len(X) * HPARAMS["val_frac"]))
    n_train = len(X) - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    print(f"[train] train={n_train}, val={n_val}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=HPARAMS["batch_size"], shuffle=True)

    # ③ Model + optimizer
    model     = DifficultyMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
    loss_fn   = nn.MSELoss()

    # ④ Train with Azure ML MLflow tracking
    with mlflow.start_run() as run:
        mlflow.log_params({**HPARAMS, "input_dim": X.shape[1],
                           "n_train": n_train, "n_val": n_val})

        best_val_loss = float("inf")
        best_state    = None

        print(f"\n{'Epoch':>8} {'Train':>12} {'Val':>12} {'Val MAE':>10}")
        print("─" * 46)

        for epoch in range(1, HPARAMS["epochs"] + 1):
            # Train
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss_fn(model(xb), yb).backward()
                optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                val_pred  = model(torch.from_numpy(X_val))
                val_loss  = loss_fn(val_pred, torch.from_numpy(y_val)).item()
                val_mae   = torch.mean(torch.abs(val_pred - torch.from_numpy(y_val))).item()
                train_out = model(torch.from_numpy(X_train))
                train_loss = loss_fn(train_out, torch.from_numpy(y_train)).item()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_mae":    val_mae,
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae  = val_mae
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 30 == 0 or epoch == 1:
                print(f"{epoch:>8}     {train_loss:>10.6f}   {val_loss:>10.6f}   {val_mae:>8.4f}")

        # ⑤ Save best model
        model.load_state_dict(best_state)
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_mae":  best_val_mae,
        })

        # Log to Azure ML model registry
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=AZURE_CONFIG["model_name"]
        )

        # Also save locally for game.py
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_DIR / "difficulty_model.pt")

        print(f"\n[train] best val loss  : {best_val_loss:.6f}")
        print(f"[train] best val MAE   : {best_val_mae:.4f}")
        print(f"[train] MLflow run id  → {run.info.run_id}")
        print(f"[train] View at        → https://ml.azure.com")
        print(f"[train] Workspace      → {HPARAMS}")


if __name__ == "__main__":
    train()