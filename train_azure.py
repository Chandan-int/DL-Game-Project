"""
Week 5 — train.py pointed at Azure ML for experiment tracking.
"""

import os
import json
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure_config import AZURE_CONFIG


# ── AUTH HELPER (CI + Local) ──────────────────────────────────────────────────

def get_credential():
    """Use service principal in CI, DefaultAzureCredential locally."""
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    tenant_id = os.environ.get("AZURE_TENANT_ID")

    if client_id and client_secret and tenant_id:
        print("[auth] Using ClientSecretCredential (CI mode)")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

    print("[auth] Using DefaultAzureCredential (local mode)")
    return DefaultAzureCredential()


# ── AZURE ML TRACKING URI ────────────────────────────────────────────────────

def get_azure_tracking_uri() -> str:
    uri_file = Path(".mlflow_uri")
    if not uri_file.exists():
        raise FileNotFoundError(".mlflow_uri not found")
    return uri_file.read_text().strip()


# ── CONFIG ───────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "level", "deaths", "accuracy",
    "avg_reaction_time_ms", "completion_time_sec",
    "score", "difficulty_level",
]

TARGET_COL = "difficulty_level"
DATA_PATH = Path("data/sessions.csv")
MODEL_DIR = Path("models")


# ── MODEL ────────────────────────────────────────────────────────────────────

class DifficultyMLP(nn.Module):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── NORMALIZATION ─────────────────────────────────────────────────────────────

FEATURE_RANGES = {
    "level": (1, 10),
    "deaths": (0, 10),
    "accuracy": (0.0, 1.0),
    "avg_reaction_time_ms": (100, 2000),
    "completion_time_sec": (10, 180),
    "score": (0, 10000),
    "difficulty_level": (0.0, 1.0),
}


def normalize_df(df):
    out = df.copy()
    for col in FEATURE_COLS:
        lo, hi = FEATURE_RANGES[col]
        out[col] = (df[col] - lo) / (hi - lo + 1e-8)
        out[col] = out[col].clip(0, 1)
    return out


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train():
    tracking_uri = get_azure_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("difficulty-adaptation")

    print(f"[train] MLflow → Azure ML ({tracking_uri[:60]}...)")

    df = pd.read_csv(DATA_PATH).dropna()
    df = df[FEATURE_COLS]

    df = normalize_df(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    n_val = max(1, int(len(X) * 0.2))
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = DifficultyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    with mlflow.start_run() as run:
        best_state = None
        best_loss = float("inf")

        for epoch in range(1, 101):
            model.train()
            for xb, yb in train_dl:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(X_val))
                val_loss = loss_fn(val_pred, torch.from_numpy(y_val)).item()

            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_DIR / "difficulty_model.pt")

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=AZURE_CONFIG["model_name"],
        )

        print(f"[train] best loss: {best_loss:.6f}")
        print(f"[train] run id  : {run.info.run_id}")


if __name__ == "__main__":
    train()