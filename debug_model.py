# debug_model.py
"""Inspect the loaded model architecture and output range."""

import torch
import mlflow.pytorch
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pytorch.load_model("models:/DifficultyMLP/latest")
model.eval()

print("── Model architecture ───────────────────")
print(model)

print("\n── Output layer ─────────────────────────")
# Check the last layer — does it have sigmoid?
layers = list(model.children())
print(f"Last layer: {layers[-1]}")

print("\n── Test with zeros input ────────────────")
with torch.no_grad():
    x = torch.zeros(1, 7)
    out = model(x).item()
    print(f"Output for all-zeros input: {out:.6f}")

print("\n── Test with random inputs ──────────────")
with torch.no_grad():
    for i in range(5):
        x = torch.randn(1, 7)
        out = model(x).item()
        print(f"  Random input {i+1}: {out:.6f}")