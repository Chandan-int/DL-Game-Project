# ci/write_mlflow_uri.py
"""
Fetches Azure ML MLflow tracking URI during CI and writes to .mlflow_uri.
Uses explicit service principal credentials from environment variables
set by the azure/login action.
"""

import os
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from pathlib import Path

# These env vars are set automatically by azure/login action
tenant_id       = os.environ["AZURE_TENANT_ID"]
client_id       = os.environ["AZURE_CLIENT_ID"]
client_secret   = os.environ["AZURE_CLIENT_SECRET"]
subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]

credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret,
)

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name="rg-game-ai-pipeline",
    workspace_name="game-ai-mlops-v2",
)

ws  = ml_client.workspaces.get("game-ai-mlops-v2")
uri = ws.mlflow_tracking_uri
Path(".mlflow_uri").write_text(uri)
print(f"[ci] ✅ MLflow URI written: {uri[:60]}...")