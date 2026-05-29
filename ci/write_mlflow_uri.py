# ci/write_mlflow_uri.py
"""
Fetches the Azure ML MLflow tracking URI during CI
and writes it to .mlflow_uri so train_azure.py can read it.
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from pathlib import Path

# In CI, azure/login action sets env vars automatically
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="18d3a1a7-94e1-471d-a137-a704f081dee6",
    resource_group_name="rg-game-ai-pipeline",
    workspace_name="game-ai-mlops",
)

ws  = ml_client.workspaces.get("game-ai-mlops")
uri = ws.mlflow_tracking_uri
Path(".mlflow_uri").write_text(uri)
print(f"[ci] ✅ MLflow URI written: {uri[:60]}...")