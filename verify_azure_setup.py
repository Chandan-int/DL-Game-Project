# verify_azure_setup.py
"""Confirm Azure ML workspace is reachable from Python SDK."""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure_config import AZURE_CONFIG

def verify() -> None:
    print("[azure] Connecting to workspace...")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_CONFIG["subscription_id"],
        resource_group_name=AZURE_CONFIG["resource_group"],
        workspace_name=AZURE_CONFIG["workspace_name"],
    )

    ws = ml_client.workspaces.get(AZURE_CONFIG["workspace_name"])
    print(f"[azure] ✅ Workspace     : {ws.name}")
    print(f"[azure] ✅ Location      : {ws.location}")
    print(f"[azure] ✅ Resource group: {ws.resource_group}")
    print(f"[azure] ✅ MLflow URI    : {ws.mlflow_tracking_uri}")

    # Save the MLflow URI — needed for Week 5 experiment tracking
    from pathlib import Path
    Path(".mlflow_uri").write_text(ws.mlflow_tracking_uri)
    print(f"[azure] ✅ MLflow URI saved to .mlflow_uri")

if __name__ == "__main__":
    verify()