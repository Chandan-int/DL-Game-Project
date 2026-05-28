# create_workspace.py
"""Creates Azure ML workspace using Python SDK — bypasses CLI extension."""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "18d3a1a7-94e1-471d-a137-a704f081dee6"
RESOURCE_GROUP  = "rg-game-ai-pipeline"
WORKSPACE_NAME  = "game-ai-mlops"
LOCATION        = "centralindia"

def create_workspace() -> None:
    print("[setup] Authenticating...")
    credential = DefaultAzureCredential()

    # MLClient without workspace_name creates at subscription level
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
    )

    print("[setup] Creating workspace — this takes 3-5 minutes...")
    ws = Workspace(
        name=WORKSPACE_NAME,
        location=LOCATION,
        display_name="Game AI DDA Pipeline",
        description="Dynamic difficulty adaptation MLOps pipeline",
        tags={"project": "game-ai-dda", "phase": "week5"},
    )

    created = ml_client.workspaces.begin_create(ws).result()  # blocks until done

    print(f"[setup] ✅ Workspace created : {created.name}")
    print(f"[setup] ✅ Location          : {created.location}")
    print(f"[setup] ✅ MLflow URI        : {created.mlflow_tracking_uri}")

    from pathlib import Path
    Path(".mlflow_uri").write_text(created.mlflow_tracking_uri)
    print(f"[setup] ✅ MLflow URI saved  → .mlflow_uri")

if __name__ == "__main__":
    create_workspace()