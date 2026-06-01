# register_model.py
"""
Day 4 — Register best model into Azure ML Model Registry.
Builds the model URI directly from run_id — no artifact_uri needed.
"""

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure_config import AZURE_CONFIG
from pathlib import Path

print("REGISTER_MODEL VERSION: 2026-06-01-FIXED")
def get_ml_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_CONFIG["subscription_id"],
        resource_group_name=AZURE_CONFIG["resource_group"],
        workspace_name=AZURE_CONFIG["workspace_name"],
    )


def find_best_run_id() -> tuple[str, float]:
    """Return (run_id, best_val_loss) for the best FINISHED run."""
    tracking_uri = Path(".mlflow_uri").read_text().strip()
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name("difficulty-adaptation")
    if experiment is None:
        raise RuntimeError("Experiment 'difficulty-adaptation' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=20,
    )

    print("\nColumns returned by MLflow:")
    for c in runs.columns:
        print(c)

    finished = runs[
        (runs["status"] == "FINISHED") &
        (runs["metrics.val_loss"].notna())
        ]

    if finished.empty:
        raise RuntimeError("No finished runs with val_loss found")

    best = finished.loc[finished["metrics.val_loss"].idxmin()]
    run_id = best["run_id"]
    val_loss = best["metrics.val_loss"]

    print(f"[register] Best run      : {run_id}")
    print(f"[register] best_val_loss : {val_loss:.6f}")
    return run_id, val_loss


def register_model(ml_client: MLClient, run_id: str) -> str:
    """
    Register model using azureml://jobs/<run_id>/outputs/artifacts/paths/model
    This is the standard Azure ML URI format for MLflow logged models.
    """
    # Standard Azure ML path for MLflow artifacts logged inside a run
    model_uri = (
        f"azureml://jobs/{run_id}/outputs/artifacts/paths/model"
    )
    print(f"[register] Model URI     : {model_uri}")

    model = Model(
        path=model_uri,
        name=AZURE_CONFIG["model_name"],
        description=f"DDA difficulty predictor — run {run_id[:8]}",
        type=AssetTypes.MLFLOW_MODEL,
        tags={
            "stage":     "champion",
            "run_id":    run_id,
            "framework": "pytorch",
            "project":   "game-ai-dda",
        },
    )

    registered = ml_client.models.create_or_update(model)
    print(f"[register] ✅ Registered : {registered.name} v{registered.version}")
    return registered.version


def main() -> None:
    print("[register] Connecting to Azure ML...")
    ml_client = get_ml_client()

    print("[register] Finding best run...")
    run_id, val_loss = find_best_run_id()

    print("[register] Registering model in Azure ML registry...")
    version = register_model(ml_client, run_id)

    print(f"\n[register] Day 4 complete!")
    print(f"           Model   : {AZURE_CONFIG['model_name']} v{version}")
    print(f"           Stage   : champion")
    print(f"           Loss    : {val_loss:.6f}")
    print(f"           View at : https://ml.azure.com")
    print(f"                     → Models → {AZURE_CONFIG['model_name']}")


if __name__ == "__main__":
    main()