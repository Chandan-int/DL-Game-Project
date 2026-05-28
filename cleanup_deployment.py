# cleanup_deployment.py
"""Delete the failed deployment so we can redeploy cleanly."""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure_config import AZURE_CONFIG


def cleanup() -> None:
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_CONFIG["subscription_id"],
        resource_group_name=AZURE_CONFIG["resource_group"],
        workspace_name=AZURE_CONFIG["workspace_name"],
    )

    # Delete failed deployment first
    try:
        print("[cleanup] Deleting failed deployment 'blue'...")
        ml_client.online_deployments.begin_delete(
            name="blue",
            endpoint_name=AZURE_CONFIG["endpoint_name"],
        ).result()
        print("[cleanup] ✅ Deployment deleted")
    except Exception as e:
        print(f"[cleanup] Deployment delete skipped: {e}")

    # Keep the endpoint — reuse it for new deployment
    print("[cleanup] ✅ Endpoint kept — ready for redeployment")


if __name__ == "__main__":
    cleanup()