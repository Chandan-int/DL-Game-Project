# upload_data.py
"""
Day 2 — Upload sessions.csv to Azure Blob Storage and
register it as a Data Asset in Azure ML.
Uses DefaultAzureCredential directly — no storage management SDK needed.
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pathlib import Path
from azure_config import AZURE_CONFIG


def get_blob_service_client() -> BlobServiceClient:
    """
    Connect to blob storage using DefaultAzureCredential.
    No keys, no connection strings — uses your az login session.
    """
    account_url = (
        f"https://{AZURE_CONFIG['storage_account']}.blob.core.windows.net"
    )
    credential = DefaultAzureCredential()
    client = BlobServiceClient(account_url=account_url, credential=credential)
    print(f"[upload] ✅ Connected to storage: {AZURE_CONFIG['storage_account']}")
    return client


def create_container_if_missing(blob_service: BlobServiceClient) -> None:
    """Create the game-telemetry container if it does not exist."""
    container_name = AZURE_CONFIG["data_container"]
    try:
        blob_service.create_container(container_name)
        print(f"[upload] ✅ Container created : {container_name}")
    except Exception:
        print(f"[upload] ✅ Container exists  : {container_name}")


def upload_csv(blob_service: BlobServiceClient) -> str:
    """Upload sessions.csv to blob storage. Returns blob URI."""
    local_path = Path("data/sessions.csv")
    if not local_path.exists():
        raise FileNotFoundError(
            "data/sessions.csv not found — play at least one game session first!"
        )

    container = blob_service.get_container_client(AZURE_CONFIG["data_container"])
    blob_name = "sessions/sessions.csv"

    with open(local_path, "rb") as f:
        container.upload_blob(blob_name, f, overwrite=True)

    blob_uri = (
        f"https://{AZURE_CONFIG['storage_account']}.blob.core.windows.net"
        f"/{AZURE_CONFIG['data_container']}/{blob_name}"
    )
    print(f"[upload] ✅ CSV uploaded  → {blob_uri}")
    return blob_uri


def register_dataset(blob_uri: str) -> None:
    """Register the uploaded CSV as an Azure ML Data Asset."""
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=AZURE_CONFIG["subscription_id"],
        resource_group_name=AZURE_CONFIG["resource_group"],
        workspace_name=AZURE_CONFIG["workspace_name"],
    )

    data_asset = Data(
        name="game-sessions",
        version="1",
        description="Player telemetry sessions from Pygame DDA game",
        type=AssetTypes.URI_FILE,
        path=blob_uri,
        tags={"project": "game-ai-dda", "phase": "week5"},
    )

    registered = ml_client.data.create_or_update(data_asset)
    print(f"[upload] ✅ Dataset registered : {registered.name} v{registered.version}")
    print(f"[upload] ✅ Dataset path       : {registered.path}")


def main() -> None:
    print("[upload] Connecting to blob storage...")
    blob_service = get_blob_service_client()

    print("[upload] Creating container if missing...")
    create_container_if_missing(blob_service)

    print("[upload] Uploading sessions.csv...")
    blob_uri = upload_csv(blob_service)

    print("[upload] Registering as Azure ML dataset...")
    register_dataset(blob_uri)

    print("\n[upload] Day 2 complete!")
    print("         View dataset at: https://ml.azure.com")
    print(f"         Workspace: {AZURE_CONFIG['workspace_name']}")


if __name__ == "__main__":
    main()