"""
Day 2 — Upload sessions.csv to Azure Blob Storage and
register it as a Data Asset in Azure ML.
"""

import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.storage.blob import BlobServiceClient

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


# ── BLOB CLIENT ───────────────────────────────────────────────────────────────

def get_blob_service_client() -> BlobServiceClient:
    """Connect to Azure Blob Storage using CI-safe authentication."""
    account_url = f"https://{AZURE_CONFIG['storage_account']}.blob.core.windows.net"

    credential = get_credential()

    client = BlobServiceClient(
        account_url=account_url,
        credential=credential
    )

    print(f"[upload] Connected to storage: {AZURE_CONFIG['storage_account']}")
    return client


# ── CONTAINER ────────────────────────────────────────────────────────────────

def create_container_if_missing(blob_service: BlobServiceClient) -> None:
    container_name = AZURE_CONFIG["data_container"]
    try:
        blob_service.create_container(container_name)
        print(f"[upload] Container created : {container_name}")
    except Exception:
        print(f"[upload] Container exists  : {container_name}")


# ── UPLOAD ────────────────────────────────────────────────────────────────────

def upload_csv(blob_service: BlobServiceClient) -> str:
    local_path = Path("data/sessions.csv")
    if not local_path.exists():
        raise FileNotFoundError("data/sessions.csv not found")

    container = blob_service.get_container_client(AZURE_CONFIG["data_container"])
    blob_name = "sessions/sessions.csv"

    with open(local_path, "rb") as f:
        container.upload_blob(blob_name, f, overwrite=True)

    blob_uri = (
        f"https://{AZURE_CONFIG['storage_account']}.blob.core.windows.net/"
        f"{AZURE_CONFIG['data_container']}/{blob_name}"
    )

    print(f"[upload] CSV uploaded → {blob_uri}")
    return blob_uri


# ── REGISTER DATASET ─────────────────────────────────────────────────────────

def register_dataset(blob_uri: str) -> None:
    credential = get_credential()

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

    print(f"[upload] Dataset registered : {registered.name} v{registered.version}")
    print(f"[upload] Dataset path       : {registered.path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("[upload] Connecting to blob storage...")
    blob_service = get_blob_service_client()

    print("[upload] Creating container if missing...")
    create_container_if_missing(blob_service)

    print("[upload] Uploading sessions.csv...")
    blob_uri = upload_csv(blob_service)

    print("[upload] Registering dataset in Azure ML...")
    register_dataset(blob_uri)

    print("\n[upload] Done!")


if __name__ == "__main__":
    main()