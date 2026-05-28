# deploy_endpoint.py
"""
Day 5 — Deploy DifficultyMLP v1 as an Azure ML Managed Endpoint.
Gives us a REST URL: POST telemetry → get difficulty score back.
"""

import json
import time
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)
from azure.identity import DefaultAzureCredential
from azure_config import AZURE_CONFIG
from pathlib import Path


def get_ml_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_CONFIG["subscription_id"],
        resource_group_name=AZURE_CONFIG["resource_group"],
        workspace_name=AZURE_CONFIG["workspace_name"],
    )


def create_endpoint(ml_client: MLClient) -> str:
    """Create the managed online endpoint. Returns endpoint URL."""
    endpoint = ManagedOnlineEndpoint(
        name=AZURE_CONFIG["endpoint_name"],
        description="Game AI difficulty prediction endpoint",
        auth_mode="key",
        tags={"project": "game-ai-dda", "phase": "week5"},
    )

    print(f"[deploy] Creating endpoint: {AZURE_CONFIG['endpoint_name']}")
    print(f"[deploy] This takes 3-5 minutes...")

    poller  = ml_client.online_endpoints.begin_create_or_update(endpoint)
    result  = poller.result()   # blocks until done

    print(f"[deploy] ✅ Endpoint created : {result.name}")
    print(f"[deploy] ✅ Endpoint URL     : {result.scoring_uri}")
    return result.scoring_uri


def create_deployment(ml_client: MLClient) -> None:
    """Deploy DifficultyMLP v1 to the endpoint."""

    # Get the registered model
    model = ml_client.models.get(
        name=AZURE_CONFIG["model_name"],
        version="1",
    )
    print(f"[deploy] Model: {model.name} v{model.version}")

    deployment = ManagedOnlineDeployment(
        name=AZURE_CONFIG["deployment_name"],   # "blue"
        endpoint_name=AZURE_CONFIG["endpoint_name"],
        model=model,
        instance_type="Standard_DS3_v2",   # smallest available
        instance_count=1,
    )

    print(f"[deploy] Creating deployment: {AZURE_CONFIG['deployment_name']}")
    print(f"[deploy] This takes 8-15 minutes...")

    poller = ml_client.online_deployments.begin_create_or_update(deployment)
    result = poller.result()   # blocks until done

    print(f"[deploy] ✅ Deployment ready : {result.name}")


def set_traffic(ml_client: MLClient) -> None:
    """Route 100% of traffic to the blue deployment."""
    endpoint = ml_client.online_endpoints.get(AZURE_CONFIG["endpoint_name"])
    endpoint.traffic = {AZURE_CONFIG["deployment_name"]: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"[deploy] ✅ Traffic → 100% to '{AZURE_CONFIG['deployment_name']}'")


def get_api_key(ml_client: MLClient) -> str:
    """Retrieve the endpoint API key."""
    keys = ml_client.online_endpoints.get_keys(AZURE_CONFIG["endpoint_name"])
    return keys.primary_key


def save_endpoint_config(scoring_uri: str, api_key: str) -> None:
    """Save endpoint URL + key so game.py can load them."""
    config = {
        "endpoint_url": scoring_uri,
        "api_key":      api_key,
    }
    Path("models/endpoint_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"[deploy] ✅ Endpoint config saved → models/endpoint_config.json")


def test_endpoint(scoring_uri: str, api_key: str) -> None:
    """Send a test prediction to confirm the endpoint is working."""
    import urllib.request

    # Sample telemetry — normalized to 0-1 range
    test_input = {
        "input_data": {
            "columns": [
                "level", "deaths", "accuracy",
                "avg_reaction_time_ms", "completion_time_sec",
                "score", "difficulty_level"
            ],
            "data": [[0.0, 0.5, 0.35, 0.184, 0.647, 0.08, 0.5]]
        }
    }

    body    = json.dumps(test_input).encode("utf-8")
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req      = urllib.request.Request(scoring_uri, data=body, headers=headers)
    response = urllib.request.urlopen(req, timeout=30)
    result   = json.loads(response.read())

    print(f"[deploy] ✅ Test prediction : {result}")
    print(f"[deploy] ✅ Endpoint is live and responding!")


def main() -> None:
    print("[deploy] Connecting to Azure ML...")
    ml_client = get_ml_client()

    # Step 1 — create endpoint
    scoring_uri = create_endpoint(ml_client)

    # Step 2 — deploy model to endpoint
    create_deployment(ml_client)

    # Step 3 — route traffic
    set_traffic(ml_client)

    # Step 4 — get API key + save config
    print("[deploy] Retrieving API key...")
    api_key = get_api_key(ml_client)
    save_endpoint_config(scoring_uri, api_key)

    # Step 5 — smoke test
    print("[deploy] Testing endpoint...")
    test_endpoint(scoring_uri, api_key)

    print(f"\n[deploy] 🎉 Day 5 complete!")
    print(f"         Endpoint : {scoring_uri}")
    print(f"         Config   : models/endpoint_config.json")
    print(f"         View at  : https://ml.azure.com → Endpoints")


if __name__ == "__main__":
    main()