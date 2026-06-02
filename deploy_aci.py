# deploy_aci.py
"""
Deploy scoring container to Azure Container Instances.
No VM quota needed — works on student subscriptions.
"""

import json
import subprocess
from pathlib import Path
from azure_config import AZURE_CONFIG

REGISTRY_NAME = "gameaidda"          # Azure Container Registry name
IMAGE_NAME    = "difficulty-predictor"
ACI_NAME      = "difficulty-predictor-aci"
PORT          = 5001


def run(cmd: str) -> str:
    """Run az CLI command, return stdout."""
    print(f"[aci] $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{result.stderr}")
    return result.stdout.strip()


def main() -> None:
    rg  = AZURE_CONFIG["resource_group"]
    loc = AZURE_CONFIG["location"]

    # 1 — Copy model into scoring folder
    import shutil
    Path("scoring").mkdir(exist_ok=True)
    shutil.copy("models/difficulty_model.pt", "scoring/difficulty_model.pt")
    print("[aci] ✅ Model copied to scoring/")

    # 2 — Create Azure Container Registry
    # 2 — Create Azure Container Registry
    print("\n[aci] Creating container registry...")
    run(
        f'az acr create '
        f'--name {REGISTRY_NAME} '
        f'--resource-group {rg} '
        f'--location {loc} '
        f'--sku Basic '
        f'--admin-enabled true'
    )
    print(f"[aci] ✅ Registry: {REGISTRY_NAME}")

    # 3 — Build + push image using ACR Tasks (no local Docker needed)
    print("\n[aci] Building image in Azure (no local Docker needed)...")
    run(
        f'az acr build '
        f'--registry {REGISTRY_NAME} '
        f'--image {IMAGE_NAME}:latest '
        f'./scoring '
        f'--no-logs'
    )
    print(f"[aci] ✅ Image built: {REGISTRY_NAME}.azurecr.io/{IMAGE_NAME}:latest")

    # 4 — Get registry credentials
    creds = json.loads(run(
        f'az acr credential show --name {REGISTRY_NAME} --output json'
    ))
    registry_password = creds["passwords"][0]["value"]
    registry_server   = f"{REGISTRY_NAME}.azurecr.io"

    # 5 — Deploy to ACI
    print("\n[aci] Deploying to Azure Container Instances...")
    run(
        f"az container create "
        f"--resource-group {RG} "
        f"--name {ACI_NAME} "
        f"--image {REGISTRY_NAME}.azurecr.io/{IMAGE_NAME}:latest "
        f"--registry-login-server {REGISTRY_NAME}.azurecr.io "
        f"--registry-username {REGISTRY_NAME} "
        f'--registry-password "{acr_password}" '
        f"--ports {PORT} "
        f"--ip-address Public "
        f"--cpu 1 --memory 1.5 "
        f"--location {LOCATION} "
        f"--os-type Linux"
    )


    # 6 — Get public IP
    ip = run(
        f'az container show '
        f'--resource-group {rg} --name {ACI_NAME} '
        f'--query ipAddress.ip --output tsv'
    )
    if not ip:
        raise RuntimeError(
            "ACI has no public IP assigned. "
            "Check deployment configuration."
        )

    endpoint_url = f"http://{ip}:{PORT}/score"
    print(f"\n[aci] ✅ Container running at : {ip}")
    print(f"[aci] ✅ Endpoint URL         : {endpoint_url}")

    # 7 — Save config for game.py
    config = {"endpoint_url": endpoint_url, "api_key": ""}
    Path("models/endpoint_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"[aci] ✅ Config saved → models/endpoint_config.json")

    print(f"\n[aci] 🎉 Day 5 complete!")
    print(f"         Test with:")
    print(f"         python test_endpoint.py")


if __name__ == "__main__":
    main()