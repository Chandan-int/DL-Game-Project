# ci/update_aci.py
"""
Rebuilds the ACI container with the latest model weights
after retraining. Called by GitHub Actions after register_model.py.
"""

import os
import subprocess
import shutil
from pathlib import Path
import json

REGISTRY_NAME = "gameaidda"
IMAGE_NAME    = "difficulty-predictor"
ACI_NAME      = "difficulty-predictor-aci"
RG            = "rg-game-ai-pipeline"
LOCATION      = "centralindia"
PORT          = 5001


def run(cmd: str) -> str:
    print(f"[aci-update] $ {cmd}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{result.stderr}"
        )

    return result.stdout.strip()


def main() -> None:
    # Copy latest model into scoring folder
    shutil.copy("models/difficulty_model.pt", "scoring/difficulty_model.pt")
    print("[aci-update] ✅ Model copied to scoring/")

    # Rebuild image with new model weights
    run(f"az acr build --registry {REGISTRY_NAME} "
        f"--image {IMAGE_NAME}:latest ./scoring")
    print("[aci-update] ✅ Image rebuilt")

    # Get registry credentials
    # import  json
    creds = json.loads(
        run(f"az acr credential show --name {REGISTRY_NAME}")
    )

    acr_username = creds["username"]
    acr_password = creds["passwords"][0]["value"]

    # Delete old container
    try:
        run(f"az container delete --resource-group {RG} "
            f"--name {ACI_NAME} --yes")
        print("[aci-update] ✅ Old container deleted")
    except Exception:
        print("[aci-update] No existing container to delete")

    # Redeploy with new image
    run(
        f"az container create "
        f"--resource-group {rg} "
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
    print("[aci-update] ✅ New container deployed")

    # Get new IP
    ip = run(
        f"az container show --resource-group {RG} "
        f"--name {ACI_NAME} --query ipAddress.ip --output tsv"
    )
    if not ip:
        raise RuntimeError(
            "ACI has no public IP assigned. "
            "Check deployment configuration."
        )

    # import json
    config = {"endpoint_url": f"http://{ip}:{PORT}/score", "api_key": ""}
    Path("models/endpoint_config.json").write_text(json.dumps(config, indent=2))
    print(f"[aci-update] ✅ New endpoint: http://{ip}:{PORT}/score")


if __name__ == "__main__":
    main()