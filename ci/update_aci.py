# ci/update_aci.py
"""
Rebuilds the ACI container with the latest model weights
after retraining. Called by GitHub Actions after register_model.py.
"""

import subprocess
import shutil
import json
import time
from pathlib import Path

REGISTRY_NAME = "gameaidda"
IMAGE_NAME = "difficulty-predictor"
ACI_NAME = "difficulty-predictor-aci"
RG = "rg-game-ai-pipeline"
LOCATION = "centralindia"
PORT = 5001


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
    shutil.copy(
        "models/difficulty_model.pt",
        "scoring/difficulty_model.pt"
    )
    print("[aci-update] ✅ Model copied to scoring/")

    # Rebuild image with new model weights
    run(
        f"az acr build "
        f"--registry {REGISTRY_NAME} "
        f"--image {IMAGE_NAME}:latest "
        f"./scoring"
    )
    print("[aci-update] ✅ Image rebuilt")

    # Get ACR credentials
    creds = json.loads(
        run(
            f"az acr credential show "
            f"--name {REGISTRY_NAME} "
            f"--output json"
        )
    )

    registry_password = creds["passwords"][0]["value"]
    registry_server = f"{REGISTRY_NAME}.azurecr.io"

    # Delete old container
    try:
        run(
            f"az container delete "
            f"--resource-group {RG} "
            f"--name {ACI_NAME} "
            f"--yes"
        )
        print("[aci-update] ✅ Old container deleted")

        # Give Azure a few seconds to release the name
        time.sleep(15)

    except Exception:
        print("[aci-update] No existing container to delete")

    # Create new container
    run(
        f"az container create "
        f"--resource-group {RG} "
        f"--name {ACI_NAME} "
        f"--image {registry_server}/{IMAGE_NAME}:latest "
        f"--registry-login-server {registry_server} "
        f"--registry-username {REGISTRY_NAME} "
        f'--registry-password "{registry_password}" '
        f"--ports {PORT} "
        f"--ip-address Public "
        f"--cpu 1 "
        f"--memory 1.5 "
        f"--location {LOCATION} "
        f"--os-type Linux"
    )

    print("[aci-update] ✅ New container deployed")

    # Get endpoint IP
    ip = run(
        f"az container show "
        f"--resource-group {RG} "
        f"--name {ACI_NAME} "
        f"--query ipAddress.ip "
        f"--output tsv"
    )

    if not ip:
        raise RuntimeError(
            "ACI has no public IP assigned. "
            "Check deployment configuration."
        )

    config = {
        "endpoint_url": f"http://{ip}:{PORT}/score",
        "api_key": ""
    }

    Path("models/endpoint_config.json").write_text(
        json.dumps(config, indent=2)
    )

    print(
        f"[aci-update] ✅ New endpoint: "
        f"http://{ip}:{PORT}/score"
    )


if __name__ == "__main__":
    main()