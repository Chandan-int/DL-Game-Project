# fix_endpoint_config.py
"""Update endpoint_config.json with the real ACI IP."""

import json
import subprocess
from pathlib import Path

result = subprocess.run(
    'az container show '
    '--resource-group rg-game-ai-pipeline '
    '--name difficulty-predictor-aci '
    '--query ipAddress.ip --output tsv',
    shell=True, capture_output=True, text=True
)

ip = result.stdout.strip()
if not ip:
    print("❌ IP still empty — container not ready yet")
    print("   Run: az container show ... to check state")
else:
    url = f"http://{ip}:5001/score"
    config = {"endpoint_url": url, "api_key": ""}
    Path("models/endpoint_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"✅ IP found     : {ip}")
    print(f"✅ Endpoint URL : {url}")
    print(f"✅ Config saved → models/endpoint_config.json")