# test_endpoint.py
"""Verify the ACI endpoint returns a valid difficulty score."""

import json
import urllib.request
from pathlib import Path

config = json.loads(Path("models/endpoint_config.json").read_text())
url    = config["endpoint_url"]

payload = {
    "level": 2, "deaths": 3, "accuracy": 0.58,
    "avg_reaction_time_ms": 320, "completion_time_sec": 95,
    "score": 1200, "difficulty_level": 0.6,
}

body    = json.dumps(payload).encode("utf-8")
req     = urllib.request.Request(
    url, data=body,
    headers={"Content-Type": "application/json"}
)
resp   = urllib.request.urlopen(req, timeout=15)
result = json.loads(resp.read())

print(f"✅ Response : {result}")
print(f"✅ Score    : {result['difficulty_score']}")
assert 0.0 < result["difficulty_score"] < 1.0
print("✅ Endpoint is working correctly!")