# azure_config.py
"""
Central config for all Azure ML settings.
Never commit real keys — these are non-secret resource identifiers only.
"""

AZURE_CONFIG = {
    # Azure identifiers
    "subscription_id":  "18d3a1a7-94e1-471d-a137-a704f081dee6",
    "resource_group":   "rg-game-ai-pipeline",
    "workspace_name":   "game-ai-mlops-v2",
    "location":         "centralindia",

    # Blob storage  ← fill in after Step 4
    "storage_account":  "gameaimlstorage6275e6439",          # e.g. "gameaimlops1234567"
    "data_container":   "game-telemetry",

    # Model registry
    "model_name":       "DifficultyMLP",

    # Endpoint (filled in Day 5)
    "endpoint_name":    "difficulty-endpoint",
    "deployment_name":  "blue",
}