import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data\config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

DATASET_CONFIG = CONFIG["datasets"]
DEFAULTS = CONFIG.get("defaults", {"labels": []})


def resolve_dataset_config(research_object: str):
    """Match dataset configuration in config.json based on research_object text"""
    ro = (research_object or "").lower()
    for cfg in DATASET_CONFIG:
        if any(alias in ro for alias in cfg.get("aliases", [])):
            labels = cfg.get("labels", DEFAULTS["labels"])
            seed = cfg.get("seed", None)   # Note: may not exist
            return {"labels": labels, "seed": seed}
    return {"labels": DEFAULTS["labels"], "seed": None}