"""
EquiTraffic-GPT MLOps Module: Model Registry & Versioning Engine (gwnet_registry.py)

Manages production model versioning, checkpoint registry manifests, and version switching:
- Registers versioned checkpoints in model_registry.json
- Manages active deployment pointers (e.g. v1.0.0, v1.0.1)
- Exposes get_active_checkpoint() for UniversalPeMSAdapter serving with portable relative path support
"""

import os
import json
import yaml
from datetime import datetime


REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_registry.json")


def load_model_config() -> dict:
    """Load model_config.yaml from backend directory with robust fallback."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, '..', 'backend', 'model_config.yaml'),
        os.path.join(base_dir, 'model_config.yaml')
    ]
    for cfg_path in candidates:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
    return {}


def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_version": "v1.0.0", "datasets": {}}


def save_registry(registry_data: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)


def get_next_version(dataset_name: str) -> str:
    reg = load_registry()
    ds_reg = reg.get("datasets", {}).get(dataset_name, {})
    versions = list(ds_reg.get("versions", {}).keys())
    if not versions:
        return "v1.0.0"
    
    latest = versions[-1]
    try:
        parts = latest.replace("v", "").split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"v{major}.{minor}.{patch + 1}"
    except Exception:
        return f"v1.0.{len(versions)}"


def register_model_version(dataset_name: str, version_str: str, pt_path: str, tar_path: str, metrics: dict, hparams: dict):
    reg = load_registry()
    if "datasets" not in reg:
        reg["datasets"] = {}
        
    if dataset_name not in reg["datasets"]:
        reg["datasets"][dataset_name] = {
            "active_version": version_str,
            "versions": {}
        }
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rel_pt = os.path.relpath(pt_path, base_dir) if os.path.isabs(pt_path) else pt_path
    rel_tar = os.path.relpath(tar_path, base_dir) if os.path.isabs(tar_path) else tar_path

    reg["datasets"][dataset_name]["active_version"] = version_str
    reg["datasets"][dataset_name]["versions"][version_str] = {
        "dataset": dataset_name,
        "version": version_str,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pt_path": pt_path,
        "pt_relative_path": rel_pt,
        "tar_path": tar_path,
        "tar_relative_path": rel_tar,
        "metrics": metrics,
        "hyperparameters": hparams
    }
    save_registry(reg)
    print(f"[+] MLOps Model Registry: Registered version '{version_str}' for dataset '{dataset_name.upper()}' in model_registry.json")


def get_active_checkpoint(dataset_name: str) -> str:
    reg = load_registry()
    ds_reg = reg.get("datasets", {}).get(dataset_name, {})
    active_v = ds_reg.get("active_version")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if active_v and active_v in ds_reg.get("versions", {}):
        ver_info = ds_reg["versions"][active_v]
        pt_rel = ver_info.get("pt_relative_path")
        pt_path = ver_info.get("pt_path")
        
        if pt_rel and os.path.exists(os.path.join(base_dir, pt_rel)):
            return os.path.join(base_dir, pt_rel)
        if pt_path and os.path.exists(pt_path):
            return pt_path
    
    clean_ds = dataset_name.lower().replace("_", "").replace("-", "")
    fallback = os.path.join(base_dir, f"gwnet_{dataset_name}.pt")
    if not os.path.exists(fallback):
        fallback = os.path.join(base_dir, f"gwnet_{clean_ds}.pt")

    return fallback
