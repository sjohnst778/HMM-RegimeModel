# models/persist.py
# Model serialisation and versioning.
# Saves fitted HMM, BOCPD config, and ensemble as a single versioned bundle
# so you can track what was trained on what data/features.

import pickle
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

MODEL_DIR = Path("models/saved")


def save_model_bundle(
    hmm,
    ensemble,
    feature_names: list[str],
    metadata: dict,
    tag: str = "latest",
) -> Path:
    """
    Save a complete model bundle with metadata.

    Parameters
    ----------
    hmm          : fitted RegimeHMM instance
    ensemble     : fitted DownturnEnsemble instance
    feature_names: list of feature column names used in training
    metadata     : dict of training info (data_source, auc, date_range, etc.)
    tag          : version tag e.g. "yfinance_v1", "eikon_v1", "latest"

    Returns
    -------
    Path to saved bundle directory
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = MODEL_DIR / f"{tag}_{timestamp}"
    bundle_dir.mkdir()

    # Save models
    with open(bundle_dir / "hmm.pkl", "wb") as f:
        pickle.dump(hmm, f)
    with open(bundle_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)

    # Save feature names
    with open(bundle_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metadata
    full_meta = {
        "tag":           tag,
        "timestamp":     timestamp,
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "feature_hash":  hashlib.md5(str(sorted(feature_names)).encode()).hexdigest()[:8],
        **metadata,
    }
    with open(bundle_dir / "metadata.json", "w") as f:
        json.dump(full_meta, f, indent=2, default=str)

    # Also update "latest" symlink
    latest = MODEL_DIR / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(bundle_dir.name)

    print(f"  Model bundle saved: {bundle_dir}")
    print(f"  Features: {len(feature_names)}  hash: {full_meta['feature_hash']}")
    return bundle_dir


def load_model_bundle(tag: str = "latest") -> dict:
    """
    Load a saved model bundle.

    Parameters
    ----------
    tag : version tag or 'latest'

    Returns
    -------
    dict with keys: hmm, ensemble, feature_names, metadata
    """
    # Resolve tag to directory
    target = MODEL_DIR / tag
    if not target.exists():
        # Try prefix match
        matches = list(MODEL_DIR.glob(f"{tag}_*"))
        if not matches:
            raise FileNotFoundError(
                f"No model bundle found for tag '{tag}' in {MODEL_DIR}. "
                f"Available: {[d.name for d in MODEL_DIR.iterdir() if d.is_dir()]}"
            )
        target = sorted(matches)[-1]  # most recent

    with open(target / "hmm.pkl", "rb") as f:
        hmm = pickle.load(f)
    with open(target / "ensemble.pkl", "rb") as f:
        ensemble = pickle.load(f)
    with open(target / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(target / "metadata.json") as f:
        metadata = json.load(f)

    print(f"  Loaded bundle: {target.name}")
    print(f"  Trained: {metadata.get('timestamp', 'unknown')}")
    print(f"  Data source: {metadata.get('data_source', 'unknown')}")
    print(f"  AUC: {metadata.get('mean_auc', 'unknown')}")
    print(f"  Features: {metadata.get('n_features')}  hash: {metadata.get('feature_hash')}")

    return {
        "hmm":           hmm,
        "ensemble":      ensemble,
        "feature_names": feature_names,
        "metadata":      metadata,
    }


def compare_bundles(tag_a: str, tag_b: str) -> None:
    """
    Print a comparison of two model bundles — useful when switching data sources.
    Shows which features were added/removed and performance delta.
    """
    a = load_model_bundle(tag_a)
    b = load_model_bundle(tag_b)

    set_a = set(a["feature_names"])
    set_b = set(b["feature_names"])

    added   = set_b - set_a
    removed = set_a - set_b
    common  = set_a & set_b

    print(f"\n── Bundle Comparison: {tag_a} vs {tag_b} ──")
    print(f"  Features in {tag_a}: {len(set_a)}")
    print(f"  Features in {tag_b}: {len(set_b)}")
    print(f"  Common:  {len(common)}")
    print(f"  Added:   {len(added)}  {sorted(added) if added else ''}")
    print(f"  Removed: {len(removed)}  {sorted(removed) if removed else ''}")

    auc_a = a["metadata"].get("mean_auc")
    auc_b = b["metadata"].get("mean_auc")
    if auc_a and auc_b:
        delta = float(auc_b) - float(auc_a)
        sign  = "+" if delta > 0 else ""
        print(f"\n  AUC {tag_a}: {auc_a:.3f}")
        print(f"  AUC {tag_b}: {auc_b:.3f}  ({sign}{delta:.3f})")
