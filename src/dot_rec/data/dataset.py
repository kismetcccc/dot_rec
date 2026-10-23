"""Strict local-only data boundary. No pickle, network, or automatic downloads."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_keys(obj, allowed, required=()):
    unknown, missing = set(obj) - set(allowed), set(required) - set(obj)
    if unknown or missing:
        raise ValueError(f"Unknown keys {sorted(unknown)}; missing keys {sorted(missing)}")


def load_array(path, key=None):
    path = Path(path)
    if path.suffix == ".npy":
        if key is not None:
            raise ValueError("NPY has no named key")
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        if key is None:
            raise ValueError("NPZ requires an explicit key")
        with np.load(path, allow_pickle=False) as archive:
            return archive[key].copy()
    raise ValueError("Arrays must be NPY or NPZ")


def load_edges(path, key=None):
    path = Path(path)
    if path.suffix == ".csv":
        if key is not None:
            raise ValueError("CSV has no NPZ key")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["user", "item"]:
                raise ValueError("Edge CSV header must be user,item")
            rows = [(int(row["user"]), int(row["item"])) for row in reader]
        return np.array(rows, dtype=np.int64).reshape(-1, 2)
    return load_array(path, key)


def load_mmrec_inter(path, user_field="userID", item_field="itemID", label_field="x_label", separator="\t"):
    """Import explicit zero-based IDs and labels only; no upstream trainer reuse."""
    splits = [[], [], []]
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=separator)
        if not {user_field, item_field, label_field}.issubset(reader.fieldnames or []):
            raise ValueError("MMRec input missing required columns")
        for row in reader:
            label = int(row[label_field])
            if label not in (0, 1, 2):
                raise ValueError("x_label must be 0/1/2")
            splits[label].append((int(row[user_field]), int(row[item_field])))
    return tuple(np.asarray(s, dtype=np.int64).reshape(-1, 2) for s in splits)


def validate_splits(train, valid, test, n_users, n_items):
    if type(n_users) is not int or type(n_items) is not int or min(n_users, n_items) <= 0:
        raise ValueError("Positive integer catalog dimensions required")
    seen = set()
    for name, edges in (("train", train), ("valid", valid), ("test", test)):
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
            raise ValueError(f"{name}: expected integer [E,2]")
        if edges.size and (np.any(edges < 0) or np.any(edges[:, 0] >= n_users) or np.any(edges[:, 1] >= n_items)):
            raise ValueError(f"{name}: ID outside declared catalog")
        pairs = set(map(tuple, edges.tolist()))
        if len(pairs) != len(edges):
            raise ValueError(f"{name}: duplicate pairs")
        if seen & pairs:
            raise ValueError(f"{name}: cross-split overlap")
        seen.update(pairs)
    if len(train) == 0:
        raise ValueError("TRAIN must be nonempty")


def split_coverage(user_degree, item_degree, edges, population="all"):
    """Target endpoint coverage against TRAIN degrees; never filter edges."""
    if population not in ("all", "warm_only"):
        raise ValueError("evaluation_population must be all or warm_only")
    users, items = np.unique(edges[:, 0]), np.unique(edges[:, 1])
    cold_u = user_degree[edges[:, 0]] == 0
    cold_i = item_degree[edges[:, 1]] == 0
    report = {
        "definition": "cold means zero TRAIN degree; warm means positive TRAIN degree",
        "population": population,
        "candidate_catalog": "all_declared_items",
        "target_interactions": int(len(edges)),
        "target_users": int(len(users)),
        "target_items": int(len(items)),
        "cold_target_users": int(np.count_nonzero(user_degree[users] == 0)),
        "cold_target_items": int(np.count_nonzero(item_degree[items] == 0)),
        "interactions_with_cold_user": int(np.count_nonzero(cold_u)),
        "interactions_with_cold_item": int(np.count_nonzero(cold_i)),
        "interactions_with_either_cold_endpoint": int(np.count_nonzero(cold_u | cold_i)),
        "interactions_with_both_cold_endpoints": int(np.count_nonzero(cold_u & cold_i)),
        "catalog_cold_users": int(np.count_nonzero(user_degree == 0)),
        "catalog_cold_items": int(np.count_nonzero(item_degree == 0)),
    }
    if population == "warm_only" and (report["cold_target_users"] or report["cold_target_items"]):
        raise ValueError("warm_only requires every evaluation target user/item to have positive TRAIN degree; no filtering performed")
    return report


@dataclass
class Dataset:
    n_users: int
    n_items: int
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    visual: np.ndarray
    text: np.ndarray
    natural_mask: np.ndarray
    experiment_mask: np.ndarray
    provenance: dict
    protocol: str = "mtmt"

    @property
    def observed(self):
        """Fixed evaluation visibility, identical for validation and test."""
        return self.natural_mask & self.experiment_mask

    @property
    def train_observed(self):
        return self.natural_mask if self.protocol == "ftmt" else self.observed


def load_manifest(path, protocol="mtmt", evaluation_population="all"):
    if protocol not in ("mtmt", "ftmt"):
        raise ValueError("protocol must be mtmt or ftmt")
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"n_users", "n_items", "metadata", "files"}
    strict_keys(manifest, required, required)
    meta = manifest["metadata"]
    meta_keys = {"dataset", "source", "version", "filtering", "split_protocol", "id_mapping", "encoders", "mask_protocol"}
    strict_keys(meta, meta_keys, meta_keys)
    if any(not isinstance(meta[k], str) or not meta[k].strip() for k in meta_keys):
        raise ValueError("Metadata must be nonempty declared strings; use explicit unknown declarations if necessary")
    files = manifest["files"]
    needed = {"train", "valid", "test", "visual", "text", "natural_mask", "experiment_mask"}
    allowed = needed | {"mmrec_inter"}
    strict_keys(files, allowed)
    if "mmrec_inter" in files:
        if set(files) != {"mmrec_inter", "visual", "text", "natural_mask", "experiment_mask"}:
            raise ValueError("Use either three splits OR mmrec_inter")
    elif set(files) != needed:
        raise ValueError("All feature/mask and split files required")
    arrays, hashes = {}, {}
    for name, spec in files.items():
        strict_keys(spec, {"path", "sha256", "key", "user_field", "item_field", "label_field", "separator"}, {"path", "sha256"})
        location = (path.parent / spec["path"]).resolve()
        digest = sha256_file(location)
        if digest != spec["sha256"]:
            raise ValueError(f"{name}: SHA256 mismatch")
        hashes[name] = digest
        if name == "mmrec_inter":
            if "key" in spec:
                raise ValueError("MMRec text input cannot have NPZ key")
            opts = {k: spec[k] for k in ("user_field", "item_field", "label_field", "separator") if k in spec}
            arrays.update(zip(("train", "valid", "test"), load_mmrec_inter(location, **opts)))
        else:
            if set(spec) - {"path", "sha256", "key"}:
                raise ValueError("MMRec options only valid for mmrec_inter")
            loader = load_edges if name in ("train", "valid", "test") else load_array
            arrays[name] = loader(location, spec.get("key"))
    u, i = manifest["n_users"], manifest["n_items"]
    validate_splits(arrays["train"], arrays["valid"], arrays["test"], u, i)
    for split in ("train", "valid", "test"):
        arrays[split] = arrays[split].astype(np.int64, copy=False)
    user_degree = np.bincount(arrays["train"][:, 0], minlength=u)
    item_degree = np.bincount(arrays["train"][:, 1], minlength=i)
    coverage = {split: split_coverage(user_degree, item_degree, arrays[split], evaluation_population)
                for split in ("valid", "test")}
    for name in ("natural_mask", "experiment_mask"):
        if arrays[name].dtype != np.bool_ or arrays[name].shape != (i, 2):
            raise ValueError(f"{name}: expected boolean [I,2]")
    effective = arrays["natural_mask"] & arrays["experiment_mask"]
    training_visible = arrays["natural_mask"] if protocol == "ftmt" else effective
    for m, name in enumerate(("visual", "text")):
        x = arrays[name]
        if x.dtype.kind != "f" or x.ndim != 2 or x.shape[0] != i or x.shape[1] < 1:
            raise ValueError(f"{name}: expected float [I,D]")
        if not np.isfinite(x[training_visible[:, m]]).all():
            raise ValueError(f"{name}: training-visible nonfinite feature under {protocol}")
    provenance = {"manifest": manifest, "manifest_sha256": sha256_file(path), "file_sha256": hashes,
                  "protocol": protocol, "evaluation_visibility": "natural_mask AND experiment_mask",
                  "training_visibility": "natural_mask" if protocol == "ftmt" else "natural_mask AND experiment_mask",
                  "evaluation_coverage": coverage}
    return Dataset(u, i, **arrays, provenance=provenance, protocol=protocol)
