import csv
import math
import re
from pathlib import Path

import yaml

from .common import require, sha

DEFAULTS = {
    "model": {
        "mode": "checkpoint",
        "pose_checkpoint": None,
        "detector_checkpoint": None,
        "batch_size": 2,
        "detector_batch_size": 2,
        "bbox_threshold": 0.9,
        "pcutoff": 0.1,
    },
    "cleaning": {
        "keypoint_threshold": 0.1,
        "bbox_threshold": 0.5,
        "smoothing_window": 5,
        "long_gap_threshold": 24,
    },
    "cuts": {"mode": "off", "quantile": 0.999, "file": None},
    "export": {"keypoints": 10, "labeled_video": True, "display_threshold": 0.2},
}


def resolve(base, value):
    return str((base / Path(value).expanduser()).resolve())


def load(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(raw, dict), "Config must be a mapping")
    require(
        set(raw) <= {"schema_version", "manifest", "output", *DEFAULTS},
        "Unknown config key",
    )
    require(raw.get("schema_version") == 1, "schema_version must be 1")
    c = dict(
        schema_version=1,
        manifest=resolve(path.parent, raw["manifest"]),
        output=resolve(path.parent, raw["output"]),
    )
    for section, default in DEFAULTS.items():
        values = raw.get(section, {})
        require(
            isinstance(values, dict) and set(values) <= set(default),
            f"Unknown {section} key",
        )
        c[section] = {**default, **values}
    m = c["model"]
    require(
        m["mode"] in {"checkpoint", "zero-shot", "precomputed"}, "Unknown model.mode"
    )
    for field in ["pose_checkpoint", "detector_checkpoint"]:
        if m[field]:
            m[field] = resolve(path.parent, m[field])
            require(Path(m[field]).is_file(), f"Missing {field}")
    require(
        (m["mode"] == "checkpoint")
        == bool(m["pose_checkpoint"] and m["detector_checkpoint"]),
        "checkpoint mode requires BOTH weights; other modes must not specify weights",
    )
    if m["mode"] != "checkpoint":
        require(
            not m["pose_checkpoint"] and not m["detector_checkpoint"],
            "Unexpected weights",
        )
    for k in ["batch_size", "detector_batch_size"]:
        require(type(m[k]) is int and m[k] > 0, f"Invalid {k}")
    for section, keys in [
        ("model", ["bbox_threshold", "pcutoff"]),
        ("cleaning", ["keypoint_threshold", "bbox_threshold"]),
        ("export", ["display_threshold"]),
    ]:
        for k in keys:
            v = c[section][k]
            require(
                type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1,
                f"Invalid {section}.{k}",
            )
    w = c["cleaning"]["smoothing_window"]
    require(
        type(w) is int and w > 0 and w % 2 == 1,
        "smoothing_window must be positive odd integer",
    )
    require(
        type(c["cleaning"]["long_gap_threshold"]) is int
        and c["cleaning"]["long_gap_threshold"] >= 0,
        "Invalid gap threshold",
    )
    require(
        c["export"]["keypoints"] in (10, 27)
        and type(c["export"]["labeled_video"]) is bool,
        "Invalid export options",
    )
    require(c["cuts"]["mode"] in {"off", "manual", "paired"}, "Unknown cuts.mode")
    require(
        type(c["cuts"]["quantile"]) in (int, float) and 0 < c["cuts"]["quantile"] < 1,
        "Invalid cut quantile",
    )
    if c["cuts"]["file"]:
        c["cuts"]["file"] = resolve(path.parent, c["cuts"]["file"])
        require(Path(c["cuts"]["file"]).is_file(), "Missing cuts file")
    require(
        (c["cuts"]["mode"] == "manual") == bool(c["cuts"]["file"]),
        "Only manual cuts accepts a file",
    )
    with open(c["manifest"], encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        require({"trial_id", "video"} <= fields, "Manifest requires trial_id,video")
        require(
            fields
            <= {
                "trial_id",
                "video",
                "animal_id",
                "group",
                "pair_id",
                "crop_x",
                "crop_y",
                "crop_width",
                "crop_height",
                "raw_json",
                "raw_h5",
            },
            "Unknown manifest column",
        )
        rows = list(reader)
    require(bool(rows), "Empty manifest")
    seen = set()
    for row in rows:
        require(
            None not in row and all(v is not None for v in row.values()),
            "Malformed CSV row",
        )
        tid = row["trial_id"]
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tid) and tid not in seen,
            f"Invalid/duplicate ID: {tid}",
        )
        seen.add(tid)
        row["animal_id"] = row.get("animal_id") or tid
        row["group"] = row.get("group", "")
        row["pair_id"] = row.get("pair_id", "")
        for k in ["video", "raw_json", "raw_h5"]:
            if row.get(k):
                row[k] = resolve(Path(c["manifest"]).parent, row[k])
                require(Path(row[k]).is_file(), f"{tid}: missing {k}")
        crop = [
            row.get(k, "") for k in ["crop_x", "crop_y", "crop_width", "crop_height"]
        ]
        require(
            all(v != "" for v in crop) or all(v == "" for v in crop),
            "Specify all four crop fields or none",
        )
        row["crop"] = [int(v) for v in crop] if crop[0] != "" else None
        if row["crop"]:
            x, y, w, h = row["crop"]
            require(x >= 0 and y >= 0 and w > 0 and h > 0, "Invalid crop")
        require(
            bool(row.get("raw_json")) == bool(row.get("raw_h5")),
            "Need both precomputed files",
        )
        require(
            (m["mode"] == "precomputed") == bool(row.get("raw_h5")),
            "raw files only supported in precomputed mode",
        )
        if m["mode"] == "precomputed":
            require(
                not row["crop"], "Precomputed mode requires an already cropped video"
            )
        # Never allow artifacts to be inputs of the same run.
        for k in ["video", "raw_json", "raw_h5"]:
            if row.get(k):
                require(
                    not Path(row[k]).is_relative_to(Path(c["output"])),
                    "Input may not be inside output directory",
                )
    if c["cuts"]["mode"] == "paired":
        require(
            all(r["pair_id"] for r in rows), "paired cuts needs pair_id on every row"
        )
        for pair in {r["pair_id"] for r in rows}:
            require(
                sum(r["pair_id"] == pair for r in rows) == 2,
                "Each pair_id must identify exactly two synchronized videos",
            )
    return c, rows


def fingerprint(config, rows):
    files = {r[k] for r in rows for k in ["video", "raw_h5", "raw_json"] if r.get(k)}
    files.update(
        config["model"][k]
        for k in ["pose_checkpoint", "detector_checkpoint"]
        if config["model"][k]
    )
    if config["cuts"]["file"]:
        files.add(config["cuts"]["file"])
    from importlib import metadata

    versions = {
        p: metadata.version(p)
        for p in ["numpy", "pandas", "tables", "opencv-python", "PyYAML"]
    }
    for p in ["deeplabcut", "torch", "torchvision"]:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "config": config,
        "trials": rows,
        "inputs": {p: sha(p) for p in sorted(files)},
        "code": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "versions": versions,
    }
