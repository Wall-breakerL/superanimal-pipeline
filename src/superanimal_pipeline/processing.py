"""Pure processing boundary around the byte-preserved historical cleaner."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import legacy
from .common import require, save

TEN_POINTS = [
    "nose",
    "left_ear_tip",
    "right_ear_tip",
    "neck",
    "mouse_center",
    "tail_base",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
]
BONES = [
    ("nose", "left_ear_tip"),
    ("nose", "right_ear_tip"),
    ("nose", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("neck", "mouse_center"),
    ("mouse_center", "tail_base"),
    ("mouse_center", "left_hip"),
    ("mouse_center", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "tail_base"),
    ("right_hip", "tail_base"),
]


def video_info(path, decode=False):
    info = legacy.video_info(Path(path))
    require(
        info["opened"] and np.isfinite(info["fps"]) and info["fps"] > 0,
        f"Invalid video: {path}",
    )
    if decode:
        cap = cv2.VideoCapture(str(path))
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                require(
                    frame.shape[:2] == (info["height"], info["width"]),
                    "Changing frame size",
                )
                count += 1
        finally:
            cap.release()
        info["decoded_frames"] = count
        require(count >= 2, "Video must decode at least two frames")
        require(
            abs(count - info["frames"]) <= max(5, np.ceil(info["frames"] * 0.001)),
            "Material decode/header discrepancy",
        )
    return info


def prepare_video(row, folder):
    info = video_info(row["video"])
    x, y, w, h = row["crop"] or [0, 0, info["width"], info["height"]]
    require(x + w <= info["width"] and y + h <= info["height"], "Crop outside video")
    require(w % 2 == 0 and h % 2 == 0, "MP4 export requires even width and height")
    dst = folder / "video.mp4"
    if not row["crop"] and Path(row["video"]).suffix.lower() == ".mp4":
        shutil.copy2(row["video"], dst)
        result = video_info(dst, decode=True)
    else:
        cap = cv2.VideoCapture(row["video"])
        writer = cv2.VideoWriter(
            str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (w, h)
        )
        require(writer.isOpened(), "Video encoder unavailable")
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame[y : y + h, x : x + w])
                count += 1
        finally:
            cap.release()
            writer.release()
        require(
            abs(count - info["frames"]) <= max(5, np.ceil(info["frames"] * 0.001)),
            "Material source decode/header discrepancy",
        )
        result = video_info(dst, decode=True)
        require(result["decoded_frames"] == count, "Crop lost frames")
    result.update(
        source_reported_frames=info["frames"], crop=row["crop"], fps_resampled=False
    )
    save(folder / "video_info.json", result)


def pose_table(path, n_frames=None):
    df = pd.read_hdf(path)
    require(
        isinstance(df.columns, pd.MultiIndex)
        and df.columns.names == ["scorer", "individuals", "bodyparts", "coords"],
        "Expected four-level DLC H5",
    )
    require(
        not df.columns.has_duplicates and df.index.is_unique,
        "Duplicate H5 columns/index",
    )
    require(
        list(df.columns.get_level_values("individuals").unique()) == ["animal0"],
        "Only single animal0 is supported",
    )
    require(
        len(df.columns.get_level_values("scorer").unique()) == 1, "Expected one scorer"
    )
    require(
        list(df.columns.get_level_values("bodyparts").unique()) == legacy.BODY_PARTS,
        "Unexpected bodypart names/order",
    )
    require(
        len(df.columns) == 81
        and set(df.columns.get_level_values("coords")) == {"x", "y", "likelihood"},
        "Expected 27 x 3 columns",
    )
    if n_frames is not None:
        require(len(df) == n_frames, f"H5/video mismatch: {len(df)} != {n_frames}")
    return df


def validate_predictions(json_path, h5_path, n_frames):
    predictions = legacy.load_json(Path(json_path))
    require(len(predictions) == n_frames, "JSON/video frame mismatch")
    raw_table = pose_table(h5_path, n_frames)
    for frame in predictions:
        require(
            all(k in frame for k in ["bboxes", "bbox_scores", "bodyparts"]),
            "Unexpected SuperAnimal JSON schema",
        )
        require(
            all(
                isinstance(frame[k], list) and len(frame[k]) <= 1
                for k in ["bboxes", "bbox_scores", "bodyparts"]
            ),
            "Only single-animal JSON supported",
        )
        if frame["bodyparts"]:
            require(
                len(frame["bodyparts"][0]) == 27
                and all(len(kp) == 3 for kp in frame["bodyparts"][0]),
                "Expected 27 x 3 JSON pose",
            )
            require(
                np.isfinite(np.asarray(frame["bodyparts"], dtype=float)).all(),
                "Nonfinite raw JSON; normalize upstream explicitly",
            )
        if frame["bboxes"]:
            require(
                len(frame["bboxes"][0]) == 4 and np.isfinite(frame["bboxes"]).all(),
                "Invalid JSON bbox",
            )
        require(np.isfinite(frame["bbox_scores"]).all(), "Invalid bbox scores")
    # Compare valid raw keypoint entries to catch same-length mismatched files.
    # DLC may use NaN or negative sentinels for absent detections.
    expected = np.asarray(
        [
            f["bodyparts"][0] if f["bodyparts"] else [[-1.0, -1.0, -1.0]] * 27
            for f in predictions
        ]
    )
    actual = raw_table.to_numpy().reshape(n_frames, 27, 3)
    valid_expected = np.isfinite(expected).all(axis=2) & (expected >= 0).all(axis=2)
    valid_actual = np.isfinite(actual).all(axis=2) & (actual >= 0).all(axis=2)
    require(
        np.array_equal(valid_expected, valid_actual),
        "Raw JSON/H5 missing-mask mismatch",
    )
    require(
        np.allclose(
            expected[valid_expected], actual[valid_expected], rtol=1e-6, atol=1e-5
        ),
        "Raw JSON/H5 value mismatch",
    )
    return predictions


def infer(row, model, video, folder):
    runtime = {"mode": model["mode"], "video_adapt": False, "model_settings": model}
    if model["mode"] == "precomputed":
        shutil.copy2(row["raw_h5"], folder / "raw.h5")
        shutil.copy2(row["raw_json"], folder / "raw.json")
    else:
        import deeplabcut
        import torch

        runtime.update(
            cuda_available=torch.cuda.is_available(),
            torch_version=torch.__version__,
            device_selection="auto",
        )
        kwargs = dict(
            superanimal_name="superanimal_topviewmouse",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            dest_folder=str(folder),
            video_adapt=False,
            max_individuals=1,
            create_labeled_video=False,
            **{
                k: model[k]
                for k in [
                    "batch_size",
                    "detector_batch_size",
                    "pcutoff",
                    "bbox_threshold",
                ]
            },
        )
        if model["mode"] == "checkpoint":
            kwargs.update(
                customized_detector_checkpoint=model["detector_checkpoint"],
                customized_pose_checkpoint=model["pose_checkpoint"],
            )
        deeplabcut.video_inference_superanimal([str(video)], **kwargs)
        h5 = list(folder.glob("*.h5"))
        js = list(folder.glob("*_before_adapt.json"))
        require(len(h5) == 1 and len(js) == 1, "Ambiguous/missing inference outputs")
        h5[0].rename(folder / "raw.h5")
        js[0].rename(folder / "raw.json")
    n = video_info(video, decode=True)["decoded_frames"]
    validate_predictions(folder / "raw.json", folder / "raw.h5", n)
    save(folder / "inference.json", runtime)


def segmented_smooth(table, window, boundaries):
    require(
        all(type(x) is int and 0 < x < len(table) for x in boundaries),
        "Cuts must be 0-based interior frame indices",
    )
    boundaries = sorted(set(boundaries))
    starts, ends = [0, *boundaries], [*boundaries, len(table)]
    out = table.copy()
    for start, end in zip(starts, ends):
        out.iloc[start:end] = legacy.smooth_keypoints(
            table.iloc[start:end], window
        ).to_numpy()
    return out


def paired_cuts(jsons, quantile):
    displacements = []
    for file in jsons:
        centers = []
        for f in legacy.load_json(Path(file)):
            b = f["bboxes"][0] if f["bboxes"] else [-1] * 4
            score = f["bbox_scores"][0] if f["bbox_scores"] else -1
            good = (
                len(b) == 4
                and all(np.isfinite(b))
                and min(b) >= 0
                and b[2] > 0
                and b[3] > 0
                and score >= 0
            )
            centers.append(
                [b[0] + b[2] / 2, b[1] + b[3] / 2] if good else [np.nan, np.nan]
            )
        d = np.linalg.norm(np.diff(np.asarray(centers), axis=0), axis=1)
        require(np.isfinite(d).any(), "No valid displacement values for paired cuts")
        displacements.append(d)
    require(
        len(displacements) == 2 and len(displacements[0]) == len(displacements[1]),
        "Paired cut length mismatch",
    )
    a, b = displacements
    ta, tb = [float(np.quantile(d[np.isfinite(d)], quantile)) for d in displacements]
    cuts = (
        np.flatnonzero(np.isfinite(a) & np.isfinite(b) & (a >= ta) & (b >= tb)) + 1
    ).tolist()
    return cuts, [ta, tb]


def serialize_predictions(predictions, bbox_df, keypoint_df):
    """Equivalent slot-0 writeback without repeated full-table pandas slicing.

    No cleaning math happens here. Keep the historical JSON round-trip copy,
    field ordering, finite-or-missing conversion and list-extension behavior.
    """
    cleaned = json.loads(json.dumps(predictions))
    bbox = bbox_df.loc[:, ["x", "y", "width", "height", "score"]].to_numpy()
    keypoints = (
        keypoint_df.loc[
            :,
            [
                (bp, coord)
                for bp in legacy.BODY_PARTS
                for coord in legacy.KEYPOINT_COORDS
            ],
        ]
        .to_numpy()
        .reshape(len(predictions), 27, 3)
    )
    for i, frame in enumerate(cleaned):
        box = [legacy.finite_or_missing(v) for v in bbox[i]]
        pose = [[legacy.finite_or_missing(v) for v in row] for row in keypoints[i]]
        for name, value in [
            ("bboxes", box[:4]),
            ("bbox_scores", box[4]),
            ("bodyparts", pose),
        ]:
            if frame[name]:
                frame[name][0] = value
            else:
                frame[name].append(value)
    return cleaned


def clean(raw, folder, settings, boundaries, video):
    info = video_info(video, decode=True)
    predictions = validate_predictions(
        raw / "raw.json", raw / "raw.h5", info["decoded_frames"]
    )
    bbox, kp, counts = legacy.extract_slot_tables(
        predictions, 0, settings["keypoint_threshold"], settings["bbox_threshold"]
    )
    gaps = legacy.collect_gaps(bbox, kp, settings["long_gap_threshold"])
    # Preserve exact historical interpolation, including interpolated confidence.
    bc, kc = legacy.interpolate_tables(bbox, kp)
    kc = segmented_smooth(kc, settings["smoothing_window"], boundaries)
    save(folder / "gaps.json", gaps)
    np.savez_compressed(
        folder / "missing_mask.npz",
        keypoints=kp.isna().to_numpy(),
        bbox=bbox.isna().to_numpy(),
        bodyparts=np.asarray(legacy.BODY_PARTS),
    )
    require(
        np.isfinite(bc.to_numpy()).all() and np.isfinite(kc.to_numpy()).all(),
        "Unfillable column: inspect gaps.json in failed attempt; no fabricated complete output",
    )
    cleaned = serialize_predictions(predictions, bc, kc)
    legacy.dump_json(folder / "cleaned.json", cleaned, pretty=False)
    legacy.write_cleaned_h5(raw / "raw.h5", folder / "cleaned_27.h5", cleaned, 0)
    df = pose_table(folder / "cleaned_27.h5", info["decoded_frames"])
    vals = df.to_numpy().reshape(len(df), 27, 3)
    require(
        np.isfinite(vals).all() and ((vals[:, :, 2] >= 0) & (vals[:, :, 2] <= 1)).all(),
        "Invalid cleaned values",
    )
    center = vals[:, legacy.BODY_PARTS.index("mouse_center"), :2]
    jumps = np.linalg.norm(np.diff(center, axis=0), axis=1)
    top = np.argsort(jumps)[-min(10, len(jumps)) :][::-1]
    xy = vals[:, :, :2]
    qc = dict(
        frames=len(df),
        keypoints=27,
        units="pixels",
        fps=info["fps"],
        counts=counts,
        parameters=settings,
        boundary_frames=sorted(set(boundaries)),
        interpolation="linear, both directions, unlimited gap length, includes likelihood; not segmented",
        smoothing="centered rolling mean on XY only, min_periods=1; segmented at candidate cuts",
        keypoint_long_gaps=sum(g["long_gap"] for g in gaps["keypoint_gaps"]),
        out_of_image_keypoints=int(
            (
                (xy[:, :, 0] < 0)
                | (xy[:, :, 0] >= info["width"])
                | (xy[:, :, 1] < 0)
                | (xy[:, :, 1] >= info["height"])
            ).sum()
        ),
        largest_center_transitions=[
            dict(
                frame=int(i + 1),
                time_s=float((i + 1) / info["fps"]),
                distance_px=float(jumps[i]),
            )
            for i in top
        ],
        warning="No ground truth accuracy claim; source cuts remain; long gaps and confidence were interpolated",
    )
    save(folder / "qc.json", qc)


def overlay(video, table, output, threshold, progress=None):
    info = video_info(video)
    names = list(table.columns.get_level_values("bodyparts").unique())
    scorer = table.columns.get_level_values("scorer")[0]
    coords = np.stack(
        [
            table.loc[
                :, [(scorer, "animal0", p, c) for c in legacy.KEYPOINT_COORDS]
            ].to_numpy()
            for p in names
        ],
        axis=1,
    )
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    require(writer.isOpened(), "Overlay encoder unavailable")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(index < len(table), "Video longer than H5")
            pose = coords[index]
            valid = np.isfinite(pose).all(axis=1) & (pose[:, 2] >= threshold)
            for a, b in BONES:
                if a in names and b in names:
                    i, j = names.index(a), names.index(b)
                    if valid[i] and valid[j]:
                        cv2.line(
                            frame,
                            tuple(np.rint(pose[i, :2]).astype(int)),
                            tuple(np.rint(pose[j, :2]).astype(int)),
                            (0, 140, 255),
                            1,
                            cv2.LINE_AA,
                        )
            for p, visible in zip(pose, valid):
                if visible:
                    cv2.circle(
                        frame,
                        tuple(np.rint(p[:2]).astype(int)),
                        3,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                frame,
                f"frame {index} | {index / info['fps']:.2f}s | {len(names)} points",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            index += 1
            if progress and index % 500 == 0:
                progress(index, len(table))
    finally:
        cap.release()
        writer.release()
    require(index == len(table), "Video shorter than H5")
    check = video_info(output, decode=True)
    require(check["decoded_frames"] == len(table), "Overlay frame mismatch")


def export(video, cleaned, folder, settings, progress=None):
    df = pose_table(cleaned / "cleaned_27.h5")
    names = TEN_POINTS if settings["keypoints"] == 10 else legacy.BODY_PARTS
    scorer = df.columns.get_level_values("scorer")[0]
    out = df.loc[
        :, [(scorer, "animal0", p, c) for p in names for c in legacy.KEYPOINT_COORDS]
    ]
    out.to_hdf(
        folder / "02_cleaned.h5", key="df_with_missing", format="table", mode="w"
    )
    if settings["keypoints"] == 10:
        shutil.copy2(cleaned / "cleaned_27.h5", folder / "02_cleaned_27kp.h5")
    shutil.copy2(video, folder / "01_original.mp4")
    shutil.copy2(cleaned / "qc.json", folder / "qc.json")
    pd.testing.assert_frame_equal(
        pd.read_hdf(folder / "02_cleaned.h5"), out, check_exact=True
    )
    if settings["labeled_video"]:
        overlay(
            video,
            out,
            folder / "03_labeled_cleaned.mp4",
            settings["display_threshold"],
            progress,
        )
