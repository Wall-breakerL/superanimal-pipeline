from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


BODY_PARTS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "neck",
    "mid_back",
    "mouse_center",
    "mid_backend",
    "mid_backend2",
    "mid_backend3",
    "tail_base",
    "tail1",
    "tail2",
    "tail3",
    "tail4",
    "tail5",
    "left_shoulder",
    "left_midside",
    "left_hip",
    "right_shoulder",
    "right_midside",
    "right_hip",
    "tail_end",
    "head_midpoint",
]

BBOX_COORDS = ["x", "y", "width", "height", "score"]
KEYPOINT_COORDS = ["x", "y", "likelihood"]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of frame predictions: {path}")
    return data


def dump_json(path: Path, data: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        else:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        f.write("\n")


def finite_or_missing(value: float) -> float:
    if value is None:
        return -1.0
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return -1.0
    if math.isnan(fvalue) or math.isinf(fvalue):
        return -1.0
    return fvalue


def video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    info = {
        "opened": bool(opened),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else None,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) if opened else None,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else None,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else None,
    }
    cap.release()
    return info


def extract_slot_tables(
    predictions: list[dict[str, Any]],
    animal_slot: int,
    keypoint_threshold: float,
    bbox_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    bbox_rows: list[list[float]] = []
    keypoint_rows: list[list[float]] = []
    counts = {
        "frames": len(predictions),
        "missing_bbox_raw": 0,
        "low_bbox_score": 0,
        "missing_keypoint_raw": 0,
        "low_keypoint_score": 0,
    }

    keypoint_columns = pd.MultiIndex.from_product(
        [BODY_PARTS, KEYPOINT_COORDS],
        names=["bodyparts", "coords"],
    )

    for frame in predictions:
        bboxes = frame.get("bboxes", [])
        bbox_scores = frame.get("bbox_scores", [])
        bodyparts = frame.get("bodyparts", [])

        if animal_slot >= len(bboxes) or animal_slot >= len(bbox_scores):
            bbox = [-1.0, -1.0, -1.0, -1.0]
            bbox_score = -1.0
        else:
            bbox = bboxes[animal_slot]
            bbox_score = bbox_scores[animal_slot]

        bbox_values = [float(v) for v in bbox[:4]] + [float(bbox_score)]
        bbox_missing = (
            len(bbox) < 4
            or any(v < 0 for v in bbox_values[:4])
            or bbox_values[2] <= 0
            or bbox_values[3] <= 0
            or bbox_values[4] < 0
        )
        if bbox_missing:
            counts["missing_bbox_raw"] += 1
            bbox_values = [np.nan] * len(BBOX_COORDS)
        elif bbox_values[4] < bbox_threshold:
            counts["low_bbox_score"] += 1
            bbox_values = [np.nan] * len(BBOX_COORDS)
        bbox_rows.append(bbox_values)

        if animal_slot >= len(bodyparts):
            keypoints = [[-1.0, -1.0, -1.0] for _ in BODY_PARTS]
        else:
            keypoints = bodyparts[animal_slot]

        row: list[float] = []
        for idx in range(len(BODY_PARTS)):
            if idx >= len(keypoints):
                x, y, score = -1.0, -1.0, -1.0
            else:
                x, y, score = [float(v) for v in keypoints[idx][:3]]

            if x < 0 or y < 0 or score < 0:
                counts["missing_keypoint_raw"] += 1
                row.extend([np.nan, np.nan, np.nan])
            elif score < keypoint_threshold:
                counts["low_keypoint_score"] += 1
                row.extend([np.nan, np.nan, np.nan])
            else:
                row.extend([x, y, score])
        keypoint_rows.append(row)

    bbox_df = pd.DataFrame(bbox_rows, columns=BBOX_COORDS)
    keypoint_df = pd.DataFrame(keypoint_rows, columns=keypoint_columns)
    return bbox_df, keypoint_df, counts


def find_nan_gaps(
    series: pd.Series,
    target: str,
    long_gap_threshold: int,
) -> list[dict[str, Any]]:
    mask = series.isna().to_numpy()
    values = series.to_numpy(dtype=float)
    gaps: list[dict[str, Any]] = []
    idx = 0
    n = len(mask)

    while idx < n:
        if not mask[idx]:
            idx += 1
            continue

        start = idx
        while idx < n and mask[idx]:
            idx += 1
        end = idx - 1
        length = end - start + 1
        left = start - 1 if start > 0 and not mask[start - 1] else None
        right = idx if idx < n and not mask[idx] else None

        if left is not None and right is not None:
            gap_type = "middle_gap"
        elif left is None and right is not None:
            gap_type = "leading_gap"
        elif left is not None and right is None:
            gap_type = "trailing_gap"
        else:
            gap_type = "all_missing"

        gaps.append(
            {
                "target": target,
                "gap_type": gap_type,
                "start_frame": int(start),
                "end_frame": int(end),
                "length": int(length),
                "long_gap": bool(length > long_gap_threshold),
                "left_anchor_frame": int(left) if left is not None else None,
                "right_anchor_frame": int(right) if right is not None else None,
                "left_anchor_value": finite_or_missing(values[left])
                if left is not None
                else None,
                "right_anchor_value": finite_or_missing(values[right])
                if right is not None
                else None,
            }
        )
    return gaps


def collect_gaps(
    bbox_df: pd.DataFrame,
    keypoint_df: pd.DataFrame,
    long_gap_threshold: int,
) -> dict[str, Any]:
    bbox_gap_mask = bbox_df[BBOX_COORDS[:4]].isna().any(axis=1)
    bbox_gap_series = pd.Series(np.where(bbox_gap_mask, np.nan, 1.0))
    bbox_gaps = find_nan_gaps(bbox_gap_series, "animal0.bbox", long_gap_threshold)

    keypoint_gaps: list[dict[str, Any]] = []
    per_bodypart_counts: dict[str, dict[str, int]] = {}
    for bodypart in BODY_PARTS:
        bp_df = keypoint_df.loc[:, (bodypart, KEYPOINT_COORDS)]
        gap_mask = bp_df.isna().any(axis=1)
        gap_series = pd.Series(np.where(gap_mask, np.nan, 1.0))
        gaps = find_nan_gaps(
            gap_series,
            f"animal0.{bodypart}",
            long_gap_threshold,
        )
        keypoint_gaps.extend(gaps)
        per_bodypart_counts[bodypart] = {
            "gaps": len(gaps),
            "long_gaps": sum(1 for gap in gaps if gap["long_gap"]),
            "missing_or_filtered_frames": int(gap_mask.sum()),
        }

    return {
        "bbox_gaps": bbox_gaps,
        "keypoint_gaps": keypoint_gaps,
        "keypoint_gap_counts_by_bodypart": per_bodypart_counts,
    }


def interpolate_tables(
    bbox_df: pd.DataFrame,
    keypoint_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bbox_clean = bbox_df.interpolate(
        method="linear",
        axis=0,
        limit_direction="both",
    )
    keypoint_clean = keypoint_df.interpolate(
        method="linear",
        axis=0,
        limit_direction="both",
    )
    return bbox_clean, keypoint_clean


def smooth_keypoints(
    keypoint_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    if window <= 1:
        return keypoint_df
    if window % 2 == 0:
        raise ValueError(f"Smoothing window must be odd, got {window}")

    smoothed = keypoint_df.copy()
    for bodypart in BODY_PARTS:
        for coord in ("x", "y"):
            col = (bodypart, coord)
            smoothed.loc[:, col] = (
                keypoint_df.loc[:, col]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )
    return smoothed


def count_remaining_missing(
    bbox_df: pd.DataFrame,
    keypoint_df: pd.DataFrame,
) -> dict[str, int]:
    bbox_missing_frames = int(bbox_df[BBOX_COORDS[:4]].isna().any(axis=1).sum())
    keypoint_missing_values = int(keypoint_df.isna().sum().sum())
    keypoint_missing_frames = int(
        keypoint_df.isna().T.groupby(level="bodyparts").any().T.any(axis=1).sum()
    )
    return {
        "remaining_bbox_missing_frames_after_interpolation": bbox_missing_frames,
        "remaining_keypoint_missing_values_after_interpolation": keypoint_missing_values,
        "remaining_keypoint_missing_frames_after_interpolation": keypoint_missing_frames,
    }


def input_alignment(
    json_frame_count: int,
    h5_path: Path | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    alignment: dict[str, Any] = {
        "json_frames": int(json_frame_count),
        "h5_frames": None,
        "video_frames": None,
        "h5_matches_json": None,
        "video_matches_json": None,
    }
    if h5_path is not None:
        h5_df = pd.read_hdf(h5_path)
        alignment["h5_frames"] = int(len(h5_df))
        alignment["h5_matches_json"] = bool(len(h5_df) == json_frame_count)
    if video_path is not None:
        info = video_info(video_path)
        alignment["video_frames"] = info["frames"]
        alignment["video_matches_json"] = (
            bool(info["frames"] == json_frame_count) if info["frames"] is not None else None
        )
    return alignment


def count_other_slots(predictions: list[dict[str, Any]], animal_slot: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in predictions:
        for idx, (bbox, score) in enumerate(
            zip(frame.get("bboxes", []), frame.get("bbox_scores", []))
        ):
            if idx == animal_slot:
                continue
            valid = (
                len(bbox) >= 4
                and score >= 0
                and not any(float(v) < 0 for v in bbox[:4])
                and float(bbox[2]) > 0
                and float(bbox[3]) > 0
            )
            if valid:
                counts[f"animal{idx}"] = counts.get(f"animal{idx}", 0) + 1
    return counts


def update_predictions(
    predictions: list[dict[str, Any]],
    bbox_df: pd.DataFrame,
    keypoint_df: pd.DataFrame,
    animal_slot: int,
) -> list[dict[str, Any]]:
    cleaned = json.loads(json.dumps(predictions))

    for frame_idx, frame in enumerate(cleaned):
        bbox_values = bbox_df.iloc[frame_idx]
        bbox = [
            finite_or_missing(bbox_values["x"]),
            finite_or_missing(bbox_values["y"]),
            finite_or_missing(bbox_values["width"]),
            finite_or_missing(bbox_values["height"]),
        ]
        bbox_score = finite_or_missing(bbox_values["score"])

        while animal_slot >= len(frame["bboxes"]):
            frame["bboxes"].append([-1.0, -1.0, -1.0, -1.0])
        while animal_slot >= len(frame["bbox_scores"]):
            frame["bbox_scores"].append(-1.0)

        frame["bboxes"][animal_slot] = bbox
        frame["bbox_scores"][animal_slot] = bbox_score

        keypoints: list[list[float]] = []
        for bodypart in BODY_PARTS:
            row = keypoint_df.loc[frame_idx, (bodypart, KEYPOINT_COORDS)]
            keypoints.append(
                [
                    finite_or_missing(row[(bodypart, "x")]),
                    finite_or_missing(row[(bodypart, "y")]),
                    finite_or_missing(row[(bodypart, "likelihood")]),
                ]
            )

        while animal_slot >= len(frame["bodyparts"]):
            frame["bodyparts"].append([[-1.0, -1.0, -1.0] for _ in BODY_PARTS])
        frame["bodyparts"][animal_slot] = keypoints

    return cleaned


def write_cleaned_h5(
    original_h5: Path,
    cleaned_h5: Path,
    cleaned_predictions: list[dict[str, Any]],
    animal_slot: int,
) -> None:
    df = pd.read_hdf(original_h5)
    scorer = df.columns.get_level_values("scorer")[0]
    animal = f"animal{animal_slot}"

    if len(df) != len(cleaned_predictions):
        raise ValueError(
            f"H5 frame count ({len(df)}) does not match JSON frame count "
            f"({len(cleaned_predictions)})"
        )

    for bp_idx, bodypart in enumerate(BODY_PARTS):
        for coord_idx, coord in enumerate(KEYPOINT_COORDS):
            col = (scorer, animal, bodypart, coord)
            if col not in df.columns:
                raise KeyError(f"Missing H5 column: {col}")
            df.loc[:, col] = [
                frame["bodyparts"][animal_slot][bp_idx][coord_idx]
                for frame in cleaned_predictions
            ]

    cleaned_h5.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(cleaned_h5, key="df_with_missing", mode="w", format="table")


def build_bboxes_list(cleaned_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bboxes": frame["bboxes"],
            "bbox_scores": frame["bbox_scores"],
        }
        for frame in cleaned_predictions
    ]


def make_labeled_video(
    video_path: Path,
    cleaned_h5: Path,
    cleaned_json: Path,
    output_video: Path,
    animal_slot: int,
    pcutoff: float,
    bboxes_pcutoff: float,
    fps: float | None,
) -> None:
    from deeplabcut.modelzoo.utils import get_superanimal_colormaps
    from deeplabcut.utils.make_labeled_video import create_video

    predictions = load_json(cleaned_json)
    bboxes_list = build_bboxes_list(predictions)
    info = video_info(video_path)
    if not info["opened"]:
        raise ValueError(f"Could not open video: {video_path}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmap = get_superanimal_colormaps()["superanimal_topviewmouse"]

    create_video(
        str(video_path),
        str(cleaned_h5),
        animals2show=[f"animal{animal_slot}"],
        pcutoff=pcutoff,
        fps=fps if fps is not None else info["fps"],
        bbox=(0, int(info["width"]), 0, int(info["height"])),
        cmap=cmap,
        output_path=str(output_video),
        plot_bboxes=True,
        bboxes_list=bboxes_list,
        bboxes_pcutoff=bboxes_pcutoff,
    )


def analyze_command(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load_json(args.json)
    bbox_df, keypoint_df, counts = extract_slot_tables(
        predictions,
        args.animal_slot,
        args.keypoint_threshold,
        args.bbox_threshold,
    )
    gaps = collect_gaps(bbox_df, keypoint_df, args.long_gap_threshold)
    summary = {
        "input_json": str(args.json),
        "video": str(args.video) if args.video else None,
        "video_info": video_info(args.video) if args.video else None,
        "parameters": {
            "animal_slot": args.animal_slot,
            "keypoint_threshold": args.keypoint_threshold,
            "bbox_threshold": args.bbox_threshold,
            "long_gap_threshold": args.long_gap_threshold,
        },
        "counts": {
            **counts,
            "bbox_gap_count": len(gaps["bbox_gaps"]),
            "bbox_long_gap_count": sum(
                1 for gap in gaps["bbox_gaps"] if gap["long_gap"]
            ),
            "keypoint_gap_count": len(gaps["keypoint_gaps"]),
            "keypoint_long_gap_count": sum(
                1 for gap in gaps["keypoint_gaps"] if gap["long_gap"]
            ),
            "other_slot_valid_frames": count_other_slots(
                predictions,
                args.animal_slot,
            ),
        },
        "gaps": gaps,
    }

    if args.output:
        dump_json(args.output, summary)
    print(json.dumps(summary["counts"], indent=2, ensure_ascii=False))
    return summary


def clean_command(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load_json(args.json)
    alignment = input_alignment(len(predictions), args.h5, getattr(args, "video", None))
    if alignment["h5_matches_json"] is False:
        raise ValueError(
            f"H5 frame count ({alignment['h5_frames']}) does not match JSON frame "
            f"count ({alignment['json_frames']}); refusing to write cleaned H5."
        )
    bbox_df, keypoint_df, counts = extract_slot_tables(
        predictions,
        args.animal_slot,
        args.keypoint_threshold,
        args.bbox_threshold,
    )
    gaps = collect_gaps(bbox_df, keypoint_df, args.long_gap_threshold)
    bbox_clean, keypoint_clean = interpolate_tables(bbox_df, keypoint_df)
    keypoint_clean = smooth_keypoints(keypoint_clean, args.smoothing_window)
    remaining_missing = count_remaining_missing(bbox_clean, keypoint_clean)
    cleaned_predictions = update_predictions(
        predictions,
        bbox_clean,
        keypoint_clean,
        args.animal_slot,
    )

    dump_json(args.output_json, cleaned_predictions, pretty=False)
    write_cleaned_h5(args.h5, args.output_h5, cleaned_predictions, args.animal_slot)

    qc = {
        "input_json": str(args.json),
        "input_h5": str(args.h5),
        "output_json": str(args.output_json),
        "output_h5": str(args.output_h5),
        "input_alignment": alignment,
        "parameters": {
            "animal_slot": args.animal_slot,
            "keypoint_threshold": args.keypoint_threshold,
            "bbox_threshold": args.bbox_threshold,
            "long_gap_threshold": args.long_gap_threshold,
            "interpolation": {
                "method": "linear",
                "axis": 0,
                "limit_direction": "both",
                "limit": None,
                "long_gaps_interpolated": True,
            },
            "smoothing": {
                "method": "rolling_mean",
                "window_frames": args.smoothing_window,
                "coords": ["x", "y"],
                "likelihood_smoothed": False,
            },
        },
        "counts": {
            **counts,
            **remaining_missing,
            "bbox_gap_count": len(gaps["bbox_gaps"]),
            "bbox_long_gap_count": sum(
                1 for gap in gaps["bbox_gaps"] if gap["long_gap"]
            ),
            "keypoint_gap_count": len(gaps["keypoint_gaps"]),
            "keypoint_long_gap_count": sum(
                1 for gap in gaps["keypoint_gaps"] if gap["long_gap"]
            ),
            "other_slot_valid_frames": count_other_slots(
                predictions,
                args.animal_slot,
            ),
        },
        "gaps": gaps,
    }
    dump_json(args.output_qc, qc)
    print(json.dumps(qc["counts"], indent=2, ensure_ascii=False))
    return qc


def make_video_command(args: argparse.Namespace) -> None:
    make_labeled_video(
        args.video,
        args.h5,
        args.json,
        args.output_video,
        args.animal_slot,
        args.pcutoff,
        args.bboxes_pcutoff,
        args.fps,
    )
    print(f"saved video -> {args.output_video}")


def run_all_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_prefix
    analyze_path = args.output_dir / f"{stem}_analysis.json"
    cleaned_json = args.output_dir / f"{stem}_cleaned.json"
    cleaned_h5 = args.output_dir / f"{stem}_cleaned.h5"
    qc_path = args.output_dir / f"{stem}_cleaning_qc.json"
    output_video = args.output_dir / f"{stem}_labeled_cleaned.mp4"

    analyze_args = argparse.Namespace(**vars(args))
    analyze_args.output = analyze_path
    analyze_command(analyze_args)

    clean_args = argparse.Namespace(**vars(args))
    clean_args.output_json = cleaned_json
    clean_args.output_h5 = cleaned_h5
    clean_args.output_qc = qc_path
    clean_command(clean_args)

    make_labeled_video(
        args.video,
        cleaned_h5,
        cleaned_json,
        output_video,
        args.animal_slot,
        args.pcutoff,
        args.bboxes_pcutoff,
        args.fps,
    )
    print(f"analysis -> {analyze_path}")
    print(f"cleaned json -> {cleaned_json}")
    print(f"cleaned h5 -> {cleaned_h5}")
    print(f"qc -> {qc_path}")
    print(f"video -> {output_video}")


def add_common_cleaning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--animal-slot", type=int, default=0)
    parser.add_argument("--keypoint-threshold", type=float, default=0.1)
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    parser.add_argument("--long-gap-threshold", type=int, default=24)
    parser.add_argument("--smoothing-window", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean SuperAnimal top-view mouse predictions for one video."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    add_common_cleaning_args(analyze)
    analyze.add_argument("--video", type=Path)
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(func=analyze_command)

    clean = subparsers.add_parser("clean")
    add_common_cleaning_args(clean)
    clean.add_argument("--h5", type=Path, required=True)
    clean.add_argument("--output-json", type=Path, required=True)
    clean.add_argument("--output-h5", type=Path, required=True)
    clean.add_argument("--output-qc", type=Path, required=True)
    clean.set_defaults(func=clean_command)

    make_video = subparsers.add_parser("make-video")
    make_video.add_argument("--video", type=Path, required=True)
    make_video.add_argument("--json", type=Path, required=True)
    make_video.add_argument("--h5", type=Path, required=True)
    make_video.add_argument("--output-video", type=Path, required=True)
    make_video.add_argument("--animal-slot", type=int, default=0)
    make_video.add_argument("--pcutoff", type=float, default=0.1)
    make_video.add_argument("--bboxes-pcutoff", type=float, default=0.1)
    make_video.add_argument("--fps", type=float, default=None)
    make_video.set_defaults(func=make_video_command)

    run_all = subparsers.add_parser("run-all")
    add_common_cleaning_args(run_all)
    run_all.add_argument("--video", type=Path, required=True)
    run_all.add_argument("--h5", type=Path, required=True)
    run_all.add_argument("--output-dir", type=Path, required=True)
    run_all.add_argument("--output-prefix", default="206885_v4")
    run_all.add_argument("--pcutoff", type=float, default=0.1)
    run_all.add_argument("--bboxes-pcutoff", type=float, default=0.1)
    run_all.add_argument("--fps", type=float, default=None)
    run_all.set_defaults(func=run_all_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

