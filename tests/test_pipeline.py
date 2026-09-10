import copy
import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import yaml

from superanimal_pipeline import legacy
from superanimal_pipeline import processing as p
from superanimal_pipeline.common import read, save, tree
from superanimal_pipeline.config import load
from superanimal_pipeline.runner import Runner


def fixture(root, n=12):
    video = root / "source.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 120)
    )
    predictions = []
    for frame in range(n):
        writer.write(np.full((120, 160, 3), 30 + frame, dtype=np.uint8))
        pose = [[float(30 + i + frame), float(40 + i % 8), 0.9] for i in range(27)]
        if frame in (2, 3):
            pose[0] = [-1.0, -1.0, -1.0]
        predictions.append(
            dict(
                bboxes=[[20.0, 20.0, 80.0, 70.0]], bbox_scores=[0.95], bodyparts=[pose]
            )
        )
    writer.release()
    save(root / "raw.json", predictions)
    columns = pd.MultiIndex.from_product(
        [["fixture"], ["animal0"], legacy.BODY_PARTS, legacy.KEYPOINT_COORDS],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(
        np.asarray([x["bodyparts"][0] for x in predictions])
        .reshape(n, 81)
        .astype(np.float32),
        columns=columns,
    )
    df.to_hdf(root / "raw.h5", key="df_with_missing", format="table")
    with (root / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trial_id", "video", "group", "raw_json", "raw_h5"]
        )
        writer.writeheader()
        writer.writerow(
            dict(
                trial_id="05",
                video="source.mp4",
                group="example",
                raw_json="raw.json",
                raw_h5="raw.h5",
            )
        )
    cfg = dict(
        schema_version=1,
        manifest="trials.csv",
        output="outputs",
        model={"mode": "precomputed"},
    )
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    return root / "config.yaml"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_config_relative_paths_and_zero_id(self):
        cfg, rows = load(self.cfg)
        self.assertEqual(rows[0]["trial_id"], "05")
        self.assertEqual(rows[0]["video"], str((self.root / "source.mp4").resolve()))
        self.assertEqual(cfg["cleaning"]["smoothing_window"], 5)

    def test_config_reject_unknown_and_even_filter(self):
        raw = yaml.safe_load(self.cfg.read_text())
        raw["cleaning"] = {"smoothing_window": 4}
        self.cfg.write_text(yaml.safe_dump(raw))
        with self.assertRaisesRegex(ValueError, "odd"):
            load(self.cfg)
        raw["cleaning"] = {"median": True}
        self.cfg.write_text(yaml.safe_dump(raw))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load(self.cfg)

    def test_path_traversal_id_rejected(self):
        path = self.root / "trials.csv"
        path.write_text(path.read_text().replace("05,", "../bad,"))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            load(self.cfg)

    def test_historical_core_and_segmented_equivalence(self):
        predictions = read(self.root / "raw.json")
        b, k, _ = legacy.extract_slot_tables(predictions, 0, 0.1, 0.5)
        bc, kc = legacy.interpolate_tables(b, k)
        pd.testing.assert_frame_equal(
            p.segmented_smooth(kc, 5, []),
            legacy.smooth_keypoints(kc, 5),
            check_exact=True,
        )
        expected = kc.copy()
        expected.iloc[:6] = legacy.smooth_keypoints(kc.iloc[:6], 5).to_numpy()
        expected.iloc[6:] = legacy.smooth_keypoints(kc.iloc[6:], 5).to_numpy()
        pd.testing.assert_frame_equal(
            p.segmented_smooth(kc, 5, [6]), expected, check_exact=True
        )
        for invalid in [[0], [len(kc)], [-1], [2.5], [True]]:
            with self.assertRaises(ValueError):
                p.segmented_smooth(kc, 5, invalid)

    def test_missing_gap_is_not_repaired_by_mean(self):
        predictions = read(self.root / "raw.json")
        b, k, _ = legacy.extract_slot_tables(predictions, 0, 0.1, 0.5)
        self.assertEqual(len(legacy.collect_gaps(b, k, 1)["keypoint_gaps"]), 1)
        self.assertTrue(legacy.collect_gaps(b, k, 1)["keypoint_gaps"][0]["long_gap"])
        bc, kc = legacy.interpolate_tables(b, k)
        self.assertFalse(kc.isna().any().any())
        self.assertEqual(kc.iloc[2][("nose", "x")], 32.0)
        np.testing.assert_array_equal(
            p.segmented_smooth(kc, 5, []).xs("likelihood", axis=1, level="coords"),
            kc.xs("likelihood", axis=1, level="coords"),
        )

    def test_fast_serialization_exactly_matches_legacy(self):
        predictions = read(self.root / "raw.json")
        predictions[1] = dict(bboxes=[], bbox_scores=[], bodyparts=[], note="preserved")
        before = copy.deepcopy(predictions)
        b, k, _ = legacy.extract_slot_tables(predictions, 0, 0.1, 0.5)
        for bbox, kp in [(b, k), legacy.interpolate_tables(b, k)]:
            self.assertEqual(
                p.serialize_predictions(predictions, bbox, kp),
                legacy.update_predictions(predictions, bbox, kp, 0),
            )
        self.assertEqual(predictions, before)

    def test_all_missing_refused(self):
        predictions = read(self.root / "raw.json")
        for f in predictions:
            f["bodyparts"][0][0] = [-1.0, -1.0, -1.0]
        save(self.root / "raw.json", predictions)
        df = pd.read_hdf(self.root / "raw.h5")
        df.iloc[:, :3] = -1.0
        df.to_hdf(self.root / "raw.h5", key="df_with_missing", mode="w")
        out = self.root / "failed"
        out.mkdir()
        with self.assertRaisesRegex(ValueError, "Unfillable"):
            p.clean(
                self.root,
                out,
                load(self.cfg)[0]["cleaning"],
                [],
                self.root / "source.mp4",
            )
        self.assertTrue((out / "gaps.json").exists())
        self.assertFalse((out / "cleaned_27.h5").exists())

    def test_end_to_end_precomputed_and_resume(self):
        c, rows = load(self.cfg)
        runner = Runner(c, rows)
        runner.run()
        self.assertEqual(runner.verify()["frames"], 12)
        output = Path(c["output"])
        before = tree(output / "receipts")
        (output / "data/05").unlink()
        runner.run()
        self.assertEqual(before, tree(output / "receipts"))
        self.assertTrue((output / "data/05").is_symlink())
        df = pd.read_hdf(output / "data/05/02_cleaned.h5")
        full = pd.read_hdf(output / "data/05/02_cleaned_27kp.h5")
        self.assertEqual(df.shape, (12, 30))
        pd.testing.assert_frame_equal(df, full.loc[:, df.columns], check_exact=True)
        self.assertEqual(
            p.video_info(output / "data/05/03_labeled_cleaned.mp4", True)[
                "decoded_frames"
            ],
            12,
        )

    def test_incomplete_step_restarts_without_adopting_it(self):
        c, rows = load(self.cfg)
        runner = Runner(c, rows)
        with patch.object(p, "clean", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runner.run()
        self.assertFalse(runner.receipt_file("05", "clean").exists())
        runner.run()
        self.assertEqual(len(list((runner.root / "work/05").glob("clean-*"))), 2)

    def test_changed_outputs_and_config_refused(self):
        c, rows = load(self.cfg)
        runner = Runner(c, rows)
        runner.run()
        (runner.root / "data/05/qc.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.verify()
        changed = copy.deepcopy(c)
        changed["export"]["keypoints"] = 27
        with self.assertRaisesRegex(ValueError, "changed"):
            Runner(changed, rows).run()

    def test_changed_inputs_refused(self):
        c, rows = load(self.cfg)
        Runner(c, rows).run()
        with (self.root / "raw.json").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            Runner(c, rows).run()

    def test_paired_cut_and_crop(self):
        f = read(self.root / "raw.json")
        for i, frame in enumerate(f):
            frame["bboxes"][0][0] = float(20 + i + (50 if i >= 6 else 0))
        save(self.root / "a.json", f)
        save(self.root / "b.json", f)
        cuts, _ = p.paired_cuts([self.root / "a.json", self.root / "b.json"], 0.999)
        self.assertEqual(cuts, [6])
        dst = self.root / "cropped"
        dst.mkdir()
        p.prepare_video(
            dict(video=str(self.root / "source.mp4"), crop=[10, 10, 100, 80]), dst
        )
        info = p.video_info(dst / "video.mp4", True)
        self.assertEqual(
            (info["width"], info["height"], info["decoded_frames"]), (100, 80, 12)
        )

    def test_27_point_export_and_no_video(self):
        c, rows = load(self.cfg)
        c["export"].update(keypoints=27, labeled_video=False)
        Runner(c, rows).run()
        folder = Path(c["output"]) / "data/05"
        self.assertEqual(pd.read_hdf(folder / "02_cleaned.h5").shape, (12, 81))
        self.assertFalse((folder / "03_labeled_cleaned.mp4").exists())

    def test_raw_schema_and_frame_mismatch(self):
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            p.validate_predictions(self.root / "raw.json", self.root / "raw.h5", 13)
        df = pd.read_hdf(self.root / "raw.h5")
        df = df.iloc[:, :30]
        df.to_hdf(self.root / "bad.h5", key="df_with_missing")
        with self.assertRaises(ValueError):
            p.pose_table(self.root / "bad.h5")
        df = pd.read_hdf(self.root / "raw.h5")
        df.iloc[0, 0] += 2
        df.to_hdf(self.root / "mismatch.h5", key="df_with_missing")
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            p.validate_predictions(
                self.root / "raw.json", self.root / "mismatch.h5", 12
            )

    def test_inference_api_is_fixed_model_not_training(self):
        c, rows = load(self.cfg)
        calls = []

        def fake(videos, **kwargs):
            calls.append(kwargs)
            dest = Path(kwargs["dest_folder"])
            shutil.copy2(self.root / "raw.h5", dest / "model.h5")
            shutil.copy2(self.root / "raw.json", dest / "model_before_adapt.json")

        for mode in ["checkpoint", "zero-shot"]:
            dest = self.root / mode
            dest.mkdir()
            model = {
                **c["model"],
                "mode": mode,
                "pose_checkpoint": "pose.pt",
                "detector_checkpoint": "detector.pt",
            }
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False), __version__="mock"
            )
            with patch.dict(
                sys.modules,
                {
                    "deeplabcut": SimpleNamespace(video_inference_superanimal=fake),
                    "torch": fake_torch,
                },
            ):
                p.infer(rows[0], model, self.root / "source.mp4", dest)
            self.assertFalse(calls[-1]["video_adapt"])
            self.assertEqual(calls[-1]["max_individuals"], 1)
            self.assertEqual(
                "customized_pose_checkpoint" in calls[-1], mode == "checkpoint"
            )

    def test_manual_cuts_pipeline(self):
        save(self.root / "cuts.json", {"05": [6]})
        raw = yaml.safe_load(self.cfg.read_text())
        raw["cuts"] = {"mode": "manual", "file": "cuts.json"}
        self.cfg.write_text(yaml.safe_dump(raw))
        c, rows = load(self.cfg)
        r = Runner(c, rows)
        r.run()
        self.assertEqual(read(r.root / "data/05/qc.json")["boundary_frames"], [6])

    def test_metadata_change_refused(self):
        c, rows = load(self.cfg)
        r = Runner(c, rows)
        r.run()
        with (r.root / "meta.csv").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "Metadata changed"):
            r.verify()
        with self.assertRaisesRegex(ValueError, "Metadata changed"):
            r.run()

    def test_example_yaml_off_is_string(self):
        raw = yaml.safe_load(
            (Path(__file__).parents[1] / "examples/config.yaml").read_text()
        )
        self.assertEqual(raw["cuts"]["mode"], "off")

    def test_explicit_adaptation_api_contract(self):
        from superanimal_pipeline.cli import adapt

        calls = []

        def fake(videos, **kwargs):
            calls.append(kwargs)
            folder = Path(kwargs["dest_folder"]) / "checkpoints"
            folder.mkdir()
            (folder / "snapshot-hrnet_w32-004.pt").write_bytes(b"TEST_NOT_A_MODEL")
            (folder / "snapshot-fasterrcnn_resnet50_fpn_v2-004.pt").write_bytes(
                b"TEST_NOT_A_MODEL"
            )

        out = self.root / "adapt"
        with (
            patch.dict(
                sys.modules,
                {"deeplabcut": SimpleNamespace(video_inference_superanimal=fake)},
            ),
            patch("superanimal_pipeline.cli.doctor", return_value={"test": True}),
        ):
            adapt(self.root / "source.mp4", out)
            self.assertEqual(calls[0]["detector_epochs"], 4)
            self.assertEqual(calls[0]["pose_epochs"], 4)
            self.assertTrue(calls[0]["video_adapt"])
            self.assertEqual(read(out / "adaptation.json")["status"], "COMPLETE")
            with self.assertRaises(ValueError):
                adapt(self.root / "source.mp4", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
