import csv
import fcntl
import os
import tempfile
import time
from pathlib import Path

from . import processing as p
from .common import digest, read, require, save, sha, tree
from .config import fingerprint


class Runner:
    def __init__(self, config, rows):
        self.config, self.rows = config, rows
        self.root = Path(config["output"])
        self.frozen = None

    def event(self, event, **extra):
        data = dict(
            time=time.strftime("%Y-%m-%d %H:%M:%S"),
            pid=os.getpid(),
            event=event,
            **extra,
        )
        print("SAP " + __import__("json").dumps(data, ensure_ascii=False), flush=True)
        save(self.root / "status.json", data)

    def freeze(self, create=False):
        current = fingerprint(self.config, self.rows)
        dest = self.root / "provenance.json"
        if dest.exists():
            require(
                read(dest) == current,
                "Frozen inputs/config/code/environment changed: choose a NEW output directory",
            )
        else:
            require(create, "Run has not started")
            require(
                not any(x.name != ".lock" for x in self.root.iterdir()),
                "Refuse to adopt a nonempty output directory",
            )
            save(dest, current)
        self.frozen = digest(current)

    def receipt_file(self, tid, stage):
        return self.root / "receipts" / f"{tid}--{stage}.json"

    def receipt(self, tid, stage):
        r = read(self.receipt_file(tid, stage))
        folder = (self.root / r["directory"]).resolve()
        require(
            folder.is_relative_to(self.root.resolve()),
            "Receipt escapes output directory",
        )
        require(
            folder.is_dir() and r["files"] and tree(folder) == r["files"],
            f"{tid}/{stage}: output hash mismatch",
        )
        require(r["frozen"] == self.frozen, "Receipt provenance mismatch")
        return r, folder

    def stage(self, tid, name, deps, execute):
        if self.receipt_file(tid, name).exists():
            r, folder = self.receipt(tid, name)
            require(r["dependencies"] == deps, "Cached dependency mismatch")
            self.event("CACHED", trial=tid, step=name)
            return folder
        parent = self.root / "work" / tid
        parent.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=name + "-", dir=parent))
        self.event("STEP_START", trial=tid, step=name, attempt=str(folder))
        execute(folder)
        save(
            self.receipt_file(tid, name),
            dict(
                directory=str(folder.relative_to(self.root)),
                files=tree(folder),
                dependencies=deps,
                frozen=self.frozen,
            ),
        )
        self.event("STEP_COMPLETE", trial=tid, step=name)
        return folder

    def dep(self, tid, stage):
        self.receipt(tid, stage)
        return sha(self.receipt_file(tid, stage))

    def cut_step(self, raw, prepared, folder):
        mode = self.config["cuts"]["mode"]
        cuts = {r["trial_id"]: [] for r in self.rows}
        evidence = {}
        if mode == "manual":
            cuts = read(self.config["cuts"]["file"])
            require(
                set(cuts) == set(raw),
                "Manual cuts must list every trial, including empty lists",
            )
        elif mode == "paired":
            for pair in {r["pair_id"] for r in self.rows}:
                ids = [r["trial_id"] for r in self.rows if r["pair_id"] == pair]
                info = [read(prepared[t] / "video_info.json") for t in ids]
                require(
                    abs(info[0]["fps"] - info[1]["fps"]) < 1e-4, "Pair FPS mismatch"
                )
                require(
                    info[0]["decoded_frames"] == info[1]["decoded_frames"],
                    "Pair frame mismatch",
                )
                b, thresholds = p.paired_cuts(
                    [raw[t] / "raw.json" for t in ids], self.config["cuts"]["quantile"]
                )
                for tid in ids:
                    cuts[tid] = b
                evidence[pair] = dict(
                    trials=ids, thresholds_px=thresholds, candidate_frames=b
                )
        for tid, indices in cuts.items():
            n = read(prepared[tid] / "video_info.json")["decoded_frames"]
            require(
                isinstance(indices, list)
                and all(type(i) is int and 0 < i < n for i in indices),
                "Invalid cut indices",
            )
        save(folder / "cuts.json", cuts)
        save(
            folder / "audit.json",
            dict(
                mode=mode,
                pairs=evidence,
                note="Candidates only; smooth within segments; interpolation is NOT segmented; no frame deletion",
            ),
        )

    def publish(self, tid, folder):
        dest = self.root / "data" / tid
        dest.parent.mkdir(exist_ok=True)
        if dest.exists() or dest.is_symlink():
            require(
                dest.is_symlink() and dest.resolve() == folder.resolve(),
                "Workspace publication mismatch",
            )
        else:
            dest.symlink_to(
                os.path.relpath(folder, dest.parent), target_is_directory=True
            )

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # No new status or artifact is written if frozen identity changed.
            self.freeze(create=True)
            try:
                self.event("RUNNING", trials=len(self.rows))
                prepared, raw = {}, {}
                for i, row in enumerate(self.rows, 1):
                    tid = row["trial_id"]
                    self.event(
                        "TRIAL",
                        index=i,
                        total=len(self.rows),
                        trial=tid,
                        current_file=row["video"],
                    )
                    prepared[tid] = self.stage(
                        tid, "prepare", {}, lambda d, row=row: p.prepare_video(row, d)
                    )
                    raw[tid] = self.stage(
                        tid,
                        "infer",
                        {"prepare": self.dep(tid, "prepare")},
                        lambda d, row=row: p.infer(
                            row,
                            self.config["model"],
                            prepared[row["trial_id"]] / "video.mp4",
                            d,
                        ),
                    )
                cuts_dir = self.stage(
                    "_cohort",
                    "cuts",
                    {t: self.dep(t, "infer") for t in raw},
                    lambda d: self.cut_step(raw, prepared, d),
                )
                cuts = read(cuts_dir / "cuts.json")
                for i, row in enumerate(self.rows, 1):
                    tid = row["trial_id"]
                    self.event(
                        "TRIAL",
                        index=i,
                        total=len(self.rows),
                        trial=tid,
                        current_file=row["video"],
                    )
                    cleaned = self.stage(
                        tid,
                        "clean",
                        {
                            "infer": self.dep(tid, "infer"),
                            "cuts": self.dep("_cohort", "cuts"),
                        },
                        lambda d: p.clean(
                            raw[tid],
                            d,
                            self.config["cleaning"],
                            cuts[tid],
                            prepared[tid] / "video.mp4",
                        ),
                    )
                    exported = self.stage(
                        tid,
                        "export",
                        {
                            "clean": self.dep(tid, "clean"),
                            "prepare": self.dep(tid, "prepare"),
                        },
                        lambda d: p.export(
                            prepared[tid] / "video.mp4",
                            cleaned,
                            d,
                            self.config["export"],
                            lambda n, total: self.event(
                                "VIDEO_PROGRESS", trial=tid, frame=n, total_frames=total
                            ),
                        ),
                    )
                    self.publish(tid, exported)
                self.write_meta()
                result = self.verify()
                save(self.root / "verification.json", result)
                self.event("COMPLETE", **result)
            except BaseException as error:
                self.event(
                    "FAILED_OR_INTERRUPTED", error=f"{type(error).__name__}: {error}"
                )
                raise

    def write_meta(self):
        path = self.root / "meta.csv"
        receipt = self.root / "meta_receipt.json"
        if path.exists() or receipt.exists():
            require(
                path.is_file() and receipt.is_file(),
                "Partial metadata publication; preserve and inspect",
            )
            require(
                read(receipt) == {"sha256": sha(path), "frozen": self.frozen},
                "Metadata changed; refuse silent repair",
            )
            return
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as f:
            fields = [
                "trial_id",
                "animal_id",
                "group",
                "species",
                "anno_type",
                "key_points_path",
                "video_file",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                t = row["trial_id"]
                writer.writerow(
                    dict(
                        trial_id=t,
                        animal_id=row["animal_id"],
                        group=row["group"],
                        species="mouse",
                        anno_type="superanimal_h5",
                        key_points_path=f"data/{t}/02_cleaned.h5",
                        video_file=f"data/{t}/01_original.mp4",
                    )
                )
        os.replace(temporary, path)
        save(
            self.root / "meta_receipt.json",
            {"sha256": sha(path), "frozen": self.frozen},
        )

    def verify(self):
        self.freeze()
        cut_receipt, _ = self.receipt("_cohort", "cuts")
        require(
            cut_receipt["dependencies"]
            == {r["trial_id"]: self.dep(r["trial_id"], "infer") for r in self.rows},
            "Cuts dependency changed",
        )
        frames = 0
        for row in self.rows:
            tid = row["trial_id"]
            expected = {
                "prepare": {},
                "infer": {"prepare": self.dep(tid, "prepare")},
                "clean": {
                    "infer": self.dep(tid, "infer"),
                    "cuts": self.dep("_cohort", "cuts"),
                },
                "export": {
                    "clean": self.dep(tid, "clean"),
                    "prepare": self.dep(tid, "prepare"),
                },
            }
            for name in expected:
                receipt, folder = self.receipt(tid, name)
                require(
                    receipt["dependencies"] == expected[name],
                    f"{tid}/{name}: dependency mismatch",
                )
            require(
                (self.root / "data" / tid).is_symlink()
                and (self.root / "data" / tid).resolve() == folder.resolve(),
                "Missing/bad published data",
            )
            frames += read(folder / "qc.json")["frames"]
        meta = read(self.root / "meta_receipt.json")
        require(
            meta == {"sha256": sha(self.root / "meta.csv"), "frozen": self.frozen},
            "Metadata changed",
        )
        return dict(
            trials=len(self.rows),
            frames=frames,
            keypoints=self.config["export"]["keypoints"],
            evidence="TECHNICALLY_VERIFIED_NOT_GROUND_TRUTH",
            outputs="data/",
            no_behavior_analysis=True,
        )
