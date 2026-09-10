import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

from .common import read, require, save, sha


def doctor():
    result = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": {},
    }
    for name in [
        "numpy",
        "pandas",
        "tables",
        "opencv-python",
        "PyYAML",
        "deeplabcut",
        "torch",
        "torchvision",
    ]:
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
    except ImportError:
        result["cuda_available"] = False
    result["scope"] = (
        "Inference needs DLC; cleaning/precomputed mode can run on CPU. No install or model download performed."
    )
    return result


def adapt(video, output):
    from .processing import video_info

    video = Path(video).resolve()
    output = Path(output).resolve()
    require(
        video.suffix.lower() == ".mp4",
        "Adapt accepts an already cropped single-mouse MP4",
    )
    info = video_info(video, decode=True)
    require(
        not output.exists(), "Adaptation output already exists; use a new directory"
    )
    output.mkdir(parents=True)
    parameters = dict(
        superanimal_name="superanimal_topviewmouse",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        video_adapt=True,
        max_individuals=1,
        batch_size=1,
        detector_batch_size=1,
        video_adapt_batch_size=8,
        pcutoff=0.1,
        bbox_threshold=0.9,
        pseudo_threshold=0.1,
        detector_epochs=4,
        pose_epochs=4,
        create_labeled_video=False,
    )
    import shutil

    work = output / "adapt_input.mp4"
    shutil.copy2(video, work)
    save(
        output / "provenance.json",
        dict(
            video=str(video),
            sha256=sha(video),
            video_info=info,
            parameters=parameters,
            environment=doctor(),
        ),
    )
    try:
        import deeplabcut

        deeplabcut.video_inference_superanimal(
            [str(work)], dest_folder=str(output), **parameters
        )
        pose = list(output.rglob("snapshot-hrnet_w32-004.pt"))
        detector = list(output.rglob("snapshot-fasterrcnn_resnet50_fpn_v2-004.pt"))
        require(
            len(pose) == 1 and len(detector) == 1,
            "Expected one 004 checkpoint for each model",
        )
        result = dict(
            status="COMPLETE",
            pose_checkpoint=str(pose[0]),
            detector_checkpoint=str(detector[0]),
            hashes={str(f.relative_to(output)): sha(f) for f in [pose[0], detector[0]]},
            note="New video adaptation; not evidence of improved pose accuracy",
        )
        save(output / "adaptation.json", result)
        print(json.dumps(result, indent=2))
    except BaseException as e:
        save(output / "status.json", dict(status="FAILED_OR_INTERRUPTED", error=str(e)))
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Single-mouse SuperAnimal pipeline. No automatic training or behavior classification."
    )
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Read-only dependency/GPU information")
    for name in ["check", "run", "verify", "status"]:
        q = subs.add_parser(name)
        q.add_argument("--config", required=True, type=Path)
    q = subs.add_parser(
        "adapt", help="Explicit one-video 4+4 epoch adaptation; never implicit in run"
    )
    q.add_argument("--video", required=True)
    q.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
        elif args.command == "adapt":
            adapt(args.video, args.output)
        else:
            from .config import fingerprint, load
            from .runner import Runner

            config, rows = load(args.config)
            runner = Runner(config, rows)
            if args.command == "check":
                result = fingerprint(config, rows)
                from .processing import video_info

                videos = {r["trial_id"]: video_info(r["video"]) for r in rows}
                print(
                    json.dumps(
                        dict(
                            status="PREFLIGHT_ONLY",
                            trials=len(rows),
                            mode=config["model"]["mode"],
                            output=config["output"],
                            inputs_hashed=len(result["inputs"]),
                            videos=videos,
                        ),
                        indent=2,
                    )
                )
            elif args.command == "run":
                runner.run()
            elif args.command == "verify":
                print(json.dumps(runner.verify(), indent=2))
            else:
                dest = runner.root / "status.json"
                print(
                    json.dumps(
                        read(dest) if dest.exists() else {"event": "NOT_STARTED"},
                        indent=2,
                    )
                )
                print(
                    "Last recorded event only; inspect tmux/logs to determine liveness."
                )
    except (Exception, KeyboardInterrupt) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(130 if isinstance(e, KeyboardInterrupt) else 1)


if __name__ == "__main__":
    main()
