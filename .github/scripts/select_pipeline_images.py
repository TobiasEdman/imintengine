#!/usr/bin/env python3
"""Select the pipeline images build-pipeline-images.yml builds for one event.

Pull requests and path-filtered pushes select changed image directories or
the explicit R5 runtime source paths; build-logic changes select every image.
Manual dispatch keeps the existing choice: the requested image, ``ladder-crop-distill`` for a
feature-branch dispatch without input, otherwise all images.

Prints two ``key=value`` lines for ``$GITHUB_OUTPUT``: ``matrix`` (JSON with
an ``include`` list, possibly empty) and ``build`` (``true`` when non-empty).
"""
from __future__ import annotations

import argparse
import json
import sys

IMAGES = {
    "cloud-models": {"context": "docker/cloud-models",
                     "dockerfile": "docker/cloud-models/Dockerfile",
                     "platforms": "linux/amd64,linux/arm64"},
    "c2rcc-snap": {"context": "docker/c2rcc-snap",
                   "dockerfile": "docker/c2rcc-snap/Dockerfile",
                   "platforms": "linux/amd64"},
    "ladder-crop-distill": {"context": ".",
                            "dockerfile": "docker/ladder-crop-distill/Dockerfile",
                            "platforms": "linux/amd64"},
}
# A change to how images are selected or built affects every image.
R5_SOURCE_PATHS = {
    "scripts/r5_tessera_pilot.py", "scripts/run_r5_tessera.py",
    "config/r5_tessera_pilot.json", "imint/training/unified_dataset.py",
}

BUILD_LOGIC = (".github/workflows/build-pipeline-images.yml",
               ".github/scripts/select_pipeline_images.py")


def select(event: str, ref: str, requested: str, changed: list[str]) -> list[str]:
    if event == "pull_request" or (event == "push" and changed):
        if any(path in BUILD_LOGIC for path in changed):
            return list(IMAGES)
        return [name for name in IMAGES
                if any(path.startswith(f"docker/{name}/") for path in changed)
                or (name == "ladder-crop-distill" and any(path in R5_SOURCE_PATHS for path in changed))]
    if not requested and event == "workflow_dispatch" and ref != "refs/heads/main":
        requested = "ladder-crop-distill"
    if requested in ("", "all"):
        return list(IMAGES)
    if requested not in IMAGES:
        raise SystemExit(f"unknown image: {requested}")
    return [requested]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--requested", default="")
    ap.add_argument("--changed-files", type=argparse.FileType(),
                    help="newline-separated paths changed by the pull request")
    args = ap.parse_args()
    if args.event == "pull_request" and args.changed_files is None:
        ap.error("pull_request requires --changed-files")
    changed = ([line.strip() for line in args.changed_files if line.strip()]
               if args.changed_files else [])
    names = select(args.event, args.ref, args.requested, changed)
    matrix = {"include": [{"image": name, **IMAGES[name]} for name in names]}
    sys.stdout.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n"
                     f"build={'true' if names else 'false'}\n")


if __name__ == "__main__":
    main()
