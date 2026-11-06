"""Pre-registered class schemes and frozen-model identity for the forest
separability diagnostic (IM-023).

Standard library only: the job renderer and the run script import this on the
operator's machine, where the probe's numpy/scikit-learn stack need not be
installed. Run as a script it validates a run before anything is rendered or
submitted and prints the evidence tag of each year.
"""
from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import re

# Ordered (name, raw NMD2023 codes) per class; the index is the probe label.
CLASS_SCHEMES = {
    "binary": (("barrblandskog", (113,)), ("lövskog (115–117)", (115, 116, 117))),
    "forest5": (("tallskog", (111,)), ("granskog", (112,)), ("barrblandskog", (113,)),
                ("lövblandad barrskog", (114,)), ("lövskog (115–117)", (115, 116, 117))),
    # Should sumpskog (unified 5) be split by tree type? Same types on wetland.
    "sumpskog5": (("tallskog på våtmark", (121,)), ("granskog på våtmark", (122,)),
                  ("barrblandskog på våtmark", (123,)),
                  ("lövblandad barrskog på våtmark", (124,)),
                  ("lövskog på våtmark (125–127)", (125, 126, 127))),
}
FEATURE_YEARS = (2022, 2023)

# The frozen model whose features are probed. R4 is the default, so a run
# without model arguments is the run the schemes were pre-registered with.
DEFAULT_MODEL_NAME = "r4"
DEFAULT_CHECKPOINT = "/data/checkpoints/ladder/tessera_r4/best_model.pt"
DEFAULT_NUM_CLASSES = 28
# 28: R4 and the IM-023 control. 29: the IM-023 main model, whose class 28
# (barrblandskog) the probe reports. These mirror NUM_CLASSES_BEFORE_BARRBLANDSKOG
# and NUM_UNIFIED_CLASSES in imint.schema.unified_schema, which needs numpy;
# a test holds them equal.
NUM_CLASSES_CHOICES = (28, 29)
BARRBLANDSKOG_CLASS = 28
# Keys of the extraction's sample bundle besides the model's own ``<name>``
# and ``<name>_prediction``; a model name must not collide with any of them.
BUNDLE_KEYS = ("tessera", "raw_nmd", "tile", "xy", "base_label", "distilled_label")
# The name also keys the report and the Job name (a DNS label).
_MODEL_NAME = re.compile(r"[a-z][a-z0-9]{0,11}")


def check_model(name: str, checkpoint: str, num_classes: int) -> None:
    """Refuse a name that cannot key the outputs, or that misnames R4.

    ``r4`` means R4's checkpoint with 28 classes, and R4's checkpoint is only
    ever called ``r4``: otherwise a new model's results would be filed as R4's
    or R4's under another name.
    """
    if not _MODEL_NAME.fullmatch(name) or name in BUNDLE_KEYS:
        raise ValueError(f"model name {name!r} must match {_MODEL_NAME.pattern} "
                         f"and not be a bundle key {BUNDLE_KEYS}")
    if num_classes not in NUM_CLASSES_CHOICES:
        raise ValueError(f"num_classes must be one of {NUM_CLASSES_CHOICES}, got {num_classes}")
    is_r4_name = name == DEFAULT_MODEL_NAME
    is_r4_checkpoint = PurePosixPath(checkpoint) == PurePosixPath(DEFAULT_CHECKPOINT)
    if is_r4_name != is_r4_checkpoint or (is_r4_name and num_classes != DEFAULT_NUM_CLASSES):
        raise ValueError(
            f"model name {DEFAULT_MODEL_NAME!r} is R4 ({DEFAULT_CHECKPOINT}, "
            f"{DEFAULT_NUM_CLASSES} classes) and nothing else; got {name!r}, "
            f"{checkpoint!r}, {num_classes} classes")


def resolve_model(name: str | None, checkpoint: str | None,
                  num_classes: int | None) -> tuple[str, str, int]:
    """All three given, or none of them (R4)."""
    given = [value is not None for value in (name, checkpoint, num_classes)]
    if not any(given):
        return DEFAULT_MODEL_NAME, DEFAULT_CHECKPOINT, DEFAULT_NUM_CLASSES
    if not all(given):
        raise ValueError("--model-name, --checkpoint and --num-classes go together: "
                         "give all three, or none for R4")
    check_model(name, checkpoint, num_classes)
    return name, checkpoint, num_classes


def run_tag(name: str, scheme: str, feature_year: int) -> str:
    """Evidence/Job key. R4 keeps its historical ``<scheme>-<year>`` layout,
    where its results already are; every other model is prefixed by name."""
    stem = f"{scheme}-{feature_year}"
    return stem if name == DEFAULT_MODEL_NAME else f"{name}-{stem}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Validate a probe run and print one evidence tag per year.")
    ap.add_argument("--model-name")
    ap.add_argument("--checkpoint")
    ap.add_argument("--num-classes", type=int)
    ap.add_argument("--scheme", required=True, choices=sorted(CLASS_SCHEMES))
    # Pass years after "--": an option among them is then a bad year, not an option.
    ap.add_argument("years", nargs="+", type=int, choices=FEATURE_YEARS)
    args = ap.parse_args(argv)
    if len(set(args.years)) != len(args.years):
        ap.error(f"a year may be given once: {args.years}")
    try:
        name, _, _ = resolve_model(args.model_name, args.checkpoint, args.num_classes)
    except ValueError as exc:
        ap.error(str(exc))
    print("\n".join(run_tag(name, args.scheme, year) for year in args.years))


if __name__ == "__main__":
    main()
