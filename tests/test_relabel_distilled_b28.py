"""IM-023: a further model's distilled labels relabelled onto the existing class 28 NMD labels.

The chain fixture of test_relabel_barrblandskog builds the stored and the
canonical labels with the real builder and distiller. ``relabel_barrblandskog
run`` then writes the NMD directory with the first cohort, as the TESSERA job
did, and a second head stands in for Prithvi-600M. Its relabelled sidecars
must be what a canonical distillation gives and what ``run`` would write,
while the NMD directory is only read.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import relabel_barrblandskog as relabel
import relabel_distilled_b28 as reuse
from tests.test_relabel_barrblandskog import args, chain  # noqa: F401  (chain is a fixture)

SECOND_HEAD = 11   # seed of the stand-in head; the fixture's stored labels use 7


@pytest.fixture
def root(chain):
    root, distil = chain
    assert relabel.main(["run", *args(root), "--nmd-out", str(root / "nmd_b28"),
                         "--distill-out", str(root / "first_b28")]) == 0
    distil(root / "nmd_old", root / "second", seed=SECOND_HEAD)        # /cephfs/distill/prithvi600m_r2
    distil(root / "canon_nmd", root / "second_canon", seed=SECOND_HEAD)
    return root


def reuse_args(root: Path) -> list[str]:
    return ["--nmd-in", str(root / "nmd_old"), "--nmd-b28", str(root / "nmd_b28"),
            "--distill-in", str(root / "second"), "--distill-out", str(root / "second_b28")]


def rewrite(path: Path, **changes) -> None:
    np.savez_compressed(path, **(relabel.load(path) | changes))


def test_outputs_are_the_canonical_distillation_and_what_run_writes(root):
    second, first = (relabel.load(root / d / "T1.npz")["label"] for d in ("second", "stored"))
    assert (second != first).any()                       # a different head, not a copy
    assert reuse.main(reuse_args(root)) == 0
    written = relabel.load(root / "second_b28" / "T1.npz")
    canonical = relabel.load(root / "second_canon" / "T1.npz")
    for key in canonical:
        np.testing.assert_array_equal(written[key], canonical[key], err_msg=key)
        assert written[key].dtype == canonical[key].dtype, key
    relabel.run(root / "nmd_old", root / "second", root / "run_nmd", root / "run_distill")
    by_run = relabel.load(root / "run_distill" / "T1.npz")
    assert written.keys() == by_run.keys()
    for key in by_run:
        np.testing.assert_array_equal(written[key], by_run[key], err_msg=key)


def test_inputs_are_only_read(root):
    def snapshot():
        return {(d, p.name): (p.read_bytes(), p.stat().st_mtime_ns)
                for d in ("nmd_b28", "nmd_old", "second", "first_b28") for p in (root / d).iterdir()}

    before = snapshot()
    assert reuse.main(reuse_args(root)) == 0
    assert snapshot() == before


def test_refuses_to_write_over_another_cohort(root):
    with pytest.raises(ValueError, match="cohort the reused manifest was written for"):
        reuse.run(root / "nmd_old", root / "nmd_b28", root / "second", root / "first_b28")
    assert reuse.main(reuse_args(root)) == 0
    with pytest.raises(ValueError, match="relabelled another distillation"):  # second_b28 is second's
        reuse.run(root / "nmd_old", root / "nmd_b28", root / "stored", root / "second_b28")


def test_a_skipped_output_must_still_be_the_rule(root):
    assert reuse.main(reuse_args(root)) == 0
    out = root / "second_b28" / "T1.npz"
    label = relabel.load(out)["label"].copy()
    label[0, 0] = 9 if label[0, 0] != 9 else 8
    rewrite(out, label=label)                                # stamp intact, content not the rule's
    with pytest.raises(ValueError, match="not the rule applied to the current NMD label"):
        reuse.main(reuse_args(root))


def test_manifest_carries_the_reused_nmd_totals_and_resumes(root):
    assert reuse.main(reuse_args(root)) == 0
    manifest = json.loads((root / "second_b28" / relabel.MANIFEST).read_text())
    nmd = json.loads((root / "nmd_b28" / relabel.MANIFEST).read_text())
    assert manifest["rule"] == relabel.RULE
    assert manifest["outputs"] == {"nmd": str(root / "nmd_b28"), "distilled": str(root / "second_b28")}
    assert manifest["nmd_reused"]["sha256"] == relabel.digest(root / "nmd_b28" / relabel.MANIFEST)
    assert {k: manifest[k] for k in reuse.NMD_TOTALS} == {k: nmd[k] for k in reuse.NMD_TOTALS}
    # The totals run reports for this cohort (the fixture's NMD directory has its one tile).
    by_run = relabel.run(root / "nmd_old", root / "second", root / "run_nmd", root / "run_distill")
    totals = ("nmd_tiles", "distilled_tiles", "nmd_changed", "distilled_changed", "class_28_pixels")
    assert {k: manifest[k] for k in totals} == {k: by_run[k] for k in totals}
    assert (manifest["written"], manifest["skipped_current"]) == (1, 0)
    assert manifest["distilled_changed"] > 0

    stamp = (root / "second_b28" / "T1.npz").stat().st_mtime_ns
    assert reuse.main(reuse_args(root)) == 0
    again = json.loads((root / "second_b28" / relabel.MANIFEST).read_text())
    assert (again["written"], again["skipped_current"]) == (0, 1)
    assert again["distilled_changed"] == manifest["distilled_changed"]
    assert (root / "second_b28" / "T1.npz").stat().st_mtime_ns == stamp


def test_refuses_nmd_labels_that_are_not_the_rule_applied(root):
    path = root / "nmd_b28" / "T1.npz"
    label = relabel.load(path)["label"].copy()
    label[0, 0] = 9 if label[0, 0] != 9 else 8
    rewrite(path, label=label)
    with pytest.raises(ValueError, match="not the rule applied"):
        reuse.main(reuse_args(root))
    assert not (root / "second_b28" / relabel.MANIFEST).exists()
    assert not (root / "second_b28" / "T1.npz").exists()
    path.unlink()
    with pytest.raises(ValueError, match="missing or not the rule applied"):
        reuse.main(reuse_args(root))


@pytest.mark.parametrize("change, message", [
    ({"rule": "other"}, "expected " + relabel.RULE),
    ({"inputs": {"nmd": "/elsewhere", "distilled": "x"}}, "not .* into"),
])
def test_refuses_a_foreign_nmd_manifest(root, change, message):
    path = root / "nmd_b28" / relabel.MANIFEST
    path.write_text(json.dumps(json.loads(path.read_text()) | change))
    with pytest.raises(ValueError, match=message):
        reuse.main(reuse_args(root))
    path.unlink()
    with pytest.raises(ValueError, match="not relabelled"):
        reuse.main(reuse_args(root))


def test_refuses_inputs_as_output_orphans_and_raw_mismatch(root):
    for directory in ("nmd_old", "nmd_b28", "second"):
        with pytest.raises(ValueError, match="never an input"):
            reuse.run(root / "nmd_old", root / "nmd_b28", root / "second", root / directory)
    raw = relabel.load(root / "second" / "T1.npz")["nmd_label_raw"]
    rewrite(root / "second" / "T1.npz", nmd_label_raw=np.zeros_like(raw))
    with pytest.raises(ValueError, match="raw NMD differs"):
        reuse.main(reuse_args(root))
    (root / "second" / "T1.npz").rename(root / "second" / "T0.npz")
    with pytest.raises(ValueError, match="without an NMD sidecar"):
        reuse.main(reuse_args(root))
