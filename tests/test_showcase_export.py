import hashlib
import json
from pathlib import Path
import pytest
from scripts.export_showcase import export_artifacts


def fixture_roots(tmp_path):
    source, site = tmp_path / "source", tmp_path / "site"
    (source / "docs/showcase/demo").mkdir(parents=True)
    (source / "docs/js").mkdir()
    (site / "docs").mkdir(parents=True)
    (site / "docs/index.html").write_text("site")
    (source / "docs/js/app.js").write_text("must not export")
    (source / "docs/showcase/demo/result.json").write_text('{"value": 1}')
    return source, site


def test_export_round_trip_and_receipt(tmp_path):
    source, site = fixture_roots(tmp_path)
    result = export_artifacts(source, site)
    path = Path("docs/showcase/demo/result.json")
    assert (site / path).read_bytes() == (source / path).read_bytes()
    assert result['files'] == [{'path': str(path), 'sha256': hashlib.sha256((site / path).read_bytes()).hexdigest()}]
    assert json.loads((site / 'ARTIFACT_IMPORT.json').read_text()) == result
    assert not (site / 'docs/js/app.js').exists()


def test_conflicting_artifact_requires_replace(tmp_path):
    source, site = fixture_roots(tmp_path)
    export_artifacts(source, site)
    (source / 'docs/showcase/demo/result.json').write_text('{"value": 2}')
    with pytest.raises(FileExistsError):
        export_artifacts(source, site)
    assert json.loads((site / 'docs/showcase/demo/result.json').read_text()) == {'value': 1}
    export_artifacts(source, site, replace=True)
    assert json.loads((site / 'docs/showcase/demo/result.json').read_text()) == {'value': 2}


def test_symlink_cannot_escape_showcase(tmp_path):
    source, site = fixture_roots(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (site / 'docs/showcase').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes showcase'):
        export_artifacts(source, site)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("receipt", [False, True])
def test_symlink_cannot_overwrite_source_or_external_receipt(tmp_path, receipt):
    source, site = fixture_roots(tmp_path)
    victim = tmp_path / "external.json" if receipt else site / "docs/js/app.js"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("keep this")
    link = site / ("ARTIFACT_IMPORT.json" if receipt else "docs/showcase/demo/result.json")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="Symlink destination"):
        export_artifacts(source, site, replace=True)
    assert victim.read_text() == "keep this"
    if receipt:
        assert not (site / "docs/showcase/demo/result.json").exists()


def test_internal_parent_symlink_is_rejected_before_copy(tmp_path):
    source, site = fixture_roots(tmp_path)
    (site / "docs/js").mkdir()
    (site / "docs/showcase").symlink_to(site / "docs/js", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink destination"):
        export_artifacts(source, site, replace=True)
    assert list((site / "docs/js").iterdir()) == []
