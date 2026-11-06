"""The HTML reports label NMD content with the product that produced it."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from imint.exporters import html_report

_SIXTY = {"pixel_count": 6, "fraction": 0.6}
_FORTY = {"pixel_count": 4, "fraction": 0.4}
_TWENTY = {"pixel_count": 2, "fraction": 0.2}

STATS_2023 = {
    "product": 2023,
    "class_stats": {
        "level1": {"forest": _SIXTY, "outside": _FORTY},
        "level2": {"4211": _TWENTY, "111": _FORTY},
    },
    "cross_reference": {
        "change_detection": {
            "4211": {"change_fraction": 0.5, "changed_pixels": 1, "total_pixels": 2, "mean_dnbr": 0.1},
        },
    },
}


@pytest.fixture(autouse=True)
def _offline_libs(monkeypatch):
    monkeypatch.setattr(html_report, "_fetch_lib", lambda *args, **kwargs: "")


def test_chart_data_uses_the_2023_names_and_colours():
    data = html_report._build_chart_data(STATS_2023, 2023)
    assert data["l2"]["labels"] == ["Tallskog på fastmark", "Torr buskdominerad mark"]
    assert data["l2"]["colors"] == ["rgba(109,139,5,0.85)", "rgba(204,214,158,0.85)"]
    assert data["change"]["labels"] == ["Torr buskdominerad mark"]
    assert data["change"]["borders"] == ["#ccd69e"]


def test_chart_data_needs_its_product():
    with pytest.raises(ValueError, match="needs its product"):
        html_report._build_chart_data(STATS_2023, None)
    assert html_report._build_chart_data({}, None) == {}


def test_map_legend_lists_the_2023_classes_present_and_outside():
    assert html_report._nmd_map_legend(2023, STATS_2023) == [
        {"color": "#6d8b05", "label": "Tallskog på fastmark"},
        {"color": "#ccd69e", "label": "Torr buskdominerad mark"},
        {"color": "#808080", "label": "Utanför NMD 2023"},
    ]
    assert html_report._nmd_map_legend(2018, {}) is html_report._NMD2018_MAP_LEGEND


def test_summary_card_names_the_2023_klassgrupp():
    summary = {"analyzers": [{"name": "nmd", "outputs": {"class_stats": STATS_2023["class_stats"]}}]}
    html = html_report._build_summary_cards(summary, 2023)
    assert "NMD 2023 Marktäcke" in html and "60.0% Skogsmark" in html


def test_engine_report_shows_the_2023_legend(tmp_path):
    rgb, nmd = tmp_path / "rgb.png", tmp_path / "nmd.png"
    for path in (rgb, nmd):
        Image.new("RGB", (4, 4)).save(path)
    out = tmp_path / "report.html"
    html_report.save_html_report(
        {"rgb": str(rgb), "nmd": str(nmd)}, STATS_2023, {}, (4, 4), "2023-07-01", str(out),
        nmd_product=2023,
    )
    html = out.read_text()
    assert "NMD 2023 Marktäcke (Nivå 2)" in html
    assert "Torr buskdominerad mark" in html and "Utanför NMD 2023" in html
    assert "Sumpsk. tall" not in html  # no NMD 2018 legend
    with pytest.raises(ValueError, match="needs its product"):
        html_report.save_html_report(
            {"nmd": str(nmd)}, STATS_2023, {}, (4, 4), "d", str(out), nmd_product=None)


def test_showcase_tabs_refuse_nmd_2023_and_read_legacy_stats_as_2018(tmp_path):
    fire, marine = tmp_path / "fire", tmp_path / "marine"
    fire.mkdir()
    marine.mkdir()
    (fire / "nmd_stats.json").write_text(json.dumps(STATS_2023))
    with pytest.raises(ValueError, match="NMD 2023; the showcase tabs have NMD 2018 legends"):
        html_report.save_tabbed_report(str(fire), str(marine), str(tmp_path / "r.html"))
    legacy = {"class_stats": {"level1": {}, "level2": {"forest_pine": _SIXTY}}}
    (fire / "nmd_stats.json").write_text(json.dumps(legacy))
    html_report.save_tabbed_report(str(fire), str(marine), str(tmp_path / "r.html"))
    assert '"l2": {"labels": ["Tallskog"], "fractions": [60.0]' in (tmp_path / "r.html").read_text()


def test_engine_report_ignores_nmd_files_of_an_earlier_run(tmp_path):
    """No NMD result this run → an older run's NMD overlay and stats are not shown."""
    from imint.engine import _generate_html_report
    from imint.job import IMINTJob

    for name in ("d_rgb.png", "d_nmd_overlay.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)
    (tmp_path / "d_nmd_stats.json").write_text(json.dumps(
        {"class_stats": {"level1": {}, "level2": {"forest_pine": _SIXTY}}}))
    job = IMINTJob(date="d", coords={}, rgb=np.zeros((4, 4, 3), dtype=np.float32),
                   output_dir=str(tmp_path))
    _generate_html_report(job, "d_", None)
    html = (tmp_path / "d_imint_report.html").read_text()
    assert 'id="map-rgb"' in html
    assert 'id="map-nmd"' not in html and "Korsreferens mot NMD" not in html
