"""
tests/test_schema.py — Regression tests for unified_schema.py (v5, majs schema)

Run:
    pytest tests/test_schema.py -v

All assertions are deterministic and require no GPU or file I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.training.unified_schema import (
    NUM_UNIFIED_CLASSES,
    UNIFIED_CLASSES,
    UNIFIED_COLORS,
    UNIFIED_CLASS_NAMES,
    SJV_TO_UNIFIED,
    _SJV_DEFAULT,
    _NMD19_TO_UNIFIED,
    HARVEST_CLASS,
    nmd19_to_unified,
    merge_nmd_sjv,
    merge_all,
    get_class_weights,
)


# ── 1. Schema identity ─────────────────────────────────────────────────────────

def test_num_classes():
    assert NUM_UNIFIED_CLASSES == 23, "Schema must have exactly 23 classes"


def test_class_21_is_majs():
    assert UNIFIED_CLASSES[21] == "majs", (
        f"Class 21 should be 'majs', got '{UNIFIED_CLASSES[21]}'"
    )


def test_class_21_color_corn_yellow():
    r, g, b = UNIFIED_COLORS[21]
    assert (r, g, b) == (220, 200, 0), (
        f"Class 21 color should be corn yellow (220,200,0), got ({r},{g},{b})"
    )


def test_no_ovrig_aker_in_classes():
    for idx, name in UNIFIED_CLASSES.items():
        assert "övrig" not in name.lower(), (
            f"'övrig åker' found at class {idx} — must be removed"
        )


# ── 2. NMD 19-class mapping ────────────────────────────────────────────────────

def test_nmd_cropland_maps_to_background():
    # NMD 19-class index 12 = cropland → must map to 0 (background), NOT 21
    assert _NMD19_TO_UNIFIED[12] == 0, (
        f"NMD cropland (19-class idx 12) should map to background (0), "
        f"got {_NMD19_TO_UNIFIED[12]}"
    )


def test_nmd_open_land_maps_to_8():
    assert _NMD19_TO_UNIFIED[13] == 8, "NMD open_land_bare (13) → öppen mark (8)"
    assert _NMD19_TO_UNIFIED[14] == 8, "NMD open_land_veg  (14) → öppen mark (8)"


def test_nmd_forest_classes_intact():
    expected = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}
    for raw, unified in expected.items():
        assert int(_NMD19_TO_UNIFIED[raw]) == unified, (
            f"NMD raw {raw} should map to unified {unified}, got {_NMD19_TO_UNIFIED[raw]}"
        )


def test_nmd_water_maps_to_10():
    assert _NMD19_TO_UNIFIED[18] == 10
    assert _NMD19_TO_UNIFIED[19] == 10


# ── 3. SJV code mappings ───────────────────────────────────────────────────────

def test_sjv_majs_maps_to_21():
    assert SJV_TO_UNIFIED.get(9) == 21, "SJV 9 (majs) must map to class 21"


def test_sjv_gronfoder_maps_to_slattervall():
    for code in (16, 80, 81):
        mapped = SJV_TO_UNIFIED.get(code)
        assert mapped == 15, (
            f"SJV {code} (grönfoder/gröngödsling) should map to slåttervall (15), got {mapped}"
        )


def test_sjv_trada_maps_to_oppen_mark():
    assert SJV_TO_UNIFIED.get(60) == 8, "SJV 60 (träda) should map to öppen mark (8)"


def test_sjv_skyddszon_not_in_dict():
    # 66=anpassad skyddszon, 77=skyddszon — sub-pixel geometry → background
    for code in (66, 77):
        assert code not in SJV_TO_UNIFIED, (
            f"SJV {code} (skyddszon) should NOT be in SJV_TO_UNIFIED (falls to background)"
        )


def test_sjv_default_is_background():
    assert _SJV_DEFAULT == 0, (
        f"_SJV_DEFAULT should be 0 (background), got {_SJV_DEFAULT}"
    )


# ── 4. merge_all gate fix ──────────────────────────────────────────────────────

def _make_cropland_tile() -> tuple[np.ndarray, np.ndarray]:
    """3×3 NMD 19-class tile: all cropland (raw 12) + a wheat parcel (SJV 4→vete=11)."""
    nmd = np.full((3, 3), 12, dtype=np.uint8)     # all cropland (NMD 19-class)
    lpis = np.full((3, 3), 4, dtype=np.uint16)    # SJV 4 = höstvete → unified 11
    return nmd, lpis


def test_merge_all_lpis_on_cropland():
    """After _NMD19_TO_UNIFIED[12]=0, LPIS must still apply on NMD cropland."""
    nmd, lpis = _make_cropland_tile()
    result = merge_all(nmd, lpis_mask=lpis, harvest_mask=None)
    assert result.shape == (3, 3)
    # Every pixel should be vete (11), not background (0) or 21
    unique_vals = set(int(v) for v in np.unique(result))
    assert unique_vals == {11}, (
        f"LPIS vete parcels on NMD cropland should give class 11 everywhere, "
        f"got classes {unique_vals}"
    )


def test_merge_all_cropland_no_lpis_becomes_background():
    """NMD cropland with no LPIS parcel → background (0)."""
    nmd = np.full((4, 4), 12, dtype=np.uint8)
    lpis = np.zeros((4, 4), dtype=np.uint16)   # no parcels
    result = merge_all(nmd, lpis_mask=lpis, harvest_mask=None)
    unique_vals = set(int(v) for v in np.unique(result))
    assert unique_vals == {0}, (
        f"NMD cropland with no LPIS should be background (0), got {unique_vals}"
    )


def test_merge_all_forest_harvest():
    """Forest + harvest mask → hygge (22). Forest class intact without harvest."""
    nmd = np.zeros((4, 4), dtype=np.uint8)
    nmd[0:2, :] = 1     # NMD 19-class 1 = forest_pine → unified 1 (tallskog)
    nmd[2:4, :] = 2     # NMD 19-class 2 = forest_spruce → unified 2 (granskog)
    harvest = np.zeros((4, 4), dtype=np.uint8)
    harvest[0, :] = 1   # top row harvested

    result = merge_all(nmd, lpis_mask=None, harvest_mask=harvest)
    assert (result[0, :] == 22).all(), "Harvested pine → hygge (22)"
    assert (result[1, :] == 1).all(),  "Non-harvested pine → tallskog (1)"
    assert (result[2, :] == 2).all(),  "Non-harvested spruce → granskog (2)"


def test_merge_all_no_lpis_on_developed():
    """Developed land (NMD 15-17 → unified 9) must not be overridden by LPIS."""
    nmd = np.full((3, 3), 15, dtype=np.uint8)    # NMD 15 = developed_buildings → bebyggelse
    lpis = np.full((3, 3), 4, dtype=np.uint16)   # SJV 4 = vete
    result = merge_all(nmd, lpis_mask=lpis, harvest_mask=None)
    unique_vals = set(int(v) for v in np.unique(result))
    assert unique_vals == {9}, (
        f"Developed (bebyggelse) should not be overridden by LPIS, got {unique_vals}"
    )


def test_merge_all_open_land_gets_lpis():
    """Open land (NMD 13/14 → öppen mark=8) can receive LPIS crop overlay."""
    nmd = np.full((3, 3), 13, dtype=np.uint8)    # NMD 13 = open_land_bare → öppen mark (8)
    lpis = np.full((3, 3), 49, dtype=np.uint16)  # SJV 49 = slåttervall → unified 15
    result = merge_all(nmd, lpis_mask=lpis, harvest_mask=None)
    unique_vals = set(int(v) for v in np.unique(result))
    assert unique_vals == {15}, (
        f"Open land with slåttervall LPIS should give class 15, got {unique_vals}"
    )


# ── 5. Merge with unmapped SJV codes ──────────────────────────────────────────

def test_unmapped_sjv_code_becomes_background():
    """SJV code not in SJV_TO_UNIFIED on cropland → background (via _SJV_DEFAULT=0)."""
    # SJV 999 = not a real code
    nmd = np.full((2, 2), 12, dtype=np.uint8)
    lpis = np.full((2, 2), 999, dtype=np.uint16)
    result = merge_all(nmd, lpis_mask=lpis, harvest_mask=None)
    unique_vals = set(int(v) for v in np.unique(result))
    assert unique_vals == {0}, (
        f"Unmapped SJV code on cropland should be background (0), got {unique_vals}"
    )


# ── 6. Color palette completeness ─────────────────────────────────────────────

def test_all_classes_have_colors():
    for i in range(NUM_UNIFIED_CLASSES):
        assert i in UNIFIED_COLORS, f"Class {i} missing from UNIFIED_COLORS"
        r, g, b = UNIFIED_COLORS[i]
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255


def test_class_names_list_length():
    assert len(UNIFIED_CLASS_NAMES) == NUM_UNIFIED_CLASSES


def test_harvest_class_is_22():
    assert HARVEST_CLASS == 22


# ── 7. nmd19_to_unified round-trip ────────────────────────────────────────────

def test_nmd19_to_unified_shape_preserved():
    inp = np.arange(20, dtype=np.uint8).reshape(4, 5)
    out = nmd19_to_unified(inp)
    assert out.shape == inp.shape
    assert out.dtype == np.uint8


def test_nmd19_to_unified_clipping():
    """Values > 19 should be clipped to 19 (water) not crash."""
    inp = np.array([[25, 0]], dtype=np.uint8)
    out = nmd19_to_unified(inp)
    assert out[0, 0] == _NMD19_TO_UNIFIED[19]  # clipped to 19 = water → 10


# ── 8. get_class_weights ──────────────────────────────────────────────────────

def test_get_class_weights_background_is_zero():
    counts = {i: 100 for i in range(1, NUM_UNIFIED_CLASSES)}
    counts[0] = 0
    weights = get_class_weights(counts, max_weight=10.0)
    assert weights[0] == 0.0, "Background weight must be 0"
    assert weights.shape == (NUM_UNIFIED_CLASSES,)


def test_get_class_weights_cap():
    # Rare class with tiny count should be capped at max_weight
    counts = {1: 1, 2: 1_000_000}
    weights = get_class_weights(counts, max_weight=10.0)
    assert weights[1] <= 10.0
    assert weights[2] <= 10.0
