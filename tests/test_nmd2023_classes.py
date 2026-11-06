"""NMD 2023 class tables against Naturvårdsverket's pinned QML style."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from imint.schema.nmd2023 import (
    NMD2023_CLASSES,
    NMD2023_LEVEL1,
    NMD2023_OUTSIDE,
    NMD2023_QML_SHA256,
)

QML = Path(__file__).parents[1] / "docs/verification/lclu-service/NMD2023bas_v2_1.qml"


def _qml_entries() -> list[tuple[int, str, str]]:
    entries = []
    for item in ET.parse(QML).iter("paletteEntry"):
        code, *label = item.attrib["label"].split()
        assert int(code) == int(item.attrib["value"])
        entries.append((int(item.attrib["value"]), " ".join(label), item.attrib["color"]))
    return entries


def test_qml_is_the_pinned_official_style():
    assert hashlib.sha256(QML.read_bytes()).hexdigest() == NMD2023_QML_SHA256


def test_every_qml_class_with_its_label_and_colour_in_qml_order():
    entries = _qml_entries()
    assert len(entries) == 55
    assert [(code, name, color) for code, (name, color) in NMD2023_CLASSES.items()] == entries


def test_codes_above_255_are_their_own_classes():
    assert max(NMD2023_CLASSES) == 4233
    assert {4211, 4212, 4213, 4221, 4222, 4223, 4231, 4232, 4233} <= set(NMD2023_CLASSES)


def test_outside_coverage_is_not_a_class():
    assert NMD2023_OUTSIDE == 0
    assert NMD2023_OUTSIDE not in NMD2023_CLASSES


def test_level1_partitions_every_code_by_klassgrupp():
    groups = [codes for _name, codes in NMD2023_LEVEL1.values()]
    assert sum(len(codes) for codes in groups) == len(NMD2023_CLASSES)
    assert frozenset().union(*groups) == set(NMD2023_CLASSES)
    assert list(NMD2023_LEVEL1) == ["forest", "wetland", "cropland", "open_land", "developed", "water"]
    assert NMD2023_LEVEL1["forest"][1] == frozenset(range(111, 119)) | frozenset(range(121, 129))
    assert NMD2023_LEVEL1["cropland"][1] == {3}
    assert NMD2023_LEVEL1["water"][1] == {61, 62}


def test_low_mountain_forest_and_peat_follow_the_product_hierarchy():
    level1 = {code: key for key, (_name, codes) in NMD2023_LEVEL1.items() for code in codes}
    assert level1[23] == level1[230] == level1[200] == "wetland"  # 2.3 Låg fjällskog på våtmark
    assert level1[43] == level1[411] == level1[4232] == "open_land"  # 4.3 Låg fjällskog på fastmark
    assert level1[54] == "developed"  # 5.4 Torvtäkt, moved out of öppen våtmark in 2023
