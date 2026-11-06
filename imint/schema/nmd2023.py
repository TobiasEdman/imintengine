"""NMD 2023 basskikt v2.1 classes, for display and statistics.

Level 2 is every class of Naturvårdsverket's QGIS style
``NMD2023bas_v2_1.qml``, the layer style shipped with the raster (pinned
copy and SHA256 in ``docs/verification/lclu-service/``). Each class keeps
its code, label and colour from that file. Level 1 groups the codes by
their first digit, which is the product's klassgrupp (Produktbeskrivning
NMD2023 basskikt v2.1, Bilaga 2): 1 Skogsmark, 2 Öppen våtmark,
3 Åkermark, 4 Öppen fastmark, 5 Anlagd och bebyggd mark, 6 Vatten. Låg
fjällskog therefore counts as öppen våtmark (23, 230) or öppen fastmark
(43), where the product places it.

Raster value 0 means outside NMD 2023's coverage (v2.1 covers about 94.5 %
of Sweden). It is not a class. Every other value missing from the table is
an error.

The product description's Bilaga 5 gives other colours than the QML for
54 (torvtäkt), and it swaps 412 and 413. Here the QML is the reference.

These tables never produce training labels; labels go through
``unified_schema.NMD2023_TO_UNIFIED``. The NMD 2018 tables stay in
``imint.analyzers.nmd`` and ``class_schema``, whose 8-bit lookup cannot
hold the 4-digit 2023 codes.
"""
from __future__ import annotations

NMD2023_QML_SHA256 = "0154ee34d2ccc150246ea4901adaac3765b9cf02b1ea5c2d35ff07fc800a4322"

NMD2023_OUTSIDE = 0
NMD2023_OUTSIDE_NAME = "Utanför NMD 2023"

#: code → (label, colour), in the QML's order.
NMD2023_CLASSES: dict[int, tuple[str, str]] = {
    3: ("Åkermark (underindelning saknas)", "#ffffbc"),
    23: ("Låg fjällskog på våtmark", "#28dc82"),
    43: ("Låg fjällskog på fastmark", "#00f000"),
    51: ("Exploaterad mark, byggnad", "#581313"),
    52: ("Exploaterad mark, ej byggnad eller väg/järnväg", "#e24549"),
    53: ("Exploaterad mark, väg/järnväg", "#161616"),
    54: ("Exploaterad mark, torvtäkt", "#8c1964"),
    61: ("Sjö och vattendrag", "#6699cc"),
    62: ("Hav", "#89ccf9"),
    111: ("Tallskog på fastmark", "#6d8b05"),
    112: ("Granskog på fastmark", "#2c5f00"),
    113: ("Barrblandskog på fastmark", "#4e6f00"),
    114: ("Lövblandad barrskog på fastmark", "#38a800"),
    115: ("Triviallövskog på fastmark", "#4be600"),
    116: ("Ädellövskog på fastmark", "#aaff00"),
    117: ("Triviallövskog med ädellövinslag på fastmark", "#96e600"),
    118: ("Temporärt ej skog på fastmark", "#cdcd66"),
    121: ("Tallskog på våtmark", "#588b54"),
    122: ("Granskog på våtmark", "#2f5e50"),
    123: ("Barrblandskog våtmark", "#22735a"),
    124: ("Lövblandad barrskog på våtmark", "#43876f"),
    125: ("Triviallövskog på våtmark", "#88cd9b"),
    126: ("Ädellövskog på våtmark", "#a4f578"),
    127: ("Triviallövskog med ädellövinslag på våtmark", "#abcd78"),
    128: ("Temporärt ej skog på våtmark", "#888844"),
    200: ("Öppen våtmark (underindelning saknas)", "#c29cd5"),
    211: ("Buskmyr", "#884464"),
    212: ("Ristuvemyr", "#cd6699"),
    213: ("Fastmattemyr, mager", "#f579b6"),
    214: ("Fastmattemyr, frodig", "#d59dbb"),
    215: ("Sumpkärr", "#73004b"),
    216: ("Mjukmattemyr", "#a80084"),
    217: ("Lösbottenmyr", "#e600a9"),
    218: ("Övrig öppen myr", "#ff00c5"),
    221: ("Våtmark med buskar", "#6d4286"),
    222: ("Risdominerad våtmark", "#a866cc"),
    223: ("Gräsdominerad våtmark, mager", "#aa66cd"),
    224: ("Gräsdominerad våtmark, frodvuxen", "#c979f5"),
    225: ("Gräsdominerad våtmark, högvuxen", "#c29ed6"),
    226: ("Mossdominerad våtmark", "#4b0073"),
    227: ("Våtmark utan växttäcke", "#8400a8"),
    228: ("Övrig öppen våtmark", "#a900e6"),
    230: ("Låg fjällskog på övrig våtmark", "#86cc66"),
    411: ("Öppen mark utan vegetation (ej glaciär eller varaktigt snöfält)", "#e0e0e0"),
    412: ("Glaciär", "#ffffff"),
    413: ("Varaktigt snöfält", "#ffbde8"),
    4211: ("Torr buskdominerad mark", "#ccd69e"),
    4212: ("Frisk buskdominerad mark", "#abc8a6"),
    4213: ("Frisk-fuktig buskdominerad mark", "#9eb591"),
    4221: ("Torr risdominerad mark", "#d6c29e"),
    4222: ("Frisk risdominerad mark", "#cdaa66"),
    4223: ("Frisk-fuktig risdominerad mark", "#886f44"),
    4231: ("Torr gräsdominerad mark", "#ffebae"),
    4232: ("Frisk gräsdominerad mark", "#ffd27e"),
    4233: ("Frisk-fuktig gräsdominerad mark", "#ffbe7e"),
}

# Klassgrupp (first code digit) → level-1 key and name. The keys match the
# NMD 2018 level-1 keys, so consumers of level-1 statistics read both.
_KLASSGRUPP = {
    "1": ("forest", "Skogsmark"),
    "2": ("wetland", "Öppen våtmark"),
    "3": ("cropland", "Åkermark"),
    "4": ("open_land", "Öppen fastmark"),
    "5": ("developed", "Anlagd och bebyggd mark"),
    "6": ("water", "Vatten"),
}

#: level-1 key → (name, codes).
NMD2023_LEVEL1: dict[str, tuple[str, frozenset[int]]] = {
    key: (name, frozenset(code for code in NMD2023_CLASSES if str(code)[0] == digit))
    for digit, (key, name) in _KLASSGRUPP.items()
}
