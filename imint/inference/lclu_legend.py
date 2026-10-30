"""Annual-map presentation; never remaps model class IDs or training labels.

Colours verified against Naturvårdsverket NMD2023bas_v2_1.qml, SHA256
0154ee34d2ccc150246ea4901adaac3765b9cf02b1ea5c2d35ff07fc800a4322.
See docs/verification/lclu-service/nmd-legend.md for aggregation caveats.
"""
from imint.schema.unified_schema import UNIFIED_CLASSES, UNIFIED_COLORS

NMD_STYLE_URL = ('https://geodata.naturvardsverket.se/nedladdning/marktacke/'
                 'NMD2023/Basskikt_v2_x/NMD2023_basskikt_v2_1.zip')

# For aggregates, the code selects a representative official colour, not a
# claim that the model distinguishes that particular NMD subclass.
NMD_SWATCHES = {
    1: (111, '#6d8b05'), 2: (112, '#2c5f00'), 3: (115, '#4be600'),
    4: (114, '#38a800'), 5: (121, '#588b54'), 6: (118, '#cdcd66'),
    7: (200, '#c29cd5'), 9: (52, '#e24549'), 10: (61, '#6699cc'),
    23: (54, '#8c1964'), 24: (4212, '#abc8a6'), 25: (4222, '#cdaa66'),
    26: (4232, '#ffd27e'), 27: (411, '#e0e0e0'),
}
MAP_COLORS = dict(UNIFIED_COLORS)
MAP_COLORS.update({key: tuple(bytes.fromhex(color[1:]))
                   for key, (_, color) in NMD_SWATCHES.items()})
MAP_NAMES = {**UNIFIED_CLASSES,
    1: 'Tallskog på fastmark', 2: 'Granskog på fastmark',
    3: 'Lövskog', 4: 'Blandskog',
    5: 'Skog på våtmark', 6: 'Temporärt ej skog på fastmark',
    7: 'Öppen våtmark', 8: 'Öppen mark (sammanslagen)',
    9: 'Exploaterad mark', 10: 'Sjö, vattendrag och hav',
    23: 'Exploaterad mark, torvtäkt',
}
NOTES = {
    0: 'Modellens bakgrundsklass; skiljs från pixlar där data saknas.',
    **{key: 'Skogstyperna är modellklasser som förfinats med fältdata och motsvarar inte NMD exakt. '
       'Barrblandskog redovisas inte separat. Färgerna är hämtade från NMD.'
       for key in (1, 2, 3, 4)},
    5: 'Sammanslagna skogstyper på våtmark, även temporärt ej skog och låg fjällskog. '
       'Färg från NMD tallskog på våtmark.',
    7: 'Våtmarkens underklasser är sammanslagna; färg från NMD öppen våtmark.',
    8: 'Bred modellklass med öppen mark, träda samt glaciär och varaktigt snöfält. '
       'Ingen entydig NMD-motsvarighet; egen färg.',
    9: 'Byggnader, övrig exploaterad mark och väg/järnväg är sammanslagna. '
       'Färg från NMD exploaterad mark, ej byggnad eller väg/järnväg.',
    10: 'Sjöar, vattendrag och hav är sammanslagna. Färg från NMD sjö och vattendrag.',
    22: 'Modellens tillägg från SKS avverkningsdata; egen färg, ingen separat NMD-klass.',
    **{key: 'Modellens tillägg från LPIS jordbruksdata; egen färg, ingen separat NMD-klass.'
       for key in range(11, 22)},
    **{key: 'Fuktighetsvarianter är sammanslagna. Färg från NMD:s friska vegetationstyp.'
       for key in (24, 25, 26)},
}


def legend() -> list[dict]:
    return [{
        'id': key, 'name': name, 'color': '#%02x%02x%02x' % MAP_COLORS[key],
        'note': NOTES.get(key, 'Benämning och färg enligt NMD 2023.'),
        **({'nmd_swatch': NMD_SWATCHES[key][0]} if key in NMD_SWATCHES else {}),
    } for key, name in MAP_NAMES.items()] + [
        {'id': 255, 'name': 'Data saknas', 'color': None,
         'note': 'Ingen klassning där användbara indata saknas.'}]
