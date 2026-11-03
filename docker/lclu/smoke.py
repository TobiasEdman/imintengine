"""Exercise the clean runtime's imports and real exporter before source fetches."""
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import rasterio
import torch
from PIL import Image

from imint.api.v1.server import app
from imint.fm.registry import build_backbone
from imint.fm.tessera_seg import TesseraSegmentationModel
from imint.inference.lclu import Grid, export_map, write_json
from imint.training.unified_dataset import UnifiedDataset
from scripts.stage_lclu_demo import select_vpp

grid = Grid.from_lonlat(18.0686, 59.3293)
with tempfile.TemporaryDirectory(prefix='lclu-smoke-') as folder:
    output = Path(folder)
    classes = np.full((504, 504), 9, np.uint8)
    valid = np.ones((504, 504), bool)
    valid[0, 0] = False
    export_map(classes, valid, grid, output)
    with rasterio.open(output/'classification.tif') as src:
        pixels = src.read(1)
        assert src.shape == (504, 504) and src.crs.to_epsg() == 3006
        assert pixels[0, 0] == 255 and np.all(pixels[1:] == 9)
        assert src.transform == grid.transform(504)
    with Image.open(output/'overlay.png') as image:
        image.verify()
    torch.set_num_threads(2)
    encoder, _ = build_backbone('tessera_v1', img_size=504, pretrained=False)
    model = TesseraSegmentationModel(encoder, num_classes=28, n_aux_channels=11,
                                     enable_tradslag_head=True, aux_fusion='gated').eval()
    # Dependency and geometry check only; no random-weight map is published.
print('Runtime imports, TESSERA head and reopened georeferenced export: PASS', flush=True)
