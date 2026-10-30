"""Explicitly labelled synthetic browser QA server; never a deployment mode.

Run: python -m tests.lclu.browser_fixture --output /tmp/lclu-browser --port 8785
Uses actual strict model loading, canonical forward and exports with RANDOM
weights and synthetic inputs. Never use its maps as classification results.
"""
import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import uvicorn
from fastapi.responses import HTMLResponse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8785)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from imint.inference.lclu import (Grid, AUX_NAMES, EMBEDDING, BundleCatalog,
                                     LcluService, TesseraR4, write_json)
    from scripts.cache_predict import file_identity
    from imint.fm.registry import build_backbone
    from imint.fm.tessera_seg import TesseraSegmentationModel
    from imint.api.v1 import server
    torch.set_num_threads(2)
    torch.manual_seed(19)
    encoder, _ = build_backbone('tessera_v1', img_size=504, pretrained=False)
    model = TesseraSegmentationModel(encoder,num_classes=28,n_aux_channels=11,
                                    enable_tradslag_head=True,aux_fusion='gated').eval()
    checkpoint=args.output/'synthetic.pt'
    if not checkpoint.exists():
        torch.save({'model_state_dict':model.state_dict(),'config':{
            'backbone_name':'tessera_v1','num_classes':28,'n_aux_channels':11,
            'enabled_aux_names':AUX_NAMES,'num_temporal_frames':1}},checkpoint)
    grid=Grid.from_lonlat(17.64,59.86)
    entries = []
    for year in (2023, 2024):
        path = args.output / f'synthetic-{year}.npz'
        if not path.exists():
            arrays={'tessera':np.full((128,512,512),.125,np.float16),
                    'valid_embedding':np.ones((512,512),bool),'crs':np.array('EPSG:3006'),
                    'transform':np.array(list(grid.transform())[:6]),
                    'year':year,'tessera_year':year,'vpp_year':year}
            arrays['tessera'][:,:,200:300]=.5
            arrays['valid_embedding'][200:260,300:360]=False
            arrays['tessera'][:,200:260,300:360]=0
            arrays.update({n:np.ones((512,512),np.float32) for n in AUX_NAMES})
            arrays['vpp_sosd'][:] = (year % 100) * 1000 + 130
            arrays['vpp_eosd'][:] = (year % 100) * 1000 + 260
            np.savez_compressed(path,**arrays)
        entries.append({'center_3006':[grid.east,grid.north],'year':year,'public_area':True,
                        'input':file_identity(path),'embedding':EMBEDDING,
                        'auxiliary_provenance':{'vpp': {'year':year, 'season':1,
                            'index': {'sha256':'a'*64, 'size':1}, 'files':[]}}})
    os.environ['IMINT_API_MAX_PIXELS']='262144'
    os.environ['IMINT_API_OUTPUT_DIR']=str(args.output/'outputs')
    os.environ.pop('IMINT_LCLU_MANIFEST',None)
    os.chdir(Path(__file__).resolve().parents[2])
    original_life=server.lifespan
    @asynccontextmanager
    async def life(app):
        async with original_life(app):
            start=time.perf_counter()
            resident=TesseraR4(file_identity(checkpoint),args.output,'cpu')
            resident.contract.update(synthetic=True, weights='random',
                                     purpose='Interaction demonstration; not an R4 classification')
            server._lclu_service=LcluService(resident,BundleCatalog(entries,args.output),args.output/'outputs')
            write_json(args.output/'startup.json',{'synthetic':True,'load_ms':1000*(time.perf_counter()-start)})
            yield
    server.app.router.lifespan_context=life
    # Replace only the test page endpoint; production code has no synthetic switch.
    page=(Path(__file__).resolve().parents[2]/'docs/lclu.html').read_text()
    for name, target in server._LCLU_ASSETS.items():
        page = page.replace('src="'+target+'"', 'src="/lclu-assets/'+name+'"').replace('href="'+target, 'href="/lclu-assets/'+name)
    page=page.replace('<body>','<body><div style="position:sticky;top:0;z-index:2000;background:#cff8e4;padding:10px;text-align:center;font-weight:700">SYNTHETIC DEMO · RANDOM WEIGHTS · NOT AN R4 CLASSIFICATION</div>')
    page=page.replace('Interaktiv demo · befintlig TESSERA R4-modell.',
                      'Syntetisk demo · båda åren använder samma testmönster, ingen historisk förändring visas.')
    page=page.replace('Hämta GeoTIFF ↓', 'Hämta syntetisk GeoTIFF ↓')
    async def synthetic_page(): return HTMLResponse(page)
    server.app.router.routes=[r for r in server.app.router.routes if getattr(r,'path',None)!='/lclu']
    server.app.add_api_route('/lclu',synthetic_page,include_in_schema=False)
    uvicorn.run(server.app,host='127.0.0.1',port=args.port,workers=1,lifespan='on')


if __name__=='__main__':
    main()
