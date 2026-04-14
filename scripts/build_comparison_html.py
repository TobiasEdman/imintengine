#!/usr/bin/env python3
"""
Regenerates dashboards/pixel_live/tile_viz/comparison.html med
upp till 6 kolumner beroende på vilken data som finns tillgänglig.

Kolumner:
  1. Prediction ep22 (gammal, från original 3-col PNG)  — alltid
  2. GT träningsetikett (stride=1)                       — alltid
  3. NMD ny schema (åker→bakgrund)                       — alltid
  4. NMD gammal schema (åker→övrig åker)                 — alltid
  5. NIR CIR false-color (sommarram)                     — alltid
  6. Prediction ny modell (M1/MPS inference)             — om col6_inference.json finns
"""
import base64, json, sys
from pathlib import Path

REPO   = Path(__file__).parent.parent
VIZ5   = REPO / "data/viz_tiles/viz_data5_stride1.json"
COL6   = REPO / "data/viz_tiles/col6_inference.json"
ORIG   = REPO / "dashboards/pixel_live/tile_viz/before_schema_change_ep28.png"
OUT    = REPO / "dashboards/pixel_live/tile_viz/comparison.html"

# Ladda befintlig viz5 data (producerades av nmd-col-gen3)
viz5_path = REPO / "data/viz_tiles" / "viz_data5.json"
if not viz5_path.exists():
    # fallback: sök i /tmp
    viz5_path = Path("/tmp/viz_data5.json")
    if not viz5_path.exists():
        print("ERROR: viz_data5.json inte hittat"); sys.exit(1)

with open(viz5_path) as f: viz5 = json.load(f)

col6_data = None
col6_epoch = "?"
if COL6.exists():
    with open(COL6) as f: col6_data = json.load(f)
    col6_epoch = col6_data.get("_epoch","?")
    print(f"Col6: epoch={col6_epoch} metric={col6_data.get('_metric','?')}")

with open(ORIG,"rb") as f:
    orig_b64 = base64.b64encode(f.read()).decode()

CLASS_NAMES = ["background","tallskog","granskog","lövskog","blandskog","sumpskog",
    "tillfälligt ej skog","våtmark","öppen mark","bebyggelse","vatten",
    "vete","korn","havre","oljeväxter","slåttervall","bete",
    "potatis","sockerbetor","trindsäd","råg","majs","hygge"]
CLASS_COLORS = ["#000000","#1a5c35","#2d8a5b","#7bc67e","#4db380","#6b8e5a",
    "#c9df6e","#9b7722","#d4b44a","#c0392b","#2471a3",
    "#e8b800","#d4780a","#f0d060","#d4c600","#91c84c","#b8de86",
    "#9b59b6","#d63381","#e07020","#8b2020","#dcc800","#00a8c6"]

tiles = [
    ("43983968","tallskog / granskog"),("45524456","korn / vete mix"),
    ("43983958","bebyggelse + vatten"),("tile_421280_7011280","agricultural zone"),
    ("45563754","forest / crop edge"),
]

col_labels = ["🤖 Pred ep22","✅ GT","🗺 NMD ny","⚠️ NMD gammal","🌿 NIR CIR"]
col_bg     = ["#fff3e0","#e8f5e9","#e3f2fd","#fce4ec","#f3e5f5"]
if col6_data:
    col_labels.append(f"🆕 Ny modell ep{col6_epoch}")
    col_bg.append("#e8eaf6")

js5 = "{\n" + "".join(f'  "{k}": "{viz5[k]}",\n'
    for k in viz5 if not k.startswith("_") and not k.endswith("_shape")) + "}"
js6 = "{}" if not col6_data else (
    "{\n" + "".join(f'  "{k}": "{col6_data[k]}",\n'
    for k in col6_data if not k.startswith("_") and k.endswith("_pred")) + "}"
)

shapes_js = json.dumps({t: viz5.get(t+"_shape",[224,224]) for t,_ in tiles})
tiles_js  = json.dumps([{"id":t,"dom":d} for t,d in tiles])
n_cols = len(col_labels)
has_col6 = "true" if col6_data else "false"

leg = "".join(
    f'<div class="li"><span class="sw" style="background:{CLASS_COLORS[i]}"></span>{i} {CLASS_NAMES[i]}</div>'
    for i in range(1, len(CLASS_NAMES))
)

html = f"""<!DOCTYPE html>
<html lang="sv"><head><meta charset="UTF-8">
<title>Pixel classifier — {n_cols} kolumner · 10m pixlar</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f0f2f5;color:#1a1a2e;padding:20px}}
h1{{font-size:17px;font-weight:700;margin-bottom:3px}}
.sub{{font-size:12px;color:#555;margin-bottom:16px;line-height:1.7}}
.badge{{display:inline-block;background:#e74c3c;color:#fff;font-size:10px;font-weight:700;padding:1px 7px;border-radius:20px;margin-left:6px;vertical-align:middle}}
.badge.new{{background:#27ae60}}
.wrap{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);overflow-x:auto}}
.ctrl{{display:flex;align-items:center;gap:12px;font-size:12px;color:#555;margin-bottom:10px;flex-wrap:wrap}}
input[type=range]{{width:110px;accent-color:#3498db}}
.hdr{{display:grid;gap:2px;margin-bottom:4px}}
.hdr>div{{font-size:11px;font-weight:700;text-align:center;padding:4px 3px;border-radius:4px}}
.hdr>.hl{{background:none;text-align:left;color:#999;font-weight:400;min-width:130px}}
.row{{display:grid;gap:2px;margin-bottom:5px;align-items:start}}
.tlbl{{display:flex;flex-direction:column;justify-content:center;padding:3px 6px;min-width:130px;font-size:10px}}
.tlbl b{{font-family:monospace;font-size:11px;color:#222}}
.tlbl span{{color:#777;margin-top:1px}}
.cell{{cursor:zoom-in;border-radius:3px;overflow:hidden}}
.cell canvas{{display:block;width:100%;image-rendering:pixelated}}
.leg{{background:#fff;border-radius:10px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-top:14px}}
.leg h2{{font-size:12px;font-weight:700;margin-bottom:8px;color:#444}}
.lg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:3px 10px}}
.li{{display:flex;align-items:center;font-size:11px;gap:5px}}
.sw{{width:13px;height:13px;border-radius:2px;flex-shrink:0;border:1px solid rgba(0,0,0,.1)}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:999;align-items:center;justify-content:center;cursor:zoom-out;flex-direction:column;gap:8px}}
#lb.on{{display:flex}}
#lb canvas{{max-width:92vw;max-height:88vh;image-rendering:pixelated;border-radius:6px}}
#lb-lbl{{color:#ddd;font-size:13px;padding:4px 12px;background:rgba(0,0,0,.5);border-radius:20px}}
</style></head><body>
<h1>Pixel classifier snapshot — ep22 <span class="badge">schema v4</span>{'<span class="badge new">+ ny modell ep'+str(col6_epoch)+'</span>' if col6_data else ''}</h1>
<div class="sub">
  best_model.pt · mIoU 0.3386 · {n_cols} kolumner · <strong>10m pixlar (stride=1)</strong><br>
  {'  ·  '.join(f'<strong>Col {i+1}:</strong> {l}' for i,l in enumerate(col_labels))}
</div>
<div class="wrap">
  <div class="ctrl">
    Cellstorlek:<input type="range" id="zoom" min="80" max="400" value="200" step="8"><span id="zv">200px</span>
    &nbsp;·&nbsp;<label style="cursor:pointer;display:flex;align-items:center;gap:5px"><input type="checkbox" id="diff"> Diff col3↔4</label>
  </div>
  <div id="hdr" class="hdr"></div>
  <div id="grid"></div>
</div>
<div class="leg"><h2>Klasskarta — schema v5 (majs=21)</h2><div class="lg">{leg}</div></div>
<div id="lb" onclick="closeLB()"><canvas id="lb-c"></canvas><div id="lb-lbl"></div></div>
<script>
const ORIG=new Image(); ORIG.src="data:image/png;base64,{orig_b64}";
const VIZ5={js5};
const VIZ6={js6};
const HAS6={has_col6};
const SHAPES={shapes_js};
const TILES={tiles_js};
const COL_LABELS={json.dumps(col_labels)};
const COL_BG={json.dumps(col_bg)};
const NCOLS={n_cols};

let cellPx=200,diffMode=false,origLoaded=false;
const O_CELL=128,O_SEP=2,O_HDR=28,O_BAR=20,O_ROW=148;

function enc(b64){{const bin=atob(b64),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}}
async function inflate(b64){{return pako.inflate(enc(b64));}}

function makeCanvas(rgbBytes,gh,gw,px){{
  const c=document.createElement("canvas");c.width=px;c.height=px;
  const t=document.createElement("canvas");t.width=gw;t.height=gh;
  const id=t.getContext("2d").createImageData(gw,gh);
  for(let i=0;i<gh*gw;i++){{id.data[i*4]=rgbBytes[i*3];id.data[i*4+1]=rgbBytes[i*3+1];id.data[i*4+2]=rgbBytes[i*3+2];id.data[i*4+3]=255;}}
  t.getContext("2d").putImageData(id,0,0);
  const ctx=c.getContext("2d");ctx.imageSmoothingEnabled=false;
  ctx.drawImage(t,0,0,gw,gh,0,0,px,px);return c;
}}
function sliceOrig(col,ri,px){{
  const c=document.createElement("canvas");c.width=px;c.height=px;
  const ctx=c.getContext("2d");ctx.imageSmoothingEnabled=false;
  ctx.drawImage(ORIG,col*(O_CELL+O_SEP),O_HDR+ri*O_ROW+O_BAR,O_CELL,O_CELL,0,0,px,px);return c;
}}
function diffCanvas(nB,oB,gh,gw,px){{
  const c=document.createElement("canvas");c.width=px;c.height=px;
  const t=document.createElement("canvas");t.width=gw;t.height=gh;
  const id=t.getContext("2d").createImageData(gw,gh);
  for(let i=0;i<gh*gw;i++){{
    if(nB[i*3]!==oB[i*3]||nB[i*3+1]!==oB[i*3+1]||nB[i*3+2]!==oB[i*3+2]){{
      id.data[i*4]=255;id.data[i*4+1]=0;id.data[i*4+2]=200;id.data[i*4+3]=210;
    }}else{{id.data[i*4]=oB[i*3];id.data[i*4+1]=oB[i*3+1];id.data[i*4+2]=oB[i*3+2];id.data[i*4+3]=255;}}
  }}
  t.getContext("2d").putImageData(id,0,0);
  const ctx=c.getContext("2d");ctx.imageSmoothingEnabled=false;
  ctx.drawImage(t,0,0,gw,gh,0,0,px,px);return c;
}}
async function buildGrid(){{
  const grid=document.getElementById("grid"),hdr=document.getElementById("hdr");
  grid.innerHTML="";hdr.innerHTML="";
  const tcols="130px repeat("+NCOLS+","+cellPx+"px)";
  hdr.style.gridTemplateColumns=tcols;
  const h0=document.createElement("div");h0.className="hl";h0.textContent="Tile";hdr.appendChild(h0);
  COL_LABELS.forEach((l,i)=>{{const d=document.createElement("div");d.textContent=l;d.style.background=COL_BG[i];hdr.appendChild(d);}});
  for(let ri=0;ri<TILES.length;ri++){{
    const t=TILES[ri],[gh,gw]=SHAPES[t.id]||[224,224];
    const row=document.createElement("div");row.className="row";row.style.gridTemplateColumns=tcols;
    const lbl=document.createElement("div");lbl.className="tlbl";
    lbl.innerHTML=`<b>${{t.id}}</b><span>${{t.dom}}</span>`;row.appendChild(lbl);
    const predC=sliceOrig(0,ri,cellPx);
    const gtB=VIZ5[t.id+"_gt"]?await inflate(VIZ5[t.id+"_gt"]):null;
    const gtC=gtB?makeCanvas(gtB,gh,gw,cellPx):predC;
    const nNB=VIZ5[t.id+"_nmd_new"]?await inflate(VIZ5[t.id+"_nmd_new"]):null;
    const nNC=nNB?makeCanvas(nNB,gh,gw,cellPx):null;
    const nOB=VIZ5[t.id+"_nmd_old"]?await inflate(VIZ5[t.id+"_nmd_old"]):null;
    const nOC=!nOB?null:diffMode?diffCanvas(nNB,nOB,gh,gw,cellPx):makeCanvas(nOB,gh,gw,cellPx);
    const cirB=VIZ5[t.id+"_cir"]?await inflate(VIZ5[t.id+"_cir"]):null;
    const cirC=cirB?makeCanvas(cirB,gh,gw,cellPx):null;
    const cells=[predC,gtC,nNC,nOC,cirC];
    if(HAS6&&VIZ6[t.id+"_pred"]){{
      const p6B=await inflate(VIZ6[t.id+"_pred"]);
      cells.push(makeCanvas(p6B,gh,gw,cellPx));
    }}else if(HAS6){{cells.push(null);}}
    cells.forEach((c,ci)=>{{
      const wrap=document.createElement("div");wrap.className="cell";wrap.style.background=COL_BG[ci];
      if(c){{c.style.width="100%";c.style.height="auto";c.onclick=e=>{{e.stopPropagation();openLB(c,COL_LABELS[ci]+" — "+t.id);}};wrap.appendChild(c);}}
      row.appendChild(wrap);
    }});
    grid.appendChild(row);
  }}
}}
document.getElementById("zoom").addEventListener("input",e=>{{cellPx=parseInt(e.target.value);document.getElementById("zv").textContent=cellPx+"px";if(origLoaded)buildGrid();}});
document.getElementById("diff").addEventListener("change",e=>{{diffMode=e.target.checked;if(origLoaded)buildGrid();}});
function openLB(src,label){{const lb=document.getElementById("lb-c");lb.width=src.width;lb.height=src.height;lb.getContext("2d").drawImage(src,0,0);document.getElementById("lb-lbl").textContent=label;document.getElementById("lb").classList.add("on");}}
function closeLB(){{document.getElementById("lb").classList.remove("on");}}
document.addEventListener("keydown",e=>{{if(e.key==="Escape")closeLB();}});
const ps=document.createElement("script");ps.src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js";
ps.onload=()=>{{ORIG.onload=()=>{{origLoaded=true;buildGrid();}};if(ORIG.complete){{origLoaded=true;buildGrid();}}}};
document.head.appendChild(ps);
</script></body></html>"""

with open(OUT,"w",encoding="utf-8") as f: f.write(html)
# Spara viz5 till rätt plats för framtida bruk
import shutil
shutil.copy(str(viz5_path), str(REPO/"data/viz_tiles/viz_data5.json"))
print(f"HTML ({n_cols} cols): {OUT}  ({len(html)//1024}KB)")
