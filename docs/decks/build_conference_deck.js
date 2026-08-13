// Scientific-conference oral deck (DES brand): field-calibrated land-cover for Sweden.
// Writes docs/decks/v8b_nfi_conference.pptx — ENGLISH ONLY. 19-slide conference arc.
// Brand + helpers mirror docs/decks/build_des_endgame_deck.js (do NOT modify that file).
// Run: NODE_PATH=$(npm root -g) node docs/decks/build_conference_deck.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const FR = path.join(__dirname, "frames");
const OUT = __dirname;
const img64 = f => "image/png;base64," + fs.readFileSync(`${FR}/${f}`).toString("base64");
const IMG = path.join(__dirname, "..", "data", "img");  // rendered example-tile channels
const dimg = f => "image/png;base64," + fs.readFileSync(`${IMG}/${f}`).toString("base64");
const FRAMES = [img64("nmd2018_frame.png"), img64("v8b_frame.png"),
                img64("nmd2023_frame.png"), img64("distill_frame.png"),
                img64("tessera_frame.png")];

// Digital Earth Sweden brand
const BG="0E433C", BG2="145043", HEAD="0A322C", MINT="CDFCE3", CORAL="FF826C",
      WHITE="FFFFFF", MUTED="9DBDB0", ROWA="123F38", ROWB="0C3A33", NMDCOL="5E8478",
      GRID="1C5248", WINBG="1E6B54";
const F="Space Grotesk";

const LEGEND_EN = [["Pine","006400"],["Spruce","228B22"],["Deciduous","32CD32"],["Mixed","3CB371"],
                   ["Swamp forest","2E4F2E"],["Clear-cut","00CED1"],["Open land","D2B48C"],["Water","0000FF"],
                   ["Built-up","FF0000"],["Grass-dom.","FFD27E"],["Dwarf-shrub","CDAA66"],["Shrub-dom.","ABC8A6"]];

function wordmark(p,s){ s.addText([{text:"◆ ",options:{color:MINT}},{text:"Digital Earth Sweden",options:{color:WHITE}}],
  {x:9.5,y:0.35,w:3.5,h:0.35,fontFace:F,fontSize:12,bold:true,align:"right",margin:0}); }

// ---- reusable primitives ----
function title(p,s,txt,sub){
  s.addText(txt,{x:0.7,y:0.42,w:11.9,h:0.85,fontFace:F,fontSize:32,color:WHITE,bold:true,align:"center",margin:0});
  if(sub) s.addText(sub,{x:0.9,y:1.28,w:11.5,h:0.5,fontFace:F,fontSize:14,color:MUTED,align:"center",margin:0});
}
function bullets(p,s,items,x,y,w,fs){
  s.addText(items.map(it=>({text:it,options:{bullet:{code:"2022",indent:14},breakLine:true,color:WHITE,fontSize:fs||14}})),
    {x:x,y:y,w:w,h:4.5,fontFace:F,color:WHITE,valign:"top",margin:0,lineSpacingMultiple:1.15,paraSpaceAfter:8});
}
function card(p,s,x,y,w,h,head,body,accent,hfs,bfs){
  s.addShape(p.ShapeType.roundRect,{x:x,y:y,w:w,h:h,rectRadius:0.1,fill:{color:BG2},line:{type:"none"},
    shadow:{type:"outer",color:"000000",opacity:0.35,blur:8,offset:3,angle:90}});
  s.addText(head,{x:x+0.25,y:y+0.15,w:w-0.5,h:0.5,fontFace:F,fontSize:hfs||16,color:accent||MINT,bold:true,valign:"top",margin:0});
  s.addText(body,{x:x+0.25,y:y+0.62,w:w-0.5,h:h-0.75,fontFace:F,fontSize:bfs||12,color:WHITE,valign:"top",margin:0,lineSpacingMultiple:1.1});
}
function statcard(p,s,x,y,w,h,big,lab,accent){
  s.addShape(p.ShapeType.roundRect,{x:x,y:y,w:w,h:h,rectRadius:0.09,fill:{color:BG2},line:{type:"none"},
    shadow:{type:"outer",color:"000000",opacity:0.35,blur:8,offset:3,angle:90}});
  s.addText(big,{x:x+0.1,y:y+0.14,w:w-0.2,h:h*0.5,fontFace:F,fontSize:28,color:accent||WHITE,bold:true,align:"center",valign:"middle",margin:0});
  s.addText(lab,{x:x+0.12,y:y+h*0.58,w:w-0.24,h:h*0.4,fontFace:F,fontSize:12,color:MUTED,align:"center",valign:"top",margin:0,lineSpacingMultiple:1.0});
}

// ---- data (from spec — exact) ----
const D = {
  facts:[["28","unified classes"],["7,882","Sentinel-2 tiles"],["944","NFI field plots"],["34,480","LUCAS points"]],
  ceilingLabels:["v8b (NMD2018)","NMD2023 labels","NMD2023 itself","Distilled","NFI head (OOF)"],
  ceiling:[46.5,46.0,49.3,52.7,63.7],
  benchLabels:["NMD2023","Distilled (28-class)","Forest-type fractions"],
  bench:[43.1,50.2,57.9],
  kappa:["0.298","0.371","0.420"],
  f1rows:[["Pine","0.59","0.61","0.74",3],["Spruce","0.56","0.63","0.55",2],
          ["Deciduous","0.30","0.43","0.59",3],["Mixed","0.28","0.30","0.29",2],
          ["Non-forest","0.24","0.32","0.00",2]],
  lucasL2b:[["Tessera","0.499"],["Distilled","0.484"],["Prithvi-600M / tradslag","0.477"]],
  lucasL2aLabels:["Tessera","Prithvi-600M"],
  lucasL2a:[0.809,0.784],
  cropLabels:["Sugar-beet","Potato","Barley","Wheat"],
  crop:[0.91,0.90,0.85,0.78],
  // head-to-head: same field-calibrated target, three independent truths
  h2hLabels:["NFI-209","LUCAS-28","LUCAS-frac"],
  h2hPrithvi:[0.579,0.477,0.784],
  h2hTessera:[0.589,0.499,0.809],
  // training-label taxonomy by source [name, id-range, class-count]
  srcGroups:[["NMD land cover","1–10",10],["LPIS crops","11–21",11],["SKS clear-cut","22",1],["NMD2023 add.","23–27",5]],
  // NFI reference forest-type distribution (944 plots, from confusion-matrix support)
  nfiDistLabels:["Pine","Spruce","Deciduous","Mixed","Non-forest"],
  nfiDist:[372,247,133,132,60],
};

function build(){
  const p=new pptxgen(); p.layout="LAYOUT_WIDE";

  // =========================================================== S1 — Title
  let s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText("Breaking the label ceiling",{x:0.65,y:1.05,w:12,h:1.2,fontFace:F,fontSize:58,color:WHITE,bold:true,margin:0});
  s.addText("Field-calibrated land-cover mapping for Sweden",{x:0.72,y:2.35,w:12,h:0.6,fontFace:F,fontSize:22,color:MINT,margin:0});
  s.addShape(p.ShapeType.roundRect,{x:0.7,y:3.25,w:11.9,h:1.55,rectRadius:0.12,fill:{color:BG2},line:{type:"none"},
    shadow:{type:"outer",color:"000000",opacity:0.4,blur:10,offset:3,angle:90}});
  s.addText([{text:"On 209 held-out field plots the model reaches ",options:{color:WHITE}},
             {text:"50.2%",options:{color:CORAL,bold:true}},
             {text:" and its forest-type layer ",options:{color:WHITE}},
             {text:"57.9%",options:{color:MINT,bold:true}},
             {text:" — versus ",options:{color:WHITE}},
             {text:"43.1%",options:{color:WHITE,bold:true}},
             {text:" for the NMD2023 map it was trained on.",options:{color:WHITE}}],
    {x:1.05,y:3.5,w:11.2,h:1.1,fontFace:F,fontSize:22,bold:true,valign:"middle",margin:0,lineSpacingMultiple:1.05});
  let fx=0.7; D.facts.forEach(([b,l],i)=>{ statcard(p,s,fx,5.2,2.9,1.55,b,l, i===0?MINT:WHITE); fx+=3.02; });
  s.addNotes("Our headline: a model trained on Sweden's national land-cover map ends up more accurate than that map, when both are scored on independent field plots. I'll show how field calibration breaks a ceiling that hard labels impose. Four datasets underpin the work: 28 unified classes, 7,882 Sentinel-2 tiles, 944 NFI plots and over 34,000 LUCAS points.");

  // =========================================================== S2 — Motivation
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"The label-ceiling question");
  s.addText("The national land-cover map (NMD) is Sweden's standard reference — but it is a modelled proxy, not field truth. Can a model trained on it beat it?",
    {x:0.8,y:1.35,w:11.7,h:0.7,fontFace:F,fontSize:16,color:MINT,align:"left",margin:0,lineSpacingMultiple:1.15});
  bullets(p,s,[
    "NMD is produced from satellite imagery and modelling — it carries its own class errors.",
    "Distillation can denoise a teacher, but only up to the teacher's own accuracy.",
    "The open question: is the ceiling the representation (features), or the supervision target (labels)?",
    "To answer it we need supervision and evaluation grounded in the field — not in NMD."],
    0.8,2.15,7.0,15);
  card(p,s,8.2,2.15,4.35,3.55,"Why field truth?",
    "NMD tells you what another model believed. National Forest Inventory (NFI) plots and LUCAS points are surveyed on the ground. Only independent field truth can tell whether a map is right — and whether a model has out-learned its teacher.",
    MINT,18,14);
  s.addNotes("The premise: NMD is everywhere used as ground truth, but it is itself a model output. A student model can denoise its teacher, yet classical distillation is bounded by the teacher's accuracy. So the scientific question is whether the ceiling lives in the representation or in the label. Answering that requires evaluation on real field data, which is what the rest of the talk uses.");

  // =========================================================== S3 — Data & truth
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Data and truth sources");
  s.addText("TRAINING SIGNAL",{x:0.8,y:1.3,w:5.8,h:0.35,fontFace:F,fontSize:13,color:MUTED,bold:true,margin:0});
  card(p,s,0.8,1.7,5.75,1.9,"Sentinel-2 + NMD2023",
    "7,882 tiles across all of Sweden, 4 temporal frames each. Pixel labels come from the NMD2023 national land-cover map — the proxy we train against.",MINT,17,13);
  s.addText("INDEPENDENT TRUTH",{x:6.85,y:1.3,w:5.7,h:0.35,fontFace:F,fontSize:13,color:CORAL,bold:true,margin:0});
  card(p,s,6.85,1.7,5.7,1.9,"NFI  ·  LUCAS  ·  LPIS",
    "944 NFI field plots (209 held out for testing), 34,480 Swedish LUCAS points, and year-matched LPIS crop parcels. None of these ever enters training.",CORAL,17,13);
  let gx=0.8; [["7,882","tiles"],["944 / 209","NFI plots / held out"],["34,480","LUCAS points"],["28","unified classes"]].forEach(([b,l])=>{
    statcard(p,s,gx,3.95,2.9,1.5,b,l,WHITE); gx+=3.02; });
  s.addText("The models see only Sentinel-2 pixels and NMD2023 labels; every accuracy number in this talk is measured against field truth the model never touched.",
    {x:0.8,y:5.7,w:11.7,h:0.6,fontFace:F,fontSize:13,italic:true,color:MUTED,margin:0,lineSpacingMultiple:1.1});
  s.addNotes("Two worlds of data. On the left, the training signal: nearly 8,000 Sentinel-2 tiles labelled by the NMD2023 map, four temporal frames each. On the right, independent field truth we never train on — NFI plots, LUCAS points and LPIS crop parcels. That separation is what makes the later comparisons honest.");

  // =========================================================== S3b — Anatomy of a training tile (visual)
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Anatomy of a training tile");
  s.addText("One example tile shown as its inputs: four seasons of Sentinel-2 plus a deep auxiliary stack — 35 channels — with one merged 28-class label.",
    {x:0.9,y:1.12,w:11.5,h:0.42,fontFace:F,fontSize:13.5,color:MINT,align:"center",margin:0,lineSpacingMultiple:1.1});
  const thumbs=[["rgb.png","Sentinel-2 RGB\n(1 of 4 frames)"],["nir_cir.png","False-colour NIR\n(vegetation red)"],
    ["dem.png","Terrain\nCopernicus DEM"],["height.png","Forest height\nSLU grunddata"],
    ["vpp.png","Phenology\nVPP start-of-season"],["label.png","Merged\n28-class label"]];
  const tsz=1.78, tgap=0.14, tx0=(13.33-(6*tsz+5*tgap))/2, ty=1.62;
  thumbs.forEach(([f,cap],i)=>{ const x=tx0+i*(tsz+tgap);
    s.addImage({data:dimg(f),x:x,y:ty,w:tsz,h:tsz,rounding:false});
    s.addText(cap,{x:x-0.08,y:ty+tsz+0.05,w:tsz+0.16,h:0.55,fontFace:F,fontSize:10,color:i===5?MINT:WHITE,bold:i===5,align:"center",valign:"top",margin:0,lineSpacingMultiple:0.95}); });
  card(p,s,0.8,4.4,5.75,1.62,"Spectral — 4 frames × 6 bands = 24",
    "Frame 0 is autumn (Sep–Oct) of the year BEFORE the label; frames 1–3 follow VPP phenology through the growing season. Each frame carries B02 B03 B04 B8A B11 B12 (Prithvi order; NIR is B8A, not B08).",MINT,15,12);
  card(p,s,6.85,4.4,5.7,1.62,"Auxiliary — 11 channels",
    "SLU forestry: canopy height, volume, basal area, stem diameter. Terrain: Copernicus DEM. Phenology (HR-VPP): start- & end-of-season, season length, peak & trough vegetation. Plus SLU soil moisture (markfukt).",CORAL,15,12);
  s.addText([{text:"Label build:  ",options:{color:MINT,bold:true}},
    {text:"NMD2023 land cover (base)  →  LPIS crop parcels (year-matched SJV codes)  →  SKS clear-cut (hygge).  QC drops tiles with >5% nodata or <3 of 4 usable frames.  ",options:{color:WHITE}},
    {text:"7,882 tiles · 10 m GSD · 2018–24 · Sweden-wide.",options:{color:MUTED}}],
    {x:0.8,y:6.28,w:11.75,h:0.7,fontFace:F,fontSize:12,align:"center",valign:"top",margin:0,lineSpacingMultiple:1.1});
  s.addNotes("Now made concrete: the same tile shown as its actual inputs. The first two panels are two renderings of one Sentinel-2 frame — true colour and false-colour near-infrared — and there are four such temporal frames, six bands each, giving 24 spectral channels. The next three panels are auxiliary layers: Copernicus terrain, SLU forest canopy height, and a Copernicus phenology metric; eleven auxiliary channels in total — four SLU forestry metrics, the DEM, five HR-VPP phenology features, and SLU soil moisture. The last panel is the training target: the 28-class label, merged from NMD2023 land cover, LPIS crop parcels rasterised from year-matched agricultural codes, and Skogsstyrelsen clear-cut notifications, after quality control.");

  // =========================================================== S3c — Label taxonomy & class distribution
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Label taxonomy and class distribution");
  s.addText("28 classes drawn from four national label sources",{x:0.8,y:1.22,w:11.7,h:0.35,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  const srcColors=[MINT,CORAL,"F2C879",NMDCOL], srcTxt=[BG,WHITE,BG,WHITE];
  let segx=0.8; const unit=11.75/27;
  D.srcGroups.forEach(([name,range,n],i)=>{ const w=unit*n;
    s.addShape(p.ShapeType.roundRect,{x:segx,y:1.68,w:w-0.05,h:0.6,rectRadius:0.06,fill:{color:srcColors[i]},line:{type:"none"}});
    s.addText(String(n),{x:segx,y:1.68,w:w-0.05,h:0.6,fontFace:F,fontSize:n===1?13:19,bold:true,color:srcTxt[i],align:"center",valign:"middle",margin:0}); segx+=w; });
  let lgx=0.85; [["NMD land cover (1–10)"],["LPIS crops (11–21)"],["SKS clear-cut (22)"],["NMD2023 additions (23–27)"]].forEach(([t],i)=>{
    s.addShape(p.ShapeType.rect,{x:lgx,y:2.46,w:0.22,h:0.22,fill:{color:srcColors[i]},line:{type:"none"}});
    s.addText(t,{x:lgx+0.3,y:2.42,w:2.7,h:0.32,fontFace:F,fontSize:11.5,color:WHITE,valign:"middle",margin:0}); lgx+=2.95; });
  s.addText("NFI reference forest types (944 field plots)",{x:0.7,y:3.05,w:6.2,h:0.35,fontFace:F,fontSize:13.5,color:CORAL,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[{name:"NFI plots",labels:D.nfiDistLabels,values:D.nfiDist}],
    {x:0.5,y:3.45,w:6.35,h:3.05,barDir:"col",
     chartColors:[MINT,"3E7D68",CORAL,"C98A5B",NMDCOL],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:12,dataLabelColor:WHITE,dataLabelFormatCode:'0',
     showTitle:false,showLegend:false,
     valAxisMinVal:0,valAxisMaxVal:400,valAxisMajorUnit:100,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:11,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:45});
  card(p,s,7.1,3.25,5.45,3.5,"Forest-dominated, and imbalanced",
    "Of 944 NFI plots, 884 are forest and only 60 are non-forest. Within forest, pine (372) and spruce (247) are 70% of plots; deciduous (133) and mixed (132) about 15% each.\n\nThe training labels inherit this skew: forest covers most of Sweden, crops occupy a thin southern agricultural band, and clear-cut (hygge) is rarer still.\n\nSo the rare classes — mixed forest, minor crops, non-forest — carry the widest confidence intervals, and are why a thin 209-plot held-out set limits statistical power.",
    CORAL,17,12.5);
  s.addNotes("The taxonomy and its balance. The bar at the top shows the 28 classes split by their source survey: ten NMD land-cover classes, eleven LPIS crop classes, a single SKS clear-cut class, and five classes new in NMD2023. The distribution is deeply uneven. On the field-truth side, of 944 NFI plots 884 are forest and only 60 non-forest; within forest, pine and spruce alone are seventy percent, with deciduous and mixed around fifteen percent each. The training labels inherit the same skew — forest everywhere, crops in a thin southern band, clear-cuts rarer still. That imbalance is why the rare classes have the widest error bars and why our thin held-out set limits statistical power.");

  // =========================================================== S4 — Architecture
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Methods: architecture");
  card(p,s,0.8,1.5,5.75,2.05,"Prithvi-EO-2.0 600M backbone",
    "Geospatial foundation encoder. Input: 4 temporal frames plus 11 auxiliary channels — forestry height, volume, basal area and diameter, a DEM, and VPP phenology.",MINT,17,13.5);
  card(p,s,0.8,3.75,5.75,2.05,"Dual output head",
    "Head 1 predicts the 28-class land-cover map. Head 2 predicts a Trädslag forest-type fraction layer — the per-pixel mix of tree species.",MINT,17,13.5);
  card(p,s,6.85,1.5,5.7,2.05,"Tessera (frozen embeddings)",
    "A second backbone that uses precomputed Tessera embeddings. No 600M encoder forward pass at train or inference time — a deployment-cheap alternative.",CORAL,17,13.5);
  card(p,s,6.85,3.75,5.7,2.05,"11 auxiliary channels",
    "Forestry stand metrics (height, volume, basal area, diameter), terrain (DEM), and Vegetation Phenology & Productivity (VPP) features guide the seasonal signal.",CORAL,17,13.5);
  s.addText("Same features, two backbones, two heads — the design lets us separate representation quality from supervision quality.",
    {x:0.8,y:5.95,w:11.7,h:0.5,fontFace:F,fontSize:13,italic:true,color:MUTED,margin:0});
  s.addNotes("Architecturally: a Prithvi-EO-2.0 600M foundation backbone consumes four temporal frames and eleven auxiliary channels — forestry stand metrics, a DEM, and phenology. It carries two heads: a 28-class land-cover head and a forest-type fraction head. We also run a Tessera backbone on frozen, precomputed embeddings so we can ask whether an expensive encoder is even necessary.");

  // =========================================================== S5 — Honest evaluation
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Methods: honest evaluation");
  const evalCards=[
    ["Grouped-tile split","735 plots for training, 209 held out. The split is by tile, so no plot near a training tile can leak into the test set."],
    ["Calibration on train only","Thresholds and the collapse rule are fit on the 735 train plots exclusively. The 209 held-out plots stay untouched."],
    ["NMD2023 on the same plots","The baseline map is scored on the exact same 209 plots — a like-for-like comparison, not a favourable subset."],
    ["Three independent truths","NFI, LUCAS and LPIS cross-check the result from three directions, none of them in the training loop."]];
  let ey=1.55; evalCards.forEach(([h,b],i)=>{ const yy=1.55+Math.floor(i/2)*2.15; const xx=i%2? 6.85:0.8;
    card(p,s,xx,yy,5.7,1.9,h,b, i%2?CORAL:MINT,17,14); });
  s.addNotes("Rigor matters more than the headline. We split by tile, not by plot, so nothing spatially adjacent leaks across the train-test boundary. Every threshold and collapse rule is calibrated only on the 735 training plots. The NMD2023 baseline is scored on the identical 209 held-out plots. And three independent field sources — NFI, LUCAS, LPIS — cross-check the conclusion.");

  // =========================================================== S6 — Label ceiling (chart)
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Result: the label ceiling");
  s.addText("NFI overall accuracy (944 plots) across the experiment series",
    {x:0.7,y:1.28,w:7.2,h:0.4,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[{name:"NFI overall",labels:D.ceilingLabels,values:D.ceiling}],
    {x:0.55,y:1.85,w:7.4,h:4.9,barDir:"col",
     chartColors:[NMDCOL,NMDCOL,WHITE,CORAL,MINT],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:13,dataLabelColor:WHITE,dataLabelFormatCode:'0.0"%"',
     showTitle:false,showLegend:false,
     valAxisMinVal:40,valAxisMaxVal:68,valAxisMajorUnit:5,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,valAxisLabelFormatCode:'0"%"',
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:10.5,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:45});
  card(p,s,8.25,1.85,4.3,2.2,"Hard labels cap ~46%",
    "Trained against NMD classes of any generation, accuracy sticks near 46%. NMD2023 is too clean to beat by imitation.",CORAL,17,13.5);
  card(p,s,8.25,4.25,4.3,2.5,"Field target breaks it",
    "The SAME features with NFI supervision reach 63.7% (leak-free out-of-fold). The supervision TARGET was the ceiling — not the representation.",MINT,17,13.5);
  s.addNotes("This is the central finding. The three left-hand bars — different NMD generations as the training target — all sit around 46 to 49 percent. Distillation nudges it to 52.7. But take the identical features and supervise with NFI field data, and out-of-fold accuracy jumps to 63.7 percent. The representation was never the bottleneck; the label was.");

  // =========================================================== S7 — Held-out benchmark
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Result: held-out benchmark");
  s.addText("209 NFI plots, grouped-tile split, calibration on train-735 only",
    {x:0.7,y:1.28,w:7.2,h:0.4,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[{name:"overall",labels:D.benchLabels,values:D.bench}],
    {x:0.55,y:1.85,w:7.3,h:4.9,barDir:"col",
     chartColors:[NMDCOL,CORAL,MINT],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:15,dataLabelColor:WHITE,dataLabelFormatCode:'0.0"%"',
     showTitle:false,showLegend:false,
     valAxisMinVal:35,valAxisMaxVal:62,valAxisMajorUnit:5,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:11,valAxisLabelFormatCode:'0"%"',
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:12,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:55});
  s.addText("Honest evaluation",{x:8.2,y:1.95,w:4.4,h:0.5,fontFace:F,fontSize:20,color:MINT,bold:true,margin:0});
  const hb=["Thresholds and head trained ONLY on the 735 train plots — the 209 are untouched.",
            "Distilled model = full 28-class product; every class is functional (+7.2 pp over NMD2023).",
            "The fraction layer reaches 57.9% on forest type — pine F1 0.74. Cohen's kappa rises 0.298 → 0.420."];
  let ry=2.65; hb.forEach(r=>{ s.addShape(p.ShapeType.ellipse,{x:8.2,y:ry+0.07,w:0.15,h:0.15,fill:{color:CORAL},line:{type:"none"}});
    s.addText(r,{x:8.5,y:ry-0.05,w:4.1,h:1.25,fontFace:F,fontSize:13.5,color:WHITE,valign:"top",margin:0,lineSpacingMultiple:1.1}); ry+=1.35; });
  s.addNotes("On the held-out plots the numbers speak for themselves: NMD2023 scores 43.1 percent, the distilled 28-class model 50.2, and the forest-type fraction layer 57.9. Kappa climbs from 0.30 to 0.42. Crucially, all thresholds were fit only on the training plots, so this is a clean generalization result, not a tuned one.");

  // =========================================================== S7b — Student beats teacher
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"The student beats the teacher");
  s.addText("In distillation a student normally cannot exceed its teacher. Trained ONLY on NMD2023 labels, on independent NFI field truth it does — and its forest-type layer more so.",
    {x:0.7,y:1.25,w:7.5,h:0.65,fontFace:F,fontSize:13.5,color:MINT,margin:0,lineSpacingMultiple:1.15});
  s.addChart(p.ChartType.bar,[{name:"NFI held-out",labels:["NMD2023\nteacher","Student\n28-class","Student\nforest-type"],values:D.bench}],
    {x:0.5,y:2.05,w:7.45,h:4.35,barDir:"col",
     chartColors:[NMDCOL,CORAL,MINT],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:15,dataLabelColor:WHITE,dataLabelFormatCode:'0.0"%"',
     showTitle:false,showLegend:false,
     valAxisMinVal:40,valAxisMaxVal:62,valAxisMajorUnit:5,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10.5,valAxisLabelFormatCode:'0"%"',
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:11.5,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:55});
  // delta pills centered in each student bar (dark fill so text reads over the coloured bar)
  [[3.77,4.75,"+7.1 pp"],[6.03,4.55,"+14.8 pp"]].forEach(([px,py,txt])=>{
    s.addShape(p.ShapeType.roundRect,{x:px,y:py,w:1.3,h:0.44,rectRadius:0.22,fill:{color:HEAD},line:{type:"none"}});
    s.addText(txt,{x:px,y:py,w:1.3,h:0.44,fontFace:F,fontSize:14,color:WHITE,bold:true,align:"center",valign:"middle",margin:0}); });
  s.addText("Cohen's kappa  0.298 → 0.371 → 0.420",{x:0.5,y:6.5,w:7.45,h:0.35,fontFace:F,fontSize:12,italic:true,color:MUTED,align:"center",margin:0});
  const why=[
    ["Label noise averages out","Random per-pixel teacher errors regularize away over 7,882 tiles — denoising you can see.",MINT],
    ["Field truth exposes the teacher","NMD's own errors are invisible when it is scored against itself; NFI plots reveal them.",CORAL],
    ["Richer than the label","4 temporal frames + 11 auxiliary channels carry what a single-date NMD label cannot.",MINT]];
  why.forEach(([h,b,ac],i)=>{ card(p,s,8.25,1.55+i*1.66,4.3,1.5,h,b,ac,15.5,12.5); });
  s.addNotes("The scientific hook: in classical distillation a student is bounded by its teacher. Here the student, trained only on NMD2023 labels, beats it on independent NFI field truth — 50.2 versus 43.1 percent overall, and 57.9 on forest type, gains of 7.1 and 14.8 points. Kappa rises from 0.30 to 0.42. Three reasons: label noise averages out over nearly 8,000 tiles, field truth exposes the teacher's own errors that self-scoring hides, and four temporal frames plus eleven aux channels encode more than a single-date label ever could.");

  // =========================================================== S8 — Four generations (frames)
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Five views, same landscape");
  s.addText("One 5×5 km tile, identical extent and unified palette (tile_331280_6541280): the 2018/2023 label sources, and three models — v8b, the distilled Prithvi, and Tessera — each trained against the field-calibrated target.",
    {x:0.9,y:1.22,w:11.5,h:0.4,fontFace:F,fontSize:12.5,color:MUTED,align:"center",margin:0});
  const isz=2.3, gap=0.13, ix0=(13.33-(5*isz+4*gap))/2, iy=1.95;
  const panelLabs=["NMD2018 label","v8b (trained on 2018)","NMD2023 label","Distilled Prithvi","Tessera"];
  const labCol=[MUTED,WHITE,MINT,CORAL,"F2C879"];
  FRAMES.forEach((im,i)=>{ const x=ix0+i*(isz+gap);
    s.addImage({data:im,x:x,y:iy,w:isz,h:isz});
    s.addText(panelLabs[i],{x:x-0.05,y:iy+isz+0.05,w:isz+0.1,h:0.35,fontFace:F,fontSize:12.5,color:labCol[i],bold:true,align:"center",margin:0}); });
  const ly=5.55, lx=0.9, lw=3.0;
  LEGEND_EN.forEach((e,i)=>{ const col=i%4, row=Math.floor(i/4); const x=lx+col*lw, y=ly+row*0.34;
    s.addShape(p.ShapeType.rect,{x:x,y:y+0.02,w:0.2,h:0.2,fill:{color:e[1]},line:{type:"none"}});
    s.addText(e[0],{x:x+0.3,y:y-0.03,w:lw-0.5,h:0.3,fontFace:F,fontSize:11.5,color:WHITE,valign:"middle",margin:0}); });
  s.addNotes("Qualitatively, the same 5-by-5 kilometre tile through five lenses. Left to right: the 2018 NMD label, our model trained on 2018, the 2023 NMD label, the distilled Prithvi model, and Tessera. Both models' outputs are visibly crisper and more coherent than the labels they learned from — denoising you can see, not just measure — and Tessera matches the 600-million-parameter Prithvi from a frozen, precomputed embedding.");

  // =========================================================== S9 — Per-class F1 table
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Per-class F1 on the held-out plots","Three sources on the exact same 209 field plots");
  // overall + kappa strip
  const cols=["NMD2023","Distilled","Fractions"];
  const strip=[["Overall accuracy",["43.1%","50.2%","57.9%"]],["Cohen's kappa",D.kappa]];
  let sx=0.8; strip.forEach(([lab,v])=>{
    s.addShape(p.ShapeType.roundRect,{x:sx,y:1.9,w:5.85,h:1.0,rectRadius:0.08,fill:{color:BG2},line:{type:"none"}});
    s.addText(lab,{x:sx+0.25,y:1.98,w:5.4,h:0.35,fontFace:F,fontSize:13,color:MINT,bold:true,margin:0});
    s.addText([{text:`${cols[0]} `,options:{color:MUTED,fontSize:12}},{text:v[0],options:{color:WHITE,fontSize:18,bold:true}},
      {text:`   ${cols[1]} `,options:{color:MUTED,fontSize:12}},{text:v[1],options:{color:CORAL,fontSize:18,bold:true}},
      {text:`   ${cols[2]} `,options:{color:MUTED,fontSize:12}},{text:v[2],options:{color:MINT,fontSize:18,bold:true}}],
      {x:sx+0.25,y:2.33,w:5.5,h:0.5,fontFace:F,margin:0}); sx+=6.05; });
  const th={fill:{color:HEAD},bold:true,fontFace:F,fontSize:13,align:"center",valign:"middle"};
  const head=[{text:"Class",options:{...th,color:WHITE,align:"left"}},
    {text:cols[0],options:{...th,color:WHITE}},{text:cols[1],options:{...th,color:CORAL}},{text:cols[2],options:{...th,color:MINT}}];
  const body=D.f1rows.map((r,i)=>{ const bg=i%2?ROWA:ROWB; const win=r[4];
    return [{text:r[0],options:{fill:{color:bg},color:WHITE,bold:true,fontFace:F,fontSize:14,align:"left",valign:"middle"}},
      ...[r[1],r[2],r[3]].map((v,j)=>{ const isWin=(j+1)===win;
        return {text:v,options:{fill:{color:isWin?WINBG:bg},color:isWin?MINT:WHITE,bold:isWin,fontFace:F,fontSize:14,align:"center",valign:"middle"}};})];});
  s.addTable([head,...body],{x:0.8,y:3.15,w:11.75,colW:[2.9,2.95,2.95,2.95],rowH:0.56,border:{type:"solid",color:GRID,pt:1}});
  s.addText("Green = best F1 per row. Distilled wins spruce and non-forest; the fraction layer wins pine and deciduous. The fraction path cannot predict non-forest (0.00) — in production a hard mask decides forest/non-forest, fractions decide type.",
    {x:0.8,y:6.65,w:11.75,h:0.6,fontFace:F,fontSize:12,italic:true,color:MUTED,margin:0,lineSpacingMultiple:1.1});
  s.addNotes("Class by class, the story is nuanced. The distilled model wins spruce and non-forest; the fraction layer wins pine, at F1 0.74, and deciduous. Non-forest is a structural zero for the fraction path — by design it only assigns forest type, so production pairs it with a hard forest mask. No single column dominates every row, which is why we ship both.");

  // =========================================================== S10 — Ensemble
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Does combining models help?");
  s.addText("Ensemble of 6 members, evaluated on the 209 held-out plots",
    {x:0.9,y:1.28,w:11.5,h:0.4,fontFace:F,fontSize:14,color:MINT,align:"center",margin:0});
  let cx=0.8; [["0.579","Single best member","(gate)",WHITE],["0.617","Best stacked combiner","OOF-locked",MINT],
              ["p = 0.37","McNemar on Δ","not significant",CORAL],["< 0.03","Tessera adds (G2)","encoder diversity",CORAL]].forEach(([b,l,l2,ac])=>{
    s.addShape(p.ShapeType.roundRect,{x:cx,y:1.85,w:2.9,h:1.85,rectRadius:0.09,fill:{color:BG2},line:{type:"none"},
      shadow:{type:"outer",color:"000000",opacity:0.35,blur:8,offset:3,angle:90}});
    s.addText(b,{x:cx+0.1,y:2.0,w:2.7,h:0.6,fontFace:F,fontSize:26,color:ac,bold:true,align:"center",margin:0});
    s.addText(l,{x:cx+0.12,y:2.62,w:2.66,h:0.4,fontFace:F,fontSize:13,color:WHITE,align:"center",bold:true,margin:0});
    s.addText(l2,{x:cx+0.12,y:3.05,w:2.66,h:0.5,fontFace:F,fontSize:11.5,color:MUTED,align:"center",margin:0}); cx+=3.02; });
  card(p,s,0.8,4.0,11.75,2.35,"Honest verdict: within noise",
    "The best stacked combiner (0.617) edges the single best member (0.579), but McNemar gives p = 0.37 and the bootstrap 95% CI on the difference is [−0.038, +0.11] — it spans zero. Encoder-diversity ablation G2 (adding the Tessera member) moved the combiner by less than 0.03. Conclusion: the ensemble does not significantly beat the single best member, and encoder diversity did not pay off.",
    CORAL,19,15);
  s.addNotes("A natural question: does ensembling help? We tried six members and an out-of-fold-locked stacked combiner. It reaches 0.617 versus 0.579 for the single best member — but McNemar's p is 0.37 and the bootstrap confidence interval on the gap spans zero. Adding an encoder-diverse Tessera member changed almost nothing. Honestly reported: the ensemble is within noise. We do not ship it.");

  // =========================================================== S11 — Prithvi vs Tessera head-to-head
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Prithvi vs Tessera: head to head");
  s.addText("Two backbones, same field-calibrated target, raced on three independent truths — Tessera ≥ Prithvi on every one.",
    {x:0.7,y:1.28,w:11.8,h:0.4,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[
      {name:"Prithvi-600M",labels:D.h2hLabels,values:D.h2hPrithvi},
      {name:"Tessera",labels:D.h2hLabels,values:D.h2hTessera}],
    {x:0.5,y:1.9,w:7.45,h:4.35,barDir:"col",barGrouping:"clustered",
     chartColors:[NMDCOL,MINT],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:10.5,dataLabelColor:WHITE,dataLabelFormatCode:'0.000',
     showTitle:false,showLegend:true,legendPos:"b",legendColor:WHITE,legendFontFace:F,legendFontSize:12,
     valAxisMinVal:0.4,valAxisMaxVal:0.85,valAxisMajorUnit:0.1,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,valAxisLabelFormatCode:'0.00',
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:12,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:40});
  const vh={fill:{color:HEAD},bold:true,fontFace:F,fontSize:12,align:"center",valign:"middle"};
  const vrows=[["Encoder","600M forward","frozen 128-d"],["Inference cost","heavy","~ free"],
               ["NFI-209 type","0.579","0.589"],["LUCAS-28","0.477","0.499"],["LUCAS-frac","0.784","0.809"]];
  const vtable=[[{text:"",options:{...vh,color:WHITE,align:"left"}},{text:"Prithvi-600M",options:{...vh,color:WHITE}},{text:"Tessera",options:{...vh,color:MINT}}],
    ...vrows.map((r,i)=>{ const bg=i%2?ROWA:ROWB;
      return [{text:r[0],options:{fill:{color:bg},color:WHITE,bold:true,fontFace:F,fontSize:12,align:"left",valign:"middle"}},
        {text:r[1],options:{fill:{color:bg},color:MUTED,fontFace:F,fontSize:12,align:"center",valign:"middle"}},
        {text:r[2],options:{fill:{color:WINBG},color:MINT,bold:true,fontFace:F,fontSize:12,align:"center",valign:"middle"}}];})];
  s.addTable(vtable,{x:8.2,y:1.9,w:4.4,colW:[1.7,1.35,1.35],rowH:0.5,border:{type:"solid",color:GRID,pt:1}});
  card(p,s,8.2,5.0,4.4,1.45,"Deployment winner: Tessera",
    "Statistical tie on accuracy across three independent truths (NFI McNemar p = 0.88) — Tessera wins decisively on compute.",CORAL,16,13);
  s.addNotes("Now race the two backbones directly on the same field-calibrated target across three independent truths. On NFI-209 forest type Tessera scores 0.589 to Prithvi's 0.579; on the LUCAS 28-class task 0.499 to 0.477; on LUCAS forest fraction 0.809 to 0.784. Tessera is at least as good on all three. But it runs on precomputed 128-dimensional embeddings — no 600-million-parameter forward pass at train or inference time. A statistical tie on accuracy, a decisive win on compute: Tessera is the deployment winner.");

  // =========================================================== S12 — LUCAS validation
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Independent validation: LUCAS","10,108 year-matched field points, never trained on");
  // L2b table
  s.addText("L2b — 28-class overall",{x:0.8,y:1.85,w:5.6,h:0.35,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  const l2bHead=[{text:"Backbone",options:{fill:{color:HEAD},color:WHITE,bold:true,fontFace:F,fontSize:12,align:"left",valign:"middle"}},
                 {text:"Overall",options:{fill:{color:HEAD},color:WHITE,bold:true,fontFace:F,fontSize:12,align:"center",valign:"middle"}}];
  const l2bBody=D.lucasL2b.map((r,i)=>{ const bg=i%2?ROWB:ROWA; const win=i===0;
    return [{text:r[0],options:{fill:{color:win?WINBG:bg},color:win?MINT:WHITE,bold:win,fontFace:F,fontSize:13,align:"left",valign:"middle"}},
            {text:r[1],options:{fill:{color:win?WINBG:bg},color:win?MINT:WHITE,bold:win,fontFace:F,fontSize:13,align:"center",valign:"middle"}}];});
  s.addTable([l2bHead,...l2bBody],{x:0.8,y:2.25,w:5.6,colW:[3.8,1.8],rowH:0.52,border:{type:"solid",color:GRID,pt:1}});
  // L2a chart
  s.addText("L2a — forest fraction (dominant-species agreement)",{x:6.75,y:1.85,w:5.8,h:0.35,fontFace:F,fontSize:14,color:CORAL,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[{name:"L2a agreement",labels:D.lucasL2aLabels,values:D.lucasL2a}],
    {x:6.55,y:2.25,w:6.0,h:2.15,barDir:"bar",
     chartColors:[MINT,NMDCOL],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:14,dataLabelColor:WHITE,dataLabelFormatCode:'0.000',
     showTitle:false,showLegend:false,
     valAxisMinVal:0.7,valAxisMaxVal:0.84,valAxisMajorUnit:0.02,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,valAxisLabelFormatCode:'0.00',
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:11,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:70});
  card(p,s,0.8,4.65,5.85,1.85,"Spruce is the hard axis — for both",
    "Per species: pine 0.83/0.81, deciduous 0.95/0.92, but spruce only 0.55/0.53. The spruce weakness is shared across backbones — a task/data limit, not a model-specific failure.",MINT,16,13);
  card(p,s,6.75,4.65,5.8,1.85,"Breadth: strong vs weak (LUCAS F1)",
    "Strong: water 0.96, maize 0.94, sugar-beet 0.93, wheat 0.78. Weak: open land 0.02, shrub 0.07, mixed forest 0.21. Both corroborate NFI: Tessera ≥ best Prithvi on independent truth.",CORAL,16,13);
  s.addNotes("An entirely separate truth source: over ten thousand year-matched LUCAS points. On the 28-class task Tessera leads at 0.499; on threshold-free forest fraction it reaches 0.809 dominant-species agreement, again ahead of Prithvi. Both backbones struggle with spruce at around 0.54 — that is a shared task limit, not a model quirk. LUCAS independently confirms the NFI ordering.");

  // =========================================================== S13 — LUCAS x LPIS
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Label cross-check: LUCAS × LPIS","Year-matched crop truth, no model in the loop");
  let lx2=0.8; [["79.8%","LUCAS crop points on a\nyear-matched LPIS parcel",WHITE],
                ["0.81","agreement excluding\npasture",MINT],
                ["0.25","pasture agreement, but\n87% still LPIS grass",CORAL]].forEach(([b,l,ac])=>{
    s.addShape(p.ShapeType.roundRect,{x:lx2,y:1.85,w:3.82,h:1.6,rectRadius:0.09,fill:{color:BG2},line:{type:"none"},
      shadow:{type:"outer",color:"000000",opacity:0.35,blur:8,offset:3,angle:90}});
    s.addText(b,{x:lx2+0.1,y:1.98,w:3.62,h:0.6,fontFace:F,fontSize:28,color:ac,bold:true,align:"center",margin:0});
    s.addText(l,{x:lx2+0.12,y:2.62,w:3.58,h:0.7,fontFace:F,fontSize:12.5,color:MUTED,align:"center",margin:0,lineSpacingMultiple:1.0}); lx2+=3.97; });
  s.addText("Crop agreement (cereals & roots)",{x:0.8,y:3.75,w:6.0,h:0.4,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[{name:"agreement",labels:D.cropLabels,values:D.crop}],
    {x:0.55,y:4.15,w:6.3,h:2.55,barDir:"bar",
     chartColors:[MINT],
     showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:13,dataLabelColor:WHITE,dataLabelFormatCode:'0.00',
     showTitle:false,showLegend:false,
     valAxisMinVal:0.7,valAxisMaxVal:0.95,valAxisMajorUnit:0.05,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,valAxisLabelFormatCode:'0.00',
     catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:11,
     valGridLine:{color:GRID,size:1},catGridLine:{style:"none"},barGapWidthPct:55});
  card(p,s,7.1,4.15,5.45,2.55,"Labels are well-founded",
    "3,496 of 4,382 LUCAS crop points fall on a year-matched LPIS parcel. Cereals and roots agree 0.69–0.91. The pasture split is a ley-vs-pasture boundary the two surveys draw differently — not an error. Year-matching is essential: crops rotate annually.",
    MINT,17,14);
  s.addNotes("A pure label sanity check, no model involved. Nearly 80 percent of LUCAS crop points land on a year-matched LPIS parcel, and excluding pasture the two surveys agree at 0.81 — sugar-beet and potato above 0.90. Pasture looks low at 0.25, but 87 percent is still grass in LPIS: the two surveys simply draw the ley-versus-pasture line differently. Our crop labels are independently well-founded.");

  // =========================================================== S14 — Discussion
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Discussion: four takeaways");
  const disc=[
    ["The TARGET was the ceiling","Not the representation. Hard NMD labels cap accuracy near 46%; NFI supervision on the same features reaches 63.7%.",MINT],
    ["Representation is cheap","Tessera's frozen embeddings tie the 600M Prithvi (p = 0.88). A large encoder buys no accuracy here.",MINT],
    ["Ensembling doesn't pay","Members are correlated: the stacked combiner is within noise (p = 0.37), and encoder diversity added < 0.03.",CORAL],
    ["Hard classes are shared","Spruce and open/shrub are weak across every backbone and every truth source — a data limit, not a model flaw.",CORAL]];
  disc.forEach(([h,b,ac],i)=>{ const xx=i%2?6.85:0.8, yy=1.6+Math.floor(i/2)*2.4;
    card(p,s,xx,yy,5.7,2.15,h,b,ac,18,15); });
  s.addNotes("Four things to take away. First, the supervision target — not the feature representation — was the ceiling. Second, representation is cheap: frozen Tessera embeddings match a 600-million-parameter encoder. Third, ensembling doesn't pay because the members are correlated. Fourth, the hard classes — spruce, open land, shrub — are hard for every backbone and every truth source, so they are a data limit we should target next.");

  // =========================================================== S15 — Limitations
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Limitations");
  bullets(p,s,[
    "Thin held-out set: 209 plots limits statistical power — several head-to-head gaps sit within noise.",
    "Spruce accuracy (~0.53–0.55) is the persistent weak axis across both backbones and both truth sources.",
    "Crop classes are year-sensitive: labels and spectra must be matched to the same year, since crops rotate.",
    "Ley-vs-pasture grass classes are not cleanly separable across truth sources — a boundary, not an error.",
    "LUCAS coverage: only ~29% of crop points fall on a year-matched LPIS parcel, bounding that cross-check."],
    0.8,1.55,7.1,15);
  card(p,s,8.2,1.55,4.35,4.85,"What would move the needle",
    "More field plots — especially spruce-dominated stands — would tighten every confidence interval and is the single highest-leverage next step.\n\nStrict year-matching of spectra to crop labels is already enforced and must stay enforced.\n\nThe grass ambiguity is a survey-definition issue, best resolved upstream in the truth sources rather than in the model.",
    MINT,18,13.5);
  s.addNotes("Being honest about the boundaries. The held-out set is thin at 209 plots, so several close comparisons are genuinely within noise. Spruce stays weak everywhere. Crops demand strict year-matching. The ley-versus-pasture ambiguity is a survey-definition problem, not a model error. And LUCAS-LPIS overlap is only about a third of crop points. More field data — especially spruce — is the clearest path forward.");

  // =========================================================== S16 — Conclusions
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  title(p,s,"Conclusions");
  const conc=[
    ["Field calibration wins","On independent field truth the field-calibrated model beats NMD2023 — 43.1% → 50.2%, and 57.9% on forest type.",MINT],
    ["Deploy the pair","Ship the distilled 28-class base plus the forest-type fraction layer: hard mask decides forest, fractions decide species.",MINT],
    ["Tessera for cost","Use Tessera's precomputed embeddings for deployment — it ties the 600M model at a fraction of the compute.",CORAL],
    ["Skip the ensemble","Combining models is not worth it: the gain is within noise and adds serving complexity for nothing.",CORAL]];
  conc.forEach(([h,b,ac],i)=>{ const xx=i%2?6.85:0.8, yy=1.5+Math.floor(i/2)*2.05;
    card(p,s,xx,yy,5.7,1.85,h,b,ac,18,14.5); });
  s.addText("Provenance: docs/data/{distill,tradslag_fraction,hybrid_nfi_head}_finding.md — all held-out figures use 209 plots, grouped-tile split, calibration on the train side only.",
    {x:0.8,y:5.75,w:11.75,h:0.6,fontFace:F,fontSize:12,italic:true,color:MUTED,margin:0,lineSpacingMultiple:1.1});
  s.addNotes("To conclude: field calibration wins on independent truth, beating the very map it learned from. For deployment we recommend the distilled 28-class base paired with the forest-type fraction layer, and Tessera's cheap embeddings to run it. Skip the ensemble — the gain is within noise. All figures trace to the finding documents on the last line. Thank you.");

  return p.writeFile({fileName:`${OUT}/v8b_nfi_conference.pptx`});
}

build().then(f=>console.log("wrote",f));
