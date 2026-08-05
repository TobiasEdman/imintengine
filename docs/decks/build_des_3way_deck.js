const pptxgen = require("pptxgenjs");
const fs = require("fs");
const FR = "/Users/tobiasedman/Developer/ImintEngine/docs/decks/frames";
const OUT = "/Users/tobiasedman/Developer/ImintEngine/docs/decks";
const NMD_IMG = "image/png;base64," + fs.readFileSync(`${FR}/nmd_frame.png`).toString("base64");
const MODEL_IMG = "image/png;base64," + fs.readFileSync(`${FR}/model_frame.png`).toString("base64");
const LEGEND = [["tallskog","006400"],["granskog","228B22"],["lövskog","32CD32"],["blandskog","3CB371"],
                ["sumpskog","2E4F2E"],["öppen mark","D2B48C"],["vatten","0000FF"],["bebyggelse","FF0000"]];
const LEGEND_EN = [["Pine","006400"],["Spruce","228B22"],["Deciduous","32CD32"],["Mixed","3CB371"],
                   ["Swamp forest","2E4F2E"],["Open land","D2B48C"],["Water","0000FF"],["Built-up","FF0000"]];

// Digital Earth Sweden brand (extracted from DES_SN2000_EDMAN.pptx theme)
const BG="0E433C", BG2="145043", HEAD="0A322C", MINT="CDFCE3", CORAL="FF826C",
      WHITE="FFFFFF", MUTED="9DBDB0", ROWA="123F38", ROWB="0C3A33", F1TINT="17564A", NMDCOL="5E8478";
const F="Space Grotesk";

// Same-ytor three-way — 912 NFI plots where NMD2023 v2.1 has coverage
// (docs/data/compare-nmd2023-nfi-sameytor.json). Row order per class:
// [class, v8bU,v8bP,v8bF1, n23U,n23P,n23F1, n18U,n18P,n18F1]
const suite = {
  overall:{v8b:"46,3 %", n23:"49,3 %", n18:"40,6 %"}, overallEN:{v8b:"46.3%", n23:"49.3%", n18:"40.6%"},
  kappa:{v8b:"0,335", n23:"0,366", n18:"0,257"}, kappaEN:{v8b:"0.335", n23:"0.366", n18:"0.257"},
  barVal:[49.3,46.3,40.6],
  rows:[["tallskog","0,81","0,44","0,57","0,82","0,48","0,61","0,71","0,44","0,54"],
        ["granskog","0,66","0,47","0,55","0,76","0,50","0,60","0,59","0,40","0,47"],
        ["lövskog","0,60","0,36","0,45","0,32","0,55","0,41","0,30","0,40","0,34"],
        ["blandskog","0,31","0,36","0,34","0,35","0,28","0,31","0,38","0,25","0,30"]],
  rowsEN:[["Pine","0.81","0.44","0.57","0.82","0.48","0.61","0.71","0.44","0.54"],
          ["Spruce","0.66","0.47","0.55","0.76","0.50","0.60","0.59","0.40","0.47"],
          ["Deciduous","0.60","0.36","0.45","0.32","0.55","0.41","0.30","0.40","0.34"],
          ["Mixed","0.31","0.36","0.34","0.35","0.28","0.31","0.38","0.25","0.30"]],
  // index (0-based within the three F1 slots {v8b:3, n23:6, n18:9}) of the winning F1 per row
  f1win:[6,6,3,3],  // tall→n23, gran→n23, löv→v8b, bland→v8b
};

const T = {
  sv:{subtitle:"Multitemporal Sentinel-2-modell för svensk marktäckning",
    h1:"v8b slår sin egen labelkälla NMD2018 — ",h2:"och är ikapp nya NMD2023",
    headsub:"Fältvalidering mot Riksskogstaxeringen, 912 gemensamma plottar: overall 46,3 % (v8b) · 40,6 % (NMD2018) · 49,3 % (NMD2023)",
    facts:[["23","klasser (unified)"],["7 882","tiles, hela Sverige"],["Prithvi-600M","backbone, 504 px"],["0,5352","val mIoU (20 epoker)"]],
    s2t:"Overall accuracy mot fältdata",s2s:"5-klass (4 skogstyper + icke-skog), samma 912 NFI-plottar där NMD2023 v2.1 finns",
    bar:["NMD2023","v8b","NMD2018"],s2h:"Studenten slår läraren — nästan ikapp nästa generation",
    s2p:["v8b är tränad PÅ NMD2018 — ändå +5,7 pp bättre vid fältpunkterna.","Nya NMD2023 (laser + 2023 S2) leder med 49,3 % — men bara +3,0 pp över v8b.","v8b vinner löv- och blandskog på F1; NMD2023 vinner tall och gran."],
    s3t:"Standardmått — user's / producer's / F1",s3s:"Confusion-matris över alla plottar (4 skogstyper + icke-skog), tre källor på exakt samma 912 fältplottar",
    thC:"Klass", ovl:"Overall accuracy (5-klass)",kpl:"Cohen's kappa",
    s3n:"User's = precision · Producer's = recall. NMD2023 vinner overall, kappa, tall & gran; v8b vinner löv & blandskog. Grön = högsta F1 per rad.",
    s4t:"Vad betyder det?",linl:"Modellhistorik — val mIoU",linc:["v6a","300M/256","600M/512","v8","v8b"],
    cards:[["Slår sin egen labelkälla","v8b tränades på NMD2018 och slår ändå NMD2018 på F1, overall och kappa vid fältpunkterna — multitemporal S2 + LiDAR-aux avbrusar per-pixel-felen."],
      ["Ikapp nya generationen","NMD2023 (laser + 2023 S2) leder overall 49,3 vs 46,3 — men v8b vinner löv- och blandskog. Gapet är litet, inte principiellt."],
      ["Nästa steg: träna på NMD2023","Samma denoising borde lyfta en NMD2023-tränad v8b över NMD2023. Bygg om labels med NMD2023-bas, matcha 2023-spektral, finetuna."]],
    foot:"Alla tal på samma 912 NFI-plottar där NMD2023 v2.1 har täckning. 4-vägs skogstyp i en 10 m-pixel mot en fältplott är genuint svårt — för alla metoder."},
  en:{subtitle:"Multitemporal Sentinel-2 land-cover model for Sweden",
    h1:"v8b beats its own label source NMD2018 — ",h2:"and keeps pace with the new NMD2023",
    headsub:"Field validation against the National Forest Inventory, 912 shared plots: overall 46.3% (v8b) · 40.6% (NMD2018) · 49.3% (NMD2023)",
    facts:[["23","classes (unified)"],["7,882","tiles, all of Sweden"],["Prithvi-600M","backbone, 504 px"],["0.5352","val mIoU (20 epochs)"]],
    s2t:"Overall accuracy vs field truth",s2s:"5-class (4 forest types + non-forest), same 912 NFI plots where NMD2023 v2.1 exists",
    bar:["NMD2023","v8b","NMD2018"],s2h:"Student beats teacher — nearly level with the next generation",
    s2p:["v8b is trained ON NMD2018 — yet +5.7 pp better at the field points.","The new NMD2023 (laser + 2023 S2) leads at 49.3% — but only +3.0 pp over v8b.","v8b wins deciduous and mixed on F1; NMD2023 wins pine and spruce."],
    s3t:"Standard measures — user's / producer's / F1",s3s:"Confusion matrix over all plots (4 forest types + non-forest), three sources on the exact same 912 field plots",
    thC:"Class",ovl:"Overall accuracy (5-class)",kpl:"Cohen's kappa",
    s3n:"User's = precision · Producer's = recall. NMD2023 wins overall, kappa, pine & spruce; v8b wins deciduous & mixed. Green = highest F1 per row.",
    s4t:"What does it mean?",linl:"Model lineage — val mIoU",linc:["v6a","300M/256","600M/512","v8","v8b"],
    cards:[["Beats its own label source","v8b was trained on NMD2018 and still beats NMD2018 on F1, overall and kappa at the field points — multitemporal S2 + LiDAR aux denoises the per-pixel errors."],
      ["Level with the new generation","NMD2023 (laser + 2023 S2) leads overall 49.3 vs 46.3 — but v8b wins deciduous and mixed. The gap is small, not fundamental."],
      ["Next: train on NMD2023","The same denoising should lift an NMD2023-trained v8b above NMD2023. Rebuild labels on the NMD2023 base, match 2023 spectral, fine-tune."]],
    foot:"All figures on the same 912 NFI plots where NMD2023 v2.1 has coverage. 4-way forest type in a 10 m pixel against a field plot is genuinely hard — for every method."},
};

function wordmark(p,s){ s.addText([{text:"◆ ",options:{color:MINT}},{text:"Digital Earth Sweden",options:{color:WHITE}}],
  {x:9.5,y:0.35,w:3.5,h:0.35,fontFace:F,fontSize:12,bold:true,align:"right",margin:0}); }

function build(t,useEN,out){
  const p=new pptxgen(); p.layout="LAYOUT_WIDE";
  const rows=useEN?suite.rowsEN:suite.rows;
  const ov=useEN?suite.overallEN:suite.overall, kp=useEN?suite.kappaEN:suite.kappa;

  // S1
  let s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText("v8b",{x:0.65,y:1.1,w:8,h:1.5,fontFace:F,fontSize:92,color:WHITE,bold:true,margin:0});
  s.addText(t.subtitle,{x:0.72,y:2.6,w:11,h:0.6,fontFace:F,fontSize:22,color:MINT,margin:0});
  s.addShape(p.ShapeType.roundRect,{x:0.7,y:3.5,w:11.9,h:2.1,rectRadius:0.12,fill:{color:BG2},line:{type:"none"},shadow:{type:"outer",color:"000000",opacity:0.4,blur:10,offset:3,angle:90}});
  s.addText([{text:t.h1,options:{color:WHITE}},{text:t.h2,options:{color:CORAL,bold:true}}],{x:1.1,y:3.8,w:11.1,h:0.9,fontFace:F,fontSize:27,bold:true,valign:"top",margin:0});
  s.addText(t.headsub,{x:1.1,y:4.9,w:11.1,h:0.7,fontFace:F,fontSize:15,color:MINT,margin:0});
  let fx=0.7; t.facts.forEach(([b,l])=>{ s.addText(b,{x:fx,y:5.95,w:2.85,h:0.6,fontFace:F,fontSize:28,color:WHITE,bold:true,margin:0});
    s.addText(l,{x:fx,y:6.6,w:2.85,h:0.4,fontFace:F,fontSize:13,color:MUTED,margin:0}); fx+=3.13; });

  // S1b — example frames (NMD reference vs v8b prediction)
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(useEN?"Example — NMD reference vs v8b prediction":"Exempel — NMD-referens vs v8b-prediktion",
    {x:0.7,y:0.5,w:11,h:0.8,fontFace:F,fontSize:32,color:WHITE,bold:true,margin:0});
  s.addText(useEN?"One 5×5 km tile, 23-class unified schema (tile_331280_6541280, northern Sweden)":
    "En 5×5 km-tile, 23-klass unified schema (tile_331280_6541280, norra Sverige)",
    {x:0.72,y:1.24,w:12,h:0.4,fontFace:F,fontSize:14,color:MUTED,margin:0});
  const iy=1.95, isz=3.9, gap=0.5, ix1=1.9, ix2=ix1+isz+gap+1.4;
  s.addImage({data:NMD_IMG,x:ix1,y:iy,w:isz,h:isz});
  s.addImage({data:MODEL_IMG,x:ix2,y:iy,w:isz,h:isz});
  s.addText(useEN?"NMD-based reference":"NMD-baserad referens",{x:ix1,y:iy+isz+0.08,w:isz,h:0.4,fontFace:F,fontSize:15,color:MINT,bold:true,align:"center",margin:0});
  s.addText("v8b",{x:ix2,y:iy+isz+0.08,w:isz,h:0.4,fontFace:F,fontSize:15,color:CORAL,bold:true,align:"center",margin:0});
  const leg=useEN?LEGEND_EN:LEGEND; const ly=6.62, lx=0.9, lw=3.0;
  leg.forEach((e,i)=>{ const col=i%4, row=Math.floor(i/4); const x=lx+col*lw, y=ly+row*0.36;
    s.addShape(p.ShapeType.rect,{x:x,y:y+0.02,w:0.22,h:0.22,fill:{color:e[1]},line:{type:"none"}});
    s.addText(e[0],{x:x+0.32,y:y-0.03,w:lw-0.5,h:0.32,fontFace:F,fontSize:12.5,color:WHITE,margin:0,valign:"middle"}); });

  // S2 — overall-accuracy bar (three sources)
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s2t,{x:0.7,y:0.5,w:11,h:0.8,fontFace:F,fontSize:36,color:WHITE,bold:true,margin:0});
  s.addText(t.s2s,{x:0.72,y:1.3,w:11,h:0.5,fontFace:F,fontSize:15,color:MUTED,margin:0});
  s.addChart(p.ChartType.bar,[{name:"overall",labels:t.bar,values:suite.barVal}],{x:0.7,y:2.0,w:7.3,h:4.9,barDir:"col",
    chartColors:[MINT,CORAL,NMDCOL],showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:15,dataLabelColor:WHITE,dataLabelFormatCode:'0.0"%"',
    showTitle:false,showLegend:false,valAxisMinVal:35,valAxisMaxVal:52,valAxisMajorUnit:5,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:11,valAxisLabelFormatCode:'0"%"',
    catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:14,valGridLine:{color:"1C5248",size:1},catGridLine:{style:"none"},barGapWidthPct:55});
  s.addText(t.s2h,{x:8.4,y:2.15,w:4.3,h:0.7,fontFace:F,fontSize:20,color:MINT,bold:true,margin:0});
  let ry=3.05; t.s2p.forEach(r=>{ s.addShape(p.ShapeType.ellipse,{x:8.4,y:ry+0.06,w:0.16,h:0.16,fill:{color:CORAL},line:{type:"none"}});
    s.addText(r,{x:8.72,y:ry-0.05,w:4.0,h:1.1,fontFace:F,fontSize:14.5,color:WHITE,margin:0,valign:"top"}); ry+=1.25; });

  // S3 — three-way standard-measures table
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s3t,{x:0.7,y:0.45,w:11,h:0.8,fontFace:F,fontSize:32,color:WHITE,bold:true,margin:0});
  s.addText(t.s3s,{x:0.72,y:1.18,w:12.2,h:0.5,fontFace:F,fontSize:13.5,color:MUTED,margin:0});
  // overall + kappa strips, three values each
  const strip=[[t.ovl,ov],[t.kpl,kp]]; let sx=0.7;
  strip.forEach(([lab,v])=>{ s.addShape(p.ShapeType.roundRect,{x:sx,y:1.75,w:5.9,h:1.0,rectRadius:0.08,fill:{color:BG2},line:{type:"none"}});
    s.addText(lab,{x:sx+0.25,y:1.83,w:5.4,h:0.35,fontFace:F,fontSize:13,color:MINT,bold:true,margin:0});
    s.addText([{text:"NMD2023 ",options:{color:MUTED,fontSize:12}},{text:v.n23,options:{color:MINT,fontSize:20,bold:true}},
      {text:"   v8b ",options:{color:MUTED,fontSize:12}},{text:v.v8b,options:{color:CORAL,fontSize:20,bold:true}},
      {text:"   NMD2018 ",options:{color:MUTED,fontSize:12}},{text:v.n18,options:{color:WHITE,fontSize:20,bold:true}}],
      {x:sx+0.25,y:2.18,w:5.5,h:0.5,fontFace:F,margin:0}); sx+=6.1; });
  const grp={fill:{color:HEAD},bold:true,fontFace:F,fontSize:11.5,align:"center",valign:"middle"};
  const th={fill:{color:HEAD},color:MUTED,bold:true,fontFace:F,fontSize:10,align:"center",valign:"middle"};
  const grphead=[{text:t.thC,options:{...grp,color:WHITE,align:"left"}},
    {text:"v8b",options:{...grp,color:CORAL,colspan:3}},{text:"NMD2023",options:{...grp,color:MINT,colspan:3}},{text:"NMD2018",options:{...grp,color:WHITE,colspan:3}}];
  const sub=["","U","P","F1","U","P","F1","U","P","F1"].map((x,i)=>({text:x,options:{...th,fill:{color:[3,6,9].includes(i)?F1TINT:HEAD}}}));
  const body=rows.map((r,i)=>{ const bg=i%2?ROWA:ROWB; const win=suite.f1win[i];
    return [{text:r[0],options:{fill:{color:bg},color:WHITE,bold:true,fontFace:F,fontSize:12,align:"left",valign:"middle"}},
      ...r.slice(1).map((v,j)=>{ const colIdx=j+1; const isF1=[3,6,9].includes(colIdx); const isWin=colIdx===win;
        return {text:v,options:{fill:{color:isWin?"1E6B54":(isF1?F1TINT:bg)},color:isWin?MINT:(isF1?WHITE:MUTED),bold:isWin||isF1,fontFace:F,fontSize:11.5,align:"center",valign:"middle"}};})];});
  s.addTable([grphead,sub,...body],{x:0.7,y:3.0,w:12.15,colW:[2.25,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1],rowH:[0.4,0.34,0.6,0.6,0.6,0.6],border:{type:"solid",color:"1C5248",pt:1}});
  s.addText(t.s3n,{x:0.7,y:6.75,w:12.2,h:0.6,fontFace:F,fontSize:12,italic:true,color:MUTED,margin:0});

  // S4
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s4t,{x:0.7,y:0.55,w:11,h:0.8,fontFace:F,fontSize:36,color:WHITE,bold:true,margin:0});
  s.addText(t.linl,{x:0.7,y:1.55,w:6,h:0.4,fontFace:F,fontSize:15,color:MINT,bold:true,margin:0});
  s.addChart(p.ChartType.line,[{name:"mIoU",labels:t.linc,values:[0.366,0.472,0.477,0.515,0.535]}],{x:0.6,y:2.0,w:6.0,h:4.6,
    chartColors:[CORAL],lineSize:3,lineSmooth:true,showValue:true,dataLabelPosition:"t",dataLabelColor:WHITE,dataLabelFontFace:F,dataLabelFontSize:12,dataLabelFormatCode:"0.000",
    showTitle:false,showLegend:false,valAxisMinVal:0.3,valAxisMaxVal:0.6,valAxisMajorUnit:0.1,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,
    catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:12,valGridLine:{color:"1C5248",size:1},catGridLine:{style:"none"},lineDataSymbol:"circle",lineDataSymbolSize:7});
  let py=1.7; t.cards.forEach(([h,b],i)=>{ const accent=i===2?CORAL:MINT;
    s.addShape(p.ShapeType.roundRect,{x:7.0,y:py,w:5.6,h:1.5,rectRadius:0.1,fill:{color:BG2},line:{type:"none"}});
    s.addText(h,{x:7.3,y:py+0.16,w:5.05,h:0.45,fontFace:F,fontSize:17,color:accent,bold:true,margin:0,valign:"top"});
    s.addText(b,{x:7.3,y:py+0.63,w:5.05,h:0.82,fontFace:F,fontSize:12.5,color:WHITE,margin:0,valign:"top"}); py+=1.68; });
  s.addText(t.foot,{x:0.7,y:6.85,w:12,h:0.5,fontFace:F,fontSize:12.5,italic:true,color:MUTED,margin:0});

  return p.writeFile({fileName:out});
}
build(T.sv,false,`${OUT}/v8b_nmd_benchmark_DES.pptx`)
  .then(()=>build(T.en,true,`${OUT}/v8b_nmd_benchmark_DES_en.pptx`))
  .then(f=>console.log("wrote",f));
