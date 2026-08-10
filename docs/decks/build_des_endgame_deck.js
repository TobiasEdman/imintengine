// Endgame benchmark deck (DES brand): field-calibrated model beats NMD2023.
// Regenerates docs/decks/v8b_nmd_benchmark_DES{,_en}.pptx (same filenames as the
// superseded three-way version; git history keeps the old story).
// Run: NODE_PATH=$(npm root -g) node docs/decks/build_des_endgame_deck.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const FR = path.join(__dirname, "frames");
const OUT = __dirname;
const NMD_IMG = "image/png;base64," + fs.readFileSync(`${FR}/nmd2023_frame.png`).toString("base64");
const MODEL_IMG = "image/png;base64," + fs.readFileSync(`${FR}/distill_frame.png`).toString("base64");

const LEGEND = [["tallskog","006400"],["granskog","228B22"],["lövskog","32CD32"],["blandskog","3CB371"],
                ["sumpskog","2E4F2E"],["hygge","00CED1"],["öppen mark","D2B48C"],["vatten","0000FF"],
                ["bebyggelse","FF0000"],["gräsdom. mark","FFD27E"],["risdom. mark","CDAA66"],["buskdom. mark","ABC8A6"]];
const LEGEND_EN = [["Pine","006400"],["Spruce","228B22"],["Deciduous","32CD32"],["Mixed","3CB371"],
                   ["Swamp forest","2E4F2E"],["Clear-cut","00CED1"],["Open land","D2B48C"],["Water","0000FF"],
                   ["Built-up","FF0000"],["Grass-dom.","FFD27E"],["Dwarf-shrub","CDAA66"],["Shrub-dom.","ABC8A6"]];

// Digital Earth Sweden brand
const BG="0E433C", BG2="145043", HEAD="0A322C", MINT="CDFCE3", CORAL="FF826C",
      WHITE="FFFFFF", MUTED="9DBDB0", ROWA="123F38", ROWB="0C3A33", F1TINT="17564A", NMDCOL="5E8478";
const F="Space Grotesk";

// Held-out 209 NFI plots (grouped tile split — never seen by head/distill/finetune).
// docs/data/distill_finding.md + tradslag_fraction_finding.md.
const T2 = {
  bar:[43.1, 50.2, 57.9],
  rows:[["tallskog","0,59","0,61","0,74"],["granskog","0,56","0,63","0,55"],
        ["lövskog","0,30","0,43","0,59"],["blandskog","0,28","0,30","0,29"],
        ["icke-skog","0,24","0,32","0,00"]],
  rowsEN:[["Pine","0.59","0.61","0.74"],["Spruce","0.56","0.63","0.55"],
          ["Deciduous","0.30","0.43","0.59"],["Mixed","0.28","0.30","0.29"],
          ["Non-forest","0.24","0.32","0.00"]],
  f1win:[3,2,3,2,2],  // col idx of winner per row (1=NMD,2=distill,3=frac)
  ov:{sv:["43,1 %","50,2 %","57,9 %"], en:["43.1%","50.2%","57.9%"]},
  kp:{sv:["0,298","0,371","0,420"], en:["0.298","0.371","0.420"]},
  journey:[46.5, 46.0, 49.3, 52.7, 63.7],
};

const T = {
  sv:{title:"v8b-NFI", subtitle:"Fältkalibrerad marktäckningsmodell för Sverige",
    h1:"Modellen slår nu NMD2023 — ",h2:"på oberoende fältdata",
    headsub:"Håll-ut-validering (209 NFI-plottar ingen träningsfas sett): full modell 50,2 % vs NMD2023 43,1 % · skogstypslager 57,9 %",
    facts:[["28","klasser (unified)"],["7 882","tiles, hela Sverige"],["944","NFI-fältplottar"],["+0,07 / +0,15","över NMD2023 (håll-ut)"]],
    s2t:"Exempel — NMD2023 vs modellen",s2s:"En 5×5 km-tile, unified färgpalett (tile_331280_6541280). Vänster: NMD2023-baserad etikett. Höger: distillerad modell.",
    lab1:"NMD2023-etikett",lab2:"v8b-NFI (distillerad)",
    s3t:"Håll-ut-resultat — overall accuracy",s3s:"209 NFI-plottar på tiles som varken huvudträning, distillering eller finetune sett. NMD2023 poängsatt på exakt samma plottar.",
    bar:["NMD2023","Distillerad modell","Skogstypslager (fraktioner)"],
    s3h:"Ärlig utvärdering",
    s3p:["Trösklar och huvud tränade ENBART på träningssplittens 735 plottar — de 209 är orörda.","Distillerad modell = full 28-klassprodukt, alla klasser fungerar (+7,2 pp).","Fraktionslagret (NFI-kollapsregel) når 57,9 % på skogstyp — tall F1 0,74."],
    s4t:"Per klass — F1 på håll-ut-plottarna",s4s:"Tre källor på exakt samma 209 fältplottar",
    thC:"Klass",cols:["NMD2023","Distill.","Fraktioner"],ovl:"Overall accuracy",kpl:"Cohen's kappa",
    s4n:"Grön = bästa F1 per rad. Distillerad vinner gran & icke-skog; fraktionslagret vinner tall & löv. Fraktionsvägen kan inte säga icke-skog (produktion: hård mask + fraktionstyp).",
    s5t:"Vägen förbi taket",s5l:"NFI overall (944 plottar) genom experimentserien",
    s5c:["v8b (NMD2018)","NMD2023-labels","NMD2023 självt","Distillerad","NFI-head (OOF)"],
    cards:[["Hårda labels = proxytak","Träning mot NMD-klasser fastnade på ~0,46 oavsett generation — denoising-effekten skalar med lärarens brus, och NMD2023 är för ren att slå via imitation."],
      ["Fältmålet bröt taket","Samma features + NFI-supervision: 0,64 (läckagefritt OOF). Representationen räckte — det var träningsMÅLET som var begränsningen."],
      ["Produktrekommendation","Distillerad 28-klassbas + fraktionshuvudet som skogstypslager (hård mask avgör skog/ej skog, fraktioner avgör typ)."]],
    foot:"Alla håll-ut-tal: 209 plottar, grupperad tile-split, kalibrering endast på träningssidan. Detalj: docs/data/{distill,tradslag_fraction,hybrid_nfi_head}_finding.md"},
  en:{title:"v8b-NFI", subtitle:"Field-calibrated land-cover model for Sweden",
    h1:"The model now beats NMD2023 — ",h2:"on independent field data",
    headsub:"Held-out validation (209 NFI plots unseen by any training stage): full model 50.2% vs NMD2023 43.1% · forest-type layer 57.9%",
    facts:[["28","classes (unified)"],["7,882","tiles, all of Sweden"],["944","NFI field plots"],["+0.07 / +0.15","over NMD2023 (held-out)"]],
    s2t:"Example — NMD2023 vs the model",s2s:"One 5×5 km tile, unified palette (tile_331280_6541280). Left: NMD2023-based label. Right: distilled model.",
    lab1:"NMD2023 label",lab2:"v8b-NFI (distilled)",
    s3t:"Held-out results — overall accuracy",s3s:"209 NFI plots on tiles unseen by head training, distillation and finetune. NMD2023 scored on the exact same plots.",
    bar:["NMD2023","Distilled model","Forest-type layer (fractions)"],
    s3h:"Honest evaluation",
    s3p:["Thresholds and head trained ONLY on the 735 train-split plots — the 209 are untouched.","Distilled model = full 28-class product, every class functional (+7.2 pp).","The fraction layer (NFI collapse rule) reaches 57.9% on forest type — pine F1 0.74."],
    s4t:"Per class — F1 on the held-out plots",s4s:"Three sources on the exact same 209 field plots",
    thC:"Class",cols:["NMD2023","Distilled","Fractions"],ovl:"Overall accuracy",kpl:"Cohen's kappa",
    s4n:"Green = best F1 per row. Distilled wins spruce & non-forest; the fraction layer wins pine & deciduous. The fraction path cannot say non-forest (production: hard mask + fraction type).",
    s5t:"Breaking the label ceiling",s5l:"NFI overall (944 plots) across the experiment series",
    s5c:["v8b (NMD2018)","NMD2023 labels","NMD2023 itself","Distilled","NFI head (OOF)"],
    cards:[["Hard labels = proxy ceiling","Training against NMD classes capped at ~0.46 regardless of generation — the denoising effect scales with teacher noise, and NMD2023 is too clean to beat by imitation."],
      ["The field target broke it","Same features + NFI supervision: 0.64 (leak-free OOF). The representation sufficed — the supervision TARGET was the constraint."],
      ["Production recommendation","Distilled 28-class base + the fraction head as a forest-type layer (hard mask decides forest/non-forest, fractions decide type)."]],
    foot:"All held-out figures: 209 plots, grouped tile split, calibration on the train side only. Detail: docs/data/{distill,tradslag_fraction,hybrid_nfi_head}_finding.md"},
};

function wordmark(p,s){ s.addText([{text:"◆ ",options:{color:MINT}},{text:"Digital Earth Sweden",options:{color:WHITE}}],
  {x:9.5,y:0.35,w:3.5,h:0.35,fontFace:F,fontSize:12,bold:true,align:"right",margin:0}); }

function build(t,useEN,out){
  const p=new pptxgen(); p.layout="LAYOUT_WIDE";
  const rows=useEN?T2.rowsEN:T2.rows, ov=useEN?T2.ov.en:T2.ov.sv, kp=useEN?T2.kp.en:T2.kp.sv;

  // S1 — title
  let s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.title,{x:0.65,y:1.0,w:9,h:1.5,fontFace:F,fontSize:80,color:WHITE,bold:true,margin:0});
  s.addText(t.subtitle,{x:0.72,y:2.45,w:11,h:0.6,fontFace:F,fontSize:22,color:MINT,margin:0});
  s.addShape(p.ShapeType.roundRect,{x:0.7,y:3.4,w:11.9,h:2.1,rectRadius:0.12,fill:{color:BG2},line:{type:"none"},shadow:{type:"outer",color:"000000",opacity:0.4,blur:10,offset:3,angle:90}});
  s.addText([{text:t.h1,options:{color:WHITE}},{text:t.h2,options:{color:CORAL,bold:true}}],{x:1.1,y:3.7,w:11.1,h:0.9,fontFace:F,fontSize:29,bold:true,valign:"top",margin:0});
  s.addText(t.headsub,{x:1.1,y:4.75,w:11.1,h:0.7,fontFace:F,fontSize:15,color:MINT,margin:0});
  let fx=0.7; t.facts.forEach(([b,l])=>{ s.addText(b,{x:fx,y:5.9,w:2.95,h:0.6,fontFace:F,fontSize:26,color:WHITE,bold:true,margin:0});
    s.addText(l,{x:fx,y:6.55,w:2.95,h:0.4,fontFace:F,fontSize:13,color:MUTED,margin:0}); fx+=3.13; });

  // S2 — example frames
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s2t,{x:0.7,y:0.5,w:11,h:0.8,fontFace:F,fontSize:32,color:WHITE,bold:true,margin:0});
  s.addText(t.s2s,{x:0.72,y:1.24,w:12,h:0.4,fontFace:F,fontSize:13.5,color:MUTED,margin:0});
  const iy=1.9, isz=3.85, ix1=1.9, ix2=ix1+isz+1.9;
  s.addImage({data:NMD_IMG,x:ix1,y:iy,w:isz,h:isz});
  s.addImage({data:MODEL_IMG,x:ix2,y:iy,w:isz,h:isz});
  s.addText(t.lab1,{x:ix1,y:iy+isz+0.08,w:isz,h:0.4,fontFace:F,fontSize:15,color:MINT,bold:true,align:"center",margin:0});
  s.addText(t.lab2,{x:ix2,y:iy+isz+0.08,w:isz,h:0.4,fontFace:F,fontSize:15,color:CORAL,bold:true,align:"center",margin:0});
  const leg=useEN?LEGEND_EN:LEGEND; const ly=6.55, lx=0.75, lw=3.0;
  leg.forEach((e,i)=>{ const col=i%4, row=Math.floor(i/4); const x=lx+col*lw, y=ly+row*0.32;
    s.addShape(p.ShapeType.rect,{x:x,y:y+0.02,w:0.2,h:0.2,fill:{color:e[1]},line:{type:"none"}});
    s.addText(e[0],{x:x+0.3,y:y-0.03,w:lw-0.5,h:0.3,fontFace:F,fontSize:11.5,color:WHITE,margin:0,valign:"middle"}); });

  // S3 — held-out bar
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s3t,{x:0.7,y:0.5,w:11,h:0.8,fontFace:F,fontSize:34,color:WHITE,bold:true,margin:0});
  s.addText(t.s3s,{x:0.72,y:1.28,w:12,h:0.5,fontFace:F,fontSize:14,color:MUTED,margin:0});
  s.addChart(p.ChartType.bar,[{name:"overall",labels:t.bar,values:T2.bar}],{x:0.7,y:2.05,w:7.3,h:4.8,barDir:"col",
    chartColors:[NMDCOL,CORAL,MINT],showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:15,dataLabelColor:WHITE,dataLabelFormatCode:'0.0"%"',
    showTitle:false,showLegend:false,valAxisMinVal:35,valAxisMaxVal:62,valAxisMajorUnit:5,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:11,valAxisLabelFormatCode:'0"%"',
    catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:12,valGridLine:{color:"1C5248",size:1},catGridLine:{style:"none"},barGapWidthPct:55});
  s.addText(t.s3h,{x:8.4,y:2.2,w:4.3,h:0.5,fontFace:F,fontSize:21,color:MINT,bold:true,margin:0});
  let ry=2.95; t.s3p.forEach(r=>{ s.addShape(p.ShapeType.ellipse,{x:8.4,y:ry+0.06,w:0.16,h:0.16,fill:{color:CORAL},line:{type:"none"}});
    s.addText(r,{x:8.72,y:ry-0.05,w:4.0,h:1.2,fontFace:F,fontSize:14,color:WHITE,margin:0,valign:"top"}); ry+=1.3; });

  // S4 — per-class table
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s4t,{x:0.7,y:0.45,w:11,h:0.8,fontFace:F,fontSize:32,color:WHITE,bold:true,margin:0});
  s.addText(t.s4s,{x:0.72,y:1.18,w:12.2,h:0.5,fontFace:F,fontSize:14,color:MUTED,margin:0});
  const strip=[[t.ovl,ov],[t.kpl,kp]]; let sx=0.7;
  strip.forEach(([lab,v])=>{ s.addShape(p.ShapeType.roundRect,{x:sx,y:1.75,w:5.9,h:1.0,rectRadius:0.08,fill:{color:BG2},line:{type:"none"}});
    s.addText(lab,{x:sx+0.25,y:1.83,w:5.4,h:0.35,fontFace:F,fontSize:13,color:MINT,bold:true,margin:0});
    s.addText([{text:`${t.cols[0]} `,options:{color:MUTED,fontSize:12}},{text:v[0],options:{color:WHITE,fontSize:19,bold:true}},
      {text:`   ${t.cols[1]} `,options:{color:MUTED,fontSize:12}},{text:v[1],options:{color:CORAL,fontSize:19,bold:true}},
      {text:`   ${t.cols[2]} `,options:{color:MUTED,fontSize:12}},{text:v[2],options:{color:MINT,fontSize:19,bold:true}}],
      {x:sx+0.25,y:2.18,w:5.5,h:0.5,fontFace:F,margin:0}); sx+=6.1; });
  const th={fill:{color:HEAD},bold:true,fontFace:F,fontSize:12,align:"center",valign:"middle"};
  const head=[{text:t.thC,options:{...th,color:WHITE,align:"left"}},
    {text:t.cols[0],options:{...th,color:WHITE}},{text:t.cols[1],options:{...th,color:CORAL}},{text:t.cols[2],options:{...th,color:MINT}}];
  const body=rows.map((r,i)=>{ const bg=i%2?ROWA:ROWB; const win=T2.f1win[i];
    return [{text:r[0],options:{fill:{color:bg},color:WHITE,bold:true,fontFace:F,fontSize:13,align:"left",valign:"middle"}},
      ...r.slice(1).map((v,j)=>{ const isWin=(j+1)===win;
        return {text:v,options:{fill:{color:isWin?"1E6B54":bg},color:isWin?MINT:WHITE,bold:isWin,fontFace:F,fontSize:13,align:"center",valign:"middle"}};})];});
  s.addTable([head,...body],{x:0.7,y:3.05,w:12.15,colW:[3.15,3.0,3.0,3.0],rowH:0.56,border:{type:"solid",color:"1C5248",pt:1}});
  s.addText(t.s4n,{x:0.7,y:6.6,w:12.2,h:0.7,fontFace:F,fontSize:12,italic:true,color:MUTED,margin:0});

  // S5 — the journey
  s=p.addSlide(); s.background={color:BG}; wordmark(p,s);
  s.addText(t.s5t,{x:0.7,y:0.55,w:11,h:0.8,fontFace:F,fontSize:36,color:WHITE,bold:true,margin:0});
  s.addText(t.s5l,{x:0.7,y:1.5,w:6.2,h:0.4,fontFace:F,fontSize:14,color:MINT,bold:true,margin:0});
  s.addChart(p.ChartType.bar,[{name:"overall",labels:t.s5c,values:T2.journey}],{x:0.55,y:1.95,w:6.3,h:4.7,barDir:"col",
    chartColors:[NMDCOL,NMDCOL,WHITE,CORAL,MINT],showValue:true,dataLabelPosition:"outEnd",dataLabelFontFace:F,dataLabelFontSize:12,dataLabelColor:WHITE,dataLabelFormatCode:'0.0"%"',
    showTitle:false,showLegend:false,valAxisMinVal:40,valAxisMaxVal:68,valAxisMajorUnit:5,valAxisLabelColor:MUTED,valAxisLabelFontFace:F,valAxisLabelFontSize:10,
    catAxisLabelColor:WHITE,catAxisLabelFontFace:F,catAxisLabelFontSize:10,valGridLine:{color:"1C5248",size:1},catGridLine:{style:"none"},barGapWidthPct:40});
  let py=1.7; t.cards.forEach(([h,b],i)=>{ const accent=i===2?CORAL:MINT;
    s.addShape(p.ShapeType.roundRect,{x:7.1,y:py,w:5.5,h:1.5,rectRadius:0.1,fill:{color:BG2},line:{type:"none"}});
    s.addText(h,{x:7.4,y:py+0.14,w:4.95,h:0.45,fontFace:F,fontSize:16.5,color:accent,bold:true,margin:0,valign:"top"});
    s.addText(b,{x:7.4,y:py+0.6,w:4.95,h:0.85,fontFace:F,fontSize:12,color:WHITE,margin:0,valign:"top"}); py+=1.68; });
  s.addText(t.foot,{x:0.7,y:6.85,w:12.2,h:0.5,fontFace:F,fontSize:11.5,italic:true,color:MUTED,margin:0});

  return p.writeFile({fileName:out});
}
build(T.sv,false,`${OUT}/v8b_nmd_benchmark_DES.pptx`)
  .then(()=>build(T.en,true,`${OUT}/v8b_nmd_benchmark_DES_en.pptx`))
  .then(f=>console.log("wrote",f));
