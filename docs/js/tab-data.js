'use strict';

// ── Sentinel-1 detection colours (radar sub-tabs) ───────────────────
// Single-hue orange ramp by AI2 score, dim → bright, ascending; a score
// takes the colour of the last bin whose min it reaches. Validated as an
// ordinal ramp (dataviz validate_palette.js --ordinal --mode dark) on the
// radar water of the Efter (#1e1e1e) and Före (#484848) scenes. Used by
// gjStyle in app.js and by the ai2_s1_vessel legend below.
var S1_SCORE_RAMP = [
    {min:0.3, max:0.5, color:'#c06a1a'},
    {min:0.5, max:0.7, color:'#e0801f'},
    {min:0.7, max:0.9, color:'#fb9a2c'},
    {min:0.9, max:1.0, color:'#ffbd5c'}
];

// ── Shared legend definitions (reused across tabs) ──────────────────
var LEGENDS = {
    nmd: [
        {color:'#006400',label:'Tallskog'},{color:'#228B22',label:'Granskog'},
        {color:'#32CD32',label:'Lövskog'},{color:'#3CB371',label:'Blandskog'},
        {color:'#90EE90',label:'Temp. ej skog'},{color:'#2E4F2E',label:'Sumpsk. tall'},
        {color:'#3A5F3A',label:'Sumpsk. gran'},{color:'#4A7F4A',label:'Sumpsk. löv'},
        {color:'#5A8F5A',label:'Sumpsk. bland'},{color:'#7AAF7A',label:'Sumpsk. temp'},
        {color:'#8B5A2B',label:'Öpp. våtmark'},{color:'#FFD700',label:'Åkermark'},
        {color:'#C8AD7F',label:'Öpp. mark bar'},{color:'#D2B48C',label:'Öpp. mark veg.'},
        {color:'#FF0000',label:'Byggnader'},{color:'#FF4500',label:'Infrastruktur'},
        {color:'#FF6347',label:'Vägar'},{color:'#0000FF',label:'Sjöar'},
        {color:'#1E90FF',label:'Hav'}
    ],
    // NFI tab: unified classes in the model maps and the background-layer
    // scales, both copied from docs/showcase/nfi_validation/MANIFEST.json
    // (map_classes_present, background_scales[*].legend).
    unified_nfi: [
        {color:'#006400',label:'Tallskog'},{color:'#228B22',label:'Granskog'},
        {color:'#32CD32',label:'Lövskog'},{color:'#3CB371',label:'Blandskog'},
        {color:'#2E4F2E',label:'Sumpskog'},{color:'#A0C878',label:'Tillfälligt ej skog'},
        {color:'#00CED1',label:'Hygge'},{color:'#8B5A2B',label:'Våtmark'},
        {color:'#D2B48C',label:'Öppen mark'},{color:'#ABC8A6',label:'Buskdominerad mark'},
        {color:'#CDAA66',label:'Risdominerad mark'},{color:'#FFD27E',label:'Gräsdominerad mark'},
        {color:'#E0E0E0',label:'Öppen mark utan vegetation'},{color:'#E6B422',label:'Vete'},
        {color:'#D48217',label:'Korn'},{color:'#FFFF64',label:'Havre'},
        {color:'#BE9650',label:'Råg'},{color:'#2DB45A',label:'Oljeväxter'},
        {color:'#8CB432',label:'Trindsäd'},{color:'#DCC800',label:'Majs'},
        {color:'#64C864',label:'Slåttervall'},{color:'#50A03C',label:'Bete'},
        {color:'#B45028',label:'Potatis'},{color:'#C864C8',label:'Sockerbetor'},
        {color:'#FF0000',label:'Bebyggelse'},{color:'#0000FF',label:'Vatten'}
    ],
    nfi_dem: [
        {color:'#FFFFE5',label:'30 m'},{color:'#FEE390',label:'52.5 m'},{color:'#FE9829',label:'75 m'},{color:'#CB4B02',label:'97.5 m'},{color:'#662506',label:'120 m'}
    ],
    nfi_height: [
        {color:'#FFFFE5',label:'0 m'},{color:'#D9F0A3',label:'7.5 m'},{color:'#77C679',label:'15 m'},{color:'#228343',label:'22.5 m'},{color:'#004529',label:'30 m'}
    ],
    nfi_vpp_length: [
        {color:'#440154',label:'60 dagar'},{color:'#3B528B',label:'105 dagar'},{color:'#21918C',label:'150 dagar'},{color:'#5EC962',label:'195 dagar'},{color:'#FDE725',label:'240 dagar'}
    ],
    nfi_lpis: [
        {color:'#D48217',label:'Korn'},{color:'#2DB45A',label:'Oljeväxter'},{color:'#64C864',label:'Slåttervall'},{color:'#50A03C',label:'Bete'},{color:'#B45028',label:'Potatis'},{color:'#8CB432',label:'Trindsäd'},{color:'#BE9650',label:'Råg'}
    ],
    nmd_grazing: [
        {color:'#FFD700',label:'Åkermark'},{color:'#D2B48C',label:'Öpp. mark veg.'},
        {color:'#50B43C',label:'Ädellövskog'},{color:'#32CD32',label:'Triviallöv'},
        {color:'#228B22',label:'Granskog'},{color:'#46A064',label:'Blandskog'},
        {color:'#006400',label:'Tallskog'},{color:'#4A7F4A',label:'Skog våtmark'},
        {color:'#8B5A2B',label:'Öpp. våtmark'},{color:'#FF0000',label:'Bebyggelse'},
        {color:'#0000FF',label:'Vatten'}
    ],
    ndvi: [
        {color:'#a50026',label:'-1.0'},{color:'#f46d43',label:'-0.5'},
        {color:'#fee08b',label:'0.0'},{color:'#a6d96a',label:'0.5'},
        {color:'#006837',label:'1.0'}
    ],
    ndwi: [
        {color:'#67001f',label:'-1.0'},{color:'#d6604d',label:'-0.5'},
        {color:'#f7f7f7',label:'0.0'},{color:'#4393c3',label:'0.5'},
        {color:'#053061',label:'1.0 Vatten'}
    ],
    evi: [
        {color:'#a50026',label:'-1.0'},{color:'#f46d43',label:'-0.5'},
        {color:'#fee08b',label:'0.0'},{color:'#a6d96a',label:'0.5'},
        {color:'#006837',label:'1.0'}
    ],
    evi_grazing: [
        {color:'#a50026',label:'-0.5'},{color:'#fee08b',label:'0.0'},
        {color:'#a6d96a',label:'0.5'},{color:'#006837',label:'1.0'}
    ],
    cot: [
        {color:'#FFFFB2',label:'0 (Klart)'},{color:'#FD8D3C',label:'0.015 (Tunt moln)'},
        {color:'#BD0026',label:'0.05 (Tjockt moln)'}
    ],
    dnbr: [
        {color:'#1a9850',label:'Hög återväxt (< -0.25)'},
        {color:'#91cf60',label:'Låg återväxt (-0.25 – -0.1)'},
        {color:'#d9ef8b',label:'Obränt (-0.1 – 0.1)'},
        {color:'#fee08b',label:'Låg svårighetsgrad (0.1 – 0.27)'},
        {color:'#fdae61',label:'Måttligt låg (0.27 – 0.44)'},
        {color:'#f46d43',label:'Måttligt hög (0.44 – 0.66)'},
        {color:'#d73027',label:'Hög svårighetsgrad (> 0.66)'}
    ],
    change_gradient: [
        {color:'#FFFFB2',label:'Liten förändring'},
        {color:'#FD8D3C',label:'Måttlig förändring'},
        {color:'#BD0026',label:'Stor förändring'}
    ],
    prithvi_fire: [
        {color:'#228B22',label:'Ej bränt'},{color:'#FF4500',label:'Bränt'}
    ],
    sam: [
        {color:'#0000FF',label:'0° (identisk spektral signatur)'},
        {color:'#00FF00',label:'~5° (liten avvikelse)'},
        {color:'#FFFF00',label:'~7° (märkbar drift)'},
        {color:'#FF0000',label:'≥10° (misstänkt hallucination)'}
    ],
    vessel: [
        {color:'#00E5FF',label:'Detekterad båt / anomali'}
    ],
    ai2_vessel: [
        {color:'#50A0FF',label:'Stationär (0 kn)'},{color:'#FFE600',label:'Låg fart (< 5 kn)'},
        {color:'#FF6600',label:'Medelfart (5–15 kn)'},{color:'#FF0000',label:'Hög fart (> 15 kn)'}
    ],
    ai2_s1_vessel: [
        {swatch:'border:2px solid #cff8e4',label:'Heldragen: poäng ≥ 0.7 (AI2:s tröskel)'},
        {swatch:'border:2px dashed #cff8e4',label:'Streckad: kandidat 0.3–0.7'}
    ].concat(S1_SCORE_RAMP.map(function(b) {
        return {color:b.color, label:'Poäng ' + b.min.toFixed(1) + '–' + b.max.toFixed(1)};
    })),
    s1_vv_db: [
        {color:'#000000',label:'≤ 28 dB (lugnt vatten)'},{color:'#808080',label:'43 dB'},
        {color:'#ffffff',label:'≥ 58 dB (fartyg, bebyggelse)'}
    ],
    heatmap: [
        {color:'#FFFFB2',label:'Låg'},{color:'#FD8D3C',label:'Medel'},
        {color:'#BD0026',label:'Hög'}
    ],
    lpis: [
        {color:'#E6119D',label:'Aktiv betesmark'},
        {color:'#111111',label:'Ingen aktivitet'},
        {color:'#888888',label:'Ej analyserad'}
    ],
    coastseg: [
        {color:'rgb(20,102,191)',label:'Djupt vatten'},
        {color:'rgb(99,181,244)',label:'Grunt vatten'},
        {color:'rgb(211,173,113)',label:'Sediment'},
        {color:'rgb(17,122,62)',label:'Land'}
    ],
    shoreline_years: [
        {color:'#e41a1c',label:'2018'},{color:'#ff7f00',label:'2019'},
        {color:'#ffd700',label:'2020'},{color:'#4daf4a',label:'2021'},
        {color:'#00ced1',label:'2022'},{color:'#377eb8',label:'2023'},
        {color:'#984ea3',label:'2024'},{color:'#e41a9d',label:'2025'}
    ],
    erosion: [
        {color:'#d73027',label:'Erosion'},{color:'#ffffbf',label:'Stabil'},
        {color:'#1a9850',label:'Ackumulation'}
    ],
    vegetation_seg: [
        {color:'#1E78B4',label:'Vatten'},{color:'#D9C799',label:'Ej vegetation'},
        {color:'#2BA22B',label:'Vegetation'}
    ],
    vegetation_edge_years: [
        {color:'#e41a1c',label:'2018'},{color:'#ff7f00',label:'2019'},
        {color:'#ffd700',label:'2020'},{color:'#4daf4a',label:'2021'},
        {color:'#00ced1',label:'2022'},{color:'#377eb8',label:'2023'},
        {color:'#984ea3',label:'2024'},{color:'#e41a9d',label:'2025'}
    ],
    lulc_grouped: [
        {color:'#006400',label:'Tallskog'},{color:'#228B22',label:'Granskog'},
        {color:'#32CD32',label:'Lövskog'},{color:'#3CB371',label:'Blandskog'},
        {color:'#2E4F2E',label:'Sumpskog'},{color:'#8B5A2B',label:'Öpp. våtmark'},
        {color:'#FFD700',label:'Åkermark'},{color:'#D2B48C',label:'Öpp. mark'},
        {color:'#FF0000',label:'Bebyggelse'},{color:'#0000FF',label:'Vatten'}
    ],
    disagree: [
        {color:'#2ecc40',label:'Korrekt'},{color:'#ff4136',label:'Fel'},
        {color:'#ff00ff',label:'Hög konfidens fel'}
    ],
    confidence: [
        {color:'#d73027',label:'Låg (0.0)'},{color:'#fee08b',label:'Medel (0.5)'},
        {color:'#1a9850',label:'Hög (1.0)'}
    ],
    entropy: [
        {color:'#FFFFB2',label:'Låg (säker)'},{color:'#FD8D3C',label:'Medel'},
        {color:'#BD0026',label:'Hög (osäker)'}
    ],
    chl: [
        {color:'#440154',label:'Låg (~0.5 mg/m³)'},
        {color:'#3b528b',label:'Måttlig (~2 mg/m³)'},
        {color:'#21918c',label:'Förhöjd (~5 mg/m³)'},
        {color:'#5ec962',label:'Blomning (~15 mg/m³)'},
        {color:'#fde725',label:'Hög blomning (>25 mg/m³)'}
    ],
    sigma: [
        {color:'#ffffff',label:'Låg osäkerhet'},
        {color:'#999999',label:'Medel'},
        {color:'#000000',label:'Hög osäkerhet'}
    ],
    tss: [
        {color:'#00224e',label:'Klart vatten (<2 g/m³)'},
        {color:'#5d508a',label:'Lätt grumligt'},
        {color:'#a36e85',label:'Grumligt'},
        {color:'#ee7e58',label:'Sediment­plym (>20 g/m³)'},
        {color:'#fff494',label:'Mycket högt'}
    ],
    cdom: [
        {color:'#000000',label:'Lågt CDOM (<0.5 m⁻¹)'},
        {color:'#73450f',label:'Måttligt'},
        {color:'#c47e16',label:'Förhöjt'},
        {color:'#ffe491',label:'Mycket högt (>2 m⁻¹)'}
    ],
    ndci: [
        {color:'#67001f',label:'-0.2 (klart vatten)'},
        {color:'#f7f7f7',label:'0.0'},
        {color:'#053061',label:'+0.6 (klorofyll)'}
    ],
    mci: [
        {color:'#000004',label:'-0.02 (klart)'},
        {color:'#721f81',label:'0'},
        {color:'#ed6925',label:'+0.04 (förhöjd)'},
        {color:'#fcffa4',label:'+0.08 (blomning)'}
    ],
    spread: [
        {color:'#000004',label:'Metoder överens'},
        {color:'#721f81',label:'Liten skillnad'},
        {color:'#ed6925',label:'Tydlig skillnad'},
        {color:'#fcffa4',label:'Stor oenighet'}
    ],
    moisture: [
        {color:'#ffffd9',label:'Torrt (0)'},
        {color:'#41b6c4',label:'Måttligt (50)'},
        {color:'#225ea8',label:'Vått (100)'}
    ],
    height_m: [
        {color:'#440154',label:'Mark (0 m)'},
        {color:'#3b528b',label:'Buske/ungskog (5 m)'},
        {color:'#21918c',label:'Medel (15 m)'},
        {color:'#5ec962',label:'Hög (25 m)'},
        {color:'#fde725',label:'Mogen (35+ m)'}
    ],
    cover_pct: [
        {color:'#f7fcf5',label:'Glest (0%)'},
        {color:'#a1d99b',label:'Måttligt'},
        {color:'#238b45',label:'Tätt (100%)'}
    ],
    elevation: [
        {color:'#1f4d2e',label:'Lågt'},
        {color:'#a89060',label:'Medel'},
        {color:'#f0f0f0',label:'Högt'}
    ]
};

// NFI tab: the model maps and the NMD2023 background use different class
// schemes, so each strip carries a text-only title item.
LEGENDS.nmd2023 = [{label:'<strong>NMD2023:</strong>'}].concat(LEGENDS.nmd);
LEGENDS.nfi_label = [{label:'<strong>Träningsetikett:</strong>'}].concat(LEGENDS.unified_nfi, [{label:'transparent = ingen etikett'}]);
LEGENDS.unified_nfi = [{label:'<strong>Modellkarta:</strong>'}].concat(LEGENDS.unified_nfi);
LEGENDS.nfi_lpis = [{label:'<strong>Grödor (LPIS):</strong>'}].concat(LEGENDS.nfi_lpis);
LEGENDS.nfi_harvest = [{label:'<strong>Avverkningar:</strong>'}, {color:'#00CED1',label:'utförd avverkning inom fem år före rutans år'}];
LEGENDS.nfi_dem = [{label:'<strong>Höjd över havet:</strong>'}].concat(LEGENDS.nfi_dem);
LEGENDS.nfi_height = [{label:'<strong>Trädhöjd:</strong>'}].concat(LEGENDS.nfi_height, [{label:'transparent = inga träd eller inga data'}]);
LEGENDS.nfi_vpp_length = [{label:'<strong>Växtsäsongens längd:</strong>'}].concat(LEGENDS.nfi_vpp_length, [{label:'transparent = ingen växtsäsong (t.ex. vatten) eller inga data'}]);
LEGENDS.nfi_embedding = [{label:'<strong>TESSERA-embedding:</strong>'}, {label:'färg = de tre första huvudkomponenterna av 128 dimensioner; lika färg ≈ liknande signatur över året'}];

// Radar sub-tabs: every panel that can show the Sjökort background carries
// the chart's attribution as a text-only legend item.
var S1_CHART_CREDITS = {
    osm: {label:'Sjökort: © OpenStreetMap contributors (ODbL), seamarks © OpenSeaMap (CC BY-SA)'},
    s57: {label:'Sjökort: Sjökortsdata © Sjöfartsverket'}
};
Object.keys(S1_CHART_CREDITS).forEach(function(k) {
    LEGENDS['s1_vv_db_' + k] = LEGENDS.s1_vv_db.concat([S1_CHART_CREDITS[k]]);
    LEGENDS['ai2_s1_vessel_' + k] = LEGENDS.ai2_s1_vessel.concat([S1_CHART_CREDITS[k]]);
});

// ── Glossary (acronym tooltip, decorated on first occurrence per tab) ──
// Added 2026-05-11 as part of the showcase-copy rewrite. Same dict serves
// every tab — acronyms mean the same thing across the dashboard. The JS
// decorator in app.js walks tab content after render and wraps the first
// matching word per tab in <span class="gloss" data-term="...">. Hover
// shows full + sv lines from this dict.
var GLOSSARY = {
    // Spectral indices
    'NDVI':  { full: 'Normalized Difference Vegetation Index',
               sv: 'Mått på grönska. Räknas ur röd och nära-infraröd reflektans; höga värden = frisk vegetation.' },
    'NDWI':  { full: 'Normalized Difference Water Index',
               sv: 'Mått på öppet vatten och fukt. Positiva värden = vatten.' },
    'EVI':   { full: 'Enhanced Vegetation Index',
               sv: 'Vegetationsindex som är mindre känsligt för atmosfär och tät biomassa än NDVI.' },
    'MNDWI': { full: 'Modified Normalized Difference Water Index',
               sv: 'Vattenindex som använder kortvågs-IR istället för nära-IR — bättre i bebyggelse.' },
    'NDCI':  { full: 'Normalized Difference Chlorophyll Index',
               sv: 'Klorofyllindex för vatten — använder rödedge-bandet (705 nm).' },
    'MCI':   { full: 'Maximum Chlorophyll Index',
               sv: 'Klorofyllindex som mäter höjden på 705 nm-toppen över en linjär baslinje.' },
    'dNBR':  { full: 'Differenced Normalized Burn Ratio',
               sv: 'Brandsvårighetsgrad — skillnaden i NBR före och efter brand. Positiva värden = bränt.' },
    'SAM':   { full: 'Spectral Angle Mapper',
               sv: 'Vinkel mellan två spektrum, i grader. Mäter färgavvikelse oberoende av ljusstyrka.' },

    // Sentinel-2 / Copernicus
    'SCL':   { full: 'Scene Classification Layer',
               sv: 'ESA:s pixel-vis klassning av Sentinel-2-scener (moln, skugga, vatten, vegetation, …).' },
    'COT':   { full: 'Cloud Optical Thickness',
               sv: 'Mått på molntäckets tjocklek. Värden nära 0 = klart; höga = ogenomskinligt moln.' },
    'L1C':   { full: 'Level-1C (Top-of-Atmosphere)',
               sv: 'Sentinel-2-produkt med reflektans uppmätt vid satelliten, före atmosfär-korrigering.' },
    'L2A':   { full: 'Level-2A (Bottom-of-Atmosphere)',
               sv: 'Sentinel-2-produkt med atmosfär-korrigerad ytreflektans (Sen2Cor).' },
    'BOA':   { full: 'Bottom-of-Atmosphere',
               sv: 'Reflektans nära markytan, efter atmosfär-korrigering.' },
    'TOA':   { full: 'Top-of-Atmosphere',
               sv: 'Reflektans uppmätt vid satelliten, före atmosfär-korrigering.' },
    'SAFE':  { full: 'Standard Archive Format for Europe',
               sv: 'ESA:s arkivformat för Sentinel-produkter (en katalog med metadata + per-band-filer).' },
    'AOI':   { full: 'Area of Interest',
               sv: 'Det geografiska område som analysen är avgränsad till.' },

    // Svenska geodata-källor
    'NMD':   { full: 'Nationella marktäckedata',
               sv: 'Naturvårdsverkets rikstäckande karta över marktäcke, 10 m upplösning, 25+ klasser.' },
    'LPIS':  { full: 'Land Parcel Identification System',
               sv: 'Jordbruksverkets blockdatabas — registrerade jordbruksskiften och betesblock.' },
    'SKS':   { full: 'Skogsstyrelsen',
               sv: 'Svensk myndighet; data om avverkningsanmälningar och skogstillstånd.' },
    'DEM':   { full: 'Digital Elevation Model',
               sv: 'Höjddata över marken (rasterkarta i meter över havet).' },
    'SMI':   { full: 'Soil Moisture Index',
               sv: 'NMD:s markfuktighetsindex (0–100, där 100 = vått).' },

    // Vattenkvalitet
    'C2RCC': { full: 'Case-2 Regional CoastColour',
               sv: 'ESA:s neurala nätverk för optiskt komplexa kustvatten — räknar fram klorofyll, sediment, CDOM.' },
    'MDN':   { full: 'Mixture Density Network',
               sv: 'Neuralt nätverk (Pahlevan et al.) som returnerar klorofyll-a för Sentinel-2-vatten.' },
    'CDOM':  { full: 'Coloured Dissolved Organic Matter',
               sv: 'Färgat löst organiskt material i vatten (humus, gelbstoff). Mäts i absorbans m⁻¹.' },
    'TSM':   { full: 'Total Suspended Matter',
               sv: 'Totalt suspenderat material i vatten (sediment, partiklar). Mäts i g/m³.' },
    'TSS':   { full: 'Total Suspended Solids',
               sv: 'Synonym med TSM — totalt suspenderat material i vatten, g/m³.' },
    'IOPs':  { full: 'Inherent Optical Properties',
               sv: 'Vattnets inneboende optiska egenskaper (absorption, spridning) — input till färg-modeller.' },

    // ML / detection
    'YOLO':  { full: 'You Only Look Once',
               sv: 'AI-modell för objektdetektering i ett enda steg (här: båtar i satellitbilder).' },
    'NMS':   { full: 'Non-Maximum Suppression',
               sv: 'Steg som filtrerar bort överlappande detektioner och behåller den med högst poäng.' },
    'CNN':   { full: 'Convolutional Neural Network',
               sv: 'Djupt neuralt nätverk för bildanalys.' },
    'FCN':   { full: 'Fully Convolutional Network',
               sv: 'CNN-variant designad för pixel-vis segmentering.' }
};

// ── GeoJSON file paths (loaded async) ────────────────────────────────
var GEOJSON_FILES = {
    vessels:              'data/vessels.geojson',
    mc_vessels:           'data/mc_vessels.geojson',
    mc_ai2_vessels:       'data/mc_ai2_vessels.geojson',
    mr_skagen_vessels:    'data/mr_skagen_vessels.geojson',
    mr_oresund_vessels:   'data/mr_oresund_vessels.geojson',
    lpis:                 'data/lpis.geojson',
    erosion:              'data/erosion.geojson',
    segformer_shorelines: 'data/segformer-shorelines.geojson',
    coastline_shorelines: 'data/coastline-shorelines.geojson',
    vegetation_edges:     'data/vegetation-edge-vectors.geojson'
};

// ── Tab configurations ───────────────────────────────────────────────
var TAB_CONFIG = {

    fire: {
        title: 'Brandanalys — 2018-07-24',
        summary: [
            {title:'Förändringsdetektering',value:'22.8%',detail:'49 regioner'},
            {title:'dNBR Hög svårighetsgrad',value:'1.2 km²',detail:'8.9% av området'},
            {title:'NMD Marktäcke',value:'82.4% Skog',detail:'6 klasser'},
            {title:'Molnanalys (COT)',value:'99.9% klart',detail:'COT medel: 0.0017'}
        ],
        copy: {
            lede: 'Bilden visar Kårbölebranden i Ljusdals kommun den 24 juli 2018 — en av de största skogsbränderna i Sverige på över 50 år. Branden startade 14 juli och bredde ut sig över cirka 9 500 hektar. Vi har analyserat samma område före och efter med flera kompletterande metoder för att kartlägga brandens utbredning och intensitet.',
            hint: 'Klicka i panelraden ovan för att hoppa till en specifik analys. Dra opacitetsreglaget i panelens topp för att fade:a brandresultaten över satellitbilden.',
            methods: [
                { heading: 'dNBR — brandsvårighetsgrad',
                  body: '<p>Skillnaden i Normalized Burn Ratio mellan bilden före och efter branden. Höga positiva värden = svår brandskada; nära noll = obränt. Tröskelvärden från Key & Benson är ursprungligen kalibrerade för västliga USA-barrskogar, så absolutgränserna för svensk borealskog ska tolkas som indikativa, inte definitiva.</p>' },
                { heading: 'NDVI / NDWI / EVI / NMD',
                  body: '<p>NDVI och EVI mäter vegetationens tillstånd; NDWI mäter fukt. NMD-marktäcket före branden visar att området var till 82,4% skog, vilket gör korsreferensen mot brandkartan informativ — vi vet exakt vilka skogstyper som drabbades hårdast.</p>' },
                { heading: 'Förändring (gradient)',
                  body: '<p>Multispektral förändringsdetektering över alla 10 m-band, inte bara index. Fångar förändringar som ett enskilt index missar (t.ex. förändringar i markstruktur efter sotning).</p>' },
                { heading: 'Molnanalys (COT)',
                  body: '<p>DES MLP5-ensemble — fem djupa nätverk tränade på syntetiska molndata från SMHI. Båda scenerna är 99,9% klara, så brandresultaten är inte påverkade av molnskuggor.</p>' }
            ]
        },
        panels: [
            {id:'f-rgb',    key:'rgb',     title:'Sentinel-2 RGB',                    legend:null,
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'}]},
            {id:'f-nmd',    key:'nmd',     title:'NMD Marktäcke',                     legend:'nmd'},
            {id:'f-dnbr',   key:'dnbr',    title:'dNBR (Brandsvårighetsgrad)',         legend:'dnbr',
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'}]},
            {id:'f-gradient',key:'gradient',title:'Förändring (gradient)',             legend:'change_gradient',
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'}]},
            {id:'f-ndvi',   key:'ndvi',    title:'NDVI (Vegetationsindex)',            legend:'ndvi'},
            {id:'f-ndwi',   key:'ndwi',    title:'NDWI (Vattenindex)',                 legend:'ndwi'},
            {id:'f-evi',    key:'evi',     title:'EVI (Enhanced Vegetation Index)',    legend:'evi'},
            {id:'f-cot',    key:'cot',     title:'Molnoptisk tjocklek (COT)',          legend:'cot'}
        ],
        images: {
            'f-rgb':       'showcase/fire/rgb.png',
            'f-nmd':       'showcase/fire/nmd_overlay.png',
            'f-ndvi':      'showcase/fire/ndvi_clean.png',
            'f-ndwi':      'showcase/fire/ndwi_clean.png',
            'f-evi':       'showcase/fire/evi_clean.png',
            'f-cot':       'showcase/fire/cot_clean.png',
            'f-dnbr':      'showcase/fire/dnbr_clean.png',
            'f-gradient':  'showcase/fire/change_gradient.png',
            'f-baseline':  'showcase/fire/baseline_rgb.png'
        },
        imgH: 559, imgW: 267,
        hasBgToggle: false,
        hasCharts: true,
        chartSectionTitle: 'Korsreferens mot NMD (Nationellt Marktackedata)'
    },

    nfi_validation: {
        title: 'Marktäckekartering — fältvalidering mot Riksskogstaxeringen (förhandsvisning)',
        summary: [
            {title:'Fältdata',value:'5 194 provyte-år',detail:'742 per år, 2018–2024'},
            {title:'Utvärderingsmängd',value:'Förseglad',detail:'låst med SHA-256 innan modellerna kördes'},
            {title:'Jämförelse',value:'Samma provytor',detail:'modeller, NMD2018 och NMD2023 parvis'},
            {title:'Statistik',value:'Bestämd i förväg',detail:'50 km-block, korrigerad för multipla jämförelser'}
        ],
        copy: {
            lede: 'Hur väl stämmer marktäckekartor med vad som faktiskt finns på marken? Vi har jämfört modellernas kartor och Naturvårdsverkets NMD med Riksskogstaxeringens provytor från sju inventeringsår. I stora drag: både Prithvi-EO-2.0 och TESSERA stämmer oftare med fältbedömningen än NMD2023 — men bara när de tränas med fältkalibrerade etiketter. Exempelrutan nedan, ca 5 × 5 km i ett skogs- och odlingslandskap, ingår inte i utvärderingsmängden. De första panelerna visar indata och träningsetiketter, de två sista modellernas kartor.',
            hint: 'Växla bakgrund mellan satellitbild och NMD2023 i panelens topp, och dra opacitetsreglaget för att tona lagret över bakgrunden. Panelerna zoomar och panorerar tillsammans. Under kartorna beskrivs varje datakälla och i vilket steg den används.',
            methods: [
                { heading: 'Utvärderingsdesign',
                  body: '<p>742 provyte-år per år 2018–2024 valdes med ett seedat, dokumenterat urval och låstes med en SHA-256-hash innan någon modell kördes; poängsättningen kontrollerar hashen innan den räknar. Modellerna, NMD2018 och NMD2023 jämförs parvis på samma provytor. Måttet är överensstämmelse i fem klasser — tallskog, granskog, lövskog, blandskog och övrigt (inklusive icke-skog). Osäkerheten skattas över oberoende 50 km-block och korrigeras för det totala antalet jämförelser. De provyte-år vars data tränade den fältkalibrerade läraren är uteslutna ur utvärderingen.</p>' },
                { heading: 'Kartlagren',
                  body: '<p>Satellitbilden är en Sentinel-2-sommarbild med kontraststräckning för visning; den svarta kilen nere till höger saknar data just det datumet, och modellerna använder flera datum. NMD2023 visas med showcasens NMD-legend; NMD2023:s nyare koder (finindelad öppen våtmark och öppen mark, låg fjällskog, torvtäkt) visas som närmaste NMD2018-klass. Modellkartorna är de slutliga, fältkalibrerade modellernas fulla klassutdata — fler klasser än de fem som utvärderas. Höjdmodell, trädhöjd och växtsäsongens längd har en färgskala från 2:a–98:e percentilen i rutan; där värdet 0 betyder att inget finns (inga träd, ingen växtsäsong) är pixeln genomskinlig. TESSERA-embeddingen visas som sina tre första huvudkomponenter. Alla lager täcker modellens 504 × 504 px-fönster.</p>' }
            ]
        },
        panels: [
            {id:'nfi-rgb',     key:'rgb',     title:'Sentinel-2 RGB / NMD2023', legend:'nmd2023',
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-dem',     key:'dem',     title:'Höjdmodell (Copernicus DEM)', legend:['nfi_dem','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-height',  key:'height',  title:'Trädhöjd (Skogsstyrelsen)', legend:['nfi_height','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-vpp',     key:'vpp',     title:'Växtsäsongens längd (HR-VPP)', legend:['nfi_vpp_length','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-emb',     key:'emb',     title:'TESSERA-embedding (årlig)', legend:['nfi_embedding','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-lpis',    key:'lpis',    title:'Jordbruksskiften (Jordbruksverket, LPIS)', legend:['nfi_lpis','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-harvest', key:'harvest', title:'Utförda avverkningar (Skogsstyrelsen)', legend:['nfi_harvest','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-label',   key:'label',   title:'Träningsetikett (NMD2023 + LPIS + avverkningar)', legend:['nfi_label','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-prithvi', key:'prithvi', title:'Prithvi-EO-2.0 600M',       legend:['unified_nfi','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]},
            {id:'nfi-tessera', key:'tessera', title:'TESSERA',                   legend:['unified_nfi','nmd2023'],
                bgToggle:[{label:'Satellit',key:'rgb',active:true},{label:'NMD2023',key:'nmd'}]}
        ],
        images: {
            'nfi-rgb':     'showcase/nfi_validation/rgb.png',
            'nfi-nmd':     'showcase/nfi_validation/nmd2023.png',
            'nfi-dem':     'showcase/nfi_validation/dem.png',
            'nfi-height':  'showcase/nfi_validation/height.png',
            'nfi-vpp':     'showcase/nfi_validation/vpp_length.png',
            'nfi-emb':     'showcase/nfi_validation/tessera_embedding.png',
            'nfi-lpis':    'showcase/nfi_validation/lpis.png',
            'nfi-harvest': 'showcase/nfi_validation/harvest.png',
            'nfi-label':   'showcase/nfi_validation/training_label.png',
            'nfi-prithvi': 'showcase/nfi_validation/prithvi600m_r4.png',
            'nfi-tessera': 'showcase/nfi_validation/tessera_r4.png'
        },
        imgH: 504, imgW: 504,
        hasBgToggle: false
    },

    marine_leisure: {
        title: 'Marin analys — Fritid (2025-07-10)',
        summary: [
            {title:'Båtdetektering',value:'130 båtar',detail:'5 datum (4 skippade)'},
            {title:'Bästa datum',value:'50 båtar',detail:'2025-07-17'},
            {title:'NMD Marktäcke',value:'64.2% Vatten',detail:'6 klasser'},
            {title:'Molnanalys (COT)',value:'99.7% klart',detail:'COT medel: 0.0019'},
            {title:'Analysområde',value:'18.6 km²',detail:'Bohuslän kustzon'}
        ],
        copy: {
            lede: 'Bilden visar skärgården utanför Hunnebostrand i Bohuslän, ett kustområde med livlig maritim trafik från fiske, kommersiell sjöfart och fritidsbåtar. En AI-modell har letat efter båtar i Sentinel-2-bilden och markerat varje träff med en cyanfärgad ruta — aktivitetskartan summerar var båtarna har rört sig under sommarsäsongen 2025.',
            hint: 'Klicka i panelraden ovan för att hoppa till en specifik analys. Dra opacitetsreglaget i panelens topp för att fade:a överlägget mot satellitbilden eller sjökortet.',
            methods: [
                { heading: 'Datakällor',
                  body: '<p>Sentinel-2 L2A-data från 2025-07-10 över ett 18,6 km² stort havsområde. NMD-marktäckedata används som landmask för att filtrera bort detektioner på land.</p>' },
                { heading: 'Båtdetektering',
                  body: '<p>YOLO11s — en CNN-baserad objektdetektor — körs på RGB-utsnittet och föreslår bounding-boxar för varje sannolik båt. Överlappande detektioner filtreras bort med NMS, och pixlar utanför vattenmasken förkastas.</p>' },
                { heading: 'Heatmap över säsongen',
                  body: '<p>Heatmappen ackumulerar båtdetektioner från 5 molnfria datum. Datum med för hög molntäckning enligt COT-analysen exkluderas — annars skulle moln feltolkas som båtar.</p>' }
            ]
        },
        panels: [
            {id:'m-rgb',           key:'rgb',           title:'Sentinel-2 RGB',           legend:null,
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'m-vessels',       key:'vessels',       title:'Båtdetektering (YOLO)',    legend:'vessel',      vector:true, geojsonFile:'vessels',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'m-vessel-heatmap',key:'vessel-heatmap',title:'Båtaktivitet (heatmap)',   legend:'heatmap',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'m-nmd',           key:'nmd',           title:'NMD Marktäcke',            legend:'nmd',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'m-ndvi',          key:'ndvi',          title:'NDVI (Vegetationsindex)',   legend:'ndvi',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'m-ndwi',          key:'ndwi',          title:'NDWI (Vattenindex)',        legend:'ndwi',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'m-cot',           key:'cot',           title:'Molnoptisk tjocklek (COT)', legend:'cot',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]}
        ],
        images: {
            'm-rgb':            'showcase/marine/rgb.png',
            'm-vessels':        'showcase/marine/vessels_clean.png',
            'm-vessel-heatmap': 'showcase/marine/vessel_heatmap_clean.png',
            'm-nmd':            'showcase/marine/nmd_overlay.png',
            'm-ndvi':           'showcase/marine/ndvi_clean.png',
            'm-ndwi':           'showcase/marine/ndwi_clean.png',
            'm-cot':            'showcase/marine/cot_clean.png',
            'm-sjokort':        'showcase/marine/sjokort.png'
        },
        imgH: 573, imgW: 324,
        hasBgToggle: true
    },

    marine_commercial: {
        title: 'Marin analys — Sjöfart (2025-07-19)',
        summary: [
            {title:'YOLO-detektering',value:'54 båtar',detail:'Primärdatum 2025-07-19'},
            {title:'AI2-detektering',value:'20 båtar',detail:'Swin V2 B + attribut'},
            {title:'Analysområde',value:'~50 km²',detail:'Helsingborg–Helsingör'},
            {title:'NMD Marktäcke',value:'28% Hav',detail:'65% utanför NMD (DK)'},
            {title:'Molnanalys (COT)',value:'94.7% klart',detail:'COT medel: 0.009'}
        ],
        intro: 'Analysområdet visar Öresunds smalaste punkt mellan Helsingborg och Helsingör — ett av världens mest trafikerade sund med intensiv kommersiell sjöfart. Här passerar dagligen lastfartyg, tankers, containerfartyg och HH Ferries passagerarfärjor över den 4 km breda sundet. Sentinel-2-data har analyserats med två kompletterande AI-modeller för fartygsdetektering: YOLO11s (snabb objektdetektering, 54 detektioner) och Allen AI:s rslearn-modell (Swin V2 B + Faster R-CNN med attributprediktion för fartygstyp, hastighet och kurs, 20 detektioner).',
        panels: [
            {id:'mc-rgb',            key:'rgb',            title:'Sentinel-2 RGB',                    legend:null,
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-vessels',        key:'vessels',        title:'Båtdetektering (YOLO)',              legend:'vessel',      vector:true, geojsonFile:'mc_vessels',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-ai2-vessels',    key:'ai2_vessels',    title:'AI2 Fartygsdetektering (Swin V2)',   legend:'ai2_vessel',  vector:true, geojsonFile:'mc_ai2_vessels',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-vessel-heatmap', key:'vessel_heatmap', title:'Båtaktivitet — YOLO (heatmap)',      legend:'heatmap',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-ai2-heatmap',    key:'ai2_heatmap',    title:'Fartygsaktivitet — AI2 (heatmap)',   legend:'heatmap',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-nmd',            key:'nmd',            title:'NMD Marktäcke',                      legend:'nmd',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-ndvi',           key:'ndvi',           title:'NDVI (Vegetationsindex)',             legend:'ndvi',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-ndwi',           key:'ndwi',           title:'NDWI (Vattenindex)',                  legend:'ndwi',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]},
            {id:'mc-cot',            key:'cot',            title:'Molnoptisk tjocklek (COT)',           legend:'cot',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'Sjökort',key:'sjokort'}]}
        ],
        images: {
            'mc-rgb':            'showcase/marine_commercial/rgb.png',
            'mc-vessels':        'showcase/marine_commercial/vessels_clean.png',
            'mc-ai2-vessels':    'showcase/marine_commercial/ai2_vessels_clean.png',
            'mc-vessel-heatmap': 'showcase/marine_commercial/vessel_heatmap_clean.png',
            'mc-ai2-heatmap':    'showcase/marine_commercial/ai2_vessel_heatmap_clean.png',
            'mc-nmd':            'showcase/marine_commercial/nmd_overlay.png',
            'mc-ndvi':           'showcase/marine_commercial/ndvi_clean.png',
            'mc-ndwi':           'showcase/marine_commercial/ndwi_clean.png',
            'mc-cot':            'showcase/marine_commercial/cot_clean.png',
            'mc-sjokort':        'showcase/marine_commercial/sjokort.png'
        },
        imgH: 588, imgW: 893,
        hasBgToggle: true
    },

    // Radar sub-tabs: generated by scripts/generate_marine_radar_showcase.py
    // --area <skagen|oresund> (MANIFEST.json beside the images). Display grid
    // = _to_nmd_grid(bbox), EPSG:3006 at 10 m. Öresund uses the same COORDS as
    // marine_commercial, so the two sub-tabs line up pixel for pixel.
    marine_radar_skagen: {
        title: 'Marin analys — Radar Skagen (2026-06-28)',
        summary: [
            {title:'Detektioner ≥ 0.7',value:'25',detail:'AI2:s tröskel; 18 med poäng ≥ 0.99'},
            {title:'Kandidater 0.3–0.7',value:'9 st',detail:'Poäng 0.34–0.697, streckade rutor'},
            {title:'Predikterad längd',value:'174 m (median)',detail:'Detektioner ≥ 0.7, spann 21–278 m'},
            {title:'Radarscen (Efter)',value:'2026-06-28',detail:'Sentinel-1D, 05:31 UTC'},
            {title:'Jämförelsescen (Före)',value:'2026-05-23',detail:'Sentinel-1D, samma bana'}
        ],
        copy: {
            lede: 'Ankringsområdet söder om Skagen, i danska vatten mellan Skagerrak och Kattegatt, där handelsfartyg väntar på order, bunkring eller lots. Sentinel-1 sänder ut egna mikrovågspulser och ser därför genom moln och mörker: lugnt vatten blir mörkt, och stålskrov lyser som ljusa punkter, ofta med ett kors av sidlober. Allen AI:s Sentinel-1-modell ger 25 detektioner över sin tröskel 0.7 i passagen 2026-06-28 kl. 05:31 UTC, 18 av dem med poäng 0.99–1.00. 16 har predikterad fart under 0,5 knop och predikterad längd 77–278 m: sannolikt fartyg till ankars, med starka radarekon. Resten rör sig, bland annat fiske- och passagerarfartyg. Mellan 0.3 och 0.7 finns bara 9 kandidater.',
            hint: 'Växla bakgrund med Efter, Före och Sjökort i panelens topp. Före-bilden från 2026-05-23 visar samma ankringsområde fem veckor tidigare, med andra fartyg på andra platser. Klicka på en ruta för poäng och predikterade attribut. Heldragna rutor har poäng ≥ 0.7, streckade är kandidater under tröskeln. Zooma in (+) för att se skillnaden. Hela bilden är 23 × 21 km, så rutorna blir små.',
            methods: [
                { heading: 'Radarbilderna',
                  body: '<p>Sentinel-1D, IW GRDH, nedåtgående bana 66: Efter = 2026-06-28 05:31 UTC, Före = 2026-05-23, AI2:s jämförelsepassage längs samma bana. Hämtade kostnadsfritt från Microsoft Planetary Computer. Båda bilderna visar VV-polarisationen som 20·log10 av de okalibrerade GRD-värdena, dvs. relativ dB och inte kalibrerad σ⁰, med samma fasta skala 28–58 dB som Öresund-fliken: svart = lugnt vatten, vitt = starka reflektorer.</p>' },
                { heading: 'Detektering',
                  body: '<p>Allen AI:s rslearn-modell för Sentinel-1 (Swin V2 B + Faster R-CNN, Apache 2.0) får VV och VH från både Efter- och Före-passagen. Före-bilden ger modellen sammanhang: AI2 tränade den med en äldre passage för att skilja fartyg från fasta strukturer. Modellen slår ihop inlärda drag från båda bilderna och har ingen fast regel om att det som syns i båda förkastas. Modellen körs här med tröskel 0.3: rutor med poäng ≥ 0.7 (AI2:s produktionströskel) är heldragna, kandidater mellan 0.3 och 0.7 är streckade, och färgen går från mörk till ljus orange med poängen. Dubbletter inom 10 pixlar slås ihop och träffar inom 200 m från känd marin infrastruktur (Satlas) tas bort. En separat attributmodell predikterar längd, bredd, kurs, fart och typ.</p>' },
                { heading: 'Tolkning',
                  body: '<p>En detektion är en kandidat, inte ett bekräftat fartyg: ingen AIS-data har matchats, och alla attribut, även farten, är modellprediktioner från en enda radarbild. 22 av de 25 detektionerna ligger inom den begärda rutan 10,58–10,95° Ö, 57,64–57,82° N. Övriga 3 ligger i bildens hörn: det visade SWEREF 99 TM-rutnätet omsluter rutan och blir därför något större.</p>' },
                { heading: 'Rutnät och sjökort',
                  body: '<p>Modellen körs på AI2:s 10 m UTM-rutnät (EPSG:32632) med 1,5 km marginal runt området. Radarbilderna, detektionerna och sjökortet läggs på samma EPSG:3006-rutnät som resten av showcasen (2327×2140 px à 10 m). Inga svenska sjökortsceller täcker Skagen, så sjökortet är OpenSeaMaps sjömärken ovanpå OpenStreetMaps baskarta. Det är sammansatt av förhämtade rutor, zoomnivå 12 och 13, som omprojicerats till rutnätet. Radarbilden är inte terrängkorrigerad: mot sjökortet ligger kustlinjen upp till cirka 40 m förskjuten mot satelliten. © OpenStreetMap contributors (ODbL), seamarks © OpenSeaMap (CC BY-SA).</p>' }
            ]
        },
        panels: [
            {id:'mrs-rgb',     key:'rgb',     title:'Sentinel-1 radar (VV, dB)',                  legend:'s1_vv_db_osm',
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'},{label:'Sjökort',key:'sjokort'}]},
            {id:'mrs-vessels', key:'vessels', title:'AI2-detektioner och kandidater (Sentinel-1)', legend:'ai2_s1_vessel_osm', vector:true, geojsonFile:'mr_skagen_vessels',
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'},{label:'Sjökort',key:'sjokort'}]}
        ],
        images: {
            'mrs-rgb':      'showcase/marine_radar_skagen/rgb.png',
            'mrs-baseline': 'showcase/marine_radar_skagen/baseline.png',
            'mrs-sjokort':  'showcase/marine_radar_skagen/sjokort.png'
        },
        imgH: 2140, imgW: 2327,
        hasBgToggle: true
    },

    marine_radar_oresund: {
        title: 'Marin analys — Radar Öresund (2025-07-19)',
        summary: [
            {title:'Detektioner ≥ 0.7',value:'0',detail:'AI2:s tröskel, efter infrastrukturfilter'},
            {title:'Kandidater 0.3–0.7',value:'4 st',detail:'Poäng 0.30–0.59, streckade rutor'},
            {title:'Predikterad längd',value:'22 m (median)',detail:'Kandidaterna, spann 21–29 m'},
            {title:'Radarscen (Efter)',value:'2025-07-19',detail:'Sentinel-1C, 16:52 UTC'},
            {title:'Jämförelsescen (Före)',value:'2025-06-01',detail:'Sentinel-1C, samma bana'}
        ],
        copy: {
            lede: 'Samma vatten som i Sjöfart-fliken — Öresunds smalaste punkt mellan Helsingborg och Helsingør — men sett med radar, samma dag som den optiska bilden men vid en annan tidpunkt (2025-07-19 kl. 16:52 UTC). Sentinel-1 sänder ut egna mikrovågspulser och ser därför genom moln och mörker: lugnt vatten blir mörkt, medan fartyg, kajer och bebyggelse lyser vitt. Allen AI:s Sentinel-1-modell ger här ingen detektion över sin tröskel 0.7. De fyra kandidaterna mellan 0.3 och 0.7 är små farkoster (predikterad längd 21–29 m) på den svenska halvan av sundet, utanför Helsingborg.',
            hint: 'Växla bakgrund med Efter, Före och Sjökort i panelens topp. Det starka målet vid färjeläget i Helsingør, med ett ljust kors av sidlober, får ingen kandidat. Det syns även i Före-bilden från 2025-06-01; att det finns i båda bilderna är en möjlig förklaring, inte en fastställd orsak. Klicka på en ruta för poäng och predikterade attribut.',
            methods: [
                { heading: 'Radarbilderna',
                  body: '<p>Sentinel-1C, IW GRDH, uppåtgående bana 146: Efter = 2025-07-19 16:52 UTC, Före = 2025-06-01, AI2:s jämförelsepassage längs samma bana. Hämtade kostnadsfritt från Microsoft Planetary Computer. Båda bilderna visar VV-polarisationen som 20·log10 av de okalibrerade GRD-värdena, dvs. relativ dB och inte kalibrerad σ⁰, med samma fasta skala 28–58 dB: svart = lugnt vatten, vitt = starka reflektorer.</p>' },
                { heading: 'Detektering',
                  body: '<p>Allen AI:s rslearn-modell för Sentinel-1 (Swin V2 B + Faster R-CNN, Apache 2.0) får VV och VH från både Efter- och Före-passagen. Före-bilden ger modellen sammanhang: AI2 tränade den med en äldre passage för att skilja fartyg från fasta strukturer. Modellen slår ihop inlärda drag från båda bilderna och har ingen fast regel om att det som syns i båda förkastas. Modellen körs här med tröskel 0.3: rutor med poäng ≥ 0.7 (AI2:s produktionströskel) är heldragna, kandidater mellan 0.3 och 0.7 är streckade, och färgen går från mörk till ljus orange med poängen. Dubbletter inom 10 pixlar slås ihop och träffar inom 200 m från känd marin infrastruktur (Satlas) tas bort. En separat attributmodell predikterar längd, bredd, kurs, fart och typ.</p>' },
                { heading: 'Tolkning',
                  body: '<p>En detektion är en kandidat, inte ett bekräftat fartyg: ingen AIS-data har matchats, och alla attribut är modellprediktioner. Två något starkare kandidater (poäng 0.61–0.62, 19–20 m) ligger strax söder om det visade området, i den marginal som modellen också ser.</p>' },
                { heading: 'Rutnät och sjökort',
                  body: '<p>Modellen körs på AI2:s 10 m UTM-rutnät (EPSG:32633) med 1 km marginal runt området. Radarbilderna, detektionerna och sjökortet läggs på samma EPSG:3006-rutnät (893×588 px à 10 m) som Sjöfart-flikens Sentinel-2-bilder. Sjökortet är Sjöfartsverkets S-57-celler renderade på exakt detta rutnät. Cellerna täcker bara svenska vatten, så den danska sidan är tom. Radarbilden är inte terrängkorrigerad: mot sjökortet ligger kustlinjen cirka 50–60 m förskjuten mot satelliten, dvs. åt väster. Sjökortet stämmer med NMD inom 10 m. Sjökortsdata © Sjöfartsverket.</p>' }
            ]
        },
        panels: [
            {id:'mro-rgb',     key:'rgb',     title:'Sentinel-1 radar (VV, dB)',                  legend:'s1_vv_db_s57',
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'},{label:'Sjökort',key:'sjokort'}]},
            {id:'mro-vessels', key:'vessels', title:'AI2-detektioner och kandidater (Sentinel-1)', legend:'ai2_s1_vessel_s57', vector:true, geojsonFile:'mr_oresund_vessels',
                bgToggle:[{label:'Efter',key:'rgb',active:true},{label:'Före',key:'baseline'},{label:'Sjökort',key:'sjokort'}]}
        ],
        images: {
            'mro-rgb':      'showcase/marine_radar_oresund/rgb.png',
            'mro-baseline': 'showcase/marine_radar_oresund/baseline.png',
            'mro-sjokort':  'showcase/marine_radar_oresund/sjokort.png'
        },
        imgH: 588, imgW: 893,
        hasBgToggle: true
    },

    grazing: {
        title: 'Betesmarksanalys — 2025-06-14',
        summary: [
            {title:'Betesanalys (AI)',value:'68/80 aktiv',detail:'Konfidens: 73%'},
            {title:'Ingen aktivitet',value:'8 block',detail:'19 molnfria datum'},
            {title:'LPIS Betesblock',value:'80 block',detail:'124 ha total areal'},
            {title:'NDVI i betesmark',value:'0.81',detail:'± 0.06 standardavvikelse'},
            {title:'NMD inom betesblock',value:'7% Våtmark',detail:'2 markklasser'}
        ],
        copy: {
            lede: 'Bilden visar jordbrukslandskapet nordost om Lund i Skåne den 14 juni 2025 — åkermark, beteshagar och skogsdungar i blandning. Vi har överlagt LPIS-betesblocken från Jordbruksverket på Sentinel-2-bilden och låtit en AI-modell bedöma om varje block är aktivt betat eller ej, baserat på vegetations- och fuktindex över säsongen.',
            hint: 'Klicka i panelraden ovan för att hoppa till en specifik analys. Klicka på ett LPIS-block i kartan för att se klassificering och konfidens.',
            methods: [
                { heading: 'Datakällor',
                  body: '<p>LPIS-polygoner (80 betesblock, totalt 124 ha) från Jordbruksverkets blockdatabas. Sentinel-2 L2A från 2025-06-14, kompletterat med 19 molnfria datum för säsongsanalysen. NMD-marktäcke som referens för markklasser inom blocken.</p>' },
                { heading: 'Vegetations- och vattenindex',
                  body: '<p>NDVI, EVI och NDWI beräknas per pixel inom varje betesblock och aggregeras till blockmedel. Hög NDVI över hela säsongen utan slåttertopp = tecken på aktivt bete. Låg NDVI över sommaren = potentiellt övergivet block.</p>' },
                { heading: 'AI-klassificering',
                  body: '<p>En CNN-biLSTM-modell tränad på multitemporala blockstatistiker klassificerar varje block som "aktivt betat" eller "ingen aktivitet". 68 av 80 block bedömdes aktiva, 8 utan aktivitet, övriga 4 osäkra (konfidens under tröskelvärde). Konfidensvärdena är klassificerarens softmax-output, medel över alla blockets pixlar.</p>' },
                { heading: 'Molnanalys (COT)',
                  body: '<p>DES MLP5-ensemble. Datum med för hög molntäckning över AOI exkluderas innan säsongsstatistiken aggregeras.</p>' }
            ]
        },
        panels: [
            {id:'g-rgb',  key:'rgb',  title:'Sentinel-2<br>RGB',           legend:null},
            {id:'g-nmd',  key:'nmd',  title:'NMD<br>Marktäcke',            legend:'nmd_grazing'},
            {id:'g-lpis', key:'lpis', title:'LPIS<br>Betesblock',          legend:'lpis', vector:true, geojsonFile:'lpis',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'NMD',key:'nmd'}]},
            {id:'g-ndvi', key:'ndvi', title:'NDVI<br>Vegetationsindex',     legend:'ndvi'},
            {id:'g-evi',  key:'evi',  title:'EVI<br>Vegetationsindex',      legend:'evi_grazing'},
            {id:'g-ndwi', key:'ndwi', title:'NDWI<br>Vattenindex',          legend:'ndwi'},
            {id:'g-cot',  key:'cot',  title:'COT<br>Molnoptisk tjocklek',   legend:'cot'}
        ],
        images: {
            'g-rgb':  'showcase/grazing/rgb.png',
            'g-nmd':  'showcase/grazing/nmd_overlay.png',
            'g-lpis': 'showcase/grazing/lpis_overlay.png',
            'g-ndvi': 'showcase/grazing/ndvi_clean.png',
            'g-evi':  'showcase/grazing/evi_clean.png',
            'g-ndwi': 'showcase/grazing/ndwi_clean.png',
            'g-cot':  'showcase/grazing/cot_clean.png'
        },
        imgH: 344, imgW: 383,
        hasBgToggle: false
    },

    kustlinje: {
        title: 'Kustlinjeanalys — 2025-06-14',
        summary: [
            {title:'Analysperiod',value:'8 år',detail:'2018–2025'},
            {title:'Strandlinjeanalys',value:'CoastSat',detail:'NDWI + Otsu'},
            {title:'Vattenandel',value:'61.9%',detail:'Land: 38.0%'},
            {title:'Molnanalys (COT)',value:'Klart',detail:'Alla datum användbara'},
            {title:'Analysområde',value:'Ystad',detail:'Sydskånska kusten'}
        ],
        intro: 'Analysområdet är beläget vid Ystads kust i sydöstra Skåne — en utsatt kuststräcka längs Östersjön med dokumenterad stranderosion. Här har Sentinel-2-data från åtta år (2018–2025) analyserats med CoastSat-metoden (NDWI/MNDWI + Otsu-tröskelvärde) och CoastSeg SegFormer (neuralt nät) för att extrahera strandlinjer och kvantifiera erosion och ackumulation.',
        panels: [
            {id:'c-rgb',             key:'rgb',             title:'Sentinel-2 RGB',                                     legend:null},
            {id:'c-coastseg',        key:'coastseg',        title:'Indexbaserad Segmentering (NDWI/MNDWI + Otsu)',      legend:'coastseg'},
            {id:'c-segformer',       key:'segformer',       title:'CoastSeg SegFormer (4-klass neuralt nät)',            legend:'coastseg'},
            {id:'c-shoreline',       key:'shoreline',       title:'Strandlinjeförändring 2018–2025 (Indexbaserad)',      legend:'shoreline_years',
                vector:true, geojsonFile:'coastline_shorelines'},
            {id:'c-sf-shoreline',    key:'sf_shoreline',    title:'SegFormer Strandlinje 2018–2025',                    legend:'shoreline_years',
                vector:true, geojsonFile:'segformer_shorelines'},
            {id:'c-shoreline-change',key:'shoreline_change',title:'Kusterosion & ackumulation (Indexbaserad)',           legend:'erosion',
                vector:true, geojsonFile:'erosion'},
            {id:'c-sf-change',       key:'sf_change',       title:'SegFormer Kustförändring 2018–2025',                 legend:'shoreline_years',
                vector:true, geojsonFile:'segformer_shorelines'},
            {id:'c-ndvi',            key:'ndvi',            title:'NDVI (Vegetationsindex)',                             legend:'ndvi'},
            {id:'c-ndwi',            key:'ndwi',            title:'NDWI (Vattenindex)',                                  legend:'ndwi'},
            {id:'c-evi',             key:'evi',             title:'EVI (Enhanced Vegetation Index)',                     legend:'evi'},
            {id:'c-cot',             key:'cot',             title:'Molnoptisk tjocklek (COT)',                           legend:'cot'}
        ],
        images: {
            'c-rgb':            'showcase/kustlinje/rgb.png',
            'c-shoreline':      'showcase/kustlinje/shoreline_overlay.png',
            'c-shoreline-change':'showcase/kustlinje/shoreline_change.png',
            'c-ndvi':           'showcase/kustlinje/ndvi_clean.png',
            'c-ndwi':           'showcase/kustlinje/ndwi_clean.png',
            'c-evi':            'showcase/kustlinje/evi_clean.png',
            'c-cot':            'showcase/kustlinje/cot_clean.png',
            'c-baseline':       'showcase/kustlinje/baseline_rgb.png',
            'c-coastseg':       'showcase/kustlinje/coastline_clean.png',
            'c-segformer':      'showcase/kustlinje/coastseg_segformer.png',
            'c-sf-change':      'showcase/kustlinje/segformer_shoreline_change.png'
        },
        imgH: 508, imgW: 577,
        hasBgToggle: false
    },

    vegetationskant: {
        title: 'Vegetationskantanalys — Vänern',
        summary: [
            {title:'Analysperiod',value:'8 år',detail:'2018–2025'},
            {title:'Metod',value:'VedgeSat',detail:'NDVI + Otsu'},
            {title:'Analysområde',value:'Vänern',detail:'Lidköping/Läckö'},
            {title:'Upplösning',value:'10 m',detail:'Sentinel-2 L2A'},
            {title:'Referens',value:'Muir et al.',detail:'2024 + Nugraha 2026'}
        ],
        intro: 'Analysområdet visar Vänerns södra strand nära Lidköping och Läckö slott — en strandzon med blandning av skog, jordbruksmark och strandvegetation. Sentinel-2-data från åtta år (2018–2025) har analyserats med VedgeSat-metoden (NDVI-tröskling + morfologisk kantextraktion) för att detektera och följa vegetationskantens förändring över tid. Metoden identifierar gränsen mellan vegeterat och icke-vegeterat område utan behov av fältdata eller modellomträning.',
        panels: [
            {id:'ve-rgb',    key:'rgb',    title:'Sentinel-2 RGB',                              legend:null},
            {id:'ve-ndvi',   key:'ndvi',   title:'NDVI (Vegetationsindex)',                     legend:'ndvi'},
            {id:'ve-seg',    key:'seg',    title:'Vegetationssegmentering (3-klass)',            legend:'vegetation_seg'},
            {id:'ve-change', key:'change', title:'Vegetationskantförändring 2018–2025',          legend:'vegetation_edge_years'},
            {id:'ve-nmd',    key:'nmd',    title:'NMD Marktäcke',                               legend:'nmd'},
            {id:'ve-stability', key:'stability', title:'Spektral stabilitet (medel 2018–2025)',  legend:'change_gradient'},
            {id:'ve-maxchange', key:'maxchange', title:'Max spektral förändring (2018–2025)',    legend:'change_gradient'},
            {id:'ve-ndwi',   key:'ndwi',   title:'NDWI (Vattenindex)',                          legend:'ndwi'},
            {id:'ve-evi',    key:'evi',    title:'EVI (Enhanced Vegetation Index)',              legend:'evi'}
        ],
        images: {
            've-rgb':    'showcase/vegetationskant/rgb.png',
            've-ndvi':   'showcase/vegetationskant/ndvi_clean.png',
            've-seg':    'showcase/vegetationskant/vegetation_seg.png',
            've-change': 'showcase/vegetationskant/vegetation_edge_change.png',
            've-nmd':    'showcase/vegetationskant/nmd_overlay.png',
            've-stability': 'showcase/vegetationskant/spectral_stability.png',
            've-maxchange': 'showcase/vegetationskant/spectral_max_change.png',
            've-ndwi':   'showcase/vegetationskant/ndwi_clean.png',
            've-evi':    'showcase/vegetationskant/evi_clean.png'
        },
        imgH: 1145, imgW: 1194,
        hasBgToggle: false
    },

    water_quality: {
        title: 'Vattenkvalitet — Stigfjorden & Mollösund',
        summary: [
            {title:'AOI',                value:'Stigfjorden + Mollösund', detail:'+ nearshore Skagerrak, 24.25 × 9.93 km'},
            {title:'Sensor &amp; passage',value:'Sentinel-2B',             detail:'2026-04-08 10:30:19 UTC, single pass'},
            {title:'Metoder',            value:'4 parallella retrievals',  detail:'MDN + C2RCC (C2X-Nets) + NDCI + MCI'},
            {title:'Frame­proportion',    value:'2.442:1 landskap',         detail:'EPSG:3006 axis-aligned, 10 m NMD-grid'}
        ],
        intro: 'Sentinel-2 vattenkvalitetsanalys för Bohuslän­kusten — Stigfjorden mellan Tjörn och Orust, vattnen utanför Mollösund och Käringön samt nära-Skagerrak. April-blomningens kiselalge­signal (<em>Skeletonema costatum</em>, <em>Chaetoceros</em>) syns som klorofyll­strimmor i kustvattnen. <strong>Fyra retrievals körs parallellt utan fusion</strong>: två neurala (Pahlevan MDN för klorofyll-a och ESA C2RCC C2X-Nets för chl/TSM/CDOM via SNAP 13) och två klassiska rödedge-index (NDCI, MCI). All data är från <strong>en enda S2B-passage 2026-04-08 10:30:19 UTC</strong> — ingen mosaik mellan dagar eller satelliter. Inom passet mosaikas dock T32VPK + T33VUE (UTM-zon 32 + 33) eftersom det är samma fysiska observation. L2A-data kommer från DES openEO (snäv temporal-extent för att blockera 10:40:41-passet); L1C SAFE-arkiv från <a href="https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2" target="_blank">Google Clouds publika bucket</a>. Färgskalan på klorofyll-panelerna är <strong>fast pinnad till 0.5–25 mg/m³</strong> så legenden gäller över olika scener. Inspirations­bild: Sentinel-2 L2A True color 2026-04-08 (digitalearth.se).',
        // Panels are the 9 retrievals we have real 2026-04-08 data for. Three
        // additional MDN products (uncertainty, TSS, aCDOM) require the upstream
        // benchmarks/tss/SOLID variants of MDN with different weights and are
        // deferred to v2.
        panels: [
            {id:'wq-rgb',                 key:'rgb',                title:'Sentinel-2 RGB',                  legend:null},
            {id:'wq-water-mask',          key:'water_mask',         title:'Vattenmask (SCL)',                legend:null},
            {id:'wq-chl-mdn',             key:'chl_mdn',            title:'Klorofyll-a (MDN, mg/m³)',         legend:'chl'},
            {id:'wq-chl-c2rcc',           key:'chl_c2rcc',          title:'Klorofyll-a (C2RCC, mg/m³)',       legend:'chl'},
            {id:'wq-tsm-c2rcc',           key:'tsm_c2rcc',          title:'TSM (C2RCC, g/m³)',                legend:'tss'},
            {id:'wq-cdom-c2rcc',          key:'cdom_c2rcc',         title:'CDOM (C2RCC, m⁻¹)',               legend:'cdom'},
            {id:'wq-ndci',                key:'ndci',               title:'NDCI (klorofyllindex)',            legend:'ndci'},
            {id:'wq-mci',                 key:'mci',                title:'MCI (klorofyllindex)',             legend:'mci'},
            {id:'wq-spread',              key:'chl_spread',         title:'Metoders oenighet (log₁₀ Chl-a)',  legend:'spread'}
        ],
        // Relative paths from docs/index.html. Outputs live at
        // outputs/showcase/water_quality/<year>/ (dev) and are mirrored to
        // docs/showcase/water_quality/<year>/ (dashboard).
        images: {
            'wq-rgb':         'showcase/water_quality/2026/rgb.png',
            'wq-water-mask':  'showcase/water_quality/2026/water_mask.png',
            'wq-chl-mdn':     'showcase/water_quality/2026/chlorophyll_a_mdn.png',
            'wq-chl-c2rcc':   'showcase/water_quality/2026/chlorophyll_a_c2rcc.png',
            'wq-tsm-c2rcc':   'showcase/water_quality/2026/tsm_c2rcc.png',
            'wq-cdom-c2rcc':  'showcase/water_quality/2026/cdom_c2rcc.png',
            'wq-ndci':        'showcase/water_quality/2026/ndci.png',
            'wq-mci':         'showcase/water_quality/2026/mci.png',
            'wq-spread':      'showcase/water_quality/2026/chlorophyll_spread.png'
        },
        // 2.442:1 landscape ratio matches AOI v4 (24.25 × 9.93 km in EPSG:3006).
        // v4 (2026-04-28): WGS84 rectangle solved iteratively so transform_bounds
        // produces a clean axis-aligned 3006 rectangle — no parallelogram tilt,
        // no wedge gaps, _to_nmd_grid retained throughout the pipeline.
        imgH: 360, imgW: 880,
        hasBgToggle: false
    },

    sr: {
        title: 'Skarpare satellitbilder — Stockholm 4×',
        summary: [
            {title:'Område',           value:'Central Stockholm', detail:'Söder · Gamla Stan · Norrmalm · Östermalm · Djurgården'},
            {title:'Förstoring',       value:'4×',                detail:'Från 10 m per pixel till 2,5 m per pixel'},
            {title:'Antal metoder',    value:'3 jämförda',        detail:'Bicubic + SEN2SR + LDSR-S2 (DiffFuSR/SR4RS saknar publika vikter)'},
            {title:'Säker gräns',      value:'~2,5 m',            detail:'Finare än så börjar AI:n hitta på detaljer'}
        ],
        copy: {
            lede: 'En vanlig Sentinel-2-bild har en pixel per 10×10 meter — Stockholms slott blir cirka 12×9 pixlar. <strong>Superresolution</strong> är samlingsnamnet på tekniker som räknar fram en skarpare version av samma bild, ungefär som när mobilen "förbättrar" ett zoomat foto. Här jämförs tre metoder över centrala Stockholm: en enkel matematisk interpolation plus två AI-modeller.',
            hint: 'Dra opacitetsreglaget på varje panel för att fade:a den skarpade bilden över originalet (10 m). Vissa AI-modeller hittar på detaljer som <em>ser</em> snygga ut men inte motsvarar verkligheten — titta efter byggnader som dyker upp där originalet bara visar suddiga pixlar.',
            methods: [
                { heading: 'Metoderna som jämförs',
                  body: '<p><strong>Bicubic</strong> är klassisk matematisk interpolation utan AI — referensgolvet att jämföra mot. <strong>SEN2SR</strong> är en CNN tränad av ESA med radiometrisk hard-constraint: den får skarpa men inte hitta på spektral information. <strong>LDSR-S2</strong> är en latent diffusion-modell (samma teknik som DALL·E och Stable Diffusion) — visuellt övertygande men kan börja gissa.</p>' },
                { heading: 'Vad är fysikaliskt möjligt?',
                  body: '<p>Oberoende forskning från fyra håll pekar på <strong>cirka 2,5 m per pixel</strong> som gränsen för vad som verkligen kan återvinnas från en Sentinel-2-bild. Optik och sensorer sätter taket — bortom 2,5 m gissar AI:n från mönster den lärt sig på andra bilder. "1 m per pixel"-produkter på marknaden är till stor del AI-genererade detaljer.</p>' },
                { heading: 'Spektral skillnadskarta — har AI:n hittat på något?',
                  body: '<p>De tre nedre panelerna mäter <em>Spectral Angle Mapper</em> (SAM): vinkeln mellan original-spektrumet och AI:ns version per pixel. Noll grader = bara skarpning, ingen färgförändring. Höga värden = modellen tillförde spektral information som inte fanns i originalet, alltså hallucination. Bicubic ska vara nästan helt blå; SEN2SR mestadels blå/grön; LDSR-S2 kan ha gulare områden längs kanter.</p>' },
                { heading: 'Hur paneler visar pixlar',
                  body: '<p>SR-panelerna är inställda så att en bildpunkt på skärmen motsvarar en bildpunkt i SR-resultatet. Originalet (10 m) renderas som blockigt — det är inte ett fel, det är hur grov satellit-upplösningen ser ut före AI-skarpning.</p>' }
            ]
        },
        // Panel ids prefixed `sr-` so app.js's prefix lookup (`sr-rgb`) resolves
        // every model panel's background to the LR tile. The opacity slider
        // (built into every panel header) then fades the SR overlay in/out
        // against the LR baseline — exactly the comparison gesture this
        // showcase exists for. The LR panel itself uses key='rgb' so the
        // initMaps special-case loads its own image as the bg.
        panels: [
            {id:'sr-rgb',          key:'rgb',         title:'Sentinel-2 RGB (10 m, original)',           legend:null},
            {id:'sr-bicubic',      key:'bicubic',     title:'Bicubic (4× interpolation)',                legend:null},
            {id:'sr-sen2sr',       key:'sen2sr',      title:'SEN2SR — CNN, radiometriskt skyddad',       legend:null},
            {id:'sr-ldsr',         key:'ldsr',        title:'LDSR-S2 — latent diffusion',                legend:null},
            {id:'sr-sam-bicubic',  key:'sam_bicubic', title:'Spektral skillnad — Bicubic',               legend:'sam'},
            {id:'sr-sam-sen2sr',   key:'sam_sen2sr',  title:'Spektral skillnad — SEN2SR',                legend:'sam'},
            {id:'sr-sam-ldsr',     key:'sam_ldsr',    title:'Spektral skillnad — LDSR-S2',               legend:'sam'}
        ],
        images: {
            'sr-rgb':         'showcase/sr/rgb_lr.png',
            'sr-bicubic':     'showcase/sr/bicubic.png',
            'sr-sen2sr':      'showcase/sr/sen2sr.png',
            'sr-ldsr':        'showcase/sr/ldsr.png',
            'sr-sam-bicubic': 'showcase/sr/sam_bicubic.png',
            'sr-sam-sen2sr':  'showcase/sr/sam_sen2sr.png',
            'sr-sam-ldsr':    'showcase/sr/sam_ldsr.png'
        },
        // Bounds set to SR-native pixel dimensions (3744 × 2616) so that at
        // zoom 0, one bound-unit equals one screen pixel. With nativeZoom
        // the maps skip fitBounds and pin to zoom 0 — the user sees actual
        // pixel-level differences between methods rather than a browser-
        // bicubic-downsampled blur where every method looks identical.
        // LR (936 × 654 native) is stretched to match these bounds and
        // rendered with image-rendering: pixelated so each LR pixel shows
        // as a sharp 4×4 block, preserving its low-resolution character.
        imgH: 3744, imgW: 2616,
        nativeZoom: true,
        hasBgToggle: false
    },

    aircraft_parallax: {
        title: 'Push-broom band-parallax — Öckerö-skärgården 2026-01-11',
        summary: [
            {title:'Plats (kondensstrimma)', value:'57.71809°N, 11.66456°E', detail:'Sundet mellan Öckerö och Björkö · ~6 km SW om Säve · L1C tile T32VPK'},
            {title:'Verifierad passage',     value:'S2B 10:43:19 UTC',       detail:'2026-01-11 · detector 10 · Δt B04→B02 1.004 s'},
            {title:'Mätt heading',           value:'≈ 051° (NÖ)',            detail:'Wedge-spets i NÖ · matchar Säve→ESOW initial bäring 51.7° inom 1°'},
            {title:'Kondensstrimma-fysik',         value:'≥ 5–8 km flyghöjd',      detail:'Schmidt-Appleman utesluter låghöjds-trafik'},
            {title:'Trolig identitet',       value:'SWE32A · Pilatus PC-24', detail:'SE-RVE · Säve→Västerås (ESOW) · climb i Kattegatt-outbound-departure'}
        ],
        intro: 'Sentinel-2 MSI är en <strong>push-broom</strong>-sensor: bandens detektorrader sitter fysiskt åtskilda på fokalplanet, så samma markpunkt registreras vid olika tidpunkter när satelliten rör sig framåt med ~6.7 km/s. Vid 57.71809°N, 11.66456°E (sundet mellan Öckerö och Björkö i norra Göteborgs skärgård, ~6 km SW om Säve flygfält) ser vi en <strong>tunn diagonal linje</strong> i alla fyra band (B02/B03/B04/B08) — bredd ~1 pixel (10 m), längd 30–50 pixlar. Det är en <strong>kondensstrimma</strong> från ett jetplan; wedge-spetsen (skarp/ljus) ligger i NÖ, alltså flyger planet <strong>nordost</strong>. Inter-band-tiderna parsade från <code>MTD_DS.xml</code> detector 10 ger Δt(B04→B03)=0.484&nbsp;s, Δt(B04→B08)=0.744&nbsp;s, Δt(B04→B02)=1.004&nbsp;s — fokalplansordningen är B04→B03→B08→B02. Heading-mätning: <strong>051°</strong> (±5°). <strong>Kondensstrimma-fysiken</strong> (Schmidt-Appleman) utesluter låghöjds-trafik: kondensstrimmor bildas typiskt vid ≥ 5–8 km höjd där T ≤ −40°C. OpenSky-spårning identifierade <strong>SWE32A (Pilatus PC-24, SE-RVE)</strong>, en svensk affärsjet i flygning Säve→Västerås (ESOW) under exakt det tidsfönstret. Säve→ESOW initial bäring är <strong>51.7°</strong> — matchar uppmätt heading inom 1°. Flygbanan förklaras av en typisk Kattegatt-outbound-SID från Säves bana 03 (NÖ-bound), klättring SW ut över vatten för bullerdämpning, sedan turn back NÖ; AOI-passagen 10:43:19 är i NÖ-leg av detta mönster, ~5–8 km höjd, kondensstrimma-kapabel. Definitiv höjd kräver OpenSky Trino-tracks-access (rate-limited i free-tier).',
        panels: [
            {id:'ap-zoom-rgb', key:'zoom_rgb', title:'RGB-närbild vid kondensstrimman (8× pixel-zoom, hard 1-99% stretch)', legend:null},
            {id:'ap-b04',      key:'b04',      title:'B04 · 665 nm (röd) · t₀ (exponeras först)', legend:null},
            {id:'ap-b03',      key:'b03',      title:'B03 · 560 nm (grön) · t₀ + 0.484 s', legend:null},
            {id:'ap-b08',      key:'b08',      title:'B08 · 842 nm (NIR) · t₀ + 0.744 s', legend:null},
            {id:'ap-b02',      key:'b02',      title:'B02 · 490 nm (blå) · t₀ + 1.004 s (exponeras sist)', legend:null},
            {id:'ap-rgb',      key:'rgb',      title:'Sentinel-2 RGB — full 5×5 km AOI (kontext)', legend:null}
        ],
        images: {
            'ap-zoom-rgb': 'showcase/aircraft_parallax/zoom_rgb.png',
            'ap-b02':      'showcase/aircraft_parallax/picker_B02.png',
            'ap-b03':      'showcase/aircraft_parallax/picker_B03.png',
            'ap-b04':      'showcase/aircraft_parallax/picker_B04.png',
            'ap-b08':      'showcase/aircraft_parallax/picker_B08.png',
            'ap-rgb':      'showcase/aircraft_parallax/rgb.png'
        },
        imgH: 960, imgW: 960,
        hasBgToggle: false
    },

    wetland_pirinen: {
        title: 'Pirinen 2023 — input-stack för svensk våtmarkssegmentering',
        summary: [
            {title:'Modell',value:'FCN 5-klass',detail:'Pirinen 2023 / RISE'},
            {title:'Beställare',value:'Naturvårdsverket',detail:'Habitatdirektiv 7110–7310'},
            {title:'AOI',value:'Stormyran',detail:'10×10 km @ 10 m · Jämtland aapamyr'},
            {title:'Lager',value:'7 av 10',detail:'#1+#2 utelämnade (<1% mIoU)'}
        ],
        intro: 'Aleksis Pirinen (RISE) genomförde 2022 en förstudie för Naturvårdsverket om semantisk segmentering av fem svenska våtmarkstyper enligt EU:s habitatdirektiv (högmosse, rikkärr, öppna mosse, aapamyr, källor). Modellen är en fully convolutional network (FCN) som tar 10 input-lager på 10 m upplösning. Pirinen släppte aldrig pre-tränade vikter — den här demon visar att ImintEngine kan reproducera hans fullständiga input-stack från svenska öppna geodatakällor över en aapamyr-AOI (Stormyran, Jämtland). Lager #1 (base vegetation) och #2 (VMI) är utelämnade — Tabell 1 i rapporten visar att de bidrar <1 % mIoU.',
        panels: [
            {id:'wp-rgb',          key:'rgb',          title:'Sentinel-2 RGB',                       legend:null},
            {id:'wp-smi',          key:'smi',          title:'#3 NMD Markfuktighetsindex',           legend:'moisture',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'DEM',key:'dem'}]},
            {id:'wp-slu',          key:'slu_markfukt', title:'#4 SLU Markfuktighetskarta (Lidberg)', legend:'moisture',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'DEM',key:'dem'}]},
            {id:'wp-bush-h',       key:'bush_height',  title:'#6 Objekthöjd 0.5–5 m (busk-höjd)',    legend:'height_m',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'DEM',key:'dem'}]},
            {id:'wp-bush-c',       key:'bush_cover',   title:'#7 Objekttäckning 0.5–5 m (busk-täckning)', legend:'cover_pct',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'DEM',key:'dem'}]},
            {id:'wp-tree-h',       key:'tree_height',  title:'#8 Trädhöjd (laser, Skogsstyrelsen)',  legend:'height_m',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'DEM',key:'dem'}]},
            {id:'wp-tree-c',       key:'tree_cover',   title:'#9 Objekttäckning 5–45 m (träd-täckning)',  legend:'cover_pct',
                bgToggle:[{label:'RGB',key:'rgb',active:true},{label:'DEM',key:'dem'}]},
            {id:'wp-dem',          key:'dem',          title:'#10 Höjdmodell (Copernicus DEM GLO-30)',    legend:'elevation'}
        ],
        images: {
            'wp-rgb':         'showcase/wetland_pirinen/rgb.png',
            'wp-smi':         'showcase/wetland_pirinen/smi.png',
            'wp-slu':         'showcase/wetland_pirinen/slu_markfukt.png',
            'wp-bush-h':      'showcase/wetland_pirinen/bush_height.png',
            'wp-bush-c':      'showcase/wetland_pirinen/bush_cover.png',
            'wp-tree-h':      'showcase/wetland_pirinen/tree_height.png',
            'wp-tree-c':      'showcase/wetland_pirinen/tree_cover.png',
            'wp-dem':         'showcase/wetland_pirinen/dem.png',
            'wp-baseline':    'showcase/wetland_pirinen/dem.png'  // 'DEM' bg-toggle reuses #10
        },
        imgH: 1000, imgW: 1000,
        hasBgToggle: true
    },

    water_quality_lilla_karlso: {
        title: 'Vattenkvalitet — Lilla Karlsö · sillgrissle-säsong 2025',
        summary: [
            {title:'AOI',                  value:'Lilla Karlsö + havshorisont väst', detail:'10 × 22 km, helt inom UTM 33N (T33VXD)'},
            {title:'Sensor &amp; passager',  value:'Sentinel-2A/2C',                   detail:'3 single-pass-scener över häckningssäsongen'},
            {title:'Metoder',              value:'4 retrievals × 3 datum',           detail:'RGB + C2RCC (C2X-Nets) chl/TSM/CDOM via SNAP 13'},
            {title:'Färgskala chl-a',      value:'log₁₀ 0.5–25 mg/m³',               detail:'pinnad → cross-scen-jämförbar'}
        ],
        intro: 'Sentinel-2 vattenkvalitetsanalys för Lilla Karlsö (Gotland) över sillgrissle-säsongen 2025 — samma C2RCC-pipeline som Stigfjorden/Mollösund-tabben, men över en tidsserie istället för en enda passage. Sillgrissle-kolonin (~2 500 par) jagar pelagisk skarpsill och sill 5–15 km västerut, och fodersäkerheten följer klorofyll-dynamiken i Östersjön. <strong>Tre molnfria scener</strong> valda av <code>optimal_fetch_dates(era5_then_scl)</code> över hela 2025-04-15..07-31 träffar perfekt en gång varje fenologisk fas: 2025-04-29 (äggläggning + vårblom-pre), 2025-06-13 (kläckning + sommarblom-peak), 2025-07-10 (ungar lämnar + post-peak). Per scen körs <strong>fyra retrievals utan fusion</strong>: Sentinel-2 RGB samt ESA C2RCC C2X-Nets för chl-a, TSM och CDOM via SNAP 13 (signerad GHCR-image, native amd64 på ICE k8s). Brockmann 2016-formler från IOPs. Färgskalan på klorofyll är <strong>fast pinnad till log₁₀ 0.5–25 mg/m³</strong> så samma färg betyder samma värde över alla tre scener. UTM-zon-prefer ligger i fetch-laget — AOI korsar 18°E (UTM 33/34-gränsen) men eftersom centrum ligger i UTM 33 plockas T33VXD-tile för alla tre datum, ingen cross-zone-reprojektion.',
        // 12 panels: 4 retrievals (RGB + chl + TSM + CDOM) × 3 datum, samma flat-grid-
        // mönster som Bohuslän-tabben. Panel-id format `lk<MMDD>-<prod>` — app.js
        // gör prefix = panel.id.split('-')[0], så `lk0429`/`lk0613`/`lk0710`
        // ger varje datum sin egen RGB-bakgrund via `<prefix>-rgb`-lookup.
        panels: [
            {id:'lk0429-rgb',  key:'rgb',  title:'2025-04-29 RGB · äggläggning',     legend:null},
            {id:'lk0429-chl',  key:'chl',  title:'2025-04-29 Chl-a (mg/m³)',         legend:'chl'},
            {id:'lk0429-tsm',  key:'tsm',  title:'2025-04-29 TSM (g/m³)',            legend:'tss'},
            {id:'lk0429-cdom', key:'cdom', title:'2025-04-29 CDOM (m⁻¹)',           legend:'cdom'},
            {id:'lk0613-rgb',  key:'rgb',  title:'2025-06-13 RGB · kläckning',       legend:null},
            {id:'lk0613-chl',  key:'chl',  title:'2025-06-13 Chl-a (mg/m³)',         legend:'chl'},
            {id:'lk0613-tsm',  key:'tsm',  title:'2025-06-13 TSM (g/m³)',            legend:'tss'},
            {id:'lk0613-cdom', key:'cdom', title:'2025-06-13 CDOM (m⁻¹)',           legend:'cdom'},
            {id:'lk0710-rgb',  key:'rgb',  title:'2025-07-10 RGB · ungar lämnar',    legend:null},
            {id:'lk0710-chl',  key:'chl',  title:'2025-07-10 Chl-a (mg/m³)',         legend:'chl'},
            {id:'lk0710-tsm',  key:'tsm',  title:'2025-07-10 TSM (g/m³)',            legend:'tss'},
            {id:'lk0710-cdom', key:'cdom', title:'2025-07-10 CDOM (m⁻¹)',           legend:'cdom'}
        ],
        images: {
            'lk0429-rgb':   'showcase/lilla_karlso_birds/frames/2025-04-29/rgb.png',
            'lk0429-chl':   'showcase/lilla_karlso_birds/frames/2025-04-29/chl.png',
            'lk0429-tsm':   'showcase/lilla_karlso_birds/frames/2025-04-29/tsm.png',
            'lk0429-cdom':  'showcase/lilla_karlso_birds/frames/2025-04-29/cdom.png',
            'lk0613-rgb':   'showcase/lilla_karlso_birds/frames/2025-06-13/rgb.png',
            'lk0613-chl':   'showcase/lilla_karlso_birds/frames/2025-06-13/chl.png',
            'lk0613-tsm':   'showcase/lilla_karlso_birds/frames/2025-06-13/tsm.png',
            'lk0613-cdom':  'showcase/lilla_karlso_birds/frames/2025-06-13/cdom.png',
            'lk0710-rgb':   'showcase/lilla_karlso_birds/frames/2025-07-10/rgb.png',
            'lk0710-chl':   'showcase/lilla_karlso_birds/frames/2025-07-10/chl.png',
            'lk0710-tsm':   'showcase/lilla_karlso_birds/frames/2025-07-10/tsm.png',
            'lk0710-cdom':  'showcase/lilla_karlso_birds/frames/2025-07-10/cdom.png'
        },
        // Frame-proportion följer faktisk PNG-output från render.py (1903 × 2304 px,
        // 0.826:1 ≈ portrait). AOI 10×22 km i WGS84 projicerar till ungefär samma
        // form i UTM 33N. Bohuslän har 2.442:1 landscape; Lilla Karlsö är portrait
        // eftersom kolonin + foderhabitatet sträcker sig nord-syd.
        imgH: 880, imgW: 730,
        hasBgToggle: false,
        hasCharts: true,
        chartSectionTitle: ''  // Chart hanteras inline i HTML, inte via mall
    }
};
