'use strict';

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
    vessel: [
        {color:'#00E5FF',label:'Detekterad båt / anomali'}
    ],
    ai2_vessel: [
        {color:'#50A0FF',label:'Stationär (0 kn)'},{color:'#FFE600',label:'Låg fart (< 5 kn)'},
        {color:'#FF6600',label:'Medelfart (5–15 kn)'},{color:'#FF0000',label:'Hög fart (> 15 kn)'}
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
    ]
};

// ── GeoJSON file paths (loaded async) ────────────────────────────────
var GEOJSON_FILES = {
    vessels:              'data/vessels.geojson',
    mc_vessels:           'data/mc_vessels.geojson',
    mc_ai2_vessels:       'data/mc_ai2_vessels.geojson',
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
        intro: 'Analysområdet är beläget i Ljusdals kommun, Gävleborgs län, och visar Kårbölebranden — en av de största skogsbränderna i Sveriges moderna historia sommaren 2018. Den 14 juli 2018 startade en skogsbrand som till slut bredde ut sig över cirka 9 500 hektar skog, vilket gjorde den till den största skogsbranden i Sverige på över 50 år. Här har Sentinel-2-data från 2018-07-24 analyserats med flera kompletterande metoder för att kartlägga brandens utbredning och intensitet.',
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

    marine_leisure: {
        title: 'Marin analys — Fritid (2025-07-10)',
        summary: [
            {title:'Båtdetektering',value:'130 båtar',detail:'5 datum (4 skippade)'},
            {title:'Bästa datum',value:'50 båtar',detail:'2025-07-17'},
            {title:'NMD Marktäcke',value:'64.2% Vatten',detail:'6 klasser'},
            {title:'Molnanalys (COT)',value:'99.7% klart',detail:'COT medel: 0.0019'},
            {title:'Analysområde',value:'18.6 km²',detail:'Bohuslän kustzon'}
        ],
        intro: 'Analysområdet visar skärgården utanför Hunnebostrand — ett område längs den norra bohuslänska kusten med intensiv maritim aktivitet från både kommersiell sjöfart, fiske och fritidsbåtar. Sentinel-2-data från 2025-07-10 har analyserats med flera kompletterande metoder för att kartlägga båtförekomst, vattenförhållanden och marktäcke i kust- och havsområdet.',
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

    grazing: {
        title: 'Betesmarksanalys — 2025-06-14',
        summary: [
            {title:'Betesanalys (AI)',value:'68/80 aktiv',detail:'Konfidens: 73%'},
            {title:'Ingen aktivitet',value:'8 block',detail:'19 molnfria datum'},
            {title:'LPIS Betesblock',value:'80 block',detail:'124 ha total areal'},
            {title:'NDVI i betesmark',value:'0.81',detail:'± 0.06 standardavvikelse'},
            {title:'NMD inom betesblock',value:'7% Våtmark',detail:'2 markklasser'}
        ],
        intro: 'Analysområdet är beläget nordost om Lund i Skåne — ett av Sveriges mest intensivt brukade jordbrukslandskap med en blandning av åkermark, beteshagar och småskaliga skogspartier. LPIS-polygoner från Jordbruksverkets blockdatabas visar registrerade betesblock i området. Sentinel-2-data från 2025-06-14 har analyserats med spektrala index (NDVI, NDWI, EVI), molnanalys (COT) och korsrefererats mot NMD marktäckedata för att kartera vegetationens tillstånd inom betesmarkerna.',
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

    lulc: {
        title: 'LULC-klassificering — Labelkvalitet',
        summary: [
            {title:'mIoU (val)',value:'43.3%',detail:'10-klass modell + 5 aux'},
            {title:'Hög konfidens fel',value:'—',detail:'Modell >80% säker, NMD avviker'},
            {title:'Låg konfidens rätt',value:'—',detail:'Modell <50% säker, NMD stämmer'},
            {title:'Analyserade tiles',value:'—',detail:'Val-split'},
            {title:'Modell',value:'Prithvi-EO 2.0',detail:'UPerNet + AuxEncoder'}
        ],
        intro: 'Prithvi-EO 2.0 foundation model med UPerNet-dekoder och 5 auxiliära kanaler (medelhöjd, volym, grundyta, diameter, DEM) har tränats för pixelvis LULC-klassificering med NMD som grundsanning. Denna analys visar var modellen avviker från NMD — särskilt pixlar där modellen är >80% säker men NMD anger en annan klass. Dessa "high-confidence wrong"-pixlar indikerar sannolika NMD-labeleringsfel och är kandidater för labelrensning. NMD har känt lägst noggrannhet för lövskog och blandskog, vilket direkt förklarar modellens låga IoU för dessa klasser.',
        panels: [
            {id:'l-nmd',        key:'nmd_label',   title:'NMD Grundsanning',           legend:'lulc_grouped'},
            {id:'l-pred',       key:'prediction',  title:'Modellprediktion',            legend:'lulc_grouped'},
            {id:'l-disagree',   key:'disagree',    title:'Avvikelser (NMD vs modell)',  legend:'disagree'},
            {id:'l-confidence', key:'confidence',  title:'Konfidens (softmax)',         legend:'confidence'},
            {id:'l-entropy',    key:'entropy',     title:'Entropi (osäkerhet)',         legend:'entropy'}
        ],
        images: {
            'l-nmd':        'showcase/lulc/nmd_label.png',
            'l-pred':       'showcase/lulc/prediction.png',
            'l-disagree':   'showcase/lulc/disagree.png',
            'l-confidence': 'showcase/lulc/confidence.png',
            'l-entropy':    'showcase/lulc/entropy.png'
        },
        imgH: 224, imgW: 224,
        hasBgToggle: false,
        hasCharts: true,
        chartSectionTitle: 'Labelkvalitet per klass'
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
        hasBgToggle: false,
        years: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    }
};
