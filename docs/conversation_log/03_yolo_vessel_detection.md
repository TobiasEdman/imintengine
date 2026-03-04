# YOLO Vessel Detection

> Fine-tuning discussion, L2A vs TCI data comparison, on-board processing simulation, TCI scaling bug fix, vessel annotation workflow, and SAHI sliding window inference.

---

Check fixed vessel image dimensions
Dimensionerna stämmer. Nu kopiera och bygga om rapporten:

Rapporten borde vara öppen nu med:
Fixad TCI-bugg: marine_vessels.py bygger nu TCI från rå reflektansband (bands["B04"] etc.) istället för percentile-stretchad RGB. Det ger 28 detektioner istället för 0.
Fixad Leaflet CSS: Bilderna borde nu ligga korrekt ovanpå varandra med opacity-slider.
Kan du kolla om kartpanelerna renderas korrekt nu?
snyggt och stryk NDBI


If we want to fine tune the model what would be the way forward?

Bra fråga. Här är läget och vägen framåt:
Nuläge
Modellen yolo11s_tci.pt är förtränad av mayrajeo (HuggingFace) på Sentinel-2 L1C-TCI. Vi använder den som-den-är via SAHI sliding-window. Det finns ingen fine-tuning-infrastruktur ännu — bara:
✅ Inferens-pipeline (marine_vessels.py)
✅ Annotationsverktyg (vessel_annotator.html)
✅ Diagnostikscript (analyze_vessel_annotations.py)
❌ Inget träningsscript för YOLO
Väg framåt
Ultralytics gör fine-tuning enkelt. Det behövs tre saker:
1. Annotationsdata i YOLO-format
Du har redan vessel_annotations.json med manuella markeringar. De behöver konverteras till YOLO-textformat:
# labels/image_001.txt  (en rad per objekt)
# class  x_center  y_center  width  height  (normaliserat 0-1)
0  0.512  0.345  0.025  0.018
2. Mer data
28 detektioner i en scen räcker inte. Du behöver:
50-200+ annoterade scener från olika kustsegment, årstider, väder
Ditt annotationsverktyg (vessel_annotator.html) är redan klart för detta
Pipeline: hämta TCI-chips → annotera → exportera YOLO-format
3. Träningsscript
Ultralytics one-liner:
from ultralytics import YOLO
model = YOLO("yolo11s_tci.pt")  # start from pretrained
model.train(
    data="vessel_dataset.yaml",
    epochs=50,
    imgsz=320,       # matchar SAHI chip_size
    batch=16,
    freeze=10,       # frys backbone, träna bara head
    lr0=1e-4,        # låg lr för fine-tuning
)
Rekommenderad plan
Steg	Vad	Uppskattning
1	Bygg datapipeline: hämta 100+ kust-scener via DES, TCI-skala, skär i 320×320 chips	Script + DES-tid
2	Annotera med vessel_annotator.html, exportera till YOLO-format	Manuellt arbete
3	train_vessels.py — fine-tune med frozen backbone	~30 min GPU / timmar CPU
4	Evaluate på held-out scener, jämför med baseline	
Vill du att jag bygger datapipelinen och träningsscriptet?
Om man vill köra direkt på S2 L2A data?
Bra tanke — det skulle förenkla hela pipelinen. Låt mig kolla vad diagnostiken visade om L2A vs TCI:

Siffrorna berättar historien tydligt:
Varför den nuvarande modellen missar på L2A
TCI (L1C)	L2A percentile-stretch
Vatten	9–28 (mörkt)	14–65 (ljusare)
Fartyg	13–58	26–151
YOLO score	mean 0.12	mean 0.02
Detektioner ≥0.286	3/23	0/23
Modellen tränades på mörka TCI-bilder. L2A-percentile-stretch ger helt annan pixelfördelning — allt är ljusare och kontrasten ändras. Modellen känner inte igen det.
Två vägar att köra direkt på L2A
Alt A: Fine-tune på L2A refl × 255 (utan percentile-stretch)
Rå L2A reflektans (0–0.15 för vatten) → × 255 → 0–38 pixelvärden
Det ger liknande mörka bilder som TCI. Enklast — minsta steget, mest likt det modellen redan kan. Liten fine-tune borde räcka.
Alt B: Fine-tune på L2A med percentile-stretch
Percentile-stretchad RGB (0–1) → × 255 → 0–255 pixelvärden
Helt annan fördelning. Kräver mer data och träning, men fördelen är att det är exakt samma RGB som resten av pipelinen använder — ingen separat TCI-konvertering alls.
Rekommendation
Alt A (refl × 255) är lägst tröskel — pixelvärdena hamnar i samma intervall som TCI, skillnaden är bara atmosfärskorrektionen. 10–20 annoterade scener borde räcka för fine-tuning. Och du slipper TCI-konverteringen i marine_vessels.py.
Vill du att jag bygger det?
En annan idé om man vill simulera on board proceesing, vad krävs på att köra direkt på simulerad sensordata
Intressant vinkel — on-board processing på satellitnivå. Låt mig kolla vad vi har att utgå ifrån:

Här är sammanfattningen:
On-board simulering med vår pipeline
Datakedjan
On-board har du bara rå DN (Digital Numbers, 12-bit). Inget DEM, ingen atmosfärskorrigering, inget NMD. Det enda du kan göra är en enkel linjär skalning:
Sensor → DN (0–4095) → gain+clip → uint8 (0–255) → YOLO
Vad vi kan simulera med DES-data
Från L2A-reflektans kan vi gå "bakåt":
Simulering	Formel	Liknar
TOA-liknande	refl × 10000 / 16	Rå DN skalat till uint8
Enkel gain	refl × K (K≈640)	On-board auto-gain
TCI	refl × 2.5 × 255	ESA L1C-TCI (det modellen redan kan)
Över vatten är skillnaden L1C↔L2A minimal (atmosfären påverkar lite), så L2A-reflektans × fast gain är en bra simulering av vad sensorn ser.
Compute-budget
YOLO11s (320px)	YOLO11n (ännu mindre)
Modellstorlek	18 MB, 9.4M params	~5 MB, 2.6M params
Per chip	5.4 GFLOP	1.5 GFLOP
Jetson Orin NX (40W)	~50 chips/s	~180 chips/s
290km swath	~4 min	~1 min
Vad som behövs för att simulera
Skalningsformel — ersätt TCI med enkel DN × gain → uint8
Fine-tune på den skalningen (inte TCI, inte percentile-stretch)
Exportera till ONNX/TensorRT för edge-deploy
Stripa pipelinen — ingen DES, inget NMD, inget SCL-filter (ingen SCL on-board)
Landmask — ersätt SCL water-filter med en förberäknad binär mask (laddas i ROM)
Det intressanta är att YOLO11s redan är litet nog för Jetson. Vill du att jag bygger en simulerad on-board-pipeline som tar numpy-band och kör inference utan DES-beroenden?
Innan vi går vidare så vill jag att du tittar på TOA i brandvisualiseringen, den ser inte ut attha justerats tillsammans med marin, viktigt att de använder samma funktion och visualisering
Låt mig kolla COT-bilderna för båda scenarierna:
