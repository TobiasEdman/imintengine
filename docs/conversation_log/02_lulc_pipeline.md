# LULC Training Pipeline

> Monitoring run_lulc_pipeline.py progress, NMD vs spectral fetch architecture, producer/consumer pattern, adaptive workers, DES API throttling, and tile caching.

---

can you check whats happening with the training process

Jag ser den! Det kör en LULC-pipeline i bakgrunden:
scripts/run_lulc_pipeline.py --data-dir data/lulc_full --grid-spacing 10000 --years 2018 2019 --dashboard --dashboard-port 8000
Startad kl 18:32, ~8 min CPU-tid. Den har en dashboard på port 8000. Låt mig kolla status:

Dashboard svarar! Låt mig hämta progress och loggen:

Här är statusen på LULC-pipelinen:
run_lulc_pipeline.py — Hämtar Sentinel-2-tiles för hela Sverige (2018 + 2019)
Progress	2050 / 4381 tiles (46.8%)
Hastighet	~210–280 tiles/h med 2–3 DES-workers
Workers	Adaptiv: pendlar mellan 2–3 (skalas ner vid DES-fel)
Molnfilter	Skippar tiles med >5% moln, provar flera datum
NMD-filter	2071 behållna, 29 rena vatten-tiles hoppade över
Uppskattad tid kvar	~8–10 timmar vid nuvarande takt
Pipelinen rullar på stabilt. Den har en dashboard på http://localhost:8000/training_dashboard.html om du vill se den visuellt.
Det ser ut som om nmd fetch och spectral data fetch ligger på samma påverkar det hur snabbt vi hämtar?
Bra fråga. Låt mig kolla hur pipelinen hämtar data.

Pipelinen anropar prepare_training_data(config). Låt mig kolla den funktionen:

Nu ser jag arkitekturen. Låt mig kolla hur spectral-fetch-workern hämtar data - specifikt om den gör ytterligare ett NMD-anrop:

Där har vi det. NMD hämtas två gånger per tile:
Rad 408 — _nmd_producer(): NMD hämtas för att pre-filtrera (skippa rena vatten-tiles)
Rad 730 — _fetch_worker(): NMD hämtas igen med target_shape för att matcha spectral-bandets upplösning
Låt mig kolla om det finns caching:

Bra — det finns NMD-cache (.nmd_cache/-mapp med .npy-filer). Men cachenyckeln baseras på koordinater, och det andra anropet (rad 730) skickar target_shape som inte ingår i nyckeln. Låt mig verifiera:

Svaret är: Det andra NMD-anropet (rad 730) träffar cache och gör INGET nätverksanrop. Det läser bara .npy från disk och resamplar eventuellt till target_shape. Så det är gratis.
Men det finns en annan fråga — NMD-producern och spectral-workern gör olika DES-anrop som konkurrerar om samma API:

Här är sammanfattningen:
Ja, NMD-fetch och spectral-fetch delar DES-anslutningen och konkurrerar om bandbredd, men det påverkar inte hastigheten negativt. Här är varför:
NMD-producer	Spectral-workers
Tråd	1 bakgrundstråd	2–3 adaptiva trådar
DES-anrop	fetch_nmd_data() — 1 band, litet	fetch_des_data() — 11 band, stort
Cache	Ja (.nmd_cache/) — efter första hämtningen cachas lokalt	Nej — alltid nytt
Timing	Producerar snabbt (cache-hits), ligger långt före	Begränsas av adaptiv semaphore

Designen är smart — NMD-producern ligger i en separat tråd som filtrerar bort vatten-tiles innan de dyra spectral-fetchen körs. De två köerna är:
NMD-producer ──filter──▶ approved_q ──▶ spectral-workers (2-3st)
                           ▲
                     skippar vatten
Den enda potentiella flaskhalsen är att NMD-producern kan blockera DES-anslutningen vid en uncached NMD-fetch, men den har time.sleep(1) mellan uncached anrop och kön fungerar som buffert. Vid det här laget (tile 2050/4381) borde i princip alla NMD-tiles vara cachade redan.
Kort svar: NMD-fetchen bromsar inte spectral-fetchen — den ligger ahead och levererar från cache.
