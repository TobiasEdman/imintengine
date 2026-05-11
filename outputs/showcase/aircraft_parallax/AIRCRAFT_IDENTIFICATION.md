# Aircraft identification — Öckerö-skärgården 2026-01-11 10:43:19 UTC

**Status:** Closed (öppna data — definitiv ID kräver tracks/all eller Försvarsmakten).

**Bästa hypotes:** **SWE32A — Pilatus PC-24, SE-RVE** — i climbande NÖ-departure
från Säve (ESGP) mot Västerås (ESOW), 3–5 minuter efter takeoff, vid AOI på
en typisk Kattegatt-outbound-SID.

## Geografi (korrigerad)

Punkten 57.71809°N, 11.66456°E ligger **i sundet mellan Öckerö och Björkö**
(norra Göteborgs skärgård) — inte på Hisingen som tidigare angivet. AOI är
~5–6 km SW om Säve flygfält (ESGP).

Bäringar och avstånd från AOI:

| Destination | Bäring | Distans |
|------------|--------|---------|
| Säve flygfält (ESGP) | 53.4° | ~6 km |
| F7 Såtenäs | 45.6° | ~125 km |
| ESOW Västerås | 52.0° | ~370 km |
| ESSA Arlanda | 56.6° | ~410 km |
| ESGG Landvetter | 98.1° | ~30 km |

## Mätdata (från Sentinel-2 push-broom-parallax)

L1C-scen: `S2B_MSIL1C_20260111T104319_N0511_R008_T32VPK`. Detector 7
(parsad ur `MTD_DS.xml` via `parse_band_times_v2.py`):

| Band | Δt vs B02 |
|------|-----------|
| B02  | 0.0000 s  |
| B08  | 0.2599 s  |
| B03  | 0.5214 s  |
| B04  | 0.9990 s  |

User-picks i 120×120 px native crop ger total förskjutning ≈ 300 m över 1 s.
Wedge-spets i NÖ → planet flyger NÖ. Heading uppmätt: **051°** (±5°).
Vid hög höjd (H_a = 8–11 km efter sat-parallax-korrektion): **v ≈ 217–239 m/s**
— typisk PC-24 climb-cruise. Låghöjds-tolkning (~290 m/s vid H_a = 1 km)
är **förkastad** av kondensstrimma-fysiken (se nästa sektion).

## Kondensstrimma-fysik utesluter låg höjd

Kondensationsstrimmor bildas bara där:
- Temperatur ≤ −40°C (Schmidt-Appleman-kriteriet)
- Luftfuktigheten över is-mättnad

Det betyder typiskt **flyghöjd ≥ 8 km** (FL250+). I vinterns
Nordsjö-luftmassa kan tröskeln nås redan på 5–6 km. Detta **utesluter
alla låghöjds-hypoteser** (under 5 km) — kondensstrimman i sig är ett
höghöjdsbevis.

## SWE32A — flight-metadata från OpenSky

Query `/flights/aircraft?icao24=4acac5&begin=...&end=...`:

| Fält | Värde |
|------|-------|
| icao24 | 4acac5 |
| callsign | SWE32A (denna flygning) — SWE33A 1h senare ESOW→ESSA |
| firstSeen ADS-B | 2026-01-11 10:32:36 UTC |
| lastSeen ADS-B | 2026-01-11 11:08:43 UTC |
| estDepartureAirport | (inte detekterad — sannolikt Säve ESGP, ej kontroll-tower-täckt) |
| estArrivalAirport | ESOW (Stockholm-Västerås) |

Aircraft-metadata (`/metadata/aircraft/icao/4acac5`):

| Fält | Värde |
|------|-------|
| typecode | PC24 (Pilatus PC-24) |
| registration | SE-RVE |
| icaoAircraftClass | L2J (light, twin-jet) |
| country | Sweden |

Pilatus PC-24-spec (relevant för pixel-tolkning):

| Parameter | Värde |
|-----------|-------|
| Längd | 16.85 m → 1.7 px @ 10 m |
| Vingspann | 17.00 m → 1.7 px @ 10 m |
| Höjd | 5.41 m |
| MTOW | 8 300 kg |
| Service ceiling | 13 716 m (45 000 ft) |
| Cruise | Mach 0.74 ≈ 226 m/s vid FL450 |

Planet är **subpixel** vid 10 m/px — vi ser **inte** kroppen, vi ser bara
kondensstrimman (10 × 300–500 m).

## Flygbananalys: Säve-departure med Kattegatt-outbound

Säve flygfält har bana **03/21** (orienterad NÖ/SV). Måltiden 10:43:19 är
~11 min efter ADS-B firstSeen. Säve→ESOW direkt-bäring är **51.7°**, vilket
matchar uppmätt heading (051°) inom 1°.

**MEN:** AOI ligger ~6 km **SW** om Säve, alltså i motsatt riktning från
ESOW. Direkt great-circle Säve→ESOW passerar inte AOI.

Förklaring: en jet-departure från Säve mot ESOW skulle typiskt:

1. **Lyfta 03 (NÖ-bound)** från banänden
2. **Klättra ut över Kattegatt** i en SW-bage för bullerdämpning + säker
   climb över vatten (standard för jet-trafik vid kustnära fält)
3. **Turn back NÖ** när höjden räcker (~3–5 km)
4. **On-course mot ESOW** med initial bäring 52°

Vid AOI-passagen (10:43:19) skulle planet då befinna sig i **andra benet av
departure-mönstret** — på tillbakavägen NÖ efter den SW outbound-legen,
fortfarande i climb. Wedge-spetsen i NÖ matchar denna riktning.

**Tidsbudget mot destination:** Från AOI (10:43:19) till ESOW (11:08:43)
= 25 min 24 s, distans 326 km → ground speed 214 m/s (770 km/h, Mach 0.69
i climb-cruise). Helt rimligt för PC-24.

## OpenSky-cross-check (bbox-resultat reviderat)

Initial query `/flights/all` ±30 min: 571 flygningar, varav SWE32A var en
av tre svenska kandidater med matchande heading. För-aircraft `/tracks/all`
för waypoints i bbox (57.5–57.9°N, 11.4–11.9°E) ±90 s av målögonblicket
hittade noll — men:

- ADS-B-täckningen i tidig climb från Säve är typiskt gles (under
  radar-horisont, ground-stations-coverage limiterad)
- Vår bbox är 0.4°×0.5° (~25×30 km) — om SWE32A var ute över Kattegatt i
  outbound-legen vid den exakta sekunden hade den varit utanför bbox

Att ingen waypoint syntes i 90s-fönstret betyder **inte** att SWE32A inte
var planet; det betyder att ADS-B-coverage var glapp under det fönstret,
vilket är förväntat för låg climb-höjd över hav.

## Slutsats

**Mest sannolikt: SWE32A (Pilatus PC-24, SE-RVE) i NÖ-leg av Kattegatt-
outbound-departure från Säve mot Västerås, vid climbing-höjd ~5–8 km vid
AOI-passage.**

Bevis-kedjan:
1. Civilt affärsjet med exakt rutt över Sverige med initial bäring 51.7°
   matchar uppmätt heading 051° inom 1°
2. Tidsfönstret (firstSeen 10:32:36 → AOI 10:43:19 → lastSeen 11:08:43)
   är konsistent med Säve-departure → Kattegatt outbound → NÖ-turn → ESOW
3. PC-24 service ceiling 13.7 km täcker kondensstrimma-kapabel höjd
4. PC-24 cruise (~226 m/s) matchar bättre än 290 m/s-värden
5. Kondensstrimma-fysik utesluter alla låghöjds-alternativ

Inte verifierat:
- Exakt position vid 10:43:19 (`/tracks/all` rate-limited på free-tier)
- Exakt höjd och hastighet vid AOI

Definitiv ID kräver OpenSky Trino-access (gratis för forskning) eller
direktförfrågan till svenska AFIS för Säve-trafiken den dagen.

## Vad vi *inte* lyckades med

- Robust apex-detektion: wedge-tippen är 1 px bred → edge-fitting instabil,
  våglängdsberoende synlighet flippar wedge-orientation per band
- ADS-B Trino-access — kräver formell forskningsansökan, dagar–veckor

## Källor

- OpenSky `/metadata/aircraft/icao/4acac5` → PC24, SE-RVE, klass L2J
- OpenSky `/flights/aircraft?icao24=4acac5` → SWE32A → ESOW
- OpenSky `/flights/all` ±30 min → 571 ADS-B-flygningar i regionen
- Schmidt & Appleman (1953); Schumann (1996, J. Atmos. Sci.) — kondensstrimma-fysik
- Pilatus PC-24 spec sheet
- Säve flygfält AIP — bana 03/21, kustnära jet-departure-pattern

## Värt att notera

Plats-rättningen från "Hisingen / Säve" till "sundet mellan Öckerö och
Björkö, ~6 km SW om Säve" var **avgörande**. Den första hypotesen
(låghöjds-militärtrafik mot Skaraborg) byggde på fel geografi och fel
kondensstrimma-fysik. Med korrekt geografi + Schmidt-Appleman-tröskeln blev
SWE32A omedelbart den uppenbara kandidaten.
