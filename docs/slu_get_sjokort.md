# Hämta sjökort från SLU GET (Geodata Extraction Tool)

## Översikt

Sjöfartsverkets sjökort finns tillgängligt via SLU:s nedladdningstjänst GET i
vektorformat (S-57). Datan täcker hela Sveriges kust och insjöar.

| Egenskap | Värde |
|----------|-------|
| **Dataset** | Sjöfartsverket: Sjökort latest (S-57) |
| **Job-ID** | `sjokort_vektor` |
| **Format** | S-57 (.000) — internationell IHO-standard |
| **Koordinatsystem** | EPSG:4326 (WGS84) |
| **Storleksgräns** | Orange (max ~2.5 km²) |

## Steg-för-steg

### 1. Logga in

1. Öppna <https://maps.slu.se/get/>
2. Välj ditt lärosäte (t.ex. RISE) via SeamlessAccess
3. Logga in med dina institutionsuppgifter

### 2. Markera område

1. Du hamnar på fliken **"Select area on Map"**
2. Zooma in till det kustområde du vill ladda ner
3. Klicka **Draw** (nederst till vänster)
4. Håll ner vänster musknapp och dra en rektangel över området
5. Rektangeln byter färg beroende på storlek:
   - **Röd** = < 1.6 km² (alla dataset)
   - **Orange** = < 2.5 km² (sjökort fungerar upp till denna)
   - **Grå** = < 62.5 km²
   - **Gul** = < 1200 km²

   > **Tips:** Sjökort har workload 2, så orange (2.5 km²) är max.
   > För större områden: gör flera beställningar med angränsande rektanglar.

### 3. Välj dataset

1. Klicka på fliken **"Select data"**
2. Scrolla till botten av listan → **Sjöfartsverket:Sjökort latest (S-57)**
3. Kontrollera att koordinatsystem är **EPSG:4326**
4. Klicka **"Add to order"**
5. En bekräftelse visas högst upp och räknaren på "Review and Finish" ökar

### 4. Beställ och ladda ner

1. Klicka på fliken **"Review and Finish"**
2. Kontrollera att sjökortet finns i listan
3. Klicka **"I Agree to the terms below"**
4. Verifiera din e-postadress
5. Klicka **"Go!"**
6. Vänta på e-post med nedladdningslänk (vanligtvis 1–10 min)
7. Klicka länken i e-posten → ladda ner ZIP-filen

### 5. Använda S-57-data

S-57-filer (.000) kan öppnas med:

```bash
# QGIS — öppna direkt via drag-and-drop eller Layer → Add Vector Layer

# Python (GDAL/OGR)
from osgeo import ogr
ds = ogr.Open("SE5XXXXX.000")
for i in range(ds.GetLayerCount()):
    layer = ds.GetLayerByIndex(i)
    print(f"{layer.GetName()}: {layer.GetFeatureCount()} features")

# Lista lager med ogrinfo
ogrinfo SE5XXXXX.000
```

Vanliga S-57-lager:

| Lager | Innehåll |
|-------|----------|
| DEPARE | Djupområden (depth areas) |
| SOUNDG | Lodpunkter (soundings) |
| LNDARE | Landområden |
| BOYCAR / BOYLAT | Bojar (kardinala/laterala) |
| LIGHTS | Fyrar och ljus |
| ACHARE | Ankringsområden |
| BERTHS | Kajplatser |
| HRBARE | Hamnområden |
| OBSTRN | Hinder (vrak, grund) |
| NAVLNE | Navigeringslinjer |
| FAIRWY | Farleder |

## API-detaljer (för programmatisk användning)

GET-tjänsten har ett internt REST API (kräver Shibboleth-session):

```
Base URL: https://maps.slu.se/get/
```

### Informationsendpoints

| Metod | Endpoint | Beskrivning |
|-------|----------|-------------|
| GET | `/legacy-api/list` | Lista alla dataset |
| GET | `/legacy-api/config` | Konfiguration (storleksgränser mm) |
| GET | `/legacy-api/user` | Inloggad användare + e-post |
| GET | `/legacy-api/geojson/{id}` | Coverage-polygon per dataset |

### Beställ sjökort

```
POST /api/job/sjokort_vektor/{north}/{south}/{east}/{west}/{email}
```

| Parameter | Beskrivning | Exempel |
|-----------|-------------|---------|
| `north` | Max Y (norr) i SWEREF99 TM (EPSG:3006), `Math.ceil()` | `6487990` |
| `south` | Min Y (söder) i SWEREF99 TM (EPSG:3006), `Math.ceil()` | `6482260` |
| `east` | Max X (öst) i SWEREF99 TM (EPSG:3006), `Math.ceil()` | `284320` |
| `west` | Min X (väst) i SWEREF99 TM (EPSG:3006), `Math.ceil()` | `281080` |
| `email` | E-postadress för nedladdningslänk | `user@example.se` |

**Viktigt:**
- Alla parametrar skickas i URL-sökvägen, **ingen** request body
- Koordinaterna är i **SWEREF99 TM (EPSG:3006)**, inte WGS84
- Värden avrundas uppåt med `Math.ceil()`
- Levererad data är i EPSG:4326 (WGS84) — oavsett beställningskoordinater
- Max area per beställning: ~2.5 km² (workload 2 = orange)

**Exempelanrop (curl):**

```bash
curl -X POST \
  -b shibboleth_cookies.txt \
  "https://maps.slu.se/get/api/job/sjokort_vektor/6487990/6482260/284320/281080/user@example.se"
```

**Lyckat svar (JSON):**

```json
{
  "get_unique_name": "sjokort_vektor",
  "MaxX": 282700.0,
  "MinX": 281080.0,
  "MaxY": 6483690.0,
  "MinY": 6482260.0,
  "Email": "user@example.se",
  "Uuid": "b4aeb394-902c-480f-814f-90d4f1173acf",
  "Area": 2316600,
  "Eppn": "username@org.se",
  "Requested": "Thu 2026-02-26 22:26 39",
  "RequestedArea": 2316600,
  "RequestedAreaRatio": 0.0000370656
}
```

| Fält | Beskrivning |
|------|-------------|
| `Uuid` | Unikt jobb-ID — **viktigast att verifiera** |
| `Area` | Beställd area i m² |
| `MaxX/MinX/MaxY/MinY` | Bekräftade SWEREF99 TM-koordinater |
| `Email` | Mottagaradress för nedladdningslänk |
| `RequestedAreaRatio` | Andel av maximal tillåten area |

Om `Uuid` saknas i svaret har beställningen misslyckats.

**Nedladdningslänk:** Skickas till angiven e-post inom 1–10 minuter. ZIP-filen
innehåller S-57-filer (.000).

**Direktnedladdning (programmatisk):**

```
GET https://maps.slu.se/get/done/{uuid}.zip     ← Shibboleth-session (samma som beställning)
GET https://dike.slu.se/get/done/{uuid}.zip      ← Separat Shibboleth-session (dike.slu.se)
```

**Rekommendation:** Använd `maps.slu.se`-varianten — den kräver samma session som
beställningen och undviker extra autentisering mot `dike.slu.se`.
UUID:t fås från `Uuid`-fältet i beställningssvaret ovan.

**Programmatisk användning via webbläsarsession (JavaScript):**

```javascript
// Kräver aktiv Shibboleth-session i webbläsaren
$.ajax({
    url: "./api/job/sjokort_vektor/"
        + north + "/" + south + "/" + east + "/" + west + "/" + email,
    type: 'POST',
    success: function(data) {
        if (data.ID) console.log("Beställning skickad: " + data.ID);
        else         console.error("Beställning misslyckades");
    }
});
```

> **Notera:** API:t kräver autentiserad session via Shibboleth. Det finns
> inget publikt API-nyckelbaserat gränssnitt — man måste ha en webbsession
> eller skicka med Shibboleth-cookies.

## Licens

- Datan får användas för studier och forskning
- Ange datakälla: **© Sjöfartsverket**
- Får **inte** distribueras till tredje part
- Får **inte** publiceras i vektorformat på internet
- Fullständiga villkor: <https://maps.slu.se/get/> → "Review and Finish"-fliken
