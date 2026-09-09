# Lupiner längs vägar — finetuning av Tessera-modellen (utforskning)

*2026-09-08. Underlag: GBIF-API-frågor (Artportalen-speglingen) + repo-genomgång.
Status: utforskning — inget beslut, ingen implementation.*

## Vad "vår Tessera-modell" är

`TesseraSegmentationModel` (`imint/fm/tessera_seg.py`) — vårt egenutvecklade
lätta segmenteringshuvud (~2–3 convlager + klassificerare, med valbar gated
aux-fusion och Trädslag-frac-huvud) som körs direkt på TESSERA:s förberäknade
årliga 128-D per-pixel-embeddings (10 m GSD, S1+S2). Ingen encoder körs vid
träning — embeddings bakas in i tile-.npz via `scripts/enrich_tiles_tessera.py`
(geotessera; Sverige 100 % täckt 2018–2024, `TESSERA_YEARS`). Tränade
checkpoints på CephFS, t.ex. `/cephfs/checkpoints/tessera_gated/`. Modellen
ingår i modellstegen (`k8s/ladder/*tessera*`) med distill-varianter.

## Varför uppgiften passar just den här modellen

1. **Ingen spektralfetch behövs.** Huvudet tränas på embeddings + etiketter.
   Embeddings hämtas fritt via geotessera (ingen CDSE PU, ingen DES-slot).
   Lupinpunkterna ligger utspridda över hela landet — att fetcha 4-frame
   spektral för dem vore en stor kampanj; embeddings-patchar är närmast gratis.
2. **Label-efficient by design.** Tessera-huvudet är byggt för små
   träningsmängder — exakt situationen med ~tiotusentals punktetiketter.
3. **Befintligt mönster för binärt huvud.** Avverkningsmogen-huvudet
   (sigmoid + BCE) är den arkitektoniska mallen; frac-/dual-head-wiring i
   trainern är redan backbone-agnostisk.

## Etikettdata (Artportalen via GBIF, taxonKey 2964355)

| Filter | Antal |
|---|---|
| Sverige totalt | 74 527 |
| — varav Artportalen-datasetet | 72 863 |
| Koordinatosäkerhet ≤25 m | 50 140 |
| ≤25 m **och** år 2018–2024 (Tessera-täckta år) | **38 759** |
| — varav juni–augusti | 16 043 |
| Fritext "vägkant" | ~3 800 (undre gräns) |

- Årsmatchning: observationsår → embeddingår (samma centraliserade
  `infer_tile_year`-disciplin som enrichern). 2025/2026-fynd (~10k) faller
  utanför testade embeddingår — clampa eller släpp.
- Åtkomst: GBIF-API öppet; SOS-API (SLU, gratis nyckel) ger fler fält
  (biotop, projekt) + SWEREF99.

## Designskiss

- **Uppgiftsform:** binär per-pixel-sannolikhet (lupin) med glespunkts-
  supervision — loss bara på etiketterade pixlar. Positiv = pixel med fynd
  (≤25 m). Antingen separat litet huvud på embeddings (renast; stör inte
  23-klass-modellen) eller extra huvud på befintlig checkpoint.
- **Pseudo-negativer:** Artportalen är presence-only. Sampla negativer från
  vägkantspixlar >~500 m från närmsta lupinfynd + slumpad bakgrund.
  Rapporteringsbias (tätortsnära, vägnära) måste balanseras i samplingen.
- **Vägmask:** NVDB-vägnät (Trafikverket, öppen data via Lastkajen),
  buffert ~20–30 m, som ROI vid inferens och för negativ-sampling.
- **Eval:** spatial block-CV (län/rutor), aldrig random split — punkterna
  klumpar. Extern sanity: Trafikverkets egna invasiv-inventeringar.

## Risker / öppna frågor

1. **Subpixel:** många vägkantsbestånd är smalare än 10 m-pixeln.
2. **Årlig embedding späder blomsignalen:** juni–juli-magentan är kanske
   framträdande i S2-tidsserien som embeddings kodar — men okänt hur mycket
   som överlever kompressionen till 128-D årsembedding. **Avgörande okänd.**
3. **Presence-only + bias:** falska negativer i pseudo-absenserna sätter
   tak på mätbar precision.

## Pilotkörning 1 (2026-09-09) — OGILTIG, år förväxlat med klass

Kördes med 2 875 punkter (1 821 lupin / 1 054 bakgrund), 100 block,
per-punkt-år, pooled out-of-fold AUC över 1°-cellsgrupper.

| Modell | AUC |
|---|---|
| logreg på embeddings | 0,908 |
| MLP på embeddings | 0,908 |
| lat/lon-kontroll (linjär) | 0,503 |
| lat/lon-kontroll (MLP) | 0,493 |
| **år-kontroll (endast årtal)** | **0,896** |

**Resultatet mäter år, inte lupin.** Samtliga negativer kom från 2024
medan positiverna spänner 2018–2024; år-kontrollen når därför nästan
embeddingmodellens AUC utan att se en enda pixel. På det väg-matchade
subsetet slår år-kontrollen (0,902) t.o.m. embeddingmodellen (0,875).
Detta är exakt domängranskarens fynd 1, realiserat i värsta form.

**Rotorsak (två samverkande fel):**
1. GBIF:s standardordning är inte slumpmässig över år, så en
   stickprovsdragning av sidor utan årsfilter returnerar ett enda år.
2. `fetch_gbif.py` saknade `if __name__ == "__main__"`-guard, så varje
   `import` från filen körde om hela hämtningen som sidoeffekt — vilket
   återskapade 2024-datat efter att det ersatts, och dolde felet genom
   att se ut som en normal loggrad.

**Vad som räddade körningen:** kontrollmodellerna och asserterna som
granskarna krävde. Utan år-kontrollen hade 0,908 rapporterats som ett
"go". Duplicate-row-asserten fångade dessutom 60 punkter med helt
nollställda vektorer — geotesseras fyllnadsvärde för pixlar utan
täckning, osynligt för NaN-kontrollen.

**Åtgärder:** negativer hämtas nu år för år med explicit årsfilter;
importguard tillagd; nollvektorer och exakta dubblettvektorer filtreras
i proben (dedup på 10 m SWEREF99-grid räcker inte — TESSERA-pixlarna
ligger på ett annat rutnät).

### Infrastrukturlärdomar (gäller även klustrets enrich-jobb)

- **geotessera-registryt flyttade:** PyPI:s senaste (0.7.5) pekar på
  `dl2.geotessera.org` som svarar HTTP 410. Upstream 0.10.2 använder
  `data.source.coop` och kräver Python ≥3.12. Repots `enrich-tessera`-
  yamls kör opinnad `pip install geotessera` och skulle faila likadant.
- **`embeddings_dir` defaultar till cwd**, inte till `cache_dir`. Sätts
  den inte explicit sparas inga tiles där man tror, och varje körning
  laddar ner allt på nytt.
- **Nedladdningen saknar timeout** — en död socket hänger processen för
  alltid (uppmätt: 55 min, 0,02 s CPU). `socket.setdefaulttimeout()` plus
  förhämtning i egen regi gör felen högljudda och parallella.
- En tile kräver **tre** filer (embedding, scales, landmask); saknas en
  enda kastar offline-sampling och tar hela batchen med sig.

## Pilotkörning 2 (2026-09-09) — GILTIG. Svag men äkta signal

Årsmatchade negativer (4 200/år × 7 år), 2 999 användbara punkter
(1 870 lupin / 1 129 bakgrund), 99 block, 32 CV-grupper.

| Modell | AUC | AP |
|---|---|---|
| **logreg på embeddings** | **0,745** | 0,804 |
| MLP på embeddings | 0,740 | 0,794 |
| lat/lon-kontroll (linjär) | 0,541 | 0,645 |
| lat/lon-kontroll (MLP) | 0,478 | 0,628 |
| år-kontroll | 0,590 | 0,664 |

Embeddingmodellen slår alla tre kontroller med tydlig marginal
(+0,155 mot år, +0,204 mot linjär geografi). År-kontrollen föll från
0,896 till 0,590 när negativerna årsmatchades — confounden är borta,
och det som återstår är äkta.

**Men habitat förklarar en stor del.** Väg-matchat subset (båda klasser
≤30 m från OSM-väg): embeddings 0,666, år 0,634, geografi 0,506. AUC
faller alltså från 0,745 till 0,666 när negativerna också är vägkant —
precis den diagnos domängranskaren efterlyste. Kvarvarande marginal mot
år-kontrollen är tunn (+0,032).

Vetosvep (negativer >100/250/500 m från känd lupin): 0,745 / 0,752 /
0,723 — stabilt, så falska negativer sätter inget hårt tak.
Månadsstratifiering: juni–juli 0,743 mot augusti 0,684, svagt förenligt
med att blomningen bidrar (jämför endast AUC; AP:n har olika prevalens).

### Tolkning

Utfallet ligger i mellanzonen 0,6–0,8, inte i go-zonen ≥0,8. Läsningen
enligt granskarens tolkningsregel: **signalen finns men är svag, och
merparten av den lätta separationen är vägkant-mot-natur snarare än
lupin-mot-annan-vägkantsvegetation.** Det är konsistent med risk 2 i
utforskningen — den årliga 128-D-embeddingen späder ut ett fenologiskt
fönster på några veckor.

Slutsatsen falsifierar "TESSERA:s **årsembedding** bär lupinsignalen
starkt nog för produktion", inte "S2 bär lupinsignal". Nästa billiga
test är därför inte ett segmenteringshuvud, utan att jämföra samma
punkter mot en ren S2-juni/juli-komposit: bär råspektrat mer än
årsembeddingen, ligger flaskhalsen i den temporala kompressionen och
inte i uppgiften.

## Rekommenderad pilot (billig, avgör risk 2 direkt)

Linear probe: hämta embeddings via geotessera för ~2 000 positiva punkter
(≤25 m, 2018–2024, juni–aug-fynd för renast signal) + ~2 000 pseudo-negativa
vägkantspunkter. Träna logistisk regression / litet MLP på 128-D-vektorerna,
mät AUC med spatial CV. Körbar på CPU på timmar. AUC ≳0,8 → gå vidare till
segmenteringshuvud + NVDB-pipeline; AUC ~0,5–0,6 → embeddingarna bär inte
lupinsignalen och spåret stannar där, till låg kostnad.
