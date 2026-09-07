# Kommunikationsfixen — plan v2

**Datum:** 2026-09-07. **Ersätter** åtgärdslistan i
[`comms_channel_analysis.md`](comms_channel_analysis.md). Mätdelen av den
rapporten står kvar; **åtgärderna gör det inte.**

**Underlag:** fyra oberoende utredningar — inventering av befintlig kod,
prior art, adversariell protokollgranskning, och människans behov — plus egen
verifiering av de två avgörande påståendena.

---

## Varför v1 underkändes

Den adversariella granskningen blockerade två av tre P0-åtgärder. Jag har
verifierat båda skälen i koden själv.

### P0-1 (höj TTL) — BLOCKERAD. Förlänger livslängden på falska påståenden.

v1 modellerade meddelanden som *information som är värdefull tills den lästs*.
Trafiken är inte sådan. En stor del är **statuspåståenden som ruttnar**:

```
21:30Z codex  "post-:17 window open … return PASS or BLOCK"
22:48Z codex  "exact SHA d81e0c2 review unanswered"
04:46Z claude PASS på d81e0c2
```

Med TTL 1h gick de ut 22:30 respektive 23:48 — **och det var korrekt**, för
kl 04:46 var påståendet falskt. Med TTL 86400 får nästa session
`"din review är obesvarad"` levererat **efter** att den redan svarat, som
färsk data.

TTL är idag den enda mekanism som pensionerar ett inaktuellt påstående. v1
tog bort den utan att ersätta den. Percentilanalysen (p95 = 3,7 h, 22 % av
svarsparen överskrider 1 h) mätte rätt sak på fel modell.

### P0-3 (kvittens) — BLOCKERAD. Skulle ljuga.

`hook_inbox` i den inkopplade implementationen gör, i denna ordning:

```python
seen.add(message_id)
_atomic_write(seen_path, …)      # bokföringen är committad
return {"hookSpecificOutput": {"additionalContext": …}}   # först nu levereras
```

Bokföringen sker **före** att texten lämnat processen, och långt före den når
modellens kontext. Användaren avbryter i **21 %** av turerna. En ACK byggd på
detta påstår "läst" om meddelanden som aldrig setts.

Idag betyder tystnad *"inte hanterat"* — sant och handlingsbart. Efter v1
hade den betytt *"läst men obesvarat"* — falskt och handlingsbart, eftersom
mottagaren nu **litar** på fältet. Ett fält som påstår mer än mekanismen kan
bevisa är en lögn med schemastöd.

### Och ACK löser inte incidenten v1 sa att den löste

v1 skrev att kvittens *"ensamt hade förhindrat hela 04:46-incidenten"*. Fel.
Incidenten är två oberoende fel:

| fel | ACK hjälper? |
|---|---|
| Codex rapporterar från en ögonblicksbild tagen 9 s före mitt PASS | **nej** — ACK säger till *avsändaren* att mottagaren läst, inte till *läsaren* att något nytt kommit |
| Jag drar *"pollar inte"* ur *"skriver inte"* (fynd 11) | ja |

ACK fixar den hälft rapporten skrev som fynd 11 — inte den som skrevs som
fynd 3.

### Netto för v1

**1,5 av 7 dokumenterade felmoder åtgärdade. Tre förvärrade** (SessionStart-
floden blir 24× längre av P0-1, ×20 i ACK-volym av P0-3, och en ny flod per
väckt session av P0-5).

---

## Fyra saker finns redan och är oanvända

Detta är inventeringens viktigaste fynd. Vi var på väg att bygga om dem.

| Vad | Var | Varför dött |
|---|---|---|
| **Komplett ACK-protokoll** — `message-read --acknowledge --ack-message-id`, heartbeat, takeover | `agentic_task/governance_cli.py:961-981` | protocol-v1 kräver `.agents/agent-governance.json` på skyddad branch. Finns i **noll** repon. |
| **Durable wake-watcher** — *"at-least-once without double invocation"* | `agentic_task/pilot_watcher.py`, installerad som `agentic-pilot-watcher` | samma odriftsatta backend |
| **Nyare `continuity.py`** med låst append, rotation, bunden tail-läsning, Codex-väckning, `<<<UNTRUSTED-INBOX>>>`-framing | `~/.agents/runtimes/multi-agentic/0.7.0-2a016252/…/continuity.py` (96 kB, 2026-09-02) | **inte inkopplad** — hookarna pekar på `~/.agents/bin/` (110 kB, 2026-08-31) |
| **`blocked_by` / `blocks`** — "vem väntar på vem" | `.agents/schema.json` | `.agents/README.md:150`: *"exist in the schema but the CLI doesn't read them yet"* |

**Spärr före all kodändring:** minst fyra kopior av `continuity` existerar och
de har **divergerat** (TTL-konstanten ligger på rad 61 i den inkopplade, rad
69 i källan). Kör man båda mot samma repo trunkerar den nya `.seen` till 4096
id:n medan den gamla läser den trunkerade listan och **återlevererar hela
historiken**. Att patcha fel fil är inte en miss — det är en ny felmod.

---

## Codex hookar körs inte

Verifierat på tre oberoende sätt:

```
active-sessions/     codex-*.json:  0     claude-*.json: 76
session-snapshots/   codex/:       saknas  claude/:      finns
.seen/               codex:  1 fil, 2026-08-27, ETT meddelande-id
```

`~/.codex/hooks.json` registrerar alla åtta hookarna. Tre hook-drivna
datastrukturer är ändå tomma sedan installationsdagen.

Rapportens fynd 4 sa *"Codex läser på något annat sätt utan att bokföra"*.
Rätt slutsats är att **dess hook-lager aldrig kör**. Om det åtgärdas tar
Codex emot meddelanden vid `SessionStart` och `PreToolUse` som jag gör, och
merparten av den uppmätta asymmetrin försvinner utan ny kod.

**Detta är den enskilt högsta nyttan per insats i hela materialet.**

---

## Omprövning: byt bärare, laga inte kanalen

Kärnfelet är att kanalen är en **oadresserad broadcast-buss av fritext med
tidsbaserad sophämtning**. Varje åtgärd i v1 lappade en konsekvens av det
valet.

**Modell: meddelanden är tillståndsövergångar på ett arbetsobjekt.**
En append-only logg per objekt (`PR36`), inte per repo, med typade poster:
`review-requested(sha)`, `verdict(sha, PASS|BLOCK, evidence)`, `executed(…)`.

| v1-åtgärd | Blir onödig därför att |
|---|---|
| P0-1 TTL | ett PASS på en exakt SHA gäller tills SHA:n ändras — det ruttnar inte. TTL finns bara för att en broadcast-buss behöver GC. |
| P0-3 ACK | avsändaren behöver inte *"läste du?"* utan *"finns verdikt för `sha=d81e0c2`?"* — en **query**, inte en inferens |
| P1-4 status | status blir en fold över loggen, inte en gissning |
| N-agent / `to`-fält / 7-sessioner | kollapsar: enheten är ett *anspråk på ett objekt*, som `agentic-task` redan modellerar |
| Fynd 5 flod | en session läser sitt objekts logg, inte 24 h repo-brett korsprat |

`send_message` har **inget `to`-fält** — varje meddelande är broadcast. Med
7 sessions-ID per runtime betyder `by: <runtime>` att den första session som
bootar kvitterar bort skyldigheten för alla andra, inklusive den som äger
granskningen. Falskt "läst" är handlingsbart; avsaknad av ACK är det inte.
**Den modellen är sämre än ingen ACK.**

---

## Prioritetsordning v2

### 0. Spärrar — inget landar före dessa

- **0a.** Avgör kanonisk implementation av `continuity`; avveckla de andra.
- **0b.** Verifiera om Codex CLI läser `~/.codex/hooks.json` alls.
- **0c.** Verifiera om `additionalContext` honoreras på `PreToolUse`.
  *Test:* posta meddelande med unik nonce, tvinga fram ett PreToolUse-Bash
  utan mellanliggande SessionStart, sök nonce i mottagarens transkript.
  Saknas den men finns id:t i `.seen` → bekräftad tyst förlust, och `Stop`
  (v1:s P1-1) skulle införa samma fel på ett tredje event.

### 1. Noll kod, högst nytta

**Förbjud ostämplade negativa påståenden.** En protokollregel i AGENTS.md /
CLAUDE.md. *"Ingen PASS finns"* är meningslöst. *"Ingen PASS finns per
inbox-id 1788756402030, 04:46:33Z"* är sant och falsifierbart.

Hade ensamt förhindrat fynd 3, fynd 11 och turerna 104/110/125 — **fyra av
fem dokumenterade incidenter.** Saknades helt i v1.

### 2. Flytta verdikt till PR:en

GitHub har redan adressering, kvittens, varaktighet, query och en
tillståndsmaskin *requested reviewer → submitted review*. Det tar den högsta
insatsen ur den svagaste bäraren. Noll ny infrastruktur.

### 3. Fixa ordningen bokföring→leverans

Markera `.seen` bara för poster som faktiskt emitterats, och bara på event
där `additionalContext` bevisligen honoreras (spärr 0c). Liten diff,
avskaffar verklig förlust — och är en **förutsättning** för att en framtida
ACK ska vara sann.

### 4. Statusytan — tre fält

`väntar_på_dig[]`, `blockerat[] {objekt, väntar_på, sedan}`, `färsk_sedan`.
Härledda ur primärkällor (inbox, `gh`, `kubectl`, `active-jobs`), **aldrig**
ur vad en agent skrivit. Renderade som en promptsträng:

```
imint ⏳2 ⛔1 ⚠1 · 3m
```

Adresserar 50 av 131 turer. Krav: varje rad bär sin källa och sin watermark,
annars reproducerar ytan fynd 3 mekaniskt och med auktoritet.

### 5. Eskalering med förval och tidsgräns

`{objekt, fråga, alternativ[], förval, förfaller_om, förvalets_konsekvens}`.
Förvalet alltid fail-closed. Människan är borta 85 % av väggklockan; **en
eskalering utan förval är ett stopp.** Då blir hans frånvaro ett beslut i
stället för en blockering.

### 6. Först därefter — arbetsobjektsloggar

Och i så fall en **headless kökonsument**, inte "väck en interaktiv session".
En väckt session får nytt sessions-ID → tom `.seen` → hela kön levereras →
och den får ändå inte agera utan användartur (CLAUDE.md §1). Nettoresultat:
*"Claude har läst allt"* plus noll handling — strikt sämre än idag.

---

## Vad ingen åtgärd löser, och som ska sägas rakt ut

**Latens.** Ingen in-band-mekanism fixar att ingen kör. De ärliga
alternativen är att acceptera människodriven kadens och göra väntan *korrekt*
i stället för snabb (punkt 1 och 2 gör det), eller att köra en verklig
arbetarprocess. Att kalla det senare för "idle-vaktare" och implementera det
som sessionsstart är att bygga det ena med det andras garantier.

---

## Acceptanskriterier

Mot den mätta baslinjen (316 meddelanden, median 7,7 min, p95 3,7 h,
22 % överskrider 1 h, 50 av 131 människoturer = overhead):

| kriterium | idag | mål |
|---|---|---|
| Negativa påståenden utan watermark | norm | **noll** — lint:bart i loggen |
| Verdikt återfinnbara utan att fråga en agent | nej | ja, via `gh` |
| `.seen` markerad utan bevisad leverans | ja | **noll** |
| Människans transport- och statusturer | 50/131 | < 5 |
| Codex hook-drivna datastrukturer | 3 av 3 tomma | 3 av 3 skrivna |

---

## Öppna frågor som kräver Codex

1. Vilken `continuity`-implementation är kanon?
2. Läser Codex CLI `~/.codex/hooks.json`? Om inte — varför registrerades den?
3. Honoreras `additionalContext` på `PreToolUse` och `Stop`?
4. Vad får en väckt session göra utan användartur? Om svaret är "inget", vad
   är väckningens värde?
5. `.agents/handoffs/` är redan den arbetsobjektsmodell punkt 6 beskriver —
   varför gick PR36-granskningen via inboxen i stället? Under tre dygn och
   316 inbox-meddelanden skrevs **noll** handoffs.
