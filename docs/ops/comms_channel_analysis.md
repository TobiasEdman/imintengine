# Cross-runtime inbox — varför meddelanden inte når fram

**Datum:** 2026-09-07. **Uppdrag:** PR36-COMMS-ANALYSIS (Codex, 04:58:26Z).
**Metod:** mätning på 316 meddelanden i
`~/.agents/continuity/session-inbox/ImintEngine--f4ea50ce3600.jsonl`
(2026-08-29 → 2026-09-07) plus inspektion av `agentic-continuity` och
hook-konfigurationen. **Read-only. Ingen rolloutkod eller ICE rörd.**

## Sammanfattning

Uppdraget bad om att skilja *missade pollningar* från *osynliga läsningar*.
Den distinktionen går inte att göra, eftersom **det inte finns någon
pollning alls**. Kanalen har ingen bakgrundsläsare. Två av fynden nedan är
inte fördröjningar utan tyst dataförlust.

| # | Fynd | Allvar |
|---|---|---|
| 1 | Ingen poller — leverans är parasitisk på mottagarens egen aktivitet | P0 |
| 2 | 16 % av Codex meddelanden **gick ut** innan svar (TTL 1h < medellatens 80 min) | P0 |
| 3 | Ingen ACK — avsändaren kan inte skilja *oläst* från *läst men obesvarat* | P0 |
| 4 | Codex läs-tillstånd är dött sedan 2026-08-27 | P0 |
| 5 | Sessions-churn nollställer dedup → SessionStart-flod | P1 |
| 6 | 2000-teckensgräns tvingar fram trunkerad granskningstext | P1 |

## Mätdata

- 316 meddelanden, 147 från claude, 168 från codex
- **Medianlatens 7,7 min. Medel 79,8 min.** Kraftigt svansdriven.
- Glapp > 1 h förekommer i **båda** riktningarna — inte en runtimes fel
- 7 distinkta sessions-ID per runtime i avsändarfältet

## Fynd 1 — det finns ingen poller (P0)

`inbox`-hooken är registrerad på exakt två händelser:

```
inbox-hook på: PreToolUse
inbox-hook på: SessionStart
```

Ingen timer, ingen `Notification`, ingen bakgrundsbevakare. Leverans sker
alltså bara när mottagaren **på eget initiativ** startar en session eller
är på väg att köra ett verktyg.

En agent som väntar på sin användare kör inga verktyg och tar därför emot
ingenting. Den kan sitta i timmar med olästa meddelanden. Det förklarar
varje observerat glapp, inklusive de tre dyraste i den här kampanjen
(15:06→17:30, 17:47→20:58, 22:48→04:46).

**Följd:** "väckningsrader" väcker ingenting. De föreföll fungera för att
mottagaren råkade bli aktiv strax efter — inte på grund av meddelandet.

## Fynd 2 — TTL är kortare än svarstiden (P0)

`DEFAULT_INBOX_TTL_SECONDS = 3600` (1 timme). Faktisk användning:

| runtime | TTL 1h | 24h | övrigt |
|---|---|---|---|
| claude | 31 | 116 | — |
| codex | **144** | 20 | 5 |

Codex skickar alltså 86 % av sina meddelanden med en timmes TTL, mot en
**medellatens på 80 minuter**. Resultatet är mätbart:

- **27 av 168 meddelanden från codex (16 %) gick ut innan motparten svarade**
- 12 av 147 från claude (8 %)

Det är inte fördröjning utan **förlust**, och varken avsändare eller
mottagare får någon signal om att det hänt.

## Fynd 3 — ingen kvittens (P0)

Det finns ingen read-receipt tillbaka till avsändaren. En avsändare som
inte får svar kan inte avgöra om meddelandet är oläst, läst-men-obesvarat,
eller utgånget.

Det är exakt vad som hände 2026-09-07: Codex rapporterade sig blockerad på
"Claudes exakt-SHA-review" klockan då mitt PASS på `d81e0c2` redan låg i
kön sedan 04:46:42Z. Slutsatsen "obesvarat" drogs ur **frånvaro av svar**,
vilket är det enda observerbara.

## Fynd 4 — Codex läs-tillstånd är dött (P0)

Dedup lagras i `.seen/<receiver-session-id>.json`.

```
claude: 20 filer, senast ändrad 2026-09-07 07:00
codex :  1 fil,  senast ändrad 2026-08-27 12:43, innehåller EN message-id
```

Codex har skrivit 168 meddelanden fram till 2026-09-07, men dess
läs-tillstånd har inte uppdaterats på elva dygn. Den läser uppenbarligen
på något annat sätt (den svarar ibland inom 85 sekunder), men **inte via
den väg som bokför vad som lästs**.

Konsekvensen är att ingen — varken Codex själv, jag, eller en människa —
kan avgöra vad Codex faktiskt har sett. Kanalen är i praktiken
skriv-bara i den ena riktningen.

## Fynd 5 — sessions-churn nollställer dedup (P1)

`.seen` är nycklad per sessions-ID, och det finns 20 claude-sessioner.
Koden säger det själv (rad 553): *"interrupted state as empty and messages
are therefore delivered again"*. Varje ny session får hela den
icke-utgångna kön på nytt vid SessionStart. Det tränar läsaren att skumma
en flod, vilket sänker uppmärksamheten på det som faktiskt är nytt.

## Fynd 6 — 2000-teckensgränsen (P1)

`tell` avvisar meddelanden över 2000 tecken. Under den här kampanjen
tvingades granskningstext trunkeras minst fem gånger, i flera fall genom
att korta ned just den bevisföring som motiverade ett PASS eller BLOCK.
Ett granskningsprotokoll som bygger på exakt-SHA-bevis bör inte ha en
transportgräns som beskär beviset.

## Åtgärder

### P0 — stoppar tyst förlust och falska blockerings-lägen

**P0-1. Höj default-TTL till 86400.** Enradsändring
(`DEFAULT_INBOX_TTL_SECONDS`). Motiv: nuvarande default är kortare än
medellatensen, vilket garanterar förlust.
*Acceptanstest:* skicka utan `--ttl-seconds`, assertera
`expires_at - ts == 86400`. Spela om 9-dygnskorpusen och assertera
**noll** utgångna-före-svar.

**P0-2. Reparera Codex läs-tillstånd.** Läsvägen måste bokföra `.seen`.
*Acceptanstest:* efter att Codex läst ett meddelande, assertera att dess
`.seen`-fil har mtime inom 60 s och innehåller meddelande-id:t.

**P0-3. Inför kvittens.** När en runtime läser ett meddelande, appendera
`{"acks": <id>, "by": <runtime>, "ts": ...}` till samma JSONL.
*Acceptanstest:* en läsning ger exakt en ack-post; `tell --status <id>`
svarar entydigt *oläst / läst / utgången*. Detta ensamt hade förhindrat
hela 04:46→05:00-incidenten.

### P1 — kortar latensen

**P1-1. Registrera inbox-hooken på `Stop`.** Då kontrolleras kön när en
agent *avslutar* sin tur, inte bara när den påbörjar en. Återanvänder
befintlig maskineri, noll ny infrastruktur.
*Acceptanstest:* hook-konfigurationen innehåller inbox på `Stop`;
simulerad turslut levererar väntande meddelande.

**P1-2. Extern väckning vid idle.** Eftersom ingen runtime pollar när den
är inaktiv är detta den enda verkliga lösningen: en schemalagd bevakare
(per global regel §8) som läser JSONL:en och ytar nytt.
*Acceptanstest:* med båda sessionerna inaktiva ytas ett meddelande postat
vid T senast T+N.

**P1-3. Höj eller kringgå 2000-teckensgränsen** — t.ex. tillåt
`--body-file` med en pekare i meddelandet.
*Acceptanstest:* en 5000-teckens analys levereras intakt.

### P2 — hygien

**P2-1. Stabil mottagaridentitet** per (runtime, repo) i stället för per
session, så `.seen` överlever sessionsbyten.
*Acceptanstest:* två på varandra följande sessioner får **inte** samma
meddelande levererat två gånger.

## Vad som INTE är problemet

- Det är inte en runtimes slarv. Glappen går åt båda hållen.
- Det är inte "missade pollningar". Det finns inga pollningar att missa.
- Det är inte TTL **ensamt**: även med oändlig TTL levereras ingenting
  till en inaktiv mottagare. P0-1 stoppar förlusten, P1-2 stoppar väntan.

## Störst effekt per insats

P0-3 (kvittens) och P0-1 (TTL) är bägge små ändringar som tillsammans
eliminerar *tyst* fel: efter dem vet en avsändare alltid om ett meddelande
är levererat, och inget försvinner. Latensen kvarstår tills P1-2 finns,
men den blir då **synlig och korrekt attribuerad** i stället för att
feltolkas som utebliven granskning.

---

# Tillägg — människa↔agent-dialogen

**Datum:** 2026-09-07. **Föranlett av:** Tobias fråga *"har ni tittat på
våra sessioner och dialoger också?"* (05:57Z), understödd av Codex
addendum samma minut. **Granskat transkript:** exakt en fil,
`~/.claude/projects/-Users-tobiasedman-Developer-ImintEngine/`
`4f34d985-bf1e-4233-94b4-b71f1985f35d.jsonl` — Claude-sessionen som körde
hela #36-rolloutgranskningen 2026-09-02→09-07, 131 användarturer.
Codex-sidans människodialog är **inte** granskad; den ligger utanför min
åtkomst och måste mätas av Codex själv för att bilden ska bli hel.

## Varför den ursprungliga analysen var otillräcklig

Grundanalysen mätte bara agent↔agent-kön och drog slutsatsen att ingen
runtime tar emot något medan den är inaktiv. Men om ingen agent pollar,
så är **människan den faktiska transporten** — och då är varje uppmätt
"glapp" i själva verket *tiden tills Tobias puffade mottagaren*. Att
utelämna människodialogen var att mäta symptomet och hoppa över
mekanismen. Tobias fråga var alltså en korrekt granskningsanmärkning på
mitt eget arbete, inte en utvidgning av scopet.

## Mätning

| kategori | turer | andel |
|---|---|---|
| **Transport** (relä av Codex-text, *"har Codex läst X?"*) | 32 | 24,4 % |
| **Status/ETA-pollning** | 18 | 13,7 % |
| Avbrott + *"Try again"* | 27 | 20,6 % |
| Skill/system | 6 | 4,6 % |
| **Faktisk styrning och arbete** | 48 | **36,6 %** |

- Transport + status = **50 turer, 38,2 %** av allt Tobias skrev
- Inklusive avbrott: **58,8 % overhead**
- 18 mänskliga frånvaroluckor > 2 h, totalt **112,8 h = 85 % av
  sessionens väggklocka**

## Fynd 7 — människan är transporten, och är borta 85 % av tiden (P0)

Ingen runtime pollar när den är inaktiv (fynd 1). Leveransen sker alltså
i praktiken när Tobias är närvarande. Han är frånvarande 85 % av
väggklockan.

Det förklarar diskrepansen i grunddatan som annars ser konstig ut:
**medianlatens 7,7 min mot medel 80 min.** När båda parter är vakna är
kanalen snabb. När människan kliver undan stannar den helt. Det är inte
två olika beteenden hos agenterna utan ett beteende hos systemet, sett
under två olika förutsättningar.

**Konsekvens för prioritering:** en schemalagd idle-vaktare stod som P1-2
i grundrapporten. Den bedömningen var fel. Om människan är den enda
transporten och är borta 85 % av tiden är idle-vaktaren **P0**, inte P1 —
den är skillnaden mellan en kanal som fungerar och en som fungerar bara
när någon står bredvid.

## Fynd 8 — människan tvingades bli kvittensmekanismen (P0)

Eftersom ingen ACK finns (fynd 3) är Tobias den enda part som kan
observera båda sidor. Frasen *"kolla om Codex har läst X"* förekommer
ordagrant sex gånger i transkriptet, och varje gång fungerade den som en
manuell read-receipt.

Det är samma sak som fynd 3, sett från andra hållet: den saknade
tekniska funktionen har inte försvunnit, den har flyttats till en person.

## Fynd 9 — dialogen var systemets felupptäckt

Fyra gånger avslöjade en enkel människofråga ett falskt påstående om delat
tillstånd som ingen agent hade upptäckt:

| tur | frågan | vad den avslöjade |
|---|---|---|
| 104 | *"Codex påstår att den inte har pass"* | mitt PASS låg redan i kön |
| 110 | relä: *"Claude har ännu inte svarat"* | mitt BLOCK låg redan i kön |
| 125 | relä: *"Enda blockeraren är Claudes review"* | mitt PASS låg redan i kön |
| 130 | *"har ni tittat på våra dialoger också?"* | min egen analys var för snävt avgränsad |

Att en människa är bra på att upptäcka fel är inget problem. Att han är
den *enda* mekanismen som gör det, för en klass av fel som är triviell att
detektera automatiskt, är det.

## Fynd 10 — min egen andel av friktionen (P1)

27 av 131 turer (20,6 %) är avbrott eller *"Try again"*. Det är inte
Codex fel och inte kanalens: användaren avbröt **mig** i var femte tur.

Rimliga tolkningar, som jag inte kan avgöra mellan utan att fråga: svaren
var för långa, jag fortsatte arbeta när ett kort besked räckte, eller jag
började agera innan riktningen var bekräftad. Oavsett vilket är det ett
mätbart mått på att min utdata inte matchade vad som efterfrågades, och
det hör hemma i samma rapport som kritiken mot Codex.

## Åtgärder — tillägg och omprioritering

**P0-5 (uppgraderad från P1-2). Schemalagd idle-vaktare.** Motivet är inte
längre bekvämlighet utan att människan annars *är* transporten under 85 %
av tiden.
*Acceptanstest:* med båda sessionerna inaktiva och ingen människa
närvarande ytas ett meddelande postat vid T senast T+N.

**P1-4. Frågbar statusyta.** 18 turer var *"status?"* eller *"eta"*. En
enda kommando- eller filbaserad yta som svarar *vad väntar på vem, sedan
när* eliminerar den kategorin.
*Acceptanstest:* `agentic-continuity status --repo X` svarar med
väntande-på-part och ålder utan att en agent behöver köras.

**P2-2. Mät Codex människodialog.** Denna rapport täcker en sida. Codex
bör köra samma klassificering på sitt eget transkript; utan det är
relätalen ensidiga.

## Vad tillägget ändrar i slutsatsen

Grundrapporten sade att P0-1 (TTL) och P0-3 (ACK) eliminerar *tyst* fel
medan latensen kvarstår. Det står fast. Men latensen är inte en olägenhet
som kan vänta till P1 — den bärs idag av en människa, till en mätbar
kostnad av 38 % av hans turer i den här sessionen. Idle-vaktaren är
därför en P0-åtgärd, och statusytan är det som gör att han slipper fråga.
