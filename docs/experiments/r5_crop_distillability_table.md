# R5-beslutsunderlag — crop-distillerbarhet per kolumn (UTKAST)

**Status:** UTKAST för oberoende sifferverifiering av ladder-sessionen
innan användning [delning accepterad 2026-09-08 06:0xZ]. **Beslutet**
(omträning R5 eller ej) är ett kostnadsbeslut → Tobias **via Codex**.

**Källa:** de sex write-once evidensfilerna
`/cephfs/distill/crop_heads/<m>_r2_crop_runs/<uid>--<m>_r2_crop_distillability.json`,
lästa 2026-09-08 ~06:2xZ. Per-fil sha256(16): clay `68e22b1cc2602a4c`,
croma `3711a16ced0a636e`, prithvi300m `06b0f8a126b53ccb`,
prithvi600m `987765531b510f01`, terramind `fa86140f42283adf`,
tessera `ec056949badd401c`.

## Jämförbarhetsgrindarna — VERIFIERADE före läsning av siffror

| grind | utfall |
|---|---|
| `y_sha256` | `356c3b0ed3d0ef2e` — **identisk i alla sex** ⇒ exakt samma 2491 etiketter |
| `n_plots` | 2491 = frysningens `n_distill` (aldrig holdout) |
| folds / seed / gruppering | 5 / 42 / `tile_name` (StratifiedGroupKFold) i alla sex |
| `git_sha` | `1fd08fad` i alla sex |
| `truth_col` | `unified_class`; klasstöd identiskt |
| kvalificeringsnycklar | `[s1_vv_vh, tessera]` — sexans gemensamma kohort |

Plandokumentets ram (rad 803–804): stadiet skapar **evidens, ingen gate** —
"Assemble six crop OOF results for the R5 decision". Ingen tröskel är
fördefinierad; tabellen är input, inte verdikt. `baselines: null` i alla
sex — ingen naiv baslinje ingår i bundlarna (notering till läsningen:
majoritetsklass-baslinjen är beräknbar ur klasstödet: 745/2491 = 29,9 % OA).

## Huvudtabellen — grupperad 5-fold OOF, mlp-huvud, n=2491

| kolumn (r2) | OA | Cohens κ | macro-F1 | svagaste klasser (F1) |
|---|---|---|---|---|
| **tessera** | **0,8294** | **0,7916** | **0,8327** | råg 0,63 · slåttervall 0,73 · trindsäd 0,78 |
| prithvi600m | 0,7965 | 0,7513 | 0,7923 | råg 0,62 · slåttervall 0,70 · trindsäd 0,75 |
| terramind | 0,7515 | 0,6959 | 0,7340 | råg 0,52 · slåttervall 0,64 · trindsäd 0,64 |
| prithvi300m | 0,7435 | 0,6858 | 0,7300 | råg 0,62 · havre 0,62 · slåttervall 0,63 |
| clay | 0,7383 | 0,6798 | 0,7238 | trindsäd 0,57 · råg 0,57 · havre 0,60 |
| croma | 0,7266 | 0,6655 | 0,7278 | havre 0,56 · råg 0,59 · slåttervall 0,59 |

Elva grödklasser per kolumn. Majoritetsbaslinje 29,9 % OA ⇒ samtliga
kolumner bär stark grödsignal i sina frysta representationer.

## Läsning (utkastets tolkning — verifieras, ej beslut)

1. **Grödinformationen är distillerbar ur alla sex kolumner** — spannet
   0,73–0,83 OA mot 0,30 baslinje. LUCAS-signalen finns i frysta
   representationer utan omträning.
2. **tessera dominerar** (+3,3 p.e. OA över tvåan, bästa F1 i 10 av 11
   klasser) — konsistent med dess dense-embedding-design.
3. **Rangordningen speglar INTE NFI-spåret rakt av** — prithvi600m är tvåa
   här; terramind slår prithvi300m och clay på gröda. En R5-tabell ger
   alltså delvis NY information, inte en skalad kopia av R1–R4.
4. **Svaga klasser är systematiska, inte modellspecifika:** råg (n=33),
   slåttervall och trindsäd är svagast nästan överallt — klasstödet och
   spektral förväxling (slåttervall↔bete) är trolig grund, vilket talar
   för att svagheten är data-, inte representationsbunden.
5. **croma sist** även här — konsistent med [user-stated 2026-08-31]
   prioriteringen.

## Vad tabellen INTE säger

- Ingen jämförelse mot **supervised** grödprestanda (rung-etiketterna
  innehåller LPIS-klasserna; en R5-omträning kan överträffa distillering —
  det är precis vad beslutet gäller).
- Inget om holdout — 1064-setet är orört och förblir så tills ett
  eventuellt R5 tränats.
- Ingen kostnadsvägning — den är beslutets andra halva och ligger hos
  Tobias via Codex.
