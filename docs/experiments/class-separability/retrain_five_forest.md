# Destillation med fem skogsklasser — TESSERA f5

IM-023, 2026-11-17, task 0175. **Beslut (Tobias):**

- Destillationen görs om med fem skogsklasser. Barrblandskog (NMD 113,
  klass 28) förfinas mot NFI-provytorna precis som tallskog, granskog,
  lövskog och blandskog, i stället för att NMD 113:s yta läggs ovanpå
  huvudets val, som i main (`retrain_barrblandskog.md`).
- 113 och 114 hanteras lika. Lövblandad barrskog (114) stannar i blandskog
  (klass 4), som huvudet redan förfinar. Ingen egen klass.
- Sanningen är utvärderingens förregistrerade regel
  (`imint/eval/nfi_amendment.derive_barrblandskog`), så huvudet lär sig den
  sanning det poängsätts mot.
- Contorta räknas som tall, som i utvärderingen. Känsligheten för contorta
  utvärderas senare.
- TESSERA tränas om på de nya etiketterna, inte Prithvi. Prithvi-kedjan är
  pausad.
- Körningen sker på en 2080 Ti (ICE har ingen H100).
- Ingen ny kontrollkörning. Jämförelsen görs mot de befintliga
  `im023_control` och main.

## Sanningen

`scripts/nfi_truth_f5.py derive` bygger en kopia av provyteindexet med
kolumnen `nfi_forest_f5`:

1. Först sanning v1 (`derive_nfi_forest_class`). Har barr eller löv minst
   0,7 av virkesförrådet blir ytan tall eller gran (efter vilket som har
   störst volym) respektive lövskog. Annars blir den blandskog. Trädlösa
   ytor får -1.
2. Därefter blir en v1-barryta där varken tall (tall + contorta) eller gran
   når 0,7 av barrvolymen barrblandskog (28).

Bara klass 28 flyttar. Tall och gran blir därmed smalare än i v1: de kräver
nu 0,7 av barrvolymen i stället för att bara ha störst volym. Klasserna 3
och 4 och de trädlösa ytorna är oförändrade.

## Kedjan (`k8s/im023/f5/distill-f5-job.yaml`, 2080 Ti)

Kedjan är R4:s TESSERA-destillation (`k8s/ladder/distill-tessera-job.yaml`)
med samma rung-2-checkpoint, provyteindex, pinnade provytor och kohort. Ett
test håller varje steg mot R4:s, med bara de planerade avvikelserna.

0. Klarsignalen `_GATE_OK` tas bort innan något körs.
1. Sanningen härleds i en kopia av indexet.
2. Egenskaperna extraheras med `--truth-col nfi_forest_f5`. `same-plots`
   vägrar fortsätta om de inte täcker exakt R4-huvudets provytor. Nyckeln
   är `(tile_name, TractID, PlotID)`, samma som de pinnade provytornas. Den
   frysta NFI-holdouten utesluter varje plot-år i R4:s egenskapsfil, så
   huvudet förblir disjunkt från den. Båda stegen vägrar också om
   provyteindexet eller R4:s egenskapsfil inte längre har den storlek och
   sha256 som frysningens manifest registrerade, eftersom det är de
   bytesen holdouten utesluter.
3. Destillerbarheten beräknas (OOF på de pinnade provytorna), med
   poängsättning i sanningens eget klassrum.
4. Huvudet tränas med klasserna 0–4 och 28.
5. Den täta passagen körs med `--forest-classes 1,2,3,4,28` på
   `/cephfs/nmd2023_labels_b28`, där NMD 113 är 28. En pixel skrivs om där
   både etiketten och huvudet säger en skogstyp. Destilleraren vägrar köra
   om mängden inte är exakt huvudets skogsklasser, eller om kohorten inte är
   exakt `tessera_r2`:s. Det kontrolleras innan första tilen.
6. Huvudet publiceras som `heads/tessera_r2_f5_head.npz`, och `_GATE_OK`
   skrivs sist.

Allt som skrivs hamnar på nya sökvägar. R4:s huvud och provytefiler,
`/cephfs/distill/tessera_r2` och b28-katalogerna läses bara.
Mellanresultaten ligger per körning i `/cephfs/im023/f5_distill/<UTC>/`.

## Träningen (`k8s/im023/f5/train-f5-job.yaml`, 2080 Ti)

`scripts/im023_retrain_f5.py` ger 29 klasser, etiketter från
`/cephfs/distill/tessera_r2_f5`, R4:s flaggor och mains klassvikter
(`b28_class_stats`). Mellan f5 och main skiljer därför bara hur klass 28
märks, och profilen.

**Profilen `2080ti` är en receptskillnad mot main:** FP32 (Turing saknar
BF16), batch 2 (tränaren saknar gradientackumulering, och batch 8 får inte
plats i 11 GB) och 6 workers. Den ingår i det som f5 minus main mäter och
redovisas tillsammans med resultatet.

Preflighten vägrar starta i fem fall:
- `_GATE_OK` saknas;
- någon etikett är skriven av ett annat huvud än det publicerade;
- huvudet har andra klasser än 0–4 och 28;
- kohorten skiljer sig från R4:s;
- en etikett i ett urval av de första 20 filerna är 29 eller högre.

Stämpeln kontrolleras på varje fil. Att varje fil i kohorten finns säkrar
destillerarens grind och kohortjämförelsen.

## Utvärdering

Utvärderingen görs inte här. Att poängsätta f5 mot den frysta NFI-holdouten
kräver ett nytt amendment, eftersom cellistan i `nfi_amendment.py` är låst
till main, control och R4, och Tobias klartecken. Contorta-känsligheten
hör till samma steg. Resultaten hålls privata till publiceringen.
