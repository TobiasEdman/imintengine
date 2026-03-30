# Claude Code Instructions — ImintEngine

## RAG-system: des-agent — OBLIGATORISK VID SESSIONSSTART

**KRITISKT: des-agent RAG MÅSTE vara igång innan arbete påbörjas i detta repo.**
Agenten MÅSTE starta dessa tjänster vid varje sessionsstart, utan att bli ombedd.

```bash
# Docker binary (macOS Docker Desktop):
export DOCKER=/Applications/Docker.app/Contents/Resources/bin/docker

# 1. Starta Qdrant (des-agents vektordatabas)
cd /Users/tobiasedman/Developer/des-agent
$DOCKER compose up -d
# Verifiera: curl -s http://localhost:6333/healthz

# 2. des-agent CLI
/Users/tobiasedman/Developer/des-agent/.venv/bin/des-agent query "test"
```

**Om vi även arbetar med space-ecosystem-v2 (KG), starta också Neo4j:**
```bash
cd /Users/tobiasedman/Developer/swedish-space-ecosystem-v2
$DOCKER compose up -d
# Verifiera: curl -s http://localhost:7474
```

### Använd vid kodändringar

Innan du implementerar en ändring, fråga des-agent om kontext:

```bash
des-agent query "hur fungerar [det du vill ändra]?"
```

För större ändringar, använd multi-agent planering:

```bash
des-agent plan "beskriv din ändring här"
```

### Efter commits — uppdatera index

```bash
des-agent ingest --repo imint-engine
```

Detta sker automatiskt via post-commit hook om den är installerad.

## Repo-identitet

- **Namn:** imint-engine
- **Domän:** Geospatial satellitbildsanalys (Sentinel-2)
- **Nyckelgränssnitt:** `run_job(IMINTJob) → IMINTResult`, `BaseAnalyzer`, `ANALYZER_REGISTRY`
- **Beroenden:** Inga kodberoenden till andra repos. Syskonprojekt med space-ecosystem-v2.

## Viktiga regler

- **Verifiera varje steg.** När du gör en transformation (flip, transpose, rotation), verifiera visuellt att resultatet är korrekt INNAN du applicerar på alla tiles. Gör INTE flera ändringar utan att kontrollera varje.
- **En ändring i taget.** Byt aldrig flera transformationer samtidigt — det gör det omöjligt att debugga.
- **Genomför instruktioner exakt.** Om användaren säger "applicera X" — gör exakt X, inte en approximation.

## Arkitekturregler

- Nya analyzers ska subklassa `BaseAnalyzer` och registreras i `ANALYZER_REGISTRY`
- Executors bygger `IMINTJob` och anropar `run_job()` — engine är executor-agnostisk
- Modifiera ALDRIG andra repos direkt härifrån
