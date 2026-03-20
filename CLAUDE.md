# Claude Code Instructions — ImintEngine

## RAG-system: des-agent

Detta repo indexeras av `des-agent`, ett RAG + multi-agent system som förstår hela Digital Earth Sweden-ekosystemet.

### Aktivera vid sessionsstart

```bash
source /Users/tobiasedman/developer/des-agent/resume.sh
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

## Arkitekturregler

- Nya analyzers ska subklassa `BaseAnalyzer` och registreras i `ANALYZER_REGISTRY`
- Executors bygger `IMINTJob` och anropar `run_job()` — engine är executor-agnostisk
- Modifiera ALDRIG andra repos direkt härifrån
