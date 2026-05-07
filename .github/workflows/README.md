# GitHub Actions workflows

## `build-pipeline-images.yml`

Auto-bygger Docker-images i `docker/<namn>/` när Dockerfile (eller
config-filer i samma katalog) ändras. Pushar till GitHub Container
Registry med [cosign](https://docs.sigstore.dev/cosign/) keyless-signering
via GitHub OIDC.

### Vad workflowen producerar

För varje image i `docker/`:

```
ghcr.io/<owner>/imint-<image>:sha-<7-char-sha>      # alltid (per commit)
ghcr.io/<owner>/imint-<image>:<branch-slug>         # på feature-branch
ghcr.io/<owner>/imint-<image>:latest                # bara på main
```

Plus cosign-signaturer för alla pushade tags. Signaturerna binder
imagen till **just denna workflow-run** via Sigstore Fulcio-certifikat
— det går inte att efter-signera retroaktivt.

### Multi-arch-stöd per image

| Image | linux/amd64 | linux/arm64 | Anledning |
|---|---|---|---|
| `cloud-models` | ✓ | ✓ | python:3.11-slim + CPU-torch är portabel |
| `c2rcc-snap` | ✓ | ✗ | mundialis/esa-snap är amd64-only (ESA SNAP Java + JNI native libs ej arm64) |

### Pull av signerad image

Lokal körning (Mac, k8s, vad som helst):

```bash
# Verifiera att imagen är signerad av denna workflow innan pull
cosign verify ghcr.io/tobiasedman/imint-c2rcc-snap:latest \
  --certificate-identity-regexp "^https://github.com/TobiasEdman/imintengine/" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

# Om verifieringen passar, pulla och kör
docker pull ghcr.io/tobiasedman/imint-c2rcc-snap:latest
docker run --rm ghcr.io/tobiasedman/imint-c2rcc-snap:latest /usr/local/snap/bin/gpt -h
```

### Pinna till digest istället för tag

För maxa reproducerbarhet i k8s-job, pinna till digest, inte tag:

```yaml
# k8s/some-job.yaml
spec:
  containers:
    - image: ghcr.io/tobiasedman/imint-c2rcc-snap@sha256:<digest>
```

Hämta digest:

```bash
docker buildx imagetools inspect ghcr.io/tobiasedman/imint-c2rcc-snap:latest \
  --format '{{json .Manifest}}' | jq -r '.digest'
```

### Trigger-conditions

- **Push till `main`** med ändring i `docker/**` → bygg + push + sign
- **PR mot `main`** med ändring i `docker/**` → bygg + smoke-test (ingen push)
- **`workflow_dispatch`** → manuell trigger för rebuild

### Cache-strategi

Använder GitHub Actions-cache (`cache-from: type=gha`) per image. Sparar
~70% bygg-tid mellan commits som inte rör Dockerfile-layers. Cache är
scoped per image-namn så cloud-models och c2rcc-snap inte bråkar om
samma cache-key.

### Vad som händer på PR

PR-builds skippar push och signering — bara smoke-test körs. Detta
verifierar att Dockerfile bygger utan att förorena GHCR med varje
work-in-progress-commit.

### Felscenarier

| Scenario | Vad workflowen gör |
|---|---|
| Dockerfile-syntaxfel | `build-push-action` failar, exit non-zero |
| Smoke-test failar (gpt -h saknar c2rcc.msi) | Job failar, ingen push sker |
| Cosign-signering failar (OIDC down) | Job failar, image kvar i GHCR utan signatur (måste manuellt rensas eller re-runnas) |
| arm64-emulering tar för lång tid | GitHub free-tier-runners har 360 min cap; om vi når den, byt till `runs-on: ubuntu-24.04-arm` (native arm64-runner från 2025) |

### Varför inte privata cosign-nycklar?

Keyless-signering via OIDC eliminerar nyckelhantering helt. Fördelar:
- Ingen privat nyckel att leak:a, rotera, eller dela med team
- Signaturen binder till en specifik workflow-run-URL — auditable
- Sigstore Rekor (transparency log) ger bevis på vem signerade när

Nackdelen: kort cert-livstid (15 min) → signatur kräver verifiering mot
Rekor-loggen, inte mot ett lokalt cert. För publik open-source kod
(som detta repo) är det rätt tradeoff.
