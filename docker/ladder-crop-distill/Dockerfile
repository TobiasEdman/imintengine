FROM python:3.11-slim@sha256:d1e9ca7c4e78d1e8ecadb5d44bfc8e956e7a65b659a9950f569f243d72b326d0 AS sources

ARG SOURCE_GIT_SHA

# Refuse an omitted, abbreviated, uppercase, or otherwise ambiguous source
# identity before Docker attempts the codeload request below.
RUN python -c "import re; value='${SOURCE_GIT_SHA}'; assert re.fullmatch(r'[0-9a-f]{40}', value), 'SOURCE_GIT_SHA must be one exact 40-hex commit'"

# All three source trees are content-addressed. CROMA and Clay have
# independently recorded archive hashes; the ImintEngine archive is anchored
# by SOURCE_GIT_SHA and sealed to a canonical 64-hex payload hash. The large
# repository archive remains in this disposable stage.
ADD https://codeload.github.com/TobiasEdman/ImintEngine/tar.gz/${SOURCE_GIT_SHA} /tmp/imintengine.tar.gz
ADD https://codeload.github.com/antofuller/CROMA/tar.gz/59505a6bcadbf36ba20767270154bf9f3067c5e7 /tmp/croma.tar.gz
ADD https://codeload.github.com/Clay-foundation/model/tar.gz/f14e698f3c237cabf8d28dec669a362d66625381 /tmp/clay.tar.gz

COPY docker/ladder-crop-distill/prepare_runtime.py /tmp/prepare_runtime.py

RUN python /tmp/prepare_runtime.py extract \
      --source-git-sha "${SOURCE_GIT_SHA}" \
      --source-archive /tmp/imintengine.tar.gz \
      --source-root /opt/imintengine \
      --croma-archive /tmp/croma.tar.gz \
      --croma-root /opt/CROMA \
      --clay-archive /tmp/clay.tar.gz \
      --clay-root /opt/clay \
      --archive-identities /opt/provenance/archive-identities.json \
    && rm /tmp/imintengine.tar.gz /tmp/croma.tar.gz /tmp/clay.tar.gz


FROM python:3.11-slim@sha256:d1e9ca7c4e78d1e8ecadb5d44bfc8e956e7a65b659a9950f569f243d72b326d0 AS runtime

ARG SOURCE_GIT_SHA

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/imintengine:/opt/CROMA:/opt/clay \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline

LABEL org.opencontainers.image.title="ImintEngine LUCAS crop-distill runtime" \
      org.opencontainers.image.source="https://github.com/TobiasEdman/ImintEngine" \
      org.opencontainers.image.revision="${SOURCE_GIT_SHA}" \
      imint.crop-distill.croma.revision="59505a6bcadbf36ba20767270154bf9f3067c5e7" \
      imint.crop-distill.clay.revision="f14e698f3c237cabf8d28dec669a362d66625381" \
      imint.crop-distill.terratorch.version="1.2.11"

RUN python -c "import re; value='${SOURCE_GIT_SHA}'; assert re.fullmatch(r'[0-9a-f]{40}', value), 'SOURCE_GIT_SHA must be one exact 40-hex commit'"

# Rasterio's manylinux wheel links against Expat and the numerical stack uses
# the GNU OpenMP runtime. The slim base deliberately omits both. Fetch the
# exact Debian artifacts directly and verify their repository SHA-256 values;
# an apt index update would otherwise make this supposedly frozen image vary.
ADD https://deb.debian.org/debian-security/pool/updates/main/e/expat/libexpat1_2.8.3-1~deb13u1_amd64.deb /tmp/libexpat1.deb
ADD https://deb.debian.org/debian/pool/main/g/gcc-14/libgomp1_14.2.0-19_amd64.deb /tmp/libgomp1.deb
RUN echo "38abe0e710a07688e9c149d74536e67cfee0364bdb64dd6d644c32a1cfad389f  /tmp/libexpat1.deb" | sha256sum --check --strict \
    && echo "72b7a1a33c48f3195efeb044111b34ae684b1e640bb6495c832443b798601cd5  /tmp/libgomp1.deb" | sha256sum --check --strict \
    && dpkg --install /tmp/libexpat1.deb /tmp/libgomp1.deb \
    && rm /tmp/libexpat1.deb /tmp/libgomp1.deb

# Only the selected runtime code and the exact model source trees enter the
# final image; docs, tracked showcase assets, data, and the codeload archives
# remain behind in the disposable stage.
COPY --from=sources /opt/imintengine /opt/imintengine
COPY --from=sources /opt/CROMA /opt/CROMA
COPY --from=sources /opt/clay /opt/clay
COPY --from=sources /opt/provenance/archive-identities.json /opt/provenance/archive-identities.json
COPY docker/ladder-crop-distill/prepare_runtime.py /tmp/prepare_runtime.py
COPY docker/ladder-crop-distill/runtime_smoke.py /tmp/runtime_smoke.py
COPY docker/ladder-crop-distill/requirements-model.lock /opt/provenance/model-requirements.lock
COPY docker/ladder-crop-distill/requirements-scoring.lock /opt/provenance/scoring-requirements.lock

# Keep the TerraTorch model stack and NumPy-1.26 scoring stack in distinct,
# stable environments.  The Python-3.11 training timestamp resolves to
# TerraTorch 1.2.11 + TorchGeo 0.8.1 and requires NumPy >=2.2; scoring
# intentionally remains on NumPy 1.26.4.
# Both lockfiles contain every transitive version and accepted distribution
# hash. Torch and torchvision use exact official CPython-3.11/linux-x86_64
# cu121 wheel URLs, so no index can silently substitute another build.
RUN python -m venv /opt/venvs/model \
    && /opt/venvs/model/bin/python -m pip install \
      --require-hashes --no-build-isolation \
      -r /opt/provenance/model-requirements.lock \
    && /opt/venvs/model/bin/python -m pip install \
      --no-deps --no-build-isolation /opt/clay \
    && /opt/venvs/model/bin/python -c "from pathlib import Path; import shutil; dst=Path('/data/model_cache/clay_metadata.yaml'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile('/opt/clay/configs/metadata.yaml', dst)" \
    && /opt/venvs/model/bin/python -m pip check \
    && /opt/venvs/model/bin/python -m pip freeze --all \
      > /tmp/model-pip-freeze.unsorted \
    && LC_ALL=C sort /tmp/model-pip-freeze.unsorted \
      > /opt/provenance/model-pip-freeze.txt \
    && rm /tmp/model-pip-freeze.unsorted

RUN python -m venv /opt/venvs/scoring \
    && /opt/venvs/scoring/bin/python -m pip install \
      --require-hashes --no-build-isolation \
      -r /opt/provenance/scoring-requirements.lock \
    && /opt/venvs/scoring/bin/python -m pip check \
    && /opt/venvs/scoring/bin/python -m pip freeze --all \
      > /tmp/scoring-pip-freeze.unsorted \
    && LC_ALL=C sort /tmp/scoring-pip-freeze.unsorted \
      > /opt/provenance/scoring-pip-freeze.txt \
    && rm /tmp/scoring-pip-freeze.unsorted

# Exercise both exact upstream packages through ImintEngine's checkpoint-
# reconstruction path with all model hubs forced offline.  Separately import
# the exact scoring ABI and compile every script used by either environment.
RUN /opt/venvs/model/bin/python /tmp/runtime_smoke.py model \
    && /opt/venvs/scoring/bin/python /tmp/runtime_smoke.py scoring

# Seal after installation and smoke so runtime validation sees the exact final
# filesystem state. PYTHONDONTWRITEBYTECODE prevents imports from dirtying the
# sealed trees with __pycache__ files.
RUN python /tmp/prepare_runtime.py seal \
      --source-root /opt/imintengine \
      --croma-root /opt/CROMA \
      --clay-root /opt/clay \
      --archive-identities /opt/provenance/archive-identities.json \
      --base-python /usr/local/bin/python \
      --model-python /opt/venvs/model/bin/python \
      --model-requirements-lock /opt/provenance/model-requirements.lock \
      --model-pip-freeze /opt/provenance/model-pip-freeze.txt \
      --scoring-python /opt/venvs/scoring/bin/python \
      --scoring-requirements-lock /opt/provenance/scoring-requirements.lock \
      --scoring-pip-freeze /opt/provenance/scoring-pip-freeze.txt \
      --runtime-manifest /opt/provenance/runtime.json \
    && python /opt/imintengine/scripts/crop_distill_provenance.py verify-runtime \
      --source-git-sha "${SOURCE_GIT_SHA}" \
      --image-ref "local-build@sha256:0000000000000000000000000000000000000000000000000000000000000000" \
      --runtime-manifest /opt/provenance/runtime.json \
    && rm /tmp/prepare_runtime.py /tmp/runtime_smoke.py \
      /opt/provenance/archive-identities.json

# Installation-time network access ends at the image boundary.  Runtime Jobs
# cannot ask pip or model hubs for mutable dependencies or upstream weights.
ENV PIP_NO_INDEX=1

WORKDIR /opt/imintengine
# Split and crop consumers run as this fixed unprivileged identity.  A
# separate one-shot storage-prep Job overrides only UID (not GID) to migrate
# three exact 0755 root:root PVC children before any data work begins.
USER 2000:2000
CMD ["/usr/local/bin/python", "-c", "import json; print(json.dumps(json.load(open('/opt/provenance/runtime.json')), indent=2, sort_keys=True))"]
