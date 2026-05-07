# ESA SNAP 13.0.0 + c2rcc.msi för Sentinel-2 vattenkvalitetsanalys.
#
# Rekonstruktion av lokal image `imint-snap13:latest` (sha256:832b51265ba8,
# byggd 2026-04-28). Den är image som verifierat fungerade för
# Mollösundscaset (Bohuslän, commit 52d19ae) och som k8s-diag mot
# Lilla Karlsö-SAFE 2025-06-13 (T33VXD) bekräftat har fungerande
# S2OrthoProductReaderPlugIn för EPSG:32633.
#
# Strategin:
#   1. Eclipse Temurin OpenJDK 11 base (officiell, signerad)
#   2. apt: wget + ca-certificates + libgomp1 (SNAP-dependencies)
#   3. Ladda ESA SNAP 13 Sentinel-installer från download.esa.int
#   4. Headless install via varfile till /opt/snap
#   5. Symlinks i /usr/local/bin → /opt/snap/bin/{gpt,snap}
#
# OBS: SNAP installeras i /opt/snap (NOT /usr/local/snap som mundialis-baserade
# images använder). All pipeline-kod måste referera /opt/snap/bin/gpt.
#
# Verifikation:
#   docker run --rm imint-snap-c2rcc:latest /opt/snap/bin/gpt -h c2rcc.msi
#   → "Performs atmospheric correction and IOP retrieval ..."
#
# Build (lokalt, native amd64 eller via docker buildx):
#   docker build --platform=linux/amd64 \
#       -t imint-snap-c2rcc:latest -f docker/c2rcc-snap/Dockerfile .
#
# Beroenden i pipeline:
#   - imint.training.optimal_fetch.optimal_fetch_dates  (datumval)
#   - imint.fetch.fetch_l1c_safe_from_gcp                (SAFE-arkiv)
#   - imint.analyzers.water_quality.analyzer             (chl/TSM/CDOM-formler)

# Eclipse Temurin OpenJDK 11.0.30+7 (LTS, 2026-04). SNAP 13 kräver Java 11+.
# Pinned via tag, men :11.0.30_7-jdk-noble är stabil release-channel.
FROM eclipse-temurin:11.0.30_7-jdk-noble

ENV DEBIAN_FRONTEND=noninteractive

# SNAP-runtime-dependencies. libgomp1 behövs för SNAP:s native libs (GDAL etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Ladda ESA SNAP 13 Sentinel-installer (~1 GB, amd64-only).
# URL från https://step.esa.int/main/download/snap-download/
ADD https://download.esa.int/step/snap/13.0/installers/esa-snap_sentinel_linux-13.0.0.sh \
    /tmp/snap-installer.sh
RUN chmod +x /tmp/snap-installer.sh

# Headless install via varfile. install4j-installer kräver att alla
# parametrar är pre-deklarerade — interaktiva prompts blockerar bygget.
RUN cat > /tmp/snap-install.varfile <<VARS && \
    /tmp/snap-installer.sh -q -varfile /tmp/snap-install.varfile && \
    rm /tmp/snap-installer.sh /tmp/snap-install.varfile
sys.programGroupDisabled\$Boolean=true
sys.languageId=en
sys.installationDir=/opt/snap
sys.symlinkDir=/usr/local/bin
deleteSnapDir=NONE
forcePython=false
VARS

# Lägg /opt/snap/bin först i PATH så `gpt` och `snap` är tillgängliga utan
# absolut sökväg. Java från base-image (/opt/java/openjdk) bibehålls.
ENV PATH="/opt/snap/bin:${PATH}"

# Verifiera att c2rcc.msi-operatorn är tillgänglig — failar bygget om inte.
# Detta fångar både SNAP-install-fel och plugin-registrerings-fel.
# Sentinel-installern inkluderar c2rcc.msi out-of-the-box (s3tbx-paketet
# följer med). Ingen separat plugin-install behövs.
RUN /opt/snap/bin/gpt -h 2>&1 | grep -E "c2rcc.msi" || \
    (echo "c2rcc.msi MISSING in SNAP 13 install" && exit 1)

WORKDIR /work

# Default till sh så `docker run --rm <image> /opt/snap/bin/gpt ...` fungerar
# direkt. Pipelinekoden kallar gpt med absolut sökväg.
CMD ["/bin/sh"]
