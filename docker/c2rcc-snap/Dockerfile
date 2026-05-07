# ESA SNAP 13 + c2rcc.msi för Sentinel-2 vattenkvalitetsanalys.
#
# Rekonstruktion av lokal image `imint-snap-c2rcc:latest` (sha256:80a519fe8023)
# vars Dockerfile försvann från versionkontroll. Strategin här är annorlunda
# än originalet (som kopierade /usr/local/snap från en host-installation):
# vi börjar från en officiell mundialis/esa-snap base som redan har SNAP
# installerat via ESA:s officiella installer, och lägger till s3tbx.c2rcc-
# pluginen.
#
# Verifikation av ekvivalens med originalet:
#   docker run --rm imint-snap-c2rcc:latest /usr/local/snap/bin/gpt -h \
#     | grep -E "c2rcc.msi"
# Båda images ska ha samma c2rcc.msi-operator tillgänglig.
#
# Build:
#   docker build -t imint-snap-c2rcc:latest -f docker/c2rcc-snap/Dockerfile .
#
# Beroenden i pipeline:
#   - imint.training.optimal_fetch.optimal_fetch_dates  (datumval)
#   - imint.fetch.fetch_l1c_safe_from_gcp                (SAFE-arkiv)
#   - imint.analyzers.water_quality.analyzer             (chl/TSM/CDOM-formler)

# Pinned via digest istället för :latest för reproducerbara builds.
# Image: mundialis/esa-snap (taggad latest 2026-03-26, linux/amd64).
# För att uppdatera: `docker pull mundialis/esa-snap:latest && docker
# inspect <id> --format '{{index .RepoDigests 0}}'`.
FROM mundialis/esa-snap@sha256:056f971a993907cee0d24e8d94c1342fa86d4017ca1a757ffcbea437365deb35

# SNAP-pluginer för Sentinel-2 + C2RCC-retrieval.
# `--update-all` säkerställer att modulindex är fresh innan install.
RUN /usr/local/snap/bin/snap --nosplash --nogui --modules --update-all && \
    /usr/local/snap/bin/snap --nosplash --nogui --modules --install \
        org.esa.s2tbx.s2tbx.kit \
        org.esa.s3tbx.s3tbx.c2rcc 2>&1 | tail -15

# Verifiera att c2rcc.msi-operatorn registrerades — failar bygget om inte.
RUN /usr/local/snap/bin/gpt -h 2>&1 | grep -E "c2rcc.msi" || \
    (echo "c2rcc.msi MISSING after plugin install" && \
     /usr/local/snap/bin/snap --nosplash --nogui --modules --list-installed 2>&1 | grep c2rcc && \
     exit 1)

# CLI-tools default ENV (LD_LIBRARY_PATH speglar originalet)
ENV LD_LIBRARY_PATH=.:
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/snap/bin"

# Default till sh; pipelinen anropar `gpt <graph.xml>` direkt.
CMD ["/bin/sh"]
