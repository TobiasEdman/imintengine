# Sen2Cor v2.12.04 — ESA's official Sentinel-2 L1C → L2A atmospheric
# correction processor. Standalone Linux64 installer from STEP.
#
# Used for back-filling frame_2016 / frame_2015 tiles where DES openEO
# has no L2A indexed (verified 2026-04-29 via probe-2017-stac-2gv4m
# and probe-l1c-2016-lktlw: CDSE STAC has 27 L1C scenes for summer
# 2016 over our test bbox, none in L2A).
#
# Pin: FROM digest, installer SHA256 (capture after first build).
# Smoke: run --help to verify binary is wired into PATH.
# Manifest: see docker/sen2cor/MANIFEST.json.

# Ubuntu 22.04 LTS — long support, glibc 2.35 compatible with sen2cor
# 02.12.04. amd64-only — sen2cor is shipped as a Linux64 .run installer
# with bundled x86_64 binaries; there is no ARM64 build. The
# ``--platform=linux/amd64`` is needed when building on Apple Silicon;
# k8s nodes are already amd64 and ignore the directive.
FROM --platform=linux/amd64 ubuntu:22.04@sha256:962f6cadeae0ea6284001009daa4cc9a8c37e75d1f5191cf0eb83fe565b63dd7

LABEL maintainer="ImintEngine pipeline"
LABEL sen2cor_version="2.12.04"
LABEL installer_url="https://step.esa.int/thirdparties/sen2cor/2.12.0/Sen2Cor-02.12.04-Linux64.run"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Sen2cor needs: bash, wget for installer, libxext6 for the GDAL
# X11 stub, libglib2.0-0 for GTK (sen2cor's PyQt remnants), libsm6/
# libxrender1 for the GDAL+rasterio path. NO graphical display.
#
# IMPORTANT: sen2cor 2.12.04 advertises a self-contained python
# runtime, but its bin/python2.7 is a symlink that resolves against
# /usr/bin/python2.7 via the system loader. Ubuntu 22.04 has no
# python2 by default. The 'python2' package lives in the 'universe'
# component — we enable that, then install python2 + python2.7-dev
# so the symlink resolves and L2A_Process can start.
#
# Verified 2026-05-15 against build-sen2cor-image-fjhx6 which failed
# with "bin/python2.7: not found" until python2 was added.
RUN sed -i 's|^# deb http|deb http|g; s|^# deb-src http|deb-src http|g' /etc/apt/sources.list \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends software-properties-common \
 && add-apt-repository -y universe \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        libxext6 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        python2 \
        python2.7-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Download the installer. We verify the file size matches what we
# observed at build time (264 206 010 bytes per 2026-02-10 release).
# If the upstream file changes, the build fails — we want to know.
ARG SEN2COR_INSTALLER_URL="https://step.esa.int/thirdparties/sen2cor/2.12.0/Sen2Cor-02.12.04-Linux64.run"
ARG SEN2COR_INSTALLER_SIZE=264206010

RUN wget --no-verbose -O /tmp/sen2cor.run "${SEN2COR_INSTALLER_URL}" \
 && actual_size=$(stat -c %s /tmp/sen2cor.run) \
 && if [ "$actual_size" != "$SEN2COR_INSTALLER_SIZE" ]; then \
        echo "FAIL: sen2cor installer size mismatch. expected=${SEN2COR_INSTALLER_SIZE} actual=${actual_size}" >&2; \
        exit 1; \
    fi \
 && chmod +x /tmp/sen2cor.run

# Install + smoke-test in a single RUN to avoid kaniko's snapshot
# layer between them losing the bundled musl-libc Python binary
# (verified 2026-05-15 via probe-sen2cor-ldd: the bundled
# bin/python2.7 is a musl-linked PIE that resolves correctly when
# L2A_Bashrc is sourced, but kaniko's redo snapshot mode appears
# to drop the binary on the next layer boundary so a subsequent
# RUN sees it as missing).
#
# Sen2cor's makeself extracts to /opt/Sen2Cor-02.12.04-Linux64/.
# L2A_Bashrc inside that dir sets PATH + GDAL_DATA + locale; it
# must be sourced before invoking python2.7, otherwise the bundled
# musl loader cannot find libpython2.7.so.1.0 / libc.musl-x86_64.
RUN /tmp/sen2cor.run --target /opt/Sen2Cor-02.12.04-Linux64 \
 && rm -f /tmp/sen2cor.run \
 && . /opt/Sen2Cor-02.12.04-Linux64/L2A_Bashrc \
 && L2A_Process --help > /tmp/help.txt \
 && grep -q "Sentinel-2 Level 2A Prototype Processor" /tmp/help.txt \
 && rm /tmp/help.txt

# Put L2A_Process on PATH for downstream consumers — Bashrc also
# does this when sourced, but adding it to ENV makes ``docker run``
# work without an explicit source step.
ENV PATH=/opt/Sen2Cor-02.12.04-Linux64/bin:${PATH}
ENV GDAL_DATA=/opt/Sen2Cor-02.12.04-Linux64/share/gdal
ENV GDAL_DRIVER_PATH=disable
ENV LC_NUMERIC=C

WORKDIR /work
ENTRYPOINT ["L2A_Process"]
CMD ["--help"]
