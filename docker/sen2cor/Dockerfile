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

# Sen2cor 2.12.04 IS fully self-contained: bin/python2.7 is a
# musl-libc-linked PIE binary with its own bundled libpython2.7.so
# under lib/. It does NOT depend on system python2.
#
# The makeself installer runs a post-extract relocation step that
# uses ``patchelf`` to rewrite each bundled binary's ELF interpreter
# path (the musl loader ld-musl-x86_64.so.1) to the chosen --target
# directory. That relocation script invokes the ``file`` command to
# discover which extracted files are ELF binaries. If ``file`` is
# absent, patchelf silently never runs, the interpreter path stays
# at the packaging-time default, and exec'ing bin/python2.7 fails
# with ENOENT → the shell reports "python2.7: not found" (exit 127).
#
# Verified 2026-05-15: build-sen2cor-image-{fjhx6,t8v82} failed with
# "bin/python2.7: not found" because ``file`` was missing; the
# probe-sen2cor-ldd job had ``file`` installed and there the same
# binary ran cleanly. The earlier python2/universe "fix" was a
# misdiagnosis and is removed.
#
# Runtime deps: wget+bzip2+ca-certificates for the installer,
# libxext6/libglib2.0-0/libsm6/libxrender1 for the GDAL+rasterio
# path, file for the installer's patchelf relocation.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        file \
        libxext6 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
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

# Install + smoke-test in a single RUN. ``file`` (installed above)
# lets the installer's patchelf step rewrite the musl ELF
# interpreter path to this --target. L2A_Bashrc sets PATH +
# GDAL_DATA + locale and must be sourced before L2A_Process.
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
