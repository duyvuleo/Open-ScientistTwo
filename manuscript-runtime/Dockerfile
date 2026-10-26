# Supply a verified Python Debian image by digest explicitly at build time.
# Builds may retrieve packages; execution remains offline and uses the resulting
# immutable image digest. Record the base digest and installed package inventory.
ARG PYTHON_BASE
FROM ${PYTHON_BASE}
ARG PYTHON_BASE
ARG DEBIAN_SNAPSHOT
RUN python -c 'import re,sys; assert re.fullmatch(r"(?:public[.]ecr[.]aws/docker/library/)?python:3[.]12[.][0-9]+-slim-bookworm@sha256:[a-f0-9]{64}", sys.argv[1]), "A pinned official Python 3.12 bookworm base is required"' "${PYTHON_BASE}" \
    && test -n "${DEBIAN_SNAPSHOT}" \
    && case "${DEBIAN_SNAPSHOT}" in *[!0-9TZ]*) exit 1 ;; esac \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm main\n' "${DEBIAN_SNAPSHOT}" > /etc/apt/sources.list \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm-updates main\n' "${DEBIAN_SNAPSHOT}" >> /etc/apt/sources.list \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "${DEBIAN_SNAPSHOT}" >> /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-fonts-recommended \
    && dpkg-query -W > /usr/local/share/scientisttwo-manuscript-packages.txt \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp TEXMFVAR=/tmp/texmf-var TEXMFCONFIG=/tmp/texmf-config
USER 65534:65534
WORKDIR /tmp
