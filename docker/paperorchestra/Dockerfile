# Supply an official Python 3.12 bookworm image by digest and a Debian snapshot.
# Preparation may use the network; all PDF execution is offline and digest-pinned.
ARG PYTHON_BASE
FROM ${PYTHON_BASE}
ARG PYTHON_BASE
ARG DEBIAN_SNAPSHOT
RUN python -c 'import re,sys; assert re.fullmatch(r"(?:public[.]ecr[.]aws/docker/library/)?python:3[.]12[.][0-9]+-slim-bookworm@sha256:[a-f0-9]{64}", sys.argv[1]), "A pinned official Python 3.12 bookworm base is required"' "${PYTHON_BASE}" \
    && python -c 'import re,sys; assert re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", sys.argv[1]), "A dated Debian snapshot is required"' "${DEBIAN_SNAPSHOT}" \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm main\n' "${DEBIAN_SNAPSHOT}" > /etc/apt/sources.list \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm-updates main\n' "${DEBIAN_SNAPSHOT}" >> /etc/apt/sources.list \
    && printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "${DEBIAN_SNAPSHOT}" >> /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended texlive-science poppler-utils \
    && dpkg-query -W > /usr/local/share/scientisttwo-paperorchestra-packages.txt \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp TEXMFVAR=/tmp/texmf-var TEXMFCONFIG=/tmp/texmf-config
USER 65534:65534
WORKDIR /tmp
