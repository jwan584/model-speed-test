FROM ubuntu:22.04
ARG TARGETARCH
LABEL org.opencontainers.image.title="GDPval timing sandbox" \
      org.opencontainers.image.description="Native-architecture GDPval tool environment"
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential git curl wget unzip zip \
    libreoffice fonts-noto fonts-noto-cjk fonts-liberation \
    ffmpeg imagemagick poppler-utils ghostscript graphviz tesseract-ocr \
    libgl1 libglib2.0-0 libmagic-dev libzbar0 pandoc wkhtmltopdf \
    nodejs npm default-jre-headless r-base ruby-full && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev libgdal-dev gdal-bin libproj-dev python3-fiona pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev && \
    rm -rf /var/lib/apt/lists/*
COPY docker/requirements-gdpval.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt
COPY docker/image-healthcheck.py /usr/local/bin/gdpval-image-healthcheck
RUN chmod 0555 /usr/local/bin/gdpval-image-healthcheck && \
    /usr/local/bin/gdpval-image-healthcheck && \
    useradd --uid 1000 --create-home --shell /bin/bash gdpval
WORKDIR /workspace
USER gdpval
CMD ["sleep", "infinity"]
