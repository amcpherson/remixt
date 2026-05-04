# syntax=docker/dockerfile:1
FROM continuumio/miniconda3

LABEL maintainer="Andrew McPherson <andrew.mcpherson@gmail.com>"

ARG REMIXT_VERSION=latest
ARG INSTALL_FROM=local
LABEL version="${REMIXT_VERSION}"

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies (build tools, htslib, boost for shapeit4)
RUN conda install -y python=3.11 && conda clean -afy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates pkg-config \
        libhts-dev libboost-iostreams-dev libboost-program-options-dev \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-gnutls-dev libssl-dev \
        bcftools tabix \
    && rm -rf /var/lib/apt/lists/*

# Build shapeit4
RUN git clone --branch v4.2.2 --depth 1 https://github.com/odelaneau/shapeit4.git /opt/shapeit4 \
    && cd /opt/shapeit4 \
    && sed -i 's|HTSLIB_INC=.*|HTSLIB_INC=/usr/include/htslib|' makefile \
    && sed -i 's|HTSLIB_LIB=.*|HTSLIB_LIB=-lhts|' makefile \
    && sed -i 's|CXXFLAG=-O3 -mavx2 -mfma|CXXFLAG=-O3|' makefile \
    && make -j$(nproc) \
    && cp bin/shapeit4.2 /usr/local/bin/shapeit4

# Build bingraphsample
RUN cd /opt/shapeit4/tools/bingraphsample \
    && sed -i 's|HTSLIB_INC=.*|HTSLIB_INC=/usr/include/htslib|' makefile \
    && sed -i 's|HTSLIB_LIB=.*|HTSLIB_LIB=-lhts|' makefile \
    && sed -i 's|CXXFLAG=.*|CXXFLAG=-O3|' makefile \
    && make -j$(nproc) \
    && cp bin/bingraphsample /usr/local/bin/bingraphsample \
    && rm -rf /opt/shapeit4

# Install remixt (from local source or GitHub depending on INSTALL_FROM)
RUN --mount=type=bind,target=/opt/remixt,rw \
    if [ "$INSTALL_FROM" = "local" ]; then \
        pip install --no-cache-dir /opt/remixt; \
    elif [ "$REMIXT_VERSION" = "latest" ]; then \
        pip install --no-cache-dir "remixt @ git+https://github.com/amcpherson/remixt.git"; \
    else \
        pip install --no-cache-dir "remixt @ git+https://github.com/amcpherson/remixt.git@v${REMIXT_VERSION}"; \
    fi && \
    pip install --no-cache-dir pysam click

# Configure matplotlib for non-interactive use
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

WORKDIR /

