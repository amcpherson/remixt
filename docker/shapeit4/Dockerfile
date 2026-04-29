FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        libhts-dev libboost-iostreams-dev libboost-program-options-dev \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-gnutls-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v4.2.2 --depth 1 https://github.com/odelaneau/shapeit4.git /opt/shapeit4

WORKDIR /opt/shapeit4

# Build with portable flags (no AVX2/FMA) for Apple Silicon compatibility
# Use shared htslib to avoid LTO version mismatch with static libhts.a
RUN sed -i 's|HTSLIB_INC=.*|HTSLIB_INC=/usr/include/htslib|' makefile && \
    sed -i 's|HTSLIB_LIB=.*|HTSLIB_LIB=-lhts|' makefile && \
    sed -i 's|CXXFLAG=-O3 -mavx2 -mfma|CXXFLAG=-O3|' makefile && \
    make -j$(nproc) && \
    cp bin/shapeit4.2 /usr/local/bin/shapeit4

ENTRYPOINT ["shapeit4"]
