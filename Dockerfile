FROM ubuntu:24.04

# avoid prompts during apt installs
ARG DEBIAN_FRONTEND=noninteractive

# 1) Install system build tools + dev‐libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        git \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        libmsgpack-dev \
        libmsgpack-cxx-dev \
        libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Build & install FlatBuffers
ARG FLATBUFFERS_VERSION=24.3.25
RUN git clone --depth 1 -b v${FLATBUFFERS_VERSION} \
      https://github.com/google/flatbuffers.git /tmp/flatbuffers \
    && mkdir /tmp/flatbuffers/build \
    && cd /tmp/flatbuffers/build \
    && cmake \
        -DFLATBUFFERS_BUILD_TESTS=OFF \
        -DFLATBUFFERS_BUILD_CPP17=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DFLATBUFFERS_ENABLE_PCH=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib \
        .. \
    && make -j"$(nproc)" install \
    && rm -rf /tmp/flatbuffers

# ensure flatc is on PATH
ENV PATH="/usr/local/bin:${PATH}"

# 3) Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    # config conda for strict, non-interactive installs
    && $CONDA_DIR/bin/conda config --set always_yes yes --set changeps1 no \
    && $CONDA_DIR/bin/conda config --set channel_priority strict \
    && $CONDA_DIR/bin/conda update -q conda

ENV PATH="$CONDA_DIR/bin:$PATH"

# 4) Copy in your environment spec
WORKDIR /project
COPY . /project/

# Create Conda environment
RUN conda create --name neat3p python=3.12 \
    && conda clean -afy

RUN conda init bash
RUN echo "conda activate neat3p" > ~/.bashrc

RUN conda install --name neat3p -c conda-forge msgpack-c
RUN conda install --name neat3p -c conda-forge spdlog

RUN conda run -n neat3p pip install -r requirements.txt \
    && conda run -n neat3p pip install --upgrade build scikit-build-core

# Defer wheel-building to container runtime
ENTRYPOINT ["conda", "run", "-n", "neat3p", "python", "-m", "build", "/project"]