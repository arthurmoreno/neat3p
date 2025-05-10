# Base image
FROM quay.io/pypa/manylinux2014_x86_64

# System tools
RUN yum install -y \
      wget \
      bzip2 \
      git \
      gcc-c++ \
      cmake3 \
      make \
      ninja-build \
    #   lmdb \
    #   lmdb-devel \
    && yum clean all

# Build & install flatbuffers
ARG FLATBUFFERS_VERSION=23.5.26
RUN git clone --depth 1 -b v${FLATBUFFERS_VERSION} \
      https://github.com/google/flatbuffers.git /tmp/flatbuffers \
 && mkdir /tmp/flatbuffers/build \
 && cd /tmp/flatbuffers/build \
 && cmake3 \
     -DFLATBUFFERS_BUILD_TESTS=OFF \
     -DFLATBUFFERS_BUILD_CPP17=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DFLATBUFFERS_ENABLE_PCH=ON \
     -DCMAKE_INSTALL_PREFIX=/usr/local \
     -DCMAKE_INSTALL_LIBDIR=lib \
     .. \
 && make -j"$(nproc)" \
 && make install \
 && rm -rf /tmp/flatbuffers

# Ensure flatc is on PATH
ENV PATH="/usr/local/bin:${PATH}"

# Miniconda installation
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda init bash \
    && $CONDA_DIR/bin/conda config --set always_yes yes --set changeps1 no \
    && $CONDA_DIR/bin/conda update -q conda

# Update PATH
ENV PATH="$CONDA_DIR/bin:$PATH"

# Copy environment spec
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