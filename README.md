# neat3p (Neat-Python Plus Plus)
Neat Python, Amplified by C++
A high-performance, C++-backed Python library for NEAT (NeuroEvolution of Augmenting Topologies) algorithms and beyond.

Overview
neat3p is designed to bring the flexibility of Python together with the performance of C++. By leveraging nanobind for C++/Python interoperability, neat3p aims to provide an efficient implementation of NEAT-based methods with the ease-of-use of a Python API.

Features
High-Performance: Core functionality implemented in modern C++ (C++20) for speed and efficiency.
Python-Friendly API: Exposes a user-friendly interface using nanobind.

## Installation
### Preparing Development

0. Clean env first (Debugging)

```bash
conda deactivate
conda env remove -n neat3p
```

1. Create a Conda Virtual Environment with Python 3.12:

```bash
conda create --name neat3p python=3.12
conda activate neat3p
```

2. Install nanobind via Conda:

(Optional) For some reason nanobind dependency is managed by build + pyproject.toml configurations. So this step can be ignored. But if built manually might be necessary to install.
```bash
conda install -c conda-forge nanobind
```

Required libraries
```bash
conda install --name neat3p -c conda-forge msgpack-c
conda install --name neat3p -c conda-forge spdlog
```

3. Install Python Package Requirements:

Ensure you have a requirements.txt file in your repository root (see below for an example). Then run:

```bash
conda run -n neat3p pip install -r requirements.txt
conda run -n neat3p pip install --upgrade build scikit-build-core
```

### Installing the Library

Before installing, clear any previous build artifacts:

```bash
rm -rf build
```

Then, install the library in your active environment:

```bash
pip install . --verbose
```

### Building and distribution with docker

```bash
docker build -t neat3p-manylinux-builder .
docker run --rm \
  -v "$(pwd)/dist":/project/dist \
  neat3p-manylinux-builder
```

## Usage
Once installed, you can import and use neat3p directly from Python. For example:

```python
import neat3p

# Create a GenomeConfig instance
config = neat3p.GenomeConfig()
config.compatibility_weight_coefficient = 1.5

# You can also work with other exposed classes, e.g., DefaultNodeGene
# node = neat3p.DefaultNodeGene(1)
# print(node.to_string())
```

Check the documentation for more detailed usage instructions and API reference.
