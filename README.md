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
1. Create a Conda Virtual Environment with Python 3.12:

```bash
conda create -n neat3p python=3.12
conda activate neat3p
```

2. Install nanobind via Conda:

```bash
conda install -c conda-forge nanobind
```

3. Install Python Package Requirements:

Ensure you have a requirements.txt file in your repository root (see below for an example). Then run:

```bash
pip install -r requirements.txt
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
