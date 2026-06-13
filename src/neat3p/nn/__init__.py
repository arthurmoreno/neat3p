"""
neat3p.nn — PyTorch neural network phenotypes for NEAT genomes.

Install with:  pip install "neat3p[nn]"
"""

try:
    import torch  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "neat3p.nn requires PyTorch.\nInstall it with:  pip install 'neat3p[nn]'\nor install torch directly."
    ) from _e

# Submodules are imported on demand (phenotypes, modules, composite).
# They are not auto-imported here to keep the core nn namespace thin.
