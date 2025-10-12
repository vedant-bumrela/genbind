"""
MoleGen Source Package
----------------------

This package contains the core code for the AI-driven molecular generator.

Modules in `src/`:
- config.yaml       -> Configuration settings
- utils.py          -> Helper functions (SMILES/SELFIES handling, RDKit utilities)
- data_prep.py      -> Preprocessing and cleaning raw SMILES
- dataset.py        -> PyTorch Dataset for SELFIES sequences
- models.py         -> Encoder, Decoder, and SeqVAE model definitions
- train.py          -> Training loop, checkpointing, and logging
- sample.py         -> Sampling/generating new molecules
- evaluate.py       -> Compute metrics: validity, uniqueness, novelty, QED, SA

Usage:
    import src
    from src import utils, dataset, models, train, sample, evaluate

Author: Vedant Bumrela
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Vedant Bumrela"
__all__ = ["utils", "data_prep", "dataset", "models", "train", "sample", "evaluate"]
