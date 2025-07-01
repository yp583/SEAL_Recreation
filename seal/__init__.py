"""
SEAL: Self-Adapting LLMs

A framework that enables language models to improve themselves by generating
their own synthetic data and optimization parameters ("self-edits") in response
to new data.
"""

__version__ = "0.1.0"

from .core.framework import SEALFramework
from .core.self_edit import SelfEditGenerator
from .core.restem import ReSTEMOptimizer

__all__ = [
    "SEALFramework",
    "SelfEditGenerator", 
    "ReSTEMOptimizer",
]