"""Wasserstein DRO two-stage multi-objective model for multi-product bulk
transportation under demand uncertainty."""

from .config import InstanceConfig, DROConfig, SCALE_PRESETS
from .data_generation import generate_instance, Instance
from .dro_model import DROSolver
from .recourse import evaluate_plan

__all__ = [
    "InstanceConfig", "DROConfig", "SCALE_PRESETS",
    "generate_instance", "Instance", "DROSolver", "evaluate_plan",
]
