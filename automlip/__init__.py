"""AutoMLIP: Automated training of machine learning interatomic potentials."""

__version__ = "0.2.0"

from automlip.config import Config
from automlip.modes import run_pipeline, train_only

__all__ = ["Config", "run_pipeline", "train_only"]
