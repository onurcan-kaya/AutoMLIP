"""AutoMLIP: Automated training of machine learning interatomic potentials."""

__version__ = "0.2.0"

from automlip.config import Config
from automlip.modes import run_pipeline, train_only
from automlip.pipeline import Pipeline

__all__ = ["Config", "Pipeline", "run_pipeline", "train_only"]
