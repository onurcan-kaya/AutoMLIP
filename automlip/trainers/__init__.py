"""Trainer factory."""

from automlip.trainers.base import BaseTrainer, PredictionResult
from automlip.config import SystemConfig, TrainerConfig

__all__ = ["BaseTrainer", "PredictionResult", "make_trainer"]


def make_trainer(system_config: SystemConfig, trainer_config: TrainerConfig) -> BaseTrainer:
    backend = trainer_config.backend.lower()
    if backend == "gap":
        from automlip.trainers.gap import GAPTrainer
        return GAPTrainer(system_config, trainer_config)
    elif backend == "mace":
        from automlip.trainers.mace import MACETrainer
        return MACETrainer(system_config, trainer_config)
    elif backend == "nequip":
        from automlip.trainers.nequip import NequIPTrainer
        return NequIPTrainer(system_config, trainer_config)
    else:
        raise ValueError(f"Unknown trainer backend: {backend!r}. Supported: gap, mace, nequip")
