"""Abstract interface for interatomic potential trainers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass
class PredictionResult:
    energies: np.ndarray  # (n_models,)
    forces: np.ndarray    # (n_models, n_atoms, 3)

    @property
    def energy_mean(self) -> float: return float(np.mean(self.energies))
    @property
    def energy_std(self) -> float: return float(np.std(self.energies))
    @property
    def forces_mean(self) -> np.ndarray: return np.mean(self.forces, axis=0)
    @property
    def forces_std(self) -> np.ndarray: return np.std(self.forces, axis=0)
    @property
    def mean_force_std(self) -> float: return float(np.mean(self.forces_std))
    @property
    def max_force_std(self) -> float: return float(np.max(self.forces_std))


class BaseTrainer(ABC):
    @staticmethod
    def resolve_device(config) -> str:
        device = getattr(config, "device", "auto")
        backend = getattr(config, "backend", "gap").lower()
        if backend == "gap":
            return "cpu"
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @abstractmethod
    def train(self, training_data: List[Atoms], model_dir: Path, model_name: str = "model") -> Path: ...
    @abstractmethod
    def train_committee(self, training_data: List[Atoms], model_dir: Path,
                        n_models: int = 3, bootstrap_fraction: float = 0.8) -> List[Path]: ...
    @abstractmethod
    def get_calculator(self, model_path: Path) -> Calculator: ...
    @abstractmethod
    def predict_committee(self, atoms_list: List[Atoms], model_paths: List[Path],
                          n_cores: int = 1) -> List[PredictionResult]: ...
    @abstractmethod
    def compute_validation_error(self, validation_data: List[Atoms],
                                 model_path: Path) -> Tuple[float, float]: ...
