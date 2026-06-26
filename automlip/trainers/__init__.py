"""Trainer interface and factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass
class Prediction:
	"""Committee prediction for one structure."""
	energies: np.ndarray   # (n_models,)
	forces: np.ndarray    # (n_models, n_atoms, 3)

	@property
	def mean_force_std(self) -> float:
		return float(np.mean(np.std(self.forces, axis=0)))


class BaseTrainer(ABC):
	@abstractmethod
	def train_committee(self, data: List[Atoms], model_dir: Path,
						n_models: int = 3, restart: bool = False,
						val_data: List[Atoms] = None) -> List[Path]:
		"""Train N models. Returns list of model paths.

		val_data, if given, is used as an explicit validation set instead of
		letting the backend take its own split of the training data.
		"""
		...

	@abstractmethod
	def get_calculator(self, model_path: Path) -> Calculator:
		"""Return an ASE calculator for inference."""
		...

	def train_student(self, data: List[Atoms], model_dir: Path,
					  distill_config, restart: bool = False) -> Path:
		"""Train a small student model for distillation. MACE only by default."""
		raise NotImplementedError(
			"Distillation student training is only implemented for the MACE "
			"backend."
		)

	def predict_committee(self, atoms_list: List[Atoms],
						  model_paths: List[Path]) -> List[Prediction]:
		"""Predict with all committee members on all structures."""
		calcs = [self.get_calculator(p) for p in model_paths]
		results = []
		for atoms in atoms_list:
			n_models = len(calcs)
			energies = np.zeros(n_models)
			forces = np.zeros((n_models, len(atoms), 3))
			for j, calc in enumerate(calcs):
				a = atoms.copy()
				a.calc = calc
				energies[j] = a.get_potential_energy()
				forces[j] = a.get_forces()
			results.append(Prediction(energies=energies, forces=forces))
		return results

	def compute_validation_error(self, val_data: List[Atoms],
								 model_path: Path) -> Tuple[float, float]:
		"""RMSE of energy/atom and forces on a validation set."""
		calc = self.get_calculator(model_path)
		e_err, f_err = [], []
		for atoms in val_data:
			ref_e = _ref_energy(atoms)
			ref_f = _ref_forces(atoms)
			if ref_e is None or ref_f is None:
				continue
			a = atoms.copy()
			a.calc = calc
			pe = a.get_potential_energy()
			pf = a.get_forces()
			n = len(atoms)
			e_err.append((pe / n - ref_e / n) ** 2)
			f_err.extend(((pf - ref_f) ** 2).flatten().tolist())
		rmse_e = float(np.sqrt(np.mean(e_err))) if e_err else float("inf")
		rmse_f = float(np.sqrt(np.mean(f_err))) if f_err else float("inf")
		return rmse_e, rmse_f


def _ref_energy(atoms):
	"""Reference energy from info or, if read from disk, the calculator."""
	if "energy" in atoms.info:
		return atoms.info["energy"]
	res = getattr(atoms.calc, "results", {}) if atoms.calc is not None else {}
	for k in ("energy", "free_energy"):
		if k in res:
			return res[k]
	return None


def _ref_forces(atoms):
	"""Reference forces from arrays or, if read from disk, the calculator."""
	if "forces" in atoms.arrays:
		return atoms.arrays["forces"]
	res = getattr(atoms.calc, "results", {}) if atoms.calc is not None else {}
	return res.get("forces")


def make_trainer(system_config, trainer_config) -> BaseTrainer:
	backend = trainer_config.backend.lower()
	if backend == "mace":
		from automlip.trainers.mace import MACETrainer
		return MACETrainer(system_config, trainer_config)
	elif backend == "nequip":
		from automlip.trainers.nequip import NequIPTrainer
		return NequIPTrainer(system_config, trainer_config)
	else:
		raise ValueError(f"Unknown backend: {backend!r}. Use 'mace' or 'nequip'.")


def resolve_device(config) -> str:
	device = config.device
	if device == "auto":
		try:
			import torch
			if not torch.cuda.is_available():
				raise RuntimeError(
					"No GPU found. Set trainer.device='cpu' to train on CPU, "
					"but GPU is strongly recommended for MACE/NequIP."
				)
			return "cuda"
		except ImportError:
			raise RuntimeError("PyTorch not installed.")
	return device
