"""NequIP/Allegro trainer. Finetuning via initialize_from_state."""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from ase import Atoms
from ase.io import write as ase_write

from automlip.config import SystemConfig, TrainerConfig
from automlip.trainers import BaseTrainer, resolve_device

logger = logging.getLogger("automlip.trainers.nequip")


class NequIPTrainer(BaseTrainer):
	def __init__(self, system_config: SystemConfig, trainer_config: TrainerConfig):
		self.system = system_config
		self.config = trainer_config
		self.device = resolve_device(trainer_config)

	def train_committee(self, data: List[Atoms], model_dir: Path,
						n_models: int = 3, restart: bool = False,
						val_data: List[Atoms] = None) -> List[Path]:
		model_dir.mkdir(parents=True, exist_ok=True)

		# Use the explicit validation set when given, otherwise split internally.
		if val_data:
			train = list(data)
			val = list(val_data)
		else:
			rng = np.random.default_rng(self.config.seed)
			indices = rng.permutation(len(data))
			n_val = max(1, int(len(data) * 0.1))
			train = [data[i] for i in indices[:-n_val]]
			val = [data[i] for i in indices[-n_val:]]
		train_path = model_dir / "train.extxyz"
		val_path = model_dir / "val.extxyz"
		ase_write(str(train_path), train, format="extxyz")
		ase_write(str(val_path), val, format="extxyz")

		paths = []
		for i in range(n_models):
			seed = self.config.seed + i
			name = f"committee_{i:02d}"
			member_dir = model_dir / name
			member_dir.mkdir(parents=True, exist_ok=True)

			deployed = self._train_and_deploy(
				train_path, val_path, member_dir, name, seed,
			)
			if deployed:
				paths.append(deployed)
			else:
				logger.error("NequIP member %d failed", i)

		if not paths:
			raise RuntimeError("All NequIP committee members failed")
		return paths

	def get_calculator(self, model_path: Path):
		from nequip.ase import NequIPCalculator
		return NequIPCalculator.from_deployed_model(
			model_path=str(model_path), device=self.device,
		)

	def _train_and_deploy(self, train_path, val_path, model_dir, name, seed):
		config_path = self._write_yaml(
			train_path, val_path, model_dir, name, seed,
		)

		# Train.
		r = subprocess.run(
			f"nequip-train {config_path}",
			shell=True, capture_output=True, text=True, cwd=str(model_dir),
		)
		if r.returncode != 0:
			logger.error("nequip-train failed:\n%s", r.stderr[:500])
			return None

		# Deploy.
		train_dir = model_dir / name
		deployed = model_dir / f"{name}_deployed.pth"
		r = subprocess.run(
			f"nequip-deploy build --train-dir {train_dir} {deployed}",
			shell=True, capture_output=True, text=True, cwd=str(model_dir),
		)
		if r.returncode != 0:
			logger.error("nequip-deploy failed:\n%s", r.stderr[:500])
			return None
		if not deployed.exists():
			return None
		return deployed

	def _write_yaml(self, train_path, val_path, model_dir, name, seed):
		c = self.config
		config = {
			"root": str(model_dir),
			"run_name": name,
			"seed": seed,
			"dataset_seed": seed,
			"r_max": c.nequip_r_max,
			"num_layers": c.nequip_num_layers,
			"l_max": c.nequip_l_max,
			"parity": True,
			"num_features": c.nequip_num_features,
			"num_basis": 8,
			"BesselBasis_trainable": True,
			"PolynomialCutoff_p": 6,
			"invariant_layers": 2,
			"invariant_neurons": 64,
			"avg_num_neighbors": "auto",
			"use_sc": True,
			"chemical_symbols": self.system.elements,
			"dataset": "ase",
			"dataset_file_name": str(train_path),
			"ase_args": {"format": "extxyz"},
			"key_mapping": {"energy": "energy", "forces": "forces"},
			"npz_fixed_field_keys": [],
			"validation_dataset": "ase",
			"validation_dataset_file_name": str(val_path),
			"validation_ase_args": {"format": "extxyz"},
			"n_train": "all",
			"n_val": "all",
			"batch_size": c.nequip_batch_size,
			"validation_batch_size": c.nequip_batch_size,
			"max_epochs": c.nequip_max_epochs,
			"learning_rate": c.nequip_learning_rate,
			"lr_scheduler_name": "ReduceLROnPlateau",
			"lr_scheduler_patience": 50,
			"lr_scheduler_factor": 0.5,
			"early_stopping_patiences": {"validation_loss": 100},
			"early_stopping_lower_bounds": {"LR": 1.0e-6},
			"loss_coeffs": {
				"forces": 100.0,
				"total_energy": [1.0, "PerAtomMSELoss"],
			},
			"default_dtype": "float64",
			"model_dtype": "float32",
			"device": self.device,
			"append": False,
		}

		# Model builders.
		builders = [
			"SimpleIrrepsConfig",
			"EnergyModel",
			"PerSpeciesRescale",
			"ForceOutput",
			"RescaleEnergyEtc",
		]

		# Finetuning: load pretrained weights.
		if c.foundation_model is not None:
			builders.append("initialize_from_state")
			config["initial_model_state"] = str(c.foundation_model)

		config["model_builders"] = builders

		path = model_dir / f"{name}.yaml"
		with open(path, "w") as f:
			yaml.dump(config, f, default_flow_style=False, sort_keys=False)
		return path
