"""
NequIP trainer backend.

Trains E(3)-equivariant neural network potentials using nequip-train.
Committee is built by training N models with different random seeds
on the same data (Option A: seed diversity, no bootstrap).

Requires:
    - nequip >= 0.6.0  (pip install nequip)
    - PyTorch with CUDA (for GPU training)

The workflow per model:
    1. Write a YAML config with the appropriate seed.
    2. Call nequip-train config.yaml (via subprocess).
    3. Call nequip-deploy build to produce a TorchScript .pth file.
    4. Load the deployed model via NequIPCalculator for inference.
"""

import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write

from automlip.trainers.base import BaseTrainer, PredictionResult
from automlip.config import TrainerConfig, SystemConfig

logger = logging.getLogger("automlip.trainers.nequip")


def _resolve_device(config: TrainerConfig) -> str:
    """
    Resolve device string.

    "auto" -> "cuda" if available, else "cpu".
    Anything else is passed through as-is.
    """
    device = getattr(config, "device", "auto")
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class NequIPTrainer(BaseTrainer):
    """
    NequIP trainer with seed-based committee construction.

    Committee members are trained on the same data with different
    random seeds. Each converges to a different local minimum,
    providing diverse predictions for query-by-committee.
    """

    def __init__(
        self,
        system_config: SystemConfig,
        trainer_config: TrainerConfig,
    ):
        self.system_config = system_config
        self.config = trainer_config
        self.device = _resolve_device(trainer_config)

        # NequIP-specific hyperparameters with defaults.
        self.r_max = getattr(trainer_config, "nequip_r_max", 5.0)
        self.num_layers = getattr(trainer_config, "nequip_num_layers", 4)
        self.l_max = getattr(trainer_config, "nequip_l_max", 2)
        self.num_features = getattr(trainer_config, "nequip_num_features", 32)
        self.num_basis = getattr(trainer_config, "nequip_num_basis", 8)
        self.max_epochs = getattr(trainer_config, "nequip_max_epochs", 1000)
        self.learning_rate = getattr(trainer_config, "nequip_learning_rate", 0.005)
        self.batch_size = getattr(trainer_config, "nequip_batch_size", 5)
        self.loss_energy_weight = getattr(
            trainer_config, "nequip_loss_energy_weight", 1.0
        )
        self.loss_force_weight = getattr(
            trainer_config, "nequip_loss_force_weight", 100.0
        )

        logger.info(
            "NequIPTrainer initialised: r_max=%.1f, num_layers=%d, "
            "l_max=%d, num_features=%d, device=%s",
            self.r_max, self.num_layers, self.l_max,
            self.num_features, self.device,
        )

    def _write_config(
        self,
        seed: int,
        train_path: Path,
        val_path: Path,
        model_dir: Path,
        model_name: str,
    ) -> Path:
        """
        Generate a NequIP YAML config file.

        Returns path to the written config.
        """
        chemical_symbols = self.system_config.elements

        config = {
            "root": str(model_dir),
            "run_name": model_name,
            "seed": seed,
            "dataset_seed": seed,

            # Network architecture.
            "model_builders": [
                "SimpleIrrepsConfig",
                "EnergyModel",
                "PerSpeciesRescale",
                "ForceOutput",
                "RescaleEnergyEtc",
            ],
            "r_max": self.r_max,
            "num_layers": self.num_layers,
            "l_max": self.l_max,
            "parity": True,
            "num_features": self.num_features,
            "num_basis": self.num_basis,
            "BesselBasis_trainable": True,
            "PolynomialCutoff_p": 6,
            "invariant_layers": 2,
            "invariant_neurons": 64,
            "avg_num_neighbors": "auto",
            "use_sc": True,

            # Chemical species.
            "chemical_symbols": chemical_symbols,

            # Data.
            "dataset": "ase",
            "dataset_file_name": str(train_path),
            "ase_args": {
                "format": "extxyz",
            },
            "key_mapping": {
                "energy": "energy",
                "forces": "forces",
            },
            "npz_fixed_field_keys": [],

            # Validation.
            "validation_dataset": "ase",
            "validation_dataset_file_name": str(val_path),
            "validation_ase_args": {
                "format": "extxyz",
            },

            # Training.
            "n_train": "all",
            "n_val": "all",
            "batch_size": self.batch_size,
            "validation_batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "lr_scheduler_name": "ReduceLROnPlateau",
            "lr_scheduler_patience": 50,
            "lr_scheduler_factor": 0.5,
            "early_stopping_patiences": {"validation_loss": 100},
            "early_stopping_lower_bounds": {"LR": 1.0e-6},

            # Loss.
            "loss_coeffs": {
                "forces": self.loss_force_weight,
                "total_energy": [self.loss_energy_weight, "PerAtomMSELoss"],
            },

            # Metrics.
            "metrics_components": [
                ["forces", "mae"],
                ["forces", "rmse"],
                ["total_energy", "mae", {"PerAtom": True}],
                ["total_energy", "rmse", {"PerAtom": True}],
            ],

            # Misc.
            "default_dtype": "float64",
            "model_dtype": "float32",
            "device": self.device,
            "log_batch_freq": 100,
            "verbose": "info",
            "append": False,
        }

        config_path = model_dir / f"{model_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.debug("Wrote NequIP config: %s", config_path)
        return config_path

    def _train_single(
        self,
        train_path: Path,
        val_path: Path,
        model_dir: Path,
        model_name: str,
        seed: int,
    ) -> Optional[Path]:
        """
        Train one NequIP model and deploy it.

        Returns path to the deployed .pth file, or None on failure.
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._write_config(
            seed, train_path, val_path, model_dir, model_name,
        )

        # ---- Train ----
        train_cmd = f"nequip-train {config_path}"
        logger.info("Training NequIP: %s (seed=%d)", model_name, seed)

        try:
            result = subprocess.run(
                train_cmd, shell=True, capture_output=True, text=True,
                cwd=str(model_dir),
            )
            if result.returncode != 0:
                logger.error(
                    "NequIP training failed for %s:\n%s",
                    model_name, result.stderr[:1000],
                )
                return None
        except Exception as e:
            logger.error("NequIP training exception: %s", e)
            return None

        # ---- Deploy ----
        # NequIP stores the trained model under root/run_name/
        train_dir = model_dir / model_name
        deployed_path = model_dir / f"{model_name}_deployed.pth"

        deploy_cmd = (
            f"nequip-deploy build "
            f"--train-dir {train_dir} "
            f"{deployed_path}"
        )

        try:
            result = subprocess.run(
                deploy_cmd, shell=True, capture_output=True, text=True,
                cwd=str(model_dir),
            )
            if result.returncode != 0:
                logger.error(
                    "NequIP deploy failed for %s:\n%s",
                    model_name, result.stderr[:1000],
                )
                return None
        except Exception as e:
            logger.error("NequIP deploy exception: %s", e)
            return None

        if not deployed_path.exists():
            logger.error("Deployed model not found: %s", deployed_path)
            return None

        logger.info("NequIP trained and deployed: %s", deployed_path)
        return deployed_path

    def train(
        self,
        training_data: List[Atoms],
        model_dir: Path,
        model_name: str = "model",
    ) -> Path:
        """Train a single NequIP potential."""
        model_dir.mkdir(parents=True, exist_ok=True)

        train_path, val_path = self._split_and_save(
            training_data, model_dir, model_name,
        )

        base_seed = getattr(self.config, "seed", 42) if hasattr(self.config, "seed") else 42

        result = self._train_single(
            train_path, val_path, model_dir, model_name, seed=base_seed,
        )

        if result is None:
            raise RuntimeError(f"NequIP training failed for {model_name}")

        return result

    def train_committee(
        self,
        training_data: List[Atoms],
        model_dir: Path,
        n_models: int = 3,
        bootstrap_fraction: float = 0.8,
    ) -> List[Path]:
        """
        Train a committee of NequIP models with different seeds.

        All models see the same training data. Diversity comes from
        different random initialisations and batch orderings.

        bootstrap_fraction is accepted for API compatibility with
        BaseTrainer but is ignored (no data resampling).
        """
        model_dir.mkdir(parents=True, exist_ok=True)

        train_path, val_path = self._split_and_save(
            training_data, model_dir, "committee",
        )

        base_seed = getattr(self.config, "seed", 42) if hasattr(self.config, "seed") else 42
        model_paths = []

        for i in range(n_models):
            seed = base_seed + i
            member_name = f"committee_{i:02d}"
            member_dir = model_dir / member_name

            result = self._train_single(
                train_path, val_path, member_dir, member_name, seed=seed,
            )

            if result is None:
                logger.error(
                    "Committee member %d failed, skipping", i,
                )
                continue

            model_paths.append(result)

        if not model_paths:
            raise RuntimeError("All NequIP committee members failed to train")

        if len(model_paths) < n_models:
            logger.warning(
                "Only %d/%d committee members trained successfully",
                len(model_paths), n_models,
            )

        return model_paths

    def get_calculator(self, model_path: Path) -> Calculator:
        """Return an ASE Calculator for a deployed NequIP model."""
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(
            model_path=str(model_path),
            device=self.device,
        )

    def predict_committee(
        self,
        atoms_list: List[Atoms],
        model_paths: List[Path],
        n_cores: int = 1,
    ) -> List[PredictionResult]:
        """
        Evaluate all committee members on all structures.

        Runs sequentially since NequIP inference uses GPU and
        parallelising across models on a single GPU has no benefit.
        """
        n_models = len(model_paths)
        results = []

        # Load all calculators once.
        calculators = [self.get_calculator(p) for p in model_paths]

        for atoms in atoms_list:
            energies = np.zeros(n_models)
            forces = np.zeros((n_models, len(atoms), 3))

            for j, calc in enumerate(calculators):
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc
                energies[j] = atoms_copy.get_potential_energy()
                forces[j] = atoms_copy.get_forces()

            results.append(PredictionResult(
                energies=energies,
                forces=forces,
            ))

        return results

    def compute_validation_error(
        self,
        validation_data: List[Atoms],
        model_path: Path,
    ) -> Tuple[float, float]:
        """Compute RMSE of energy/atom and forces on a validation set."""
        calc = self.get_calculator(model_path)

        energy_errors = []
        force_errors = []

        for atoms in validation_data:
            ref_energy = atoms.info.get("energy", atoms.info.get("REF_energy"))
            ref_forces = atoms.arrays.get("forces", atoms.arrays.get("REF_forces"))

            if ref_energy is None or ref_forces is None:
                continue

            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            pred_energy = atoms_copy.get_potential_energy()
            pred_forces = atoms_copy.get_forces()

            n_atoms = len(atoms)
            energy_errors.append(
                (pred_energy / n_atoms - ref_energy / n_atoms) ** 2
            )
            force_errors.extend(
                ((pred_forces - ref_forces) ** 2).flatten().tolist()
            )

        rmse_energy = float(np.sqrt(np.mean(energy_errors))) if energy_errors else float("inf")
        rmse_forces = float(np.sqrt(np.mean(force_errors))) if force_errors else float("inf")

        logger.info(
            "Validation RMSE: energy=%.4f eV/atom, forces=%.4f eV/A",
            rmse_energy, rmse_forces,
        )

        return rmse_energy, rmse_forces

    def _split_and_save(
        self,
        data: List[Atoms],
        out_dir: Path,
        prefix: str,
        val_fraction: float = 0.1,
    ) -> Tuple[Path, Path]:
        """
        Split data into train/val and write as extxyz.

        Returns (train_path, val_path).
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        n = len(data)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val

        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(n)

        train_atoms = [data[i] for i in indices[:n_train]]
        val_atoms = [data[i] for i in indices[n_train:]]

        train_path = out_dir / f"{prefix}_train.extxyz"
        val_path = out_dir / f"{prefix}_val.extxyz"

        ase_write(str(train_path), train_atoms, format="extxyz")
        ase_write(str(val_path), val_atoms, format="extxyz")

        logger.info(
            "Data split: %d train, %d val -> %s, %s",
            len(train_atoms), len(val_atoms), train_path, val_path,
        )

        return train_path, val_path
