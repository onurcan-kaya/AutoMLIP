"""MACE trainer. Committee built by training with different random seeds."""

import logging, subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from automlip.trainers.base import BaseTrainer, PredictionResult
from automlip.config import TrainerConfig, SystemConfig

logger = logging.getLogger("automlip.trainers.mace")


class MACETrainer(BaseTrainer):
    def __init__(self, system_config: SystemConfig, trainer_config: TrainerConfig):
        self.system_config = system_config
        self.config = trainer_config
        self.device = self.resolve_device(trainer_config)
        logger.info("MACETrainer: r_max=%.1f, channels=%d, device=%s",
                     trainer_config.mace_r_max, trainer_config.mace_num_channels, self.device)

    def train(self, training_data: List[Atoms], model_dir: Path, model_name: str = "model") -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        training_file = model_dir / f"{model_name}_train.extxyz"
        valid_file = model_dir / f"{model_name}_val.extxyz"
        n_val = max(1, int(len(training_data) * 0.1))
        ase_write(str(training_file), training_data[:-n_val], format="extxyz")
        ase_write(str(valid_file), training_data[-n_val:], format="extxyz")
        seed = self.config.seed
        cmd = self._build_command(training_file, valid_file, model_dir, model_name, seed)
        logger.info("Training MACE: %s (seed=%d)", model_name, seed)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(model_dir))
        if result.returncode != 0:
            logger.error("MACE training failed:\n%s", result.stderr[:500])
            raise RuntimeError(f"MACE training failed for {model_name}")
        model_path = model_dir / f"{model_name}.model"
        if not model_path.exists():
            candidates = list(model_dir.glob(f"{model_name}*.model"))
            if candidates:
                model_path = candidates[0]
            else:
                raise RuntimeError(f"No .model file found in {model_dir}")
        return model_path

    def train_committee(self, training_data: List[Atoms], model_dir: Path,
                        n_models: int = 3, bootstrap_fraction: float = 0.8) -> List[Path]:
        model_dir.mkdir(parents=True, exist_ok=True)
        training_file = model_dir / "committee_train.extxyz"
        valid_file = model_dir / "committee_val.extxyz"
        n_val = max(1, int(len(training_data) * 0.1))
        ase_write(str(training_file), training_data[:-n_val], format="extxyz")
        ase_write(str(valid_file), training_data[-n_val:], format="extxyz")
        base_seed = self.config.seed
        model_paths = []
        for i in range(n_models):
            seed = base_seed + i
            member_name = f"committee_{i:02d}"
            member_dir = model_dir / member_name
            member_dir.mkdir(parents=True, exist_ok=True)
            cmd = self._build_command(training_file, valid_file, member_dir, member_name, seed)
            logger.info("Training MACE committee member %d (seed=%d)", i, seed)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(member_dir))
            if result.returncode != 0:
                logger.error("MACE member %d failed:\n%s", i, result.stderr[:500])
                continue
            mp = member_dir / f"{member_name}.model"
            if not mp.exists():
                candidates = list(member_dir.glob(f"{member_name}*.model"))
                if candidates:
                    mp = candidates[0]
                else:
                    continue
            model_paths.append(mp)
        if not model_paths:
            raise RuntimeError("All MACE committee members failed")
        return model_paths

    def get_calculator(self, model_path: Path):
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=str(model_path), device=self.device)

    def predict_committee(self, atoms_list: List[Atoms], model_paths: List[Path],
                          n_cores: int = 1) -> List[PredictionResult]:
        calculators = [self.get_calculator(p) for p in model_paths]
        results = []
        for atoms in atoms_list:
            energies = np.zeros(len(model_paths))
            forces = np.zeros((len(model_paths), len(atoms), 3))
            for j, calc in enumerate(calculators):
                a = atoms.copy(); a.calc = calc
                energies[j] = a.get_potential_energy()
                forces[j] = a.get_forces()
            results.append(PredictionResult(energies=energies, forces=forces))
        return results

    def compute_validation_error(self, validation_data: List[Atoms], model_path: Path) -> Tuple[float, float]:
        calc = self.get_calculator(model_path)
        e_err, f_err = [], []
        for atoms in validation_data:
            ref_e = atoms.info.get("energy"); ref_f = atoms.arrays.get("forces")
            if ref_e is None or ref_f is None: continue
            a = atoms.copy(); a.calc = calc
            pe = a.get_potential_energy(); pf = a.get_forces()
            n = len(atoms)
            e_err.append((pe / n - ref_e / n) ** 2)
            f_err.extend(((pf - ref_f) ** 2).flatten().tolist())
        return float(np.sqrt(np.mean(e_err))) if e_err else float("inf"), \
               float(np.sqrt(np.mean(f_err))) if f_err else float("inf")

    def _build_command(self, training_file, valid_file, model_dir, model_name, seed):
        c = self.config
        return (
            f"{c.mace_run_command} "
            f"--name={model_name} --train_file={training_file} --valid_file={valid_file} "
            f"--model={c.mace_model_type} --max_L={c.mace_max_L} --num_channels={c.mace_num_channels} "
            f"--r_max={c.mace_r_max} --max_num_epochs={c.mace_epochs} --lr={c.mace_lr} "
            f"--batch_size={c.mace_batch_size} --seed={seed} --device={self.device} "
            f"--energy_key=energy --forces_key=forces "
            f"--results_dir={model_dir} --checkpoints_dir={model_dir} --default_dtype=float64"
        )
