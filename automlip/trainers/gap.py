"""GAP trainer. Committee built via bootstrap resampling of training data."""

import logging, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from ase import Atoms
from ase.io import write as ase_write, read as ase_read
from automlip.trainers.base import BaseTrainer, PredictionResult
from automlip.config import TrainerConfig, SystemConfig

logger = logging.getLogger("automlip.trainers.gap")


def _auto_soap_params(system_config):
    from ase.data import covalent_radii, atomic_numbers
    max_r = max(covalent_radii[atomic_numbers[e]] for e in system_config.elements)
    cutoff = round(2.5 * max_r, 1)
    return {"cutoff": cutoff, "lmax": 6, "nmax": 8, "n_sparse": 1000, "delta": 1.0}


def save_atoms_list(atoms_list, path):
    ase_write(str(path), atoms_list, format="extxyz")


class GAPTrainer(BaseTrainer):
    def __init__(self, system_config: SystemConfig, trainer_config: TrainerConfig):
        self.system_config = system_config
        self.trainer_config = trainer_config
        auto = _auto_soap_params(system_config)
        self.soap_params = {
            "cutoff": trainer_config.soap_cutoff or auto["cutoff"],
            "lmax": trainer_config.soap_lmax or auto["lmax"],
            "nmax": trainer_config.soap_nmax or auto["nmax"],
            "n_sparse": trainer_config.soap_n_sparse or auto["n_sparse"],
            "delta": trainer_config.soap_delta or auto["delta"],
        }
        self.sigma = trainer_config.sigma or [0.002, 0.2, 0.0, 0.0]
        logger.info("GAPTrainer: cutoff=%.1f, lmax=%d, nmax=%d, n_sparse=%d",
                     self.soap_params["cutoff"], self.soap_params["lmax"],
                     self.soap_params["nmax"], self.soap_params["n_sparse"])

    def train(self, training_data: List[Atoms], model_dir: Path, model_name: str = "model") -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        training_file = model_dir / f"{model_name}_train.extxyz"
        model_path = model_dir / f"{model_name}.xml"
        save_atoms_list(training_data, training_file)
        cmd = self._build_gap_command(training_file, model_path)
        logger.info("Training GAP: %s", model_name)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(model_dir))
        if result.returncode != 0:
            logger.error("GAP training failed:\n%s", result.stderr[:500])
            raise RuntimeError(f"GAP training failed for {model_name}")
        return model_path

    def train_committee(self, training_data: List[Atoms], model_dir: Path,
                        n_models: int = 3, bootstrap_fraction: float = 0.8) -> List[Path]:
        model_dir.mkdir(parents=True, exist_ok=True)
        n_data = len(training_data)
        n_sample = max(1, int(n_data * bootstrap_fraction))
        rng = np.random.default_rng(self.trainer_config.seed)
        model_paths = []
        for i in range(n_models):
            indices = rng.choice(n_data, size=n_sample, replace=True)
            subset = [training_data[j] for j in indices]
            member_dir = model_dir / f"committee_{i:02d}"
            try:
                mp = self.train(subset, member_dir, model_name=f"committee_{i:02d}")
                model_paths.append(mp)
            except RuntimeError:
                logger.error("Committee member %d failed", i)
        if not model_paths:
            raise RuntimeError("All GAP committee members failed")
        return model_paths

    def get_calculator(self, model_path: Path):
        from quippy.potential import Potential
        return Potential(param_filename=str(model_path))

    def predict_committee(self, atoms_list: List[Atoms], model_paths: List[Path],
                          n_cores: int = 1) -> List[PredictionResult]:
        calculators = [self.get_calculator(p) for p in model_paths]
        results = []
        for atoms in atoms_list:
            energies = np.zeros(len(model_paths))
            forces = np.zeros((len(model_paths), len(atoms), 3))
            for j, calc in enumerate(calculators):
                a = atoms.copy()
                a.calc = calc
                energies[j] = a.get_potential_energy()
                forces[j] = a.get_forces()
            results.append(PredictionResult(energies=energies, forces=forces))
        return results

    def compute_validation_error(self, validation_data: List[Atoms], model_path: Path) -> Tuple[float, float]:
        calc = self.get_calculator(model_path)
        e_err, f_err = [], []
        for atoms in validation_data:
            ref_e = atoms.info.get("energy")
            ref_f = atoms.arrays.get("forces")
            if ref_e is None or ref_f is None:
                continue
            a = atoms.copy(); a.calc = calc
            pe = a.get_potential_energy(); pf = a.get_forces()
            n = len(atoms)
            e_err.append((pe / n - ref_e / n) ** 2)
            f_err.extend(((pf - ref_f) ** 2).flatten().tolist())
        return float(np.sqrt(np.mean(e_err))) if e_err else float("inf"), \
               float(np.sqrt(np.mean(f_err))) if f_err else float("inf")

    def _build_gap_command(self, training_file, model_path):
        sp = self.soap_params
        s = self.sigma
        elements_str = " ".join(self.system_config.elements)
        cmd = (
            f"gap_fit atoms_filename={training_file} "
            f"gap={{distance_2b cutoff=3.0 delta=2.0 n_sparse=15 covariance_type=ard_se : "
            f"angle_3b cutoff=3.0 delta=0.1 n_sparse=200 : "
            f"soap cutoff={sp['cutoff']} l_max={sp['lmax']} n_max={sp['nmax']} "
            f"delta={sp['delta']} n_sparse={sp['n_sparse']} "
            f"atom_sigma=0.5 zeta=4 covariance_type=dot_product "
            f"sparse_method=cur_points}} "
            f"default_sigma={{{s[0]} {s[1]} {s[2]} {s[3]}}} "
            f"energy_parameter_name=energy force_parameter_name=forces "
            f"gp_file={model_path}"
        )
        extra = self.trainer_config.extra_gap_args
        for k, v in extra.items():
            cmd += f" {k}={v}"
        return cmd
