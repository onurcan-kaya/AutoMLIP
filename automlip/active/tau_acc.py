"""
tau_acc: time-to-failure convergence metric.

Runs MD with the trained potential and periodically compares against
DFT energy. Accumulates error above a tolerance. tau_acc is the
simulation time at which cumulative error exceeds a threshold.

Higher tau_acc = more stable potential = longer reliable MD runs.
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

logger = logging.getLogger("automlip.active.tau_acc")


@dataclass
class TauResult:
    value: float           # tau_acc in fs
    per_config: List[float]  # tau_acc for each starting config
    converged: bool        # True if all configs reached max_time


def _run_single_tau(
    atoms: Atoms,
    ml_calculator: Calculator,
    dft_calculator,
    e_lower: float,
    e_thresh: float,
    max_time_fs: float,
    interval_fs: float,
    temp: float,
    dt_fs: float,
    seed: int,
) -> float:
    """Run one tau_acc trajectory. Returns tau_acc in fs."""
    atoms = atoms.copy()
    atoms.calc = ml_calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp,
                                 rng=np.random.default_rng(seed))

    dyn = VelocityVerlet(atoms, dt_fs * units.fs)
    steps_per_interval = max(1, int(interval_fs / dt_fs))
    cumulative_error = 0.0
    current_time = 0.0

    while current_time < max_time_fs:
        dyn.run(steps_per_interval)
        current_time += steps_per_interval * dt_fs

        ml_energy = atoms.get_potential_energy()
        dft_energy = _get_dft_energy(atoms.copy(), dft_calculator)

        if dft_energy is None:
            logger.warning("DFT failed at t=%.1f fs, returning current time", current_time)
            return current_time

        error = abs(ml_energy - dft_energy)
        excess = max(error - e_lower, 0.0)
        cumulative_error += excess

        logger.debug("t=%.1f fs: |E_ML - E_DFT| = %.4f eV, cumul = %.4f eV",
                     current_time, error, cumulative_error)

        if cumulative_error > e_thresh:
            logger.info("tau_acc = %.1f fs (threshold exceeded)", current_time)
            return current_time

    logger.info("tau_acc = %.1f fs (reached max time)", max_time_fs)
    return max_time_fs


def _get_dft_energy(atoms: Atoms, dft_calculator) -> Optional[float]:
    """Compute DFT energy for a single frame using the pipeline's calculator."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            calc_dir = Path(tmpdir)
            input_path = dft_calculator.write_input(atoms, calc_dir)
            run_cmd = dft_calculator.get_run_command(input_path)

            result = subprocess.run(
                run_cmd, shell=True, cwd=str(calc_dir),
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                return None

            parsed = dft_calculator.parse_output(calc_dir)
            if parsed is None:
                return None
            return parsed.info.get("energy")

    except Exception as e:
        logger.warning("DFT energy evaluation failed: %s", e)
        return None


def compute_tau_acc(
    training_data: List[Atoms],
    ml_calculator: Calculator,
    dft_calculator,
    e_lower: float = 0.043,
    e_thresh: Optional[float] = None,
    max_time_fs: float = 1000.0,
    interval_fs: float = 20.0,
    temp: float = 300.0,
    dt_fs: float = 0.5,
    n_configs: int = 3,
    seed: int = 42,
) -> TauResult:
    """
    Compute tau_acc averaged over multiple starting configurations.
    Picks the n_configs lowest-energy structures from training data.
    """
    if e_thresh is None:
        e_thresh = 10.0 * e_lower

    # Pick starting configs by lowest energy per atom.
    energies_per_atom = []
    for a in training_data:
        e = a.info.get("energy")
        if e is not None:
            energies_per_atom.append(e / len(a))
        else:
            energies_per_atom.append(float("inf"))

    sorted_indices = np.argsort(energies_per_atom)
    selected = sorted_indices[:n_configs]

    tau_values = []
    for i, idx in enumerate(selected):
        atoms = training_data[idx]
        logger.info("tau_acc config %d/%d (%d atoms, E/atom=%.4f eV)",
                     i + 1, n_configs, len(atoms), energies_per_atom[idx])

        tau = _run_single_tau(
            atoms, ml_calculator, dft_calculator,
            e_lower=e_lower, e_thresh=e_thresh,
            max_time_fs=max_time_fs, interval_fs=interval_fs,
            temp=temp, dt_fs=dt_fs, seed=seed + i,
        )
        tau_values.append(tau)

    mean_tau = float(np.mean(tau_values))
    all_converged = all(t >= max_time_fs for t in tau_values)

    logger.info("tau_acc = %.1f fs (mean of %d configs, converged=%s)",
                mean_tau, len(tau_values), all_converged)

    return TauResult(value=mean_tau, per_config=tau_values, converged=all_converged)
