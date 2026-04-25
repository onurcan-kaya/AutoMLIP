"""MD sampling with current MLIP to generate candidate structures."""

import logging
from typing import List, Optional
import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units

logger = logging.getLogger("automlip.active.sampler")


def run_md_sampling(
    atoms: Atoms,
    calculator,
    temp: float = 300.0,
    n_steps: int = 5000,
    timestep: float = 1.0,
    interval: int = 10,
    seed: int = 42,
    friction: float = 0.01,
) -> List[Atoms]:
    """
    Run Langevin MD and collect snapshots at regular intervals.

    Args:
        atoms: starting structure with cell and PBC set.
        calculator: ASE calculator (MLIP).
        temp: temperature in K.
        n_steps: total MD steps.
        timestep: timestep in fs.
        interval: collect a snapshot every this many steps.
        seed: random seed for velocities.
        friction: Langevin friction coefficient.

    Returns:
        List of Atoms snapshots.
    """
    atoms = atoms.copy()
    atoms.calc = calculator

    np.random.seed(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

    dyn = Langevin(atoms, timestep * units.fs, temperature_K=temp, friction=friction)

    snapshots = []

    def _collect():
        snap = atoms.copy()
        snap.info["md_step"] = dyn.nsteps
        snap.info["md_temp"] = temp
        snapshots.append(snap)

    dyn.attach(_collect, interval=interval)

    logger.info("Running MD: %d steps, T=%.0f K, dt=%.1f fs, collecting every %d steps",
                n_steps, temp, timestep, interval)

    dyn.run(n_steps)

    logger.info("MD complete: collected %d snapshots", len(snapshots))
    return snapshots


def run_multi_md(
    structures: List[Atoms],
    calculator,
    temp: float = 300.0,
    n_steps: int = 5000,
    timestep: float = 1.0,
    interval: int = 10,
    seed: int = 42,
) -> List[Atoms]:
    """Run MD on multiple starting structures, collect all snapshots."""
    all_snapshots = []
    for i, atoms in enumerate(structures):
        snaps = run_md_sampling(
            atoms, calculator, temp=temp, n_steps=n_steps,
            timestep=timestep, interval=interval, seed=seed + i,
        )
        all_snapshots.extend(snaps)
    logger.info("Multi-MD: %d total snapshots from %d structures",
                len(all_snapshots), len(structures))
    return all_snapshots
