"""MD sampling to generate candidate structures for active learning."""

import logging
from typing import List

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from automlip.utils.sampling import sample_scalar

logger = logging.getLogger("automlip.samplers.md")


def run_md(
	atoms: Atoms,
	calculator,
	temp=300.0,
	n_steps=5000,
	timestep: float = 1.0,
	interval: int = 10,
	seed: int = 42,
	friction: float = 0.01,
) -> List[Atoms]:
	"""Run Langevin MD, collect snapshots at regular intervals.

	temp and n_steps each take a single value or a (min, max) range drawn once
	per call.
	"""
	atoms = atoms.copy()
	atoms.calc = calculator

	rng = np.random.default_rng(seed)
	temp = sample_scalar(temp, rng)
	n_steps = sample_scalar(n_steps, rng, as_int=True)
	MaxwellBoltzmannDistribution(atoms, temperature_K=temp, rng=rng)

	dyn = Langevin(atoms, timestep * units.fs, temperature_K=temp,
				   friction=friction, rng=rng)

	snapshots = []

	def _collect():
		snap = atoms.copy()
		snap.info["md_step"] = dyn.nsteps
		# Drop any energy/forces/stress inherited from the seed frame. These
		# are stale for the MD geometry; QE overwrites them in the AL path, but
		# a snapshot read before labelling must not carry the seed's values.
		for k in ("energy", "free_energy", "stress"):
			snap.info.pop(k, None)
		for k in ("forces", "stress"):
			snap.arrays.pop(k, None)
		snapshots.append(snap)

	dyn.attach(_collect, interval=interval)
	dyn.run(n_steps)

	logger.info("MD: %d snapshots from %d steps at %.0f K",
				len(snapshots), n_steps, temp)
	return snapshots
