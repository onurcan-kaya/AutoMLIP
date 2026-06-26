"""Random structure generation with optional pre-relaxation."""

import logging
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, covalent_radii

from automlip.utils.sampling import sample_composition, sample_scalar

logger = logging.getLogger("automlip.samplers.rss")


def generate_rss(
	elements: List[str],
	composition: Dict,
	n_atoms=100,
	n_structures: int = 20,
	density=2.0,
	min_distance: Optional[float] = None,
	seed: int = 42,
	prerelax_model: Optional[str] = None,
	prerelax_fmax: float = 5.0,
	prerelax_steps: int = 50,
	device: str = "cpu",
) -> List[Atoms]:
	"""Generate random periodic structures, optionally pre-relax.

	composition values, n_atoms and density each take a single value or a
	(min, max) range. A single value is fixed across structures; a range is
	drawn uniformly per structure (composition fractions are renormalised to
	sum to 1). The drawn values are recorded in atoms.info.

	Pre-relaxation runs a short geometry optimisation with a foundation
	model (e.g. MACE-MP) to remove unphysical configurations before
	expensive DFT labelling. Structures where relaxation fails or the
	final energy is an outlier are discarded.
	"""
	rng = np.random.default_rng(seed)

	if min_distance is None:
		radii = [covalent_radii[atomic_numbers[e]] for e in elements]
		min_distance = 0.8 * (min(radii) + max(radii))

	raw = []
	for i in range(n_structures):
		comp_i = sample_composition(composition, elements, rng)
		n_i = sample_scalar(n_atoms, rng, as_int=True)
		symbols_i = _build_symbol_list(elements, comp_i, n_i)
		density_i = sample_scalar(density, rng)
		box_i = _box_from_density(symbols_i, density_i)

		positions = _place_atoms(len(symbols_i), box_i, min_distance, rng)
		if positions is None:
			logger.warning("Structure %d: placement failed, skipping", i)
			continue
		atoms = Atoms(
			symbols=symbols_i,
			positions=positions,
			cell=np.eye(3) * box_i,
			pbc=True,
		)
		atoms.info["rss_density"] = float(density_i)
		atoms.info["rss_n_atoms"] = int(n_i)
		atoms.info["rss_composition"] = {
			k: float(v) for k, v in comp_i.items()}
		raw.append(atoms)

	logger.info("Generated %d/%d random structures", len(raw), n_structures)

	if prerelax_model is None:
		return raw

	return _prerelax_batch(raw, prerelax_model, prerelax_fmax,
						   prerelax_steps, device)


def _prerelax_batch(structures, model_spec, fmax, steps, device="cpu"):
	"""Run short relaxation with a foundation model, discard failures."""
	calc = _load_prerelax_calculator(model_spec, device)
	relaxed = []

	n = len(structures)
	for i, atoms in enumerate(structures):
		try:
			result = _relax_single(atoms, calc, fmax, steps)
			if result is not None:
				relaxed.append(result)
		except Exception as e:
			logger.warning("Pre-relax failed for structure %d: %s", i, e)
		if (i + 1) % 25 == 0 or (i + 1) == n:
			logger.info("Pre-relaxation progress: %d/%d", i + 1, n)

	if not relaxed:
		logger.warning("All pre-relaxations failed, returning raw structures")
		return structures

	# Filter energy outliers (beyond 3 sigma from mean).
	energies = np.array([a.info["prerelax_energy"] / len(a) for a in relaxed])
	mean, std = np.mean(energies), np.std(energies)
	if std > 0:
		keep = [a for a, e in zip(relaxed, energies)
				if abs(e - mean) < 3 * std]
	else:
		keep = relaxed

	logger.info(
		"Pre-relaxation: %d/%d survived (discarded %d failures, %d outliers)",
		len(keep), len(structures),
		len(structures) - len(relaxed), len(relaxed) - len(keep),
	)
	return keep


def _relax_single(atoms, calc, fmax, steps):
	"""Relax one structure. Returns relaxed Atoms or None."""
	from ase.optimize import LBFGS

	atoms = atoms.copy()
	atoms.calc = calc

	opt = LBFGS(atoms, logfile=None)
	try:
		opt.run(fmax=fmax, steps=steps)
	except Exception:
		return None

	atoms.info["prerelax_energy"] = atoms.get_potential_energy()
	atoms.calc = None  # detach calculator
	return atoms


def _load_prerelax_calculator(model_spec: str, device: str = "cpu"):
	"""Load a calculator for pre-relaxation on the given device.

	Accepts:
	  - "mace_mp" or "mace_mp_small" etc: load MACE-MP from mace
	  - any file path: load as MACE model
	"""
	dev = device
	if dev == "auto":
		import torch
		dev = "cuda" if torch.cuda.is_available() else "cpu"

	spec = model_spec.lower().strip()

	if spec.startswith("mace_mp") or spec.startswith("mace-mp"):
		from mace.calculators import mace_mp
		# mace_mp() returns a calculator for the MACE-MP-0 model.
		size = "small"
		if "medium" in spec:
			size = "medium"
		elif "large" in spec:
			size = "large"
		logger.info("Loading MACE-MP (%s) for pre-relaxation on %s", size, dev)
		return mace_mp(model=size, default_dtype="float64", device=dev)

	# Assume it's a path to a MACE model file.
	from mace.calculators import MACECalculator
	logger.info("Loading %s for pre-relaxation on %s", model_spec, dev)
	return MACECalculator(model_paths=model_spec, default_dtype="float64",
						  device=dev)


def _build_symbol_list(elements, composition, n_atoms):
	sorted_els = sorted(composition.keys(), key=lambda e: composition[e])
	counts = {}
	remaining = n_atoms
	for i, el in enumerate(sorted_els):
		if i == len(sorted_els) - 1:
			counts[el] = remaining
		else:
			counts[el] = max(1, int(n_atoms * composition[el]))
			remaining -= counts[el]
	symbols = []
	for el in elements:
		symbols.extend([el] * counts.get(el, 0))
	return symbols


def _box_from_density(symbols, density):
	total_mass_amu = sum(atomic_masses[atomic_numbers[s]] for s in symbols)
	total_mass_g = total_mass_amu * 1.66054e-24
	volume_ang3 = (total_mass_g / density) * 1e24
	return volume_ang3 ** (1.0 / 3.0)


def _place_atoms(n, box, min_dist, rng, max_attempts=1000):
	positions = []
	for _ in range(n):
		for _ in range(max_attempts):
			pos = rng.random(3) * box
			if all(
				np.linalg.norm(pos - o - box * np.round((pos - o) / box))
				>= min_dist
				for o in positions
			):
				positions.append(pos)
				break
		else:
			return None
	return np.array(positions)
