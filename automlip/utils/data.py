"""Load, validate, split extxyz datasets."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

logger = logging.getLogger("automlip.utils.data")

_ENERGY_KEYS = [
	"energy", "Energy", "REF_energy", "dft_energy", "DFT_energy",
	"free_energy", "total_energy", "potential_energy",
]
_FORCE_KEYS = [
	"forces", "Forces", "REF_forces", "dft_forces", "DFT_forces", "force",
]


def load_dataset(
	source: Union[str, Path, List[str]],
	energy_key: Optional[str] = None,
	force_key: Optional[str] = None,
) -> List[Atoms]:
	"""Load extxyz data, normalise keys to 'energy' and 'forces'.

	Skips frames without energy/forces or without PBC.
	"""
	files = _resolve_files(source)
	if not files:
		raise FileNotFoundError(f"No extxyz files found at: {source}")

	raw = []
	for f in files:
		try:
			frames = ase_read(str(f), index=":", format="extxyz")
			if isinstance(frames, Atoms):
				frames = [frames]
			raw.extend(frames)
		except Exception as e:
			logger.warning("Failed to read %s: %s", f, e)

	if not raw:
		raise ValueError("No structures loaded from any file.")

	validated = []
	for atoms in raw:
		if not all(atoms.pbc):
			continue
		energy = _get_energy(atoms, energy_key)
		forces = _get_forces(atoms, force_key)
		if energy is None or forces is None:
			continue
		clean = atoms.copy()
		clean.calc = None
		for k in _ENERGY_KEYS:
			clean.info.pop(k, None)
		clean.info["energy"] = float(energy)
		for k in _FORCE_KEYS:
			if k in clean.arrays and k != "forces":
				del clean.arrays[k]
		clean.arrays["forces"] = np.array(forces, dtype=np.float64)
		validated.append(clean)

	if not validated:
		raise ValueError(
			"No valid structures after filtering. Check key names."
		)

	logger.info("Loaded %d/%d valid structures", len(validated), len(raw))
	return validated


def _get_energy(atoms, key):
	"""Energy from info, an explicit key, or the attached calculator."""
	if key is not None:
		if key in atoms.info:
			return atoms.info[key]
		res = getattr(atoms.calc, "results", {}) if atoms.calc else {}
		return res.get(key)
	for k in _ENERGY_KEYS:
		if k in atoms.info:
			return atoms.info[k]
	res = getattr(atoms.calc, "results", {}) if atoms.calc else {}
	for k in ("energy", "free_energy"):
		if k in res:
			return res[k]
	return None


def _get_forces(atoms, key):
	"""Forces from arrays, an explicit key, or the attached calculator."""
	if key is not None:
		if key in atoms.arrays:
			return atoms.arrays[key]
		res = getattr(atoms.calc, "results", {}) if atoms.calc else {}
		return res.get(key)
	for k in _FORCE_KEYS:
		if k in atoms.arrays:
			return atoms.arrays[k]
	res = getattr(atoms.calc, "results", {}) if atoms.calc else {}
	if "forces" in res:
		return res["forces"]
	return None


def split_dataset(data: List[Atoms], val_fraction: float = 0.1,
				  seed: int = 42):
	"""Shuffle and split into (train, val)."""
	rng = np.random.default_rng(seed)
	indices = rng.permutation(len(data))
	n_train = max(1, int(len(data) * (1.0 - val_fraction)))
	n_train = min(n_train, len(data) - 1)
	train = [data[i] for i in indices[:n_train]]
	val = [data[i] for i in indices[n_train:]]
	logger.info("Split: %d train, %d val", len(train), len(val))
	return train, val


def _resolve_files(source):
	if isinstance(source, (list, tuple)):
		return [Path(f) for f in source if Path(f).is_file()]
	path = Path(source)
	if path.is_file():
		return [path]
	if path.is_dir():
		return sorted(path.glob("*.extxyz")) + sorted(path.glob("*.xyz"))
	return []
