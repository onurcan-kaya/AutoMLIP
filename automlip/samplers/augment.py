"""Synthetic structure generation for distillation.

Two sources, both labelled later by the teacher (not DFT):
  - a rattle-relax family tree grown from each real seed structure,
  - dimers, for short-range two-body coverage.

The generator uses a single calculator (the teacher mean, or one committee
member) for relaxation and parent selection. Labelling and the committee
disagreement gate happen separately in labellers/teacher.py.
"""

import logging
from typing import List

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from automlip.utils.sampling import sample_scalar

logger = logging.getLogger("automlip.samplers.augment")

_KB_EV = 8.617333262e-5  # eV/K


def generate_synthetic(seeds: List[Atoms], calc, cfg, seed: int = 42) -> List[Atoms]:
	"""Grow a rattle-relax family tree from each seed. Returns unlabelled Atoms.

	Each returned structure carries info['synth_energy'] from the generation
	calculator, used only for diagnostics; the training labels are attached
	later by the teacher committee.
	"""
	if not seeds:
		raise ValueError("No seed structures for synthetic generation.")

	rng = np.random.default_rng(seed)
	per_seed = max(1, cfg.n_synthetic // len(seeds))
	out: List[Atoms] = []
	for s_idx, seed_atoms in enumerate(seeds):
		children = _grow_tree(seed_atoms, calc, cfg, per_seed, rng)
		out.extend(children)
		logger.info("Seed %d/%d: generated %d synthetic structures",
					s_idx + 1, len(seeds), len(children))

	logger.info("Synthetic generation: %d structures from %d seeds",
				len(out), len(seeds))
	return out


def _grow_tree(seed_atoms, calc, cfg, target, rng):
	periodic = bool(np.any(seed_atoms.pbc))
	c0 = seed_atoms.get_cell().array.copy() if periodic else None
	temp = sample_scalar(cfg.temperature, rng)
	kT = max(1e-9, temp * _KB_EV)

	seed_e = _safe_energy(seed_atoms, calc)
	nodes = [{"atoms": seed_atoms.copy(), "energy": seed_e, "gen": 1}]
	children = []

	attempts = 0
	max_attempts = max(50, target * 20)
	while len(children) < target and attempts < max_attempts:
		attempts += 1
		parent = _select_parent(nodes, cfg.beta, kT, rng)
		child = _make_child(parent["atoms"], c0, cfg, periodic, rng)
		relaxed = _relax(child, calc, cfg, parent["energy"], kT, rng)
		if relaxed is None:
			continue
		e = relaxed.info.get("synth_energy")
		if e is None:
			e = parent["energy"]
		nodes.append({"atoms": relaxed, "energy": e, "gen": parent["gen"] + 1})
		children.append(relaxed)

	if len(children) < target:
		logger.warning("Tree from seed produced %d/%d structures (relaxations "
					   "may be failing)", len(children), target)
	return children


def _select_parent(nodes, beta, kT, rng):
	energies = np.array([n["energy"] for n in nodes], dtype=float)
	gens = np.array([n["gen"] for n in nodes], dtype=float)

	e_shift = energies - np.min(energies)
	w = np.exp(-e_shift / kT)
	if not np.all(np.isfinite(w)) or w.sum() <= 0:
		w = np.ones_like(w)
	p_boltz = w / w.sum()
	p_gen = gens / gens.sum()
	p = beta * p_boltz + (1.0 - beta) * p_gen
	p = p / p.sum()
	idx = int(rng.choice(len(nodes), p=p))
	return nodes[idx]


def _make_child(parent_atoms, c0, cfg, periodic, rng):
	a = parent_atoms.copy()
	a.calc = None
	R = a.get_positions()
	n = len(a)

	if periodic:
		sigma_cell = sample_scalar(cfg.rattle_sigma_cell, rng)
		A = rng.normal(0.0, sigma_cell, size=(3, 3))
	else:
		A = np.zeros((3, 3))
	M = A + np.eye(3)
	sigma_pos = sample_scalar(cfg.rattle_sigma_pos, rng)
	B = rng.normal(0.0, sigma_pos, size=(n, 3))

	a.set_positions(R @ M.T + B)
	if periodic and c0 is not None:
		a.set_cell(c0 @ M.T, scale_atoms=False)
		a.set_pbc(True)
	return a


def _relax(atoms, calc, cfg, parent_energy, kT, rng):
	"""Robbins-Monro-style relaxation using the generation calculator."""
	a = atoms.copy()
	a.calc = calc
	sigma = sample_scalar(cfg.rattle_sigma_pos, rng)
	steps = sample_scalar(cfg.relax_steps, rng, as_int=True)
	last_e = None
	try:
		for step in range(1, steps + 1):
			forces = a.get_forces()
			last_e = float(a.get_potential_energy())
			fmax = float(np.max(np.linalg.norm(forces, axis=1)))

			if fmax < cfg.max_force_stop and parent_energy is not None:
				dE = last_e - parent_energy
				if dE <= 0:
					p_stop = 0.25
				else:
					p_stop = min(0.25, float(np.exp(-dE / kT)))
				if rng.random() < p_stop:
					break

			norms = np.linalg.norm(forces, axis=1, keepdims=True)
			unit = np.divide(forces, norms, out=np.zeros_like(forces),
							 where=norms > 1e-12)
			a.set_positions(a.get_positions() + (sigma / step) * unit)
	except Exception as exc:
		logger.debug("Relaxation failed: %s", exc)
		return None

	a.calc = None
	if last_e is not None:
		a.info["synth_energy"] = last_e
	return a


def _safe_energy(atoms, calc):
	a = atoms.copy()
	a.calc = calc
	try:
		return float(a.get_potential_energy())
	except Exception:
		return 0.0


def generate_dimers(elements: List[str], cfg, seed: int = 42) -> List[Atoms]:
	"""Isolated dimers per element pair, distances scaled by covalent radii."""
	els = list(dict.fromkeys(elements))
	if not els:
		logger.warning("No elements given; skipping dimer generation.")
		return []

	rng = np.random.default_rng(seed + 777)
	box = cfg.dimer_box
	out: List[Atoms] = []
	for i in range(len(els)):
		for j in range(i, len(els)):
			e1, e2 = els[i], els[j]
			d = (covalent_radii[atomic_numbers[e1]]
				 + covalent_radii[atomic_numbers[e2]])
			for _ in range(cfg.dimers_per_pair):
				r = rng.uniform(cfg.dimer_min_factor * d, cfg.dimer_max_factor * d)
				out.append(Atoms(
					[e1, e2],
					positions=[[box / 2, box / 2, box / 2],
							   [box / 2 + r, box / 2, box / 2]],
					cell=np.eye(3) * box,
					pbc=False,
				))

	n_pairs = len(els) * (len(els) + 1) // 2
	logger.info("Generated %d dimer structures (%d element pairs)",
				len(out), n_pairs)
	return out
