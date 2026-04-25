"""Random structure generation for initial training sets."""

import logging
from typing import Dict, List, Optional
import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, covalent_radii

logger = logging.getLogger("automlip.generators.random")


def generate_random_structures(
    elements: List[str],
    composition: Dict[str, float],
    n_atoms: int = 100,
    n_structures: int = 20,
    density: float = 2.0,
    min_distance: Optional[float] = None,
    seed: int = 42,
) -> List[Atoms]:
    """
    Generate random periodic structures with given composition and density.

    Args:
        elements: list of element symbols.
        composition: fractional composition per element (must sum to 1).
        n_atoms: atoms per structure.
        n_structures: how many to generate.
        density: target mass density in g/cm3.
        min_distance: minimum interatomic distance in Angstrom.
                      If None, uses 0.8 * sum of covalent radii.
        seed: random seed.

    Returns:
        List of Atoms objects with cubic cells.
    """
    rng = np.random.default_rng(seed)

    # Build symbol list from composition.
    counts = {}
    remaining = n_atoms
    sorted_elements = sorted(composition.keys(), key=lambda e: composition[e])
    for i, el in enumerate(sorted_elements):
        if i == len(sorted_elements) - 1:
            counts[el] = remaining
        else:
            counts[el] = max(1, int(n_atoms * composition[el]))
            remaining -= counts[el]

    symbols = []
    for el in elements:
        symbols.extend([el] * counts.get(el, 0))

    # Compute box size from density.
    total_mass_amu = sum(atomic_masses[atomic_numbers[s]] for s in symbols)
    total_mass_g = total_mass_amu * 1.66054e-24
    volume_cm3 = total_mass_g / density
    volume_ang3 = volume_cm3 * 1e24
    box_length = volume_ang3 ** (1.0 / 3.0)

    if min_distance is None:
        radii = [covalent_radii[atomic_numbers[e]] for e in elements]
        min_distance = 0.8 * (min(radii) + max(radii))

    structures = []
    for i in range(n_structures):
        positions = _place_atoms(len(symbols), box_length, min_distance, rng)
        if positions is None:
            logger.warning("Structure %d: could not place all atoms, skipping", i)
            continue
        cell = np.eye(3) * box_length
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        structures.append(atoms)

    logger.info("Generated %d/%d random structures (%.1f g/cm3, %d atoms, L=%.2f A)",
                len(structures), n_structures, density, n_atoms, box_length)
    return structures


def _place_atoms(n_atoms, box_length, min_dist, rng, max_attempts=1000):
    """Place atoms randomly with minimum distance constraint."""
    positions = []
    for i in range(n_atoms):
        for _ in range(max_attempts):
            pos = rng.random(3) * box_length
            if _check_min_dist(pos, positions, min_dist, box_length):
                positions.append(pos)
                break
        else:
            return None
    return np.array(positions)


def _check_min_dist(pos, existing, min_dist, box_length):
    """Check minimum image distance against all existing atoms."""
    for other in existing:
        diff = pos - other
        diff -= box_length * np.round(diff / box_length)
        if np.linalg.norm(diff) < min_dist:
            return False
    return True
