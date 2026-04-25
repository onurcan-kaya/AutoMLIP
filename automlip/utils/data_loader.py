"""
Data loader for user-provided training datasets.

Reads extxyz files (single file or directory of files), validates
that each frame has the required properties (energy, forces), and
normalises key names so the rest of the pipeline sees a consistent
interface.

Usage:
    atoms_list = load_dataset("my_data.extxyz")
    atoms_list = load_dataset("/path/to/data_dir/")
    atoms_list = load_dataset(["file1.extxyz", "file2.extxyz"])
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

logger = logging.getLogger("automlip.utils.data_loader")

# Known key aliases for energy and forces across different codes
# and conventions. The loader normalises everything to "energy"
# and "forces".
_ENERGY_KEYS = [
    "energy",
    "Energy",
    "REF_energy",
    "dft_energy",
    "DFT_energy",
    "free_energy",
    "total_energy",
    "potential_energy",
]

_FORCE_KEYS = [
    "forces",
    "Forces",
    "REF_forces",
    "dft_forces",
    "DFT_forces",
    "force",
]

_STRESS_KEYS = [
    "stress",
    "Stress",
    "REF_stress",
    "virial",
    "virials",
    "REF_virial",
]


def load_dataset(
    source: Union[str, Path, List[str], List[Path]],
    energy_key: Optional[str] = None,
    force_key: Optional[str] = None,
    stress_key: Optional[str] = None,
    require_forces: bool = True,
    require_stress: bool = False,
    require_pbc: bool = True,
    verbose: bool = True,
) -> List[Atoms]:
    """
    Load and validate a dataset from extxyz files.

    Args:
        source: one of:
            - path to a single extxyz file
            - path to a directory (all .extxyz and .xyz files are read)
            - list of file paths
        energy_key: explicit key name for energy in atoms.info.
                    If None, auto-detected from known aliases.
        force_key: explicit key name for forces in atoms.arrays.
                   If None, auto-detected.
        stress_key: explicit key for stress/virial. If None, auto-detected.
        require_forces: if True, skip frames without forces.
        require_stress: if True, skip frames without stress.
        require_pbc: if True, skip non-periodic frames.
        verbose: log summary statistics.

    Returns:
        List of Atoms, each with:
            - atoms.info["energy"]: total energy (float, eV)
            - atoms.arrays["forces"]: forces (n_atoms x 3, eV/A)
            - atoms.info["stress"]: stress (6-vector, if available)
    """
    files = _resolve_files(source)

    if not files:
        raise FileNotFoundError(
            f"No extxyz files found at: {source}"
        )

    logger.info("Loading dataset from %d file(s)...", len(files))

    raw_atoms = []
    for f in files:
        try:
            frames = ase_read(str(f), index=":", format="extxyz")
            if isinstance(frames, Atoms):
                frames = [frames]
            raw_atoms.extend(frames)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)

    if not raw_atoms:
        raise ValueError("No structures loaded from any file.")

    logger.info("Read %d raw frames", len(raw_atoms))

    # Auto-detect keys from first frame if not specified.
    if energy_key is None:
        energy_key = _detect_key(raw_atoms[0].info, _ENERGY_KEYS, "energy")
    if force_key is None:
        force_key = _detect_key(raw_atoms[0].arrays, _FORCE_KEYS, "forces")
    if stress_key is None:
        stress_key = _detect_key(
            raw_atoms[0].info, _STRESS_KEYS, "stress", required=False
        )

    logger.info(
        "Key mapping: energy='%s', forces='%s', stress='%s'",
        energy_key, force_key, stress_key or "none",
    )

    # Validate and normalise.
    validated = []
    n_skipped_energy = 0
    n_skipped_forces = 0
    n_skipped_stress = 0
    n_skipped_pbc = 0

    for i, atoms in enumerate(raw_atoms):
        # Check PBC.
        if require_pbc and not all(atoms.pbc):
            n_skipped_pbc += 1
            continue

        # Extract and normalise energy.
        energy = atoms.info.get(energy_key)
        if energy is None:
            n_skipped_energy += 1
            continue

        # Extract and normalise forces.
        forces = atoms.arrays.get(force_key)
        if forces is None and require_forces:
            n_skipped_forces += 1
            continue

        # Extract stress if available.
        stress = None
        if stress_key:
            stress = atoms.info.get(stress_key)
            if stress is None and require_stress:
                n_skipped_stress += 1
                continue

        # Normalise keys.
        clean = atoms.copy()

        # Clear old keys to avoid duplication.
        for k in _ENERGY_KEYS:
            clean.info.pop(k, None)
        clean.info["energy"] = float(energy)

        if forces is not None:
            for k in _FORCE_KEYS:
                if k in clean.arrays and k != "forces":
                    del clean.arrays[k]
            clean.arrays["forces"] = np.array(forces, dtype=np.float64)

        if stress is not None:
            for k in _STRESS_KEYS:
                clean.info.pop(k, None)
            clean.info["stress"] = np.array(stress, dtype=np.float64)

        validated.append(clean)

    if verbose:
        _log_summary(
            len(raw_atoms), len(validated),
            n_skipped_energy, n_skipped_forces,
            n_skipped_stress, n_skipped_pbc,
            validated,
        )

    if not validated:
        raise ValueError(
            "No valid structures after filtering. Check your key names "
            "and data. Use energy_key/force_key arguments if your file "
            "uses non-standard names."
        )

    return validated


def _resolve_files(
    source: Union[str, Path, List[str], List[Path]],
) -> List[Path]:
    """Resolve source to a list of file paths."""
    if isinstance(source, (list, tuple)):
        return [Path(f) for f in source if Path(f).is_file()]

    path = Path(source)

    if path.is_file():
        return [path]

    if path.is_dir():
        files = sorted(path.glob("*.extxyz")) + sorted(path.glob("*.xyz"))
        return files

    # Try glob pattern.
    from glob import glob as globfn
    matches = globfn(str(source))
    return [Path(m) for m in sorted(matches) if Path(m).is_file()]


def _detect_key(
    container: dict,
    candidates: List[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    """Find which key name is used in the data."""
    for key in candidates:
        if key in container:
            return key

    if required:
        available = list(container.keys())
        raise KeyError(
            f"Could not find {label} key in data. "
            f"Tried: {candidates}. "
            f"Available keys: {available}. "
            f"Use {label}_key argument to specify explicitly."
        )

    return None


def _log_summary(
    n_raw, n_valid,
    n_skip_e, n_skip_f, n_skip_s, n_skip_pbc,
    validated,
):
    """Log dataset summary statistics."""
    logger.info("Dataset summary:")
    logger.info("  Total frames read: %d", n_raw)
    logger.info("  Valid frames: %d", n_valid)

    if n_skip_e:
        logger.warning("  Skipped (no energy): %d", n_skip_e)
    if n_skip_f:
        logger.warning("  Skipped (no forces): %d", n_skip_f)
    if n_skip_s:
        logger.warning("  Skipped (no stress): %d", n_skip_s)
    if n_skip_pbc:
        logger.warning("  Skipped (non-periodic): %d", n_skip_pbc)

    if validated:
        n_atoms_list = [len(a) for a in validated]
        elements = set()
        for a in validated:
            elements.update(a.get_chemical_symbols())

        energies = [a.info["energy"] / len(a) for a in validated]

        logger.info("  Elements: %s", sorted(elements))
        logger.info(
            "  Atoms per frame: %d-%d (mean %.0f)",
            min(n_atoms_list), max(n_atoms_list), np.mean(n_atoms_list),
        )
        logger.info(
            "  Energy/atom range: %.4f to %.4f eV",
            min(energies), max(energies),
        )


def split_dataset(
    data: List[Atoms],
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple:
    """
    Split a dataset into training and validation sets.

    Args:
        data: list of Atoms.
        train_fraction: fraction for training (rest goes to validation).
        seed: random seed for reproducibility.

    Returns:
        (train_data, val_data)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    indices = rng.permutation(n)

    n_train = max(1, int(n * train_fraction))
    n_train = min(n_train, n - 1)  # at least 1 for validation

    train_data = [data[i] for i in indices[:n_train]]
    val_data = [data[i] for i in indices[n_train:]]

    logger.info(
        "Split: %d train, %d validation (%.0f%%/%.0f%%)",
        len(train_data), len(val_data),
        100 * len(train_data) / n, 100 * len(val_data) / n,
    )

    return train_data, val_data
