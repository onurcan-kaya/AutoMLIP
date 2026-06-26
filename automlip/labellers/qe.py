"""Quantum ESPRESSO labeller.

Writes pw.x inputs, submits via scheduler, parses outputs. A failed job is
discarded after its reason is logged; there are no retries.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses, atomic_numbers

from automlip.config import DFTConfig, SchedulerConfig

logger = logging.getLogger("automlip.labellers.qe")

RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177249


# ---------- Input / Output --------------------------------------------------

def write_qe_input(atoms: Atoms, calc_dir: Path, dft: DFTConfig) -> Path:
	"""Write a pw.x input file. Returns path to pw.in."""
	calc_dir.mkdir(parents=True, exist_ok=True)
	path = calc_dir / "pw.in"
	ecutrho = dft.ecutrho if dft.ecutrho is not None else 8 * dft.ecutwfc
	symbols = atoms.get_chemical_symbols()
	unique = list(dict.fromkeys(symbols))

	lines = []
	lines.append("&CONTROL")
	lines.append(" calculation = 'scf',")
	lines.append(f"  pseudo_dir = '{dft.pseudo_dir}',")
	lines.append(" outdir = './tmp/',")
	lines.append(" tprnfor = .true.,")
	lines.append(f"  tstress = {'.true.' if dft.tstress else '.false.'},")
	lines.append("/\n")
	lines.append("&SYSTEM")
	lines.append(" ibrav = 0,")
	lines.append(f"  nat = {len(atoms)},")
	lines.append(f"  ntyp = {len(unique)},")
	lines.append(f"  ecutwfc = {dft.ecutwfc},")
	lines.append(f"  ecutrho = {ecutrho},")
	lines.append(" occupations = 'smearing',")
	lines.append(f"  smearing = '{dft.smearing}',")
	lines.append(f"  degauss = {dft.degauss},")
	lines.append("/\n")
	lines.append("&ELECTRONS")
	lines.append(f"  electron_maxstep = {dft.electron_maxstep},")
	lines.append(f"  conv_thr = {dft.conv_thr},")
	lines.append(f"  mixing_beta = {dft.mixing_beta},")
	lines.append("/\n")
	lines.append("ATOMIC_SPECIES")
	for sp in unique:
		mass = atomic_masses[atomic_numbers[sp]]
		pp = dft.pseudopotentials.get(sp, f"{sp}.UPF")
		lines.append(f"  {sp}  {mass:.4f}  {pp}")
	lines.append("")
	lines.append("CELL_PARAMETERS angstrom")
	for row in atoms.get_cell():
		lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
	lines.append("")
	lines.append("ATOMIC_POSITIONS angstrom")
	for sym, pos in zip(symbols, atoms.get_positions()):
		lines.append(f"  {sym} {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
	lines.append("")
	kpts = dft.kpoints
	lines.append("K_POINTS automatic")
	lines.append(f"  {kpts[0]} {kpts[1]} {kpts[2]} 0 0 0")
	lines.append("")

	path.write_text("\n".join(lines))
	return path


def parse_qe_output(calc_dir: Path, atoms: Atoms) -> Optional[Atoms]:
	"""Parse pw.out for energy and forces, attach them to the input atoms.

	The geometry is taken from the input structure that was sent to QE, not
	re-parsed from the output. Returns a labelled copy, or None on failure.
	"""
	out = calc_dir / "pw.out"
	if not out.exists():
		return None
	content = out.read_text(errors="replace")
	if "JOB DONE." not in content:
		return None

	energy_ry = _extract_energy(content)
	if energy_ry is None:
		return None

	forces = _extract_forces(content, len(atoms))
	if forces is None:
		return None

	labelled = atoms.copy()
	labelled.info["energy"] = energy_ry * RY_TO_EV
	labelled.arrays["forces"] = forces
	stress = _extract_stress(content)
	if stress is not None:
		labelled.info["stress"] = stress
	return labelled


def build_run_command(mpi_command: str, nprocs: int, dft_executable: str,
					  infile: str = "pw.in", outfile: str = "pw.out") -> str:
	"""Assemble the pw.x run command from parts."""
	if mpi_command:
		return (f"{mpi_command} -np {nprocs} {dft_executable} "
				f"-i {infile} > {outfile} 2>&1")
	return f"{dft_executable} -i {infile} > {outfile} 2>&1"


def get_run_command(sched: SchedulerConfig, nprocs: Optional[int] = None) -> str:
	n = sched.cores if nprocs is None else nprocs
	return build_run_command(sched.mpi_command, n, sched.dft_executable)


# ---------- Failure reporting ------------------------------------------------

def failure_reason(calc_dir: Path) -> str:
	"""Return a short human-readable reason a QE run failed."""
	out = calc_dir / "pw.out"
	crash = calc_dir / "CRASH"

	if not out.exists():
		if crash.exists():
			return "CRASH file present, no pw.out (likely out of memory)"
		return "no pw.out written (job did not start or was killed)"

	content = out.read_text(errors="replace")

	if "convergence NOT achieved" in content:
		return "SCF convergence not achieved"
	if "S matrix not positive definite" in content:
		return "S matrix not positive definite"
	if "eigenvalues not converged" in content:
		return "eigenvalues not converged"
	if "charge is wrong" in content:
		return "wrong charge"
	if crash.exists():
		return "CRASH file present"
	if "JOB DONE." not in content:
		return "run did not finish (no JOB DONE)"
	return "JOB DONE but energy or forces could not be parsed"


# ---------- QE output parsing helpers ----------------------------------------

def _extract_energy(content):
	matches = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", content)
	return float(matches[-1]) if matches else None


def _extract_stress(content):
	"""Parse the QE stress tensor into a Voigt 6-vector (xx, yy, zz, yz, xz, xy)
	in eV/Ang^3, matching ASE's QE reader so MACE receives stress in the
	convention it expects. Returns None if no stress block is present.
	"""
	m = re.search(
		r"total\s+stress\s+\(Ry/bohr\*\*3\).*?\n((?:.*\n){3})",
		content,
	)
	if not m:
		return None
	rows = [line.split() for line in m.group(1).rstrip("\n").split("\n")]
	if len(rows) < 3 or any(len(r) < 3 for r in rows):
		return None
	try:
		sxx, sxy, sxz = (float(x) for x in rows[0][:3])
		syy, syz = float(rows[1][1]), float(rows[1][2])
		szz = float(rows[2][2])
	except (ValueError, IndexError):
		return None
	voigt = np.array([sxx, syy, szz, syz, sxz, sxy])
	voigt *= -1.0 * units.Ry / units.Bohr ** 3
	return voigt


def _extract_structure(content):
	"""Extract cell, symbols, positions from QE output."""
	# Cell from CELL_PARAMETERS block.
	cell_match = re.search(
		r"CELL_PARAMETERS.*\n((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n){3})",
		content,
	)
	if cell_match:
		cell = np.array([
			[float(x) for x in line.split()]
			for line in cell_match.group(1).strip().split("\n")
		])
	else:
		# Fallback: lattice vectors from a(i) lines.
		alat_m = re.search(r"celldm\(1\)\s*=\s+([\d.]+)", content)
		alat = float(alat_m.group(1)) * BOHR_TO_ANG if alat_m else 1.0
		vecs = re.findall(
			r"a\(\d\)\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)",
			content,
		)
		cell = np.array([[float(x) for x in v] for v in vecs]) * alat

	# Positions from Cartesian axes block.
	pos_block = re.search(
		r"Cartesian axes.*?site.*?\n((?:.*\n)*?)(?:\n\s*\n|$)", content
	)
	symbols, positions = [], []
	if pos_block:
		for line in pos_block.group(1).strip().split("\n"):
			parts = line.split()
			if len(parts) >= 6:
				sym = re.sub(r'\d+', '', parts[1])
				positions.append(
					[float(parts[-4]), float(parts[-3]), float(parts[-2])]
				)
				symbols.append(sym)
		alat_m = re.search(r"celldm\(1\)\s*=\s+([\d.]+)", content)
		alat = float(alat_m.group(1)) * BOHR_TO_ANG if alat_m else 1.0
		positions = np.array(positions) * alat

	return cell, symbols, positions


def _extract_forces(content, n_atoms):
	# Match the per-atom force lines directly so a blank line between the
	# "Forces acting on atoms" header and the atom lines does not break parsing.
	pat = re.compile(
		r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*"
		r"(-?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+"
		r"(-?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+"
		r"(-?\d+\.\d+(?:[Ee][-+]?\d+)?)"
	)
	matches = pat.findall(content)
	if len(matches) < n_atoms:
		return None
	forces = [[float(x) for x in row] for row in matches[-n_atoms:]]
	return np.array(forces) * RY_TO_EV / BOHR_TO_ANG
