"""Quantum ESPRESSO pw.x calculator."""

import logging, re
from pathlib import Path
from typing import Optional
import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from automlip.calculators.base import BaseCalculator, CalcStatus
from automlip.config import DFTConfig, SchedulerConfig

logger = logging.getLogger("automlip.calculators.qe")
RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177249
INPUT_FILE = "pw.in"
OUTPUT_FILE = "pw.out"


class QECalculator(BaseCalculator):
    def __init__(self, dft_config: DFTConfig, scheduler_config: SchedulerConfig):
        self.dft = dft_config
        self.sched = scheduler_config

    def write_input(self, atoms: Atoms, calc_dir: Path) -> Path:
        calc_dir.mkdir(parents=True, exist_ok=True)
        input_path = calc_dir / INPUT_FILE
        ecutrho = self.dft.ecutrho if self.dft.ecutrho is not None else 8 * self.dft.ecutwfc
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        unique_elements = list(dict.fromkeys(symbols))
        n_atoms = len(atoms)
        n_types = len(unique_elements)
        kpts = self.dft.kpoints

        lines = []
        lines.append("&CONTROL")
        lines.append(f"  calculation = '{self.dft.calc_type}',")
        lines.append(f"  pseudo_dir = '{self.dft.pseudo_dir}',")
        lines.append(f"  outdir = './tmp/',")
        lines.append(f"  tprnfor = .true.,")
        lines.append(f"  tstress = .true.,")
        for k, v in self.dft.extra_params.items():
            if k.startswith("control_"):
                lines.append(f"  {k.replace('control_', '')} = {v},")
        lines.append("/")
        lines.append("")
        lines.append("&SYSTEM")
        lines.append(f"  ibrav = 0,")
        lines.append(f"  nat = {n_atoms},")
        lines.append(f"  ntyp = {n_types},")
        lines.append(f"  ecutwfc = {self.dft.ecutwfc},")
        lines.append(f"  ecutrho = {ecutrho},")
        lines.append(f"  occupations = 'smearing',")
        lines.append(f"  smearing = '{self.dft.smearing}',")
        lines.append(f"  degauss = {self.dft.degauss},")
        lines.append("/")
        lines.append("")
        lines.append("&ELECTRONS")
        lines.append(f"  electron_maxstep = {self.dft.electron_maxstep},")
        lines.append(f"  conv_thr = {self.dft.conv_thr},")
        lines.append(f"  mixing_beta = {self.dft.mixing_beta},")
        lines.append("/")
        lines.append("")
        lines.append("ATOMIC_SPECIES")
        for sp in unique_elements:
            mass = atomic_masses[atomic_numbers[sp]]
            pp = self.dft.pseudopotentials.get(sp, f"{sp}.UPF")
            lines.append(f"  {sp}  {mass:.4f}  {pp}")
        lines.append("")
        lines.append("CELL_PARAMETERS angstrom")
        for row in cell:
            lines.append(f"  {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}")
        lines.append("")
        lines.append("ATOMIC_POSITIONS angstrom")
        for sym, pos in zip(symbols, positions):
            lines.append(f"  {sym}  {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}")
        lines.append("")
        lines.append("K_POINTS automatic")
        lines.append(f"  {kpts[0]} {kpts[1]} {kpts[2]}  0 0 0")
        lines.append("")
        with open(input_path, "w") as f:
            f.write("\n".join(lines))
        return input_path

    def parse_output(self, calc_dir: Path) -> Optional[Atoms]:
        output_path = calc_dir / OUTPUT_FILE
        if not output_path.exists():
            return None
        try:
            content = output_path.read_text(errors="replace")
            if "JOB DONE." not in content:
                return None
            energy_ry = self._extract_energy(content)
            if energy_ry is None:
                return None
            energy_ev = energy_ry * RY_TO_EV
            cell, symbols, positions = self._extract_structure(content)
            forces = self._extract_forces(content, len(symbols))
            if forces is None:
                return None
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            atoms.info["energy"] = energy_ev
            atoms.arrays["forces"] = forces
            stress = self._extract_stress(content)
            if stress is not None:
                atoms.info["stress"] = stress
            return atoms
        except Exception as e:
            logger.error("Parse error in %s: %s", calc_dir, e)
            return None

    def check_status(self, calc_dir: Path) -> CalcStatus:
        output_path = calc_dir / OUTPUT_FILE
        if not output_path.exists():
            return CalcStatus.NOT_STARTED
        content = output_path.read_text(errors="replace")
        if "JOB DONE." in content:
            return CalcStatus.CONVERGED
        if "convergence NOT achieved" in content:
            return CalcStatus.FAILED
        return CalcStatus.RUNNING

    def get_run_command(self, input_path: Path) -> str:
        nprocs = self.sched.cores_per_node * self.sched.nodes
        mpi = self.sched.mpi_command
        exe = self.sched.dft_executable
        if mpi:
            return f"{mpi} -np {nprocs} {exe} -i {input_path} > {OUTPUT_FILE} 2>&1"
        return f"{exe} -i {input_path} > {OUTPUT_FILE} 2>&1"

    def _extract_energy(self, content):
        matches = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", content)
        if matches:
            return float(matches[-1])
        return None

    def _extract_structure(self, content):
        alat_match = re.search(r"celldm\(1\)\s*=\s+([\d.]+)", content)
        alat = float(alat_match.group(1)) * BOHR_TO_ANG if alat_match else 1.0

        cell_lines = []
        cell_match = re.search(r"CELL_PARAMETERS.*\n((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n){3})", content)
        if cell_match:
            for line in cell_match.group(1).strip().split("\n"):
                vals = [float(x) for x in line.split()]
                cell_lines.append(vals)
            cell = np.array(cell_lines)
        else:
            cell_match2 = re.findall(r"a\(\d\)\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", content)
            cell = np.array([[float(x) for x in m] for m in cell_match2]) * alat

        pos_block = re.search(r"Cartesian axes.*?site.*?\n((?:.*\n)*?)(?:\n\s*\n|$)", content)
        symbols, positions = [], []
        if pos_block:
            for line in pos_block.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 6:
                    sym = re.sub(r'\d+', '', parts[1])
                    positions.append([float(parts[-4]), float(parts[-3]), float(parts[-2])])
                    symbols.append(sym)
            positions = np.array(positions) * alat

        return cell, symbols, positions

    def _extract_forces(self, content, n_atoms):
        force_block = re.search(r"Forces acting on atoms.*?\n((?:.*force.*\n)+)", content)
        if not force_block:
            return None
        forces = []
        for line in force_block.group(1).strip().split("\n"):
            match = re.search(r"force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if match:
                forces.append([float(match.group(i)) for i in (1, 2, 3)])
        if len(forces) != n_atoms:
            return None
        return np.array(forces) * RY_TO_EV / BOHR_TO_ANG

    def _extract_stress(self, content):
        stress_match = re.search(r"total\s+stress.*\n((?:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n){3})", content)
        if not stress_match:
            return None
        lines = stress_match.group(1).strip().split("\n")
        tensor = np.array([[float(x) for x in l.split()[:3]] for l in lines])
        return tensor.flatten()[[0, 4, 8, 5, 2, 1]]
