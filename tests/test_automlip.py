"""Tests for data loader and QE error handler."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write


def _make_test_atoms(n=5, energy_key="energy", force_key="forces"):
    """Create a list of dummy Atoms with energy and forces."""
    atoms_list = []
    for i in range(n):
        a = Atoms("BN", positions=[[0, 0, 0], [1.5, 0, 0]],
                  cell=[5, 5, 5], pbc=True)
        a.info[energy_key] = -10.0 + i * 0.1
        a.arrays[force_key] = np.random.randn(2, 3) * 0.1
        atoms_list.append(a)
    return atoms_list


class TestDataLoader:
    def test_load_standard_keys(self, tmp_path):
        from automlip.utils.data_loader import load_dataset
        atoms = _make_test_atoms(5, "energy", "forces")
        path = tmp_path / "test.extxyz"
        ase_write(str(path), atoms, format="extxyz")
        loaded = load_dataset(path)
        assert len(loaded) == 5
        assert "energy" in loaded[0].info
        assert "forces" in loaded[0].arrays

    def test_load_ref_keys(self, tmp_path):
        from automlip.utils.data_loader import load_dataset
        atoms = _make_test_atoms(5, "REF_energy", "REF_forces")
        path = tmp_path / "test.extxyz"
        ase_write(str(path), atoms, format="extxyz")
        loaded = load_dataset(path)
        assert len(loaded) == 5
        # Should be normalised to "energy" and "forces"
        assert "energy" in loaded[0].info
        assert "forces" in loaded[0].arrays

    def test_load_custom_keys(self, tmp_path):
        from automlip.utils.data_loader import load_dataset
        atoms = _make_test_atoms(5, "my_E", "my_F")
        path = tmp_path / "test.extxyz"
        ase_write(str(path), atoms, format="extxyz")
        loaded = load_dataset(path, energy_key="my_E", force_key="my_F")
        assert len(loaded) == 5

    def test_load_directory(self, tmp_path):
        from automlip.utils.data_loader import load_dataset
        for i in range(3):
            atoms = _make_test_atoms(2)
            ase_write(str(tmp_path / f"batch_{i}.extxyz"), atoms, format="extxyz")
        loaded = load_dataset(tmp_path)
        assert len(loaded) == 6

    def test_split_dataset(self):
        from automlip.utils.data_loader import split_dataset
        atoms = _make_test_atoms(20)
        train, val = split_dataset(atoms, train_fraction=0.8, seed=42)
        assert len(train) == 16
        assert len(val) == 4

    def test_missing_energy_skipped(self, tmp_path):
        from automlip.utils.data_loader import load_dataset
        atoms = []
        for i in range(5):
            a = Atoms("BN", positions=[[0, 0, 0], [1.5, 0, 0]],
                      cell=[5, 5, 5], pbc=True)
            if i < 3:
                a.info["energy"] = -10.0
            a.arrays["forces"] = np.zeros((2, 3))
            atoms.append(a)
        path = tmp_path / "test.extxyz"
        ase_write(str(path), atoms, format="extxyz")
        loaded = load_dataset(path)
        assert len(loaded) == 3


class TestQEErrors:
    def test_converged(self, tmp_path):
        from automlip.calculators.qe_errors import QEErrorHandler
        output = tmp_path / "pw.out"
        output.write_text("! total energy = -100.0 Ry\nForces acting on atoms\nJOB DONE.\n")
        handler = QEErrorHandler()
        diag = handler.diagnose(tmp_path)
        assert diag.converged
        assert not diag.failed

    def test_scf_not_converged(self, tmp_path):
        from automlip.calculators.qe_errors import QEErrorHandler
        output = tmp_path / "pw.out"
        output.write_text("iteration #  1\nconvergence NOT achieved\n")
        handler = QEErrorHandler()
        diag = handler.diagnose(tmp_path)
        assert diag.failed
        assert diag.retryable
        assert "mixing_beta" in diag.fixes

    def test_no_output(self, tmp_path):
        from automlip.calculators.qe_errors import QEErrorHandler
        handler = QEErrorHandler()
        diag = handler.diagnose(tmp_path)
        assert diag.failed

    def test_crash_file(self, tmp_path):
        from automlip.calculators.qe_errors import QEErrorHandler
        (tmp_path / "CRASH").write_text("segfault")
        handler = QEErrorHandler()
        diag = handler.diagnose(tmp_path)
        assert diag.failed
        assert "memory" in diag.reason.lower() or "oom" in diag.reason.lower() or "crash" in diag.reason.lower()

    def test_s_matrix(self, tmp_path):
        from automlip.calculators.qe_errors import QEErrorHandler
        output = tmp_path / "pw.out"
        output.write_text("S matrix not positive definite\n")
        handler = QEErrorHandler()
        diag = handler.diagnose(tmp_path)
        assert diag.failed
        assert "ecutwfc" in diag.fixes

    def test_max_retries_respected(self, tmp_path):
        from automlip.calculators.qe_errors import QEErrorHandler
        output = tmp_path / "pw.out"
        output.write_text("convergence NOT achieved\n")
        handler = QEErrorHandler(max_retries=2)
        diag = handler.diagnose(tmp_path, retry_count=2)
        assert not diag.retryable


class TestGenerator:
    def test_random_generation(self):
        from automlip.generators.random import generate_random_structures
        structures = generate_random_structures(
            elements=["B", "N"],
            composition={"B": 0.5, "N": 0.5},
            n_atoms=10,
            density=2.0,
            n_structures=5,
            seed=42,
        )
        assert len(structures) == 5
        for s in structures:
            assert len(s) == 10
            assert all(s.pbc)
            symbols = s.get_chemical_symbols()
            assert "B" in symbols
            assert "N" in symbols


class TestTrainerFactory:
    def test_factory_gap(self):
        from automlip.trainers import make_trainer
        from automlip.config import SystemConfig, TrainerConfig
        t = make_trainer(SystemConfig(elements=["Si"]), TrainerConfig(backend="gap"))
        assert t.__class__.__name__ == "GAPTrainer"

    def test_factory_mace(self):
        from automlip.trainers import make_trainer
        from automlip.config import SystemConfig, TrainerConfig
        t = make_trainer(SystemConfig(elements=["Si"]), TrainerConfig(backend="mace"))
        assert t.__class__.__name__ == "MACETrainer"

    def test_factory_nequip(self):
        from automlip.trainers import make_trainer
        from automlip.config import SystemConfig, TrainerConfig
        t = make_trainer(SystemConfig(elements=["Si"]), TrainerConfig(backend="nequip"))
        assert t.__class__.__name__ == "NequIPTrainer"

    def test_factory_unknown(self):
        from automlip.trainers import make_trainer
        from automlip.config import SystemConfig, TrainerConfig
        with pytest.raises(ValueError):
            make_trainer(SystemConfig(elements=["Si"]), TrainerConfig(backend="unknown"))


class TestDeviceResolution:
    def test_gap_always_cpu(self):
        from automlip.trainers.base import BaseTrainer
        from automlip.config import TrainerConfig
        assert BaseTrainer.resolve_device(TrainerConfig(backend="gap", device="cuda")) == "cpu"

    def test_auto_without_torch(self):
        from automlip.trainers.base import BaseTrainer
        from automlip.config import TrainerConfig
        # "auto" should return something without crashing
        result = BaseTrainer.resolve_device(TrainerConfig(backend="mace", device="auto"))
        assert result in ("cpu", "cuda")
