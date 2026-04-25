"""
Example: Train a GAP potential for amorphous boron nitride.

Usage:
    python example_abn.py
"""

from automlip import Config
from automlip.config import (
    SystemConfig, DFTConfig, TrainerConfig,
    ActiveLearningConfig, SchedulerConfig,
)
from automlip.pipeline import Pipeline

config = Config(
    mode="full",

    system=SystemConfig(
        elements=["B", "N"],
        composition={"B": 0.5, "N": 0.5},
        n_atoms=100,
        density=2.1,
    ),

    dft=DFTConfig(
        code="qe",
        pseudopotentials={"B": "B.pbe-n-kjpaw_psl.1.0.0.UPF", "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF"},
        pseudo_dir="/path/to/pseudopotentials/",
        ecutwfc=80.0, ecutrho=640.0,
        kpoints=[1, 1, 1],
    ),

    trainer=TrainerConfig(
        backend="gap",
        committee_size=3,
        bootstrap_fraction=0.8,
        soap_cutoff=4.5, soap_lmax=6, soap_nmax=8,
        soap_n_sparse=1000, soap_delta=1.0,
        sigma=[0.002, 0.2, 0.0, 0.0],
    ),

    active=ActiveLearningConfig(
        n_initial=20, md_temp=300.0, md_steps=5000,
        qbc_select_fraction=0.1, max_new_per_iter=50,
        max_iterations=50, rmse_energy_tol=0.005, rmse_force_tol=0.1,
    ),

    scheduler=SchedulerConfig(
        backend="slurm", queue="batch", walltime="02:00:00",
        nodes=1, cores_per_node=36, mpi_command="mpirun",
        dft_executable="pw.x", account="your_account",
        modules=["module load espresso/7.2"],
    ),

    work_dir="./abn_gap_run", seed=42,
)

if __name__ == "__main__":
    pipeline = Pipeline(config)
    pipeline.run()
