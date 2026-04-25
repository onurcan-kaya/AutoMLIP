"""All user-facing configuration in one place."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SystemConfig:
    """What the system is made of and how big the simulation box should be."""
    elements: List[str] = field(default_factory=list)
    composition: Dict[str, float] = field(default_factory=dict)
    n_atoms: int = 100
    density: float = 2.0
    min_distance: Optional[float] = None
    prerelax_model: Optional[str] = None


@dataclass
class DFTConfig:
    """Reference DFT calculation settings (Quantum ESPRESSO)."""
    code: str = "qe"
    pseudopotentials: Dict[str, str] = field(default_factory=dict)
    pseudo_dir: str = "./"
    ecutwfc: float = 80.0
    ecutrho: Optional[float] = None
    kpoints: List[int] = field(default_factory=lambda: [1, 1, 1])
    functional: str = "PBE"
    calc_type: str = "scf"
    electron_maxstep: int = 200
    conv_thr: float = 1.0e-8
    degauss: float = 0.01
    smearing: str = "mv"
    mixing_beta: float = 0.7
    extra_params: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """MLIP training settings."""
    backend: str = "gap"  # "gap", "mace", "nequip"
    committee_size: int = 3
    bootstrap_fraction: float = 0.8
    seed: int = 42
    n_cores: int = 4

    # Device selection (shared across MACE and NequIP, ignored by GAP).
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0"

    # GAP SOAP parameters.
    soap_cutoff: Optional[float] = None
    soap_lmax: Optional[int] = None
    soap_nmax: Optional[int] = None
    soap_n_sparse: Optional[int] = None
    soap_delta: Optional[float] = None
    sigma: Optional[List[float]] = None
    do_hyperopt: bool = False
    hyperopt_trials: int = 20
    extra_gap_args: Dict[str, object] = field(default_factory=dict)

    # MACE parameters.
    mace_model_type: str = "MACE"
    mace_max_L: int = 1
    mace_num_channels: int = 128
    mace_r_max: float = 5.0
    mace_epochs: int = 500
    mace_lr: float = 0.01
    mace_batch_size: int = 10
    mace_run_command: str = "mace_run_train"

    # NequIP parameters.
    nequip_r_max: float = 5.0
    nequip_num_layers: int = 4
    nequip_l_max: int = 2
    nequip_num_features: int = 32
    nequip_num_basis: int = 8
    nequip_max_epochs: int = 1000
    nequip_learning_rate: float = 0.005
    nequip_batch_size: int = 5
    nequip_loss_energy_weight: float = 1.0
    nequip_loss_force_weight: float = 100.0


@dataclass
class ActiveLearningConfig:
    """Active learning loop settings."""
    n_initial: int = 20
    validation_fraction: float = 0.1
    md_temp: float = 300.0
    md_steps: int = 5000
    md_timestep: float = 1.0
    md_interval: int = 10
    qbc_energy_threshold: Optional[float] = None
    qbc_force_threshold: Optional[float] = None
    qbc_select_fraction: float = 0.1
    max_new_per_iter: int = 50
    max_iterations: int = 50
    rmse_energy_tol: float = 0.005
    rmse_force_tol: float = 0.1
    stop_on_no_new: bool = True


@dataclass
class SchedulerConfig:
    """HPC job scheduler settings."""
    backend: str = "local"  # "local", "slurm", "pbs"
    queue: str = "default"
    walltime: str = "02:00:00"
    nodes: int = 1
    cores_per_node: int = 4
    cores_per_dft_job: Optional[int] = None
    mpi_command: str = "mpirun"
    dft_executable: str = "pw.x"
    poll_interval: float = 30.0
    modules: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    account: Optional[str] = None


@dataclass
class Config:
    """Top-level configuration."""
    system: SystemConfig = field(default_factory=SystemConfig)
    dft: DFTConfig = field(default_factory=DFTConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    active: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    work_dir: str = "./automlip_run"
    seed: int = 42
    log_level: str = "INFO"

    # Pipeline mode.
    mode: str = "full"  # "full", "al_from_data", "train_only"
    initial_data_path: Optional[str] = None
    energy_key: Optional[str] = None
    force_key: Optional[str] = None

    # tau_acc convergence settings.
    use_tau_acc: bool = False
    tau_check_interval: int = 5
    tau_e_lower: float = 0.01
    tau_e_thresh: float = 0.1
    tau_max_fs: float = 1000.0
    tau_interval_fs: float = 10.0
    tau_temp: float = 300.0
    tau_dt_fs: float = 1.0
    tau_n_configs: int = 5
