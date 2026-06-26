"""Configuration. One dataclass per concern, minimal fields."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# A sampled knob is either a single value (fixed) or a (min, max) pair drawn
# uniformly per draw. See utils/sampling.py.
NumOrRange = Union[float, Tuple[float, float]]
IntOrRange = Union[int, Tuple[int, int]]


@dataclass
class SystemConfig:
	elements: List[str] = field(default_factory=list)
	# composition values, n_atoms and density each take a single value (fixed)
	# or a (min, max) range drawn per RSS structure.
	composition: Dict[str, NumOrRange] = field(default_factory=dict)
	n_atoms: IntOrRange = 100
	density: NumOrRange = 2.0  # g/cm3
	min_distance: Optional[float] = None  # Angstrom, auto if None


@dataclass
class DFTConfig:
	pseudopotentials: Dict[str, str] = field(default_factory=dict)
	pseudo_dir: str = "./"
	ecutwfc: float = 80.0
	ecutrho: Optional[float] = None  # defaults to 8 * ecutwfc
	kpoints: List[int] = field(default_factory=lambda: [1, 1, 1])
	smearing: str = "mv"
	degauss: float = 0.01
	mixing_beta: float = 0.7
	electron_maxstep: int = 200
	conv_thr: float = 1.0e-8
	tstress: bool = True  # compute stress in QE; set False to save cost


@dataclass
class TrainerConfig:
	backend: str = "mace"  # "mace" or "nequip"
	device: str = "auto"  # "auto", "cpu", "cuda"
	committee_size: int = 3
	seed: int = 42

	# Foundation model for finetuning. None = train from scratch.
	# MACE: "small", "medium", "large", or path to .model file.
	# NequIP: path to .pth weights file.
	foundation_model: Optional[str] = None

	# Finetuning strategy (MACE only, used when foundation_model is set).
	# "naive"  - continue training all parameters from the checkpoint.
	# "multihead" - replay finetuning with a target and a pretraining head.
	finetune_strategy: str = "naive"
	# E0s for finetuning. "estimated" reestimates against the foundation
	# model; "average" fits per-element averages; or a dict string like
	# "{5:-77.5,7:-...}" for explicit isolated-atom energies. From-scratch
	# always uses "average".
	e0s: str = "estimated"
	weight_decay: float = 0.0  # 0 recommended for finetuning
	# Multihead replay options (used when finetune_strategy == "multihead").
	pt_train_file: Optional[str] = None  # replay xyz path, or "mp"
	pseudolabel_replay: bool = False  # label replay set with the foundation model
	weight_pt: float = 1.0     # replay-head loss weight
	weight_ft: float = 1.0     # target-head loss weight

	# MACE-specific (ignored by NequIP).
	mace_r_max: float = 5.0
	mace_num_channels: int = 128
	mace_max_L: int = 1
	mace_epochs: int = 500
	mace_lr: float = 0.01
	mace_batch_size: int = 10
	# Previously hardcoded in the command builder. Defaults reproduce the
	# original command exactly.
	mace_valid_fraction: float = 0.1
	mace_default_dtype: str = "float64"  # "float64" or "float32"
	mace_model: str = "MACE"    # e.g. "MACE" or "ScaleShiftMACE"
	mace_e0s_scratch: str = "average"  # from-scratch E0s (was forced average)
	mace_scaling: str = "rms_forces_scaling"  # finetune scaling
	mace_ema: bool = True     # finetune EMA
	mace_ema_decay: float = 0.99
	mace_amsgrad: bool = True
	# Loss and per-term weights. None leaves the MACE default.
	mace_loss: Optional[str] = None
	mace_energy_weight: Optional[float] = None
	mace_forces_weight: Optional[float] = None
	mace_stress_weight: Optional[float] = None
	# Stress training. Opt-in and pairs with dft.tstress plus parsed stress.
	mace_compute_stress: bool = False
	mace_stress_key: str = "stress"
	# SWA (stochastic weight averaging) tail.
	mace_swa: bool = False
	mace_start_swa: Optional[int] = None
	mace_swa_lr: Optional[float] = None
	mace_swa_energy_weight: Optional[float] = None
	mace_swa_forces_weight: Optional[float] = None
	# Optimiser and LR scheduler.
	mace_optimizer: Optional[str] = None
	mace_scheduler: Optional[str] = None
	mace_lr_scheduler_gamma: Optional[float] = None
	mace_lr_factor: Optional[float] = None
	mace_scheduler_patience: Optional[int] = None
	# Architecture.
	mace_correlation: Optional[int] = None
	mace_num_interactions: Optional[int] = None
	mace_hidden_irreps: Optional[str] = None
	mace_mlp_irreps: Optional[str] = None
	mace_num_radial_basis: Optional[int] = None
	mace_distance_transform: Optional[str] = None
	# Training control.
	mace_patience: Optional[int] = None
	mace_eval_interval: Optional[int] = None
	mace_clip_grad: Optional[float] = None
	mace_keep_checkpoints: bool = False
	mace_save_cpu: bool = False
	# Escape hatch for any flag not modelled above. {"flag": "value"} emits
	# "--flag=value"; an empty or None value emits a bare "--flag".
	mace_extra_args: Dict[str, Optional[str]] = field(default_factory=dict)

	# NequIP-specific (ignored by MACE).
	nequip_r_max: float = 5.0
	nequip_num_layers: int = 4
	nequip_l_max: int = 2
	nequip_num_features: int = 32
	nequip_max_epochs: int = 1000
	nequip_learning_rate: float = 0.005
	nequip_batch_size: int = 5


@dataclass
class ActiveLearningConfig:
	max_iterations: int = 30
	md_temp: NumOrRange = 300.0   # K, single value or (min, max) drawn per iteration
	md_steps: IntOrRange = 5000
	md_timestep: float = 1.0  # fs
	md_interval: int = 10  # collect every N steps
	qbc_select_fraction: float = 0.1
	# Absolute disagreement floor for selection (force std, eV/Ang). Structures
	# below this are not selected. 0 keeps the old behaviour of always taking at
	# least the top structure. Set > 0 so stop_on_no_new can trigger.
	qbc_min_disagreement: float = 0.0
	max_new_per_iter: int = 50
	rmse_energy_tol: float = 0.005 # eV/atom
	rmse_force_tol: float = 0.1  # eV/Ang
	validation_fraction: float = 0.1
	stop_on_no_new: bool = True


@dataclass
class SchedulerConfig:
	backend: str = "local" # "local" or "pbs"
	cores: int = 4
	cores_per_dft_job: Optional[int] = None  # for local parallelism
	mpi_command: str = "mpirun"
	dft_executable: str = "pw.x"
	walltime: str = "02:00:00"
	queue: str = ""
	account: Optional[str] = None
	modules: List[str] = field(default_factory=list)
	env_vars: Dict[str, str] = field(default_factory=dict)
	poll_interval: float = 30.0
	# PBS DFT submission. One job per structure, nodes=1:ppn=cores_per_dft_job,
	# launched with mpi_command. pbs_max_concurrent caps how many run at once,
	# enforced by the driver, not a job-array slot limit. pbs backend only.
	pbs_max_concurrent: int = 4


@dataclass
class RSSConfig:
	"""Random structure sampling settings."""
	n_structures: int = 20
	# Foundation model for pre-relaxation. None = no pre-relax.
	# e.g. "mace_mp" to use mace-mp-0, or a path to a model file.
	prerelax_model: Optional[str] = None
	prerelax_fmax: float = 5.0 # eV/Ang, loose on purpose
	prerelax_steps: int = 50


@dataclass
class DistillConfig:
	"""Synthetic-data distillation: compress a teacher into a small student.

	A trained teacher (or committee) labels a large set of cheaply generated
	structures, and a small student is trained from scratch on those labels.
	No DFT is used at the labelling step except for structures the teacher
	committee disagrees on, which are optionally sent to QE.
	"""
	teacher_model_paths: List[str] = field(default_factory=list)
	seed_structures: Optional[str] = None  # extxyz of real (amorphous) seeds

	# Synthetic generation (rattle-relax family tree per seed). The sigmas,
	# relax_steps and temperature each take a single value or a (min, max) range.
	n_synthetic: int = 2000
	rattle_sigma_pos: NumOrRange = 0.1   # Angstrom, per-atom displacement std
	rattle_sigma_cell: NumOrRange = 0.03  # cell strain std (dimensionless)
	relax_steps: IntOrRange = 20
	temperature: NumOrRange = 1000.0   # K, explore/early-stop temperature
	beta: float = 0.5     # explore (energy) vs exploit (generation)
	max_force_stop: float = 30.0  # eV/Angstrom, force ceiling for early stop

	# Dimers (short-range two-body coverage).
	add_dimers: bool = True
	dimers_per_pair: int = 50
	dimer_min_factor: float = 0.6  # times sum of covalent radii
	dimer_max_factor: float = 3.0
	dimer_box: float = 20.0    # Angstrom cubic box for isolated dimers

	# Committee disagreement gate.
	force_std_threshold: float = 0.2  # eV/Angstrom; above this is flagged
	route_flagged_to_qe: bool = True  # send flagged structures to QE, else drop

	# Student model (trained from scratch, small, with ZBL repulsion).
	student_num_channels: int = 32
	student_max_L: int = 0
	student_r_max: float = 4.5
	student_epochs: int = 400
	student_lr: float = 0.01
	student_batch_size: int = 10
	student_pair_repulsion: bool = True  # add ZBL short-range repulsion
	# Loss weights for the main (stage one) training.
	student_loss: str = "weighted"
	student_energy_weight: float = 1.0
	student_forces_weight: float = 100.0
	# SWA tail (stage two). Off by default.
	student_swa: bool = False
	student_start_swa: int = 300
	student_swa_lr: float = 1.0e-3
	student_swa_energy_weight: float = 1000.0
	student_swa_forces_weight: float = 100.0
	validation_fraction: float = 0.1


@dataclass
class Config:
	"""Top-level config. Five modes:

	finetune:  have data, finetune a foundation model, done.
	train:   have data, train from scratch, done.
	finetune_al: no data (or seed data), finetune with active learning.
	train_al:  no data (or seed data), train from scratch with AL.
	distill:  have a trained teacher, distil it into a small student.
	"""
	mode: str = "finetune" # finetune, train, finetune_al, train_al, distill
	# "new" starts a fresh run. "resume" continues an interrupted run in the
	# same work_dir: it skips DFT jobs that already finished, lets MACE pick up
	# from its latest checkpoint, and skips synthetic generation if the
	# distillation set is already written.
	start: str = "new" # "new" or "resume"
	data_path: Optional[str] = None  # path to extxyz, dir, or list
	work_dir: str = "./automlip_run"
	seed: Optional[int] = None # None draws a random seed at runtime
	cleanup_dft_scratch: bool = False

	system: SystemConfig = field(default_factory=SystemConfig)
	dft: DFTConfig = field(default_factory=DFTConfig)
	trainer: TrainerConfig = field(default_factory=TrainerConfig)
	active: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
	scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
	rss: RSSConfig = field(default_factory=RSSConfig)
	distill: DistillConfig = field(default_factory=DistillConfig)

	# Override auto-detected keys in extxyz files.
	energy_key: Optional[str] = None
	force_key: Optional[str] = None

	def __post_init__(self):
		if self.seed is None:
			import secrets
			self.seed = secrets.randbits(32)
