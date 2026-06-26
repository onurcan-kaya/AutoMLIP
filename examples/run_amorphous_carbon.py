"""
Amorphous carbon
"""
from automlip import Config, Pipeline
from automlip.config import (
	SystemConfig, TrainerConfig, RSSConfig, DFTConfig, SchedulerConfig,
)
FOUNDATION = "/home/okaya/foundation_models/mace-omat-0-medium.model"  # set this
cfg = Config(
	mode="finetune",			# RSS -> relax -> DFT -> finetune, no AL
	start="new",				# "resume" continues an interrupted run in work_dir
	data_path=None,				# None triggers RSS generation. do NOT set a file here
	work_dir="./",
	seed=42,					# sampling seed (RSS, relax). delete for a random seed
	cleanup_dft_scratch=False,	# keep QE wavefunctions after each job
	system=SystemConfig(
		elements=["C"],
		composition={"C": 1.0},		# single value or per-element (min, max)
		n_atoms=64,					# fixed size keeps DFT cost predictable. (min,max) for a spread
		density=(1.0, 3.6),			# g/cm3, drawn per structure. a single value is fixed
		min_distance=1.1,			# Angstrom, auto from covalent radii if None
	),
	rss=RSSConfig(
		n_structures=400,			 # initial structures generated and labelled
		prerelax_model=FOUNDATION,	# relax each RSS structure with this, on trainer.device
		prerelax_fmax=0.1,			# eV/Angstrom relaxation target
		prerelax_steps=100,			# lower for a quicker generation phase
	),
	trainer=TrainerConfig(
		backend="mace",
		device="cuda",				# also the device the RSS relax runs on
		committee_size=1,			# single model
		seed=256,					# MACE training seed
		foundation_model=FOUNDATION,
		finetune_strategy="naive",
		e0s="average",				# fit C E0 from your QE labels. or "{6: <QE isolated C energy>}"
		weight_decay=0.0,
		mace_valid_fraction=0.4,	# the only split, MACE validates on exactly this
		# Architecture below is ignored while finetuning, the foundation model
		# sets it. Left here only for reference.
		mace_r_max=6.0,
		mace_num_channels=64,
		mace_max_L=1,
		mace_epochs=400,
		mace_lr=1.0e-3,
		mace_batch_size=8,
		# Loss, forces dominate.
		mace_loss="huber",
		mace_energy_weight=1.0,
		mace_forces_weight=1000.0,
		# SWA tail, refine energies in the last quarter of training.
		mace_swa=True,
		mace_start_swa=300,
		mace_swa_lr=1.0e-3,
		mace_swa_energy_weight=1000.0,
		mace_swa_forces_weight=100.0,
		# Training control.
		mace_patience=50,			# early stop if validation stalls
		mace_keep_checkpoints=False,
		mace_default_dtype="float64",
	),
	dft=DFTConfig(
		pseudopotentials={
			"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",	  # set to your actual C UPF filename
		},
		pseudo_dir="/home/okaya/pseudopotentials",	 # set this
		ecutwfc=80.0,
		ecutrho=640.0,				# 8 * ecutwfc for PAW
		kpoints=[1, 1, 1],			# gamma may be enough for a 64-atom cell
		smearing="mv",
		degauss=0.02,
		mixing_beta=0.7,
		electron_maxstep=200,
		conv_thr=1.0e-8,
		tstress=False,				# not training on stress
	),
	scheduler=SchedulerConfig(
		backend="pbs",
		cores=8,
		cores_per_dft_job=8,		# cores per DFT job (nodes=1:ppn=8)
		mpi_command="mpiexec.hydra -machinefile $PBS_NODEFILE -np $(wc -l < $PBS_NODEFILE)",
		dft_executable="pw.x",
		walltime="4:00:00",
		account="cnm84219",
		pbs_max_concurrent=4,		# DFT jobs running at once
		modules=[
			"source /etc/profile",
			"module load quantum-espresso/7.3/impi-2021/intel-2021/7.3.1-5",
		],
		poll_interval=10.0,
	),
)
if __name__ == "__main__":
	Pipeline(cfg).run()
