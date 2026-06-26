"""AutoMLIP run on Argonne CNM 'Carbon' cluster (PBS).

This is the DRIVER. It runs on one node and does two things:
  - trains MACE committee members as local subprocesses on that node,
  - submits the DFT labelling as a PBS job array via qsub.
So launch it inside a GPU job that also has qsub access (see submit_driver.pbs).

Re-running this exact script in the same work_dir resumes the active-learning
loop from the last completed iteration. See the notes at the bottom for what
does and does not resume within an iteration.

Fill every <...> placeholder for your environment before running.
"""

from automlip import Config, Pipeline
from automlip.config import (
	SystemConfig, DFTConfig, TrainerConfig,
	ActiveLearningConfig, SchedulerConfig, RSSConfig,
)


cfg = Config(
	mode="train_al",      # from scratch + active learning
	start="new",       # set to "resume" to continue this run
	work_dir="<SCRATCH>/abno_run",   # persistent dir; resume reads from here
	seed=42,
	cleanup_dft_scratch=True,    # delete QE wavefunctions after each job

	system=SystemConfig(
		elements=["B", "N", "O"],
		composition={"B": 0.45, "N": 0.45, "O": 0.10},
		n_atoms=96,
		density=2.1,      # g/cm3
	),

	dft=DFTConfig(
		# PAW PBE pseudopotentials, filenames as they sit in pseudo_dir.
		pseudopotentials={
			"B": "<B.pbe-n-kjpaw_psl.UPF>",
			"N": "<N.pbe-n-kjpaw_psl.UPF>",
			"O": "<O.pbe-n-kjpaw_psl.UPF>",
		},
		pseudo_dir="<PATH/TO/pseudopotentials>",
		ecutwfc=60.0,
		ecutrho=480.0,      # PAW: ~8x ecutwfc
		kpoints=[2, 2, 2],
		smearing="mv",
		degauss=0.01,
		mixing_beta=0.7,
		electron_maxstep=200,
		conv_thr=1.0e-8,
	),

	trainer=TrainerConfig(
		backend="mace",
		device="cuda",
		committee_size=3,
		mace_r_max=5.0,
		mace_num_channels=128,
		mace_max_L=1,
		mace_epochs=400,
		mace_lr=0.01,
		mace_batch_size=10,
		# To finetune instead of train from scratch, pre-download a foundation
		# model to a local path (compute nodes usually have no internet) and set:
		# foundation_model="<PATH/TO/mace_foundation.model>",
		# finetune_strategy="naive",  # or "multihead" (needs pt_train_file)
		# e0s="estimated",
	),

	active=ActiveLearningConfig(
		max_iterations=10,
		md_temp=1000.0,      # hot MD to sample disorder around seeds
		md_steps=2000,
		md_timestep=1.0,
		md_interval=50,
		qbc_select_fraction=0.1,
		max_new_per_iter=30,
		rmse_energy_tol=0.005,    # eV/atom
		rmse_force_tol=0.10,    # eV/Angstrom
		validation_fraction=0.1,
		stop_on_no_new=True,
	),

	scheduler=SchedulerConfig(
		backend="pbs",
		dft_executable="pw.x",
		cores_per_dft_job=8,    # cores per DFT job (nodes=1:ppn=8)
		mpi_command="mpiexec.hydra -machinefile $PBS_NODEFILE -np $(wc -l < $PBS_NODEFILE)",
		pbs_max_concurrent=4,    # DFT jobs running at once
		walltime="06:00:00",    # per DFT array task
		queue="<your_queue>",
		account="cnm84219",     # confirm your current CNM allocation
		modules=[
			# shell lines run at the top of the array script, e.g.:
			# "module load quantum-espresso",
		],
		env_vars={
			# "ESPRESSO_PSEUDO": "<PATH/TO/pseudopotentials>",
		},
		poll_interval=60.0,
	),

	rss=RSSConfig(
		n_structures=30,     # initial random structures to DFT-label
		prerelax_model=None,    # set to a local model path to pre-relax
		prerelax_fmax=5.0,
		prerelax_steps=50,
	),
)


if __name__ == "__main__":
	Pipeline(cfg).run()
