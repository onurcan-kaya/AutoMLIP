"""Pipeline: the single entry point.

Modes:
	finetune     - have data, finetune foundation model, done.
	train        - have data, train from scratch, done.
	finetune_al  - generate/load seed data, finetune with active learning.
	train_al     - generate/load seed data, train from scratch with AL.
	distill      - have a trained teacher, distil it into a small student.

start = "new" (default) or "resume". Resume continues an interrupted run in
the same work_dir: finished DFT jobs are skipped, MACE continues from its
latest checkpoint, and distillation skips generation if the synthetic set is
already written.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from automlip.config import Config
from automlip.utils.data import load_dataset, split_dataset
from automlip.utils.checkpoint import (
	PipelineState, save_checkpoint, load_checkpoint,
)
from automlip.trainers import make_trainer
from automlip.samplers.rss import generate_rss
from automlip.samplers.md import run_md
from automlip.samplers.augment import generate_synthetic, generate_dimers
from automlip.labellers.batch import label_batch
from automlip.labellers.teacher import label_with_teacher
from automlip.selectors.committee import select_by_disagreement

logger = logging.getLogger("automlip")


class Pipeline:
	def __init__(self, config: Config):
		self.config = config
		self.work_dir = Path(config.work_dir)
		self.state = PipelineState()
		self.train_data: List[Atoms] = []
		self.val_data: List[Atoms] = []
		self.model_paths: List[Path] = []
		self.resume = False

		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
		)

	def run(self):
		mode = self.config.mode.lower().replace("-", "_")
		self.work_dir.mkdir(parents=True, exist_ok=True)
		self._setup_file_logging()
		self.resume = self._resolve_start()

		if mode in ("finetune", "train"):
			self._run_train_only()
		elif mode in ("finetune_al", "train_al"):
			self._run_active_learning()
		elif mode == "distill":
			self._run_distillation()
		else:
			raise ValueError(
				f"Unknown mode: {mode!r}. "
				f"Use: finetune, train, finetune_al, train_al, distill"
			)

	def _resolve_start(self) -> bool:
		"""Turn config.start ('new' or 'resume') into a resume flag.

		Neither value ever hard-stops. 'resume' with nothing to resume falls
		back to a new run; 'new' on top of an existing run warns and proceeds.
		"""
		start = self.config.start.lower()
		has_prior = self._has_prior_run()
		if start == "resume":
			if not has_prior:
				logger.warning("start='resume' but no previous run found in "
							   "%s; starting a new run.", self.work_dir)
				return False
			self._log_resume_point()
			return True
		if start == "new":
			if has_prior:
				logger.warning("start='new' but %s already contains a run; "
							   "proceeding and overwriting as it progresses.",
							   self.work_dir)
			return False
		raise ValueError(
			f"Unknown start: {self.config.start!r}. Use 'new' or 'resume'."
		)

	def _has_prior_run(self) -> bool:
		w = self.work_dir
		markers = ["checkpoint.json", "synthetic_data.extxyz",
				   "final_training_set.extxyz"]
		if any((w / m).exists() for m in markers):
			return True
		if (w / "student").is_dir():
			return True
		if any(w.glob("iter_*")):
			return True
		return False

	def _log_resume_point(self):
		"""Read what is on disk and report where the run stopped."""
		import json
		w = self.work_dir
		logger.info("Resuming run in %s", w)
		ckpt = w / "checkpoint.json"
		if ckpt.exists():
			try:
				meta = json.loads(ckpt.read_text())
				logger.info("Last checkpoint: iteration %s, %s training "
							"structures, %s model(s).",
							meta.get("iteration"),
							_count_extxyz(w / "training_data.extxyz"),
							len(meta.get("model_paths", [])))
				if meta.get("converged"):
					logger.info("Previous run already converged: %s",
								meta.get("reason"))
			except Exception:
				logger.info("Checkpoint present but could not be summarised.")
		syn = w / "synthetic_data.extxyz"
		if syn.exists():
			logger.info("Distillation: %s labelled synthetic structures "
						"present, generation will be skipped.",
						_count_extxyz(syn))

	def _setup_file_logging(self):
		"""Write all progress to work_dir/automlip.log as well as the console."""
		log_path = self.work_dir / "automlip.log"
		pkg_logger = logging.getLogger("automlip")
		for h in pkg_logger.handlers:
			if getattr(h, "_automlip_logfile", False):
				return
		fh = logging.FileHandler(log_path)
		fh.setLevel(logging.INFO)
		fh.setFormatter(logging.Formatter(
			"%(asctime)s [%(name)s] %(levelname)s: %(message)s"
		))
		fh._automlip_logfile = True
		pkg_logger.addHandler(fh)
		logger.info("Logging progress to %s", log_path)
		logger.info("Seed: %d", self.config.seed)

	# ----------------------------------------------------------------
	# Train-only: load data, train committee, validate, done.
	# ----------------------------------------------------------------

	def _obtain_dataset(self):
		"""Return a labelled dataset with no active learning: load from a file,
		or generate via RSS and label with DFT when data_path is not set."""
		if self.config.data_path is not None:
			return load_dataset(
				self.config.data_path,
				energy_key=self.config.energy_key,
				force_key=self.config.force_key,
			)
		sc = self.config.system
		rc = self.config.rss
		structures = generate_rss(
			elements=sc.elements,
			composition=sc.composition,
			n_atoms=sc.n_atoms,
			n_structures=rc.n_structures,
			density=sc.density,
			min_distance=sc.min_distance,
			seed=self.config.seed,
			prerelax_model=rc.prerelax_model,
			prerelax_fmax=rc.prerelax_fmax,
			prerelax_steps=rc.prerelax_steps,
			device=self.config.trainer.device,
		)
		data, _ = label_batch(
			structures, self.work_dir / "iter_000",
			self.config.dft, self.config.scheduler,
			cleanup=self.config.cleanup_dft_scratch,
			resume=self.resume,
		)
		if not data:
			raise RuntimeError("No structures were labelled")
		return data

	def _run_train_only(self):
		logger.info("=== Mode: %s ===", self.config.mode)

		data = self._obtain_dataset()
		# train/finetune only: MACE owns the split via trainer.mace_valid_fraction.
		# the active-learning block is never read here.
		self.train_data = data
		self.val_data = []

		trainer = make_trainer(self.config.system, self.config.trainer)
		model_dir = self.work_dir / "models"
		self.model_paths = trainer.train_committee(
			self.train_data, model_dir,
			n_models=self.config.trainer.committee_size,
			restart=self.resume,
		)

		ase_write(str(self.work_dir / "training_data.extxyz"),
				  self.train_data, format="extxyz")
		logger.info("Done. Models in %s", model_dir)

	# ----------------------------------------------------------------
	# Active learning: generate/load seed data, train, AL loop.
	# ----------------------------------------------------------------

	def _run_active_learning(self):
		logger.info("=== Mode: %s ===", self.config.mode)

		# Restore from checkpoint only when resuming.
		if self.resume:
			state, train, val = load_checkpoint(self.work_dir)
			if state is not None:
				self.state = state
				self.train_data = train
				self.val_data = val
				self.model_paths = [Path(p) for p in state.model_paths]
				logger.info("Restored from checkpoint: iter %d", state.iteration)

		# If starting fresh, build initial dataset.
		if self.state.iteration == 0:
			self._build_initial_data()
			self._train_initial_committee()

		# Run AL loop.
		self._al_loop()
		self._finalise()

	def _build_initial_data(self):
		"""Get initial training data: from file or from RSS + DFT."""
		data = self._obtain_dataset()
		self.train_data, self.val_data = split_dataset(
			data, self.config.active.validation_fraction, self.config.seed,
		)

	def _train_initial_committee(self):
		trainer = make_trainer(self.config.system, self.config.trainer)
		model_dir = self.work_dir / "models" / "iter_000"
		self.model_paths = trainer.train_committee(
			self.train_data, model_dir,
			n_models=self.config.trainer.committee_size,
			restart=self.resume,
		)
		self.state.iteration = 1
		self.state.model_paths = [str(p) for p in self.model_paths]
		save_checkpoint(
			self.state, self.train_data, self.val_data, self.work_dir,
		)

	def _al_loop(self):
		ac = self.config.active
		trainer = make_trainer(self.config.system, self.config.trainer)
		iteration = self.state.iteration

		while True:
			if iteration > ac.max_iterations:
				reason = f"Max iterations ({ac.max_iterations})"
				logger.info("CONVERGED: %s", reason)
				self.state.converged = True
				self.state.reason = reason
				break

			logger.info("--- AL iteration %d ---", iteration)

			# MD sampling. md_temp and md_steps may be single values or ranges,
			# run_md draws them per call.
			lead_calc = trainer.get_calculator(self.model_paths[0])
			rng = np.random.default_rng(self.config.seed + iteration)
			seed_atoms = self.train_data[
				rng.integers(0, len(self.train_data))
			]
			snapshots = run_md(
				seed_atoms, lead_calc, temp=ac.md_temp,
				n_steps=ac.md_steps, timestep=ac.md_timestep,
				interval=ac.md_interval,
				seed=self.config.seed + iteration,
			)

			# Committee prediction and selection.
			predictions = trainer.predict_committee(
				snapshots, self.model_paths,
			)
			selected_idx, scores = select_by_disagreement(
				predictions,
				select_fraction=ac.qbc_select_fraction,
				max_new=ac.max_new_per_iter,
				min_disagreement=ac.qbc_min_disagreement,
			)
			candidates = [snapshots[i] for i in selected_idx]

			# Validate.
			rmse_e, rmse_f = trainer.compute_validation_error(
				self.val_data, self.model_paths[0],
			)
			logger.info("RMSE: E=%.4f eV/atom, F=%.4f eV/A", rmse_e, rmse_f)

			# Check convergence.
			converged, reason = _check_convergence(
				len(candidates), rmse_e, rmse_f, ac,
			)
			self.state.history.append({
				"iteration": iteration,
				"n_training": len(self.train_data),
				"n_selected": len(candidates),
				"rmse_energy": rmse_e,
				"rmse_force": rmse_f,
			})

			if converged:
				logger.info("CONVERGED: %s", reason)
				self.state.converged = True
				self.state.reason = reason
				break

			# Label and extend training set.
			iter_dir = self.work_dir / f"iter_{iteration:03d}"
			new_data, _ = label_batch(
				candidates, iter_dir,
				self.config.dft, self.config.scheduler,
				cleanup=self.config.cleanup_dft_scratch,
				resume=self.resume,
			)
			if new_data:
				n_new = len(new_data)
				val_rng = np.random.default_rng(
					self.config.seed + iteration + 10000)
				perm = val_rng.permutation(n_new)
				n_val = int(n_new * ac.validation_fraction)
				val_idx = set(perm[:n_val].tolist())
				new_val = [new_data[i] for i in range(n_new) if i in val_idx]
				new_train = [new_data[i] for i in range(n_new)
							 if i not in val_idx]
				self.train_data.extend(new_train)
				self.val_data.extend(new_val)
				logger.info(
					"Added %d structures (%d train, %d val; totals %d/%d)",
					n_new, len(new_train), len(new_val),
					len(self.train_data), len(self.val_data))

			# Retrain.
			model_dir = self.work_dir / "models" / f"iter_{iteration:03d}"
			self.model_paths = trainer.train_committee(
				self.train_data, model_dir,
				n_models=self.config.trainer.committee_size,
				restart=self.resume,
			)

			self.state.iteration = iteration + 1
			self.state.model_paths = [str(p) for p in self.model_paths]
			save_checkpoint(
				self.state, self.train_data, self.val_data, self.work_dir,
			)
			iteration += 1

	def _finalise(self):
		logger.info("=== Pipeline complete ===")
		logger.info("Training structures: %d", len(self.train_data))
		logger.info("Models: %s", [str(p) for p in self.model_paths])
		ase_write(str(self.work_dir / "final_training_set.extxyz"),
				  self.train_data, format="extxyz")
		save_checkpoint(
			self.state, self.train_data, self.val_data, self.work_dir,
		)

	# ----------------------------------------------------------------
	# Distillation: teacher labels synthetic structures, train a student.
	# ----------------------------------------------------------------

	def _run_distillation(self):
		from ase.io import read as ase_read

		d = self.config.distill
		logger.info("=== Mode: distill ===")
		if not d.teacher_model_paths:
			raise ValueError(
				"distill mode requires distill.teacher_model_paths "
				"(one or more trained teacher models)."
			)
		if d.seed_structures is None:
			raise ValueError(
				"distill mode requires distill.seed_structures "
				"(extxyz of real seed structures)."
			)

		trainer = make_trainer(self.config.system, self.config.trainer)
		syn_file = self.work_dir / "synthetic_data.extxyz"

		if self.resume and syn_file.exists():
			labelled = list(ase_read(str(syn_file), index=":"))
			logger.info("Resume: loaded %d labelled synthetic structures, "
						"skipping generation and labelling", len(labelled))
		else:
			teacher_calcs = [
				trainer.get_calculator(Path(p)) for p in d.teacher_model_paths
			]
			logger.info("Loaded %d teacher model(s)", len(teacher_calcs))
			if len(teacher_calcs) < 2:
				logger.warning("Single teacher model: the disagreement gate "
							   "is inactive, no structures flagged to QE.")

			seeds = ase_read(d.seed_structures, index=":", format="extxyz")
			if isinstance(seeds, Atoms):
				seeds = [seeds]
			logger.info("Loaded %d seed structures from %s",
						len(seeds), d.seed_structures)

			gen_calc = teacher_calcs[0]
			synthetic = generate_synthetic(seeds, gen_calc, d,
										   seed=self.config.seed)
			if d.add_dimers:
				synthetic += generate_dimers(
					self.config.system.elements, d, seed=self.config.seed,
				)
			logger.info("Total synthetic structures to label: %d",
						len(synthetic))

			labelled, stats = label_with_teacher(
				synthetic, teacher_calcs, d,
				dft=self.config.dft, sched=self.config.scheduler,
				qe_work_dir=self.work_dir / "distill_qe",
			)
			if not labelled:
				raise RuntimeError("No synthetic structures were labelled.")
			logger.info("Labelled set: %d total (%d teacher, %d QE)",
						len(labelled), stats["kept_teacher"], stats["qe_ok"])
			ase_write(str(syn_file), labelled, format="extxyz")

		model_dir = self.work_dir / "student"
		student_path = trainer.train_student(labelled, model_dir, d,
											 restart=self.resume)
		self.model_paths = [student_path]
		logger.info("Student model: %s", student_path)
		logger.info("Done. Student in %s", model_dir)


def _count_extxyz(path):
	try:
		from ase.io import read as ase_read
		return len(list(ase_read(str(path), index=":")))
	except Exception:
		return "?"


def _check_convergence(n_selected, rmse_e, rmse_f, ac):
	if ac.stop_on_no_new and n_selected == 0:
		return True, "No structures selected by QBC"
	if rmse_e <= ac.rmse_energy_tol and rmse_f <= ac.rmse_force_tol:
		return True, (f"RMSE below tolerance: E={rmse_e:.4f}, F={rmse_f:.4f}")
	return False, ""