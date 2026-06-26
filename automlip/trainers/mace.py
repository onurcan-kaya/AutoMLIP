"""MACE trainer. Supports training from scratch and foundation model finetuning."""

import logging
import shlex
import subprocess
from pathlib import Path
from typing import List

from ase import Atoms
from ase.io import write as ase_write

from automlip.config import SystemConfig, TrainerConfig
from automlip.trainers import BaseTrainer, resolve_device

logger = logging.getLogger("automlip.trainers.mace")


def _stderr_tail(text, n=2000):
	"""Tail of a subprocess stderr. The argparse 'error:' line and any Python
	traceback are at the end, so the tail is what actually identifies the
	failure; the head is mostly torch import warnings."""
	text = (text or "").strip()
	return text[-n:] if len(text) > n else text


class MACETrainer(BaseTrainer):
	def __init__(self, system_config: SystemConfig, trainer_config: TrainerConfig):
		self.system = system_config
		self.config = trainer_config
		self.device = resolve_device(trainer_config)

	def train_committee(self, data: List[Atoms], model_dir: Path,
						n_models: int = 3, restart: bool = False,
						val_data: List[Atoms] = None) -> List[Path]:
		"""Train N MACE models with different seeds. Same data, seed diversity.

		With restart=True each member resumes from its latest checkpoint via
		MACE's --restart_latest (no effect if no checkpoint exists yet).

		If val_data is given it is written out and passed to MACE as an explicit
		validation file, so MACE validates on exactly that set and does not take
		a second internal split. If val_data is None, MACE falls back to its own
		--valid_fraction split of the training data.
		"""
		model_dir = Path(model_dir).resolve()
		model_dir.mkdir(parents=True, exist_ok=True)

		train_file = model_dir / "train.extxyz"
		ase_write(str(train_file), data, format="extxyz")
		valid_file = None
		if val_data:
			valid_file = model_dir / "valid.extxyz"
			ase_write(str(valid_file), val_data, format="extxyz")

		paths = []
		for i in range(n_models):
			seed = self.config.seed + i
			name = f"committee_{i:02d}"
			member_dir = model_dir / name
			member_dir.mkdir(parents=True, exist_ok=True)

			cmd = self._build_command(train_file, member_dir, name, seed,
									  restart=restart, valid_file=valid_file)
			logger.info("Training MACE member %d (seed=%d)", i, seed)

			result = subprocess.run(
				cmd, shell=True, capture_output=True, text=True,
				cwd=str(member_dir),
			)
			if result.returncode != 0:
				logger.error("MACE member %d failed:\n%s", i,
							 _stderr_tail(result.stderr))
				continue

			model_path = self._find_model(member_dir, name)
			if model_path:
				paths.append(model_path)

		if not paths:
			raise RuntimeError("All MACE committee members failed")
		return paths

	def train_student(self, data: List[Atoms], model_dir: Path,
					  distill_config, restart: bool = False) -> Path:
		"""Train one small student model from scratch on synthetic data.

		Used for distillation. Returns the path to the student model. With
		restart=True the student resumes from its latest checkpoint.
		"""
		model_dir = Path(model_dir).resolve()
		model_dir.mkdir(parents=True, exist_ok=True)
		train_file = model_dir / "synthetic.extxyz"
		ase_write(str(train_file), data, format="extxyz")

		name = "student"
		member_dir = model_dir / name
		member_dir.mkdir(parents=True, exist_ok=True)

		cmd = self._build_student_command(
			train_file, member_dir, name, distill_config, restart=restart,
		)
		logger.info("Training distilled student model")
		result = subprocess.run(
			cmd, shell=True, capture_output=True, text=True,
			cwd=str(member_dir),
		)
		if result.returncode != 0:
			logger.error("Student training failed:\n%s",
						 _stderr_tail(result.stderr))
			raise RuntimeError("Student training failed")

		model_path = self._find_model(member_dir, name)
		if model_path is None:
			raise RuntimeError("No student .model file produced")
		return model_path

	def get_calculator(self, model_path: Path):
		from mace.calculators import MACECalculator
		return MACECalculator(
			model_paths=str(model_path), device=self.device,
			default_dtype=self.config.mace_default_dtype,
		)

	def _build_command(self, train_file, model_dir, name, seed, restart=False,
					   valid_file=None):
		c = self.config
		parts = [
			"mace_run_train",
			f"--name={name}",
			f"--train_file={train_file}",
		]
		if valid_file is not None:
			parts.append(f"--valid_file={valid_file}")
		else:
			parts.append(f"--valid_fraction={c.mace_valid_fraction}")
		parts += [
			f"--model={c.mace_model}",
			f"--max_L={c.mace_max_L}",
			f"--num_channels={c.mace_num_channels}",
			f"--r_max={c.mace_r_max}",
			f"--max_num_epochs={c.mace_epochs}",
			f"--lr={c.mace_lr}",
			f"--batch_size={c.mace_batch_size}",
			f"--seed={seed}",
			f"--device={self.device}",
			f"--energy_key=energy",
			f"--forces_key=forces",
			f"--results_dir={model_dir}",
			f"--checkpoints_dir={model_dir}",
			f"--default_dtype={c.mace_default_dtype}",
			f"--weight_decay={c.weight_decay}",
		]
		if restart:
			parts.append("--restart_latest")

		# E0s: from-scratch uses the configured scratch option (average by
		# default), finetuning uses the configured e0s (estimated by default).
		if c.foundation_model is None:
			parts.append(f"--E0s={c.mace_e0s_scratch}")
		else:
			parts.append(f"--foundation_model={c.foundation_model}")
			parts.append(f"--E0s={c.e0s}")
			parts.append(f"--scaling={c.mace_scaling}")
			if c.mace_ema:
				parts.append("--ema")
				parts.append(f"--ema_decay={c.mace_ema_decay}")
			if c.mace_amsgrad:
				parts.append("--amsgrad")

			strategy = c.finetune_strategy.lower()
			if strategy == "naive":
				parts.append("--multiheads_finetuning=False")
			elif strategy == "multihead":
				if not c.pt_train_file:
					raise ValueError(
						"finetune_strategy='multihead' requires "
						"trainer.pt_train_file (replay xyz path or 'mp')."
					)
				parts.append("--multiheads_finetuning=True")
				parts.append(f"--pt_train_file={c.pt_train_file}")
				if c.pseudolabel_replay:
					parts.append("--pseudolabel_replay=True")
				# MACE exposes only the replay-head weight (--weight_pt_head);
				# there is no separate target-head weight flag. weight_ft is
				# kept in the config but not emitted.
				parts.append(f"--weight_pt_head={c.weight_pt}")
				if c.mace_lr > 1.0e-3:
					logger.warning(
						"multihead replay usually needs lr ~1e-4; current "
						"mace_lr=%s may converge poorly", c.mace_lr,
					)
			else:
				raise ValueError(
					f"Unknown finetune_strategy: {c.finetune_strategy!r}. "
					f"Use 'naive' or 'multihead'."
				)

		# Optional knobs. Each emits nothing unless explicitly set, so the
		# defaults above reproduce the original command exactly.
		if c.mace_loss is not None:
			parts.append(f"--loss={c.mace_loss}")
		if c.mace_energy_weight is not None:
			parts.append(f"--energy_weight={c.mace_energy_weight}")
		if c.mace_forces_weight is not None:
			parts.append(f"--forces_weight={c.mace_forces_weight}")

		# Stress training, opt-in. Pairs with dft.tstress and parsed stress.
		if c.mace_compute_stress:
			parts.append("--compute_stress=True")
			parts.append(f"--stress_key={c.mace_stress_key}")
		if c.mace_stress_weight is not None:
			parts.append(f"--stress_weight={c.mace_stress_weight}")

		# SWA tail.
		if c.mace_swa:
			parts.append("--swa")
			if c.mace_start_swa is not None:
				parts.append(f"--start_swa={c.mace_start_swa}")
			if c.mace_swa_lr is not None:
				parts.append(f"--swa_lr={c.mace_swa_lr}")
			if c.mace_swa_energy_weight is not None:
				parts.append(f"--swa_energy_weight={c.mace_swa_energy_weight}")
			if c.mace_swa_forces_weight is not None:
				parts.append(f"--swa_forces_weight={c.mace_swa_forces_weight}")

		# Optimiser and LR scheduler.
		if c.mace_optimizer is not None:
			parts.append(f"--optimizer={c.mace_optimizer}")
		if c.mace_scheduler is not None:
			parts.append(f"--scheduler={c.mace_scheduler}")
		if c.mace_lr_scheduler_gamma is not None:
			parts.append(f"--lr_scheduler_gamma={c.mace_lr_scheduler_gamma}")
		if c.mace_lr_factor is not None:
			parts.append(f"--lr_factor={c.mace_lr_factor}")
		if c.mace_scheduler_patience is not None:
			parts.append(f"--scheduler_patience={c.mace_scheduler_patience}")

		# Architecture.
		if c.mace_correlation is not None:
			parts.append(f"--correlation={c.mace_correlation}")
		if c.mace_num_interactions is not None:
			parts.append(f"--num_interactions={c.mace_num_interactions}")
		if c.mace_hidden_irreps is not None:
			parts.append(f"--hidden_irreps={c.mace_hidden_irreps}")
		if c.mace_mlp_irreps is not None:
			parts.append(f"--MLP_irreps={c.mace_mlp_irreps}")
		if c.mace_num_radial_basis is not None:
			parts.append(f"--num_radial_basis={c.mace_num_radial_basis}")
		if c.mace_distance_transform is not None:
			parts.append(f"--distance_transform={c.mace_distance_transform}")

		# Training control.
		if c.mace_patience is not None:
			parts.append(f"--patience={c.mace_patience}")
		if c.mace_eval_interval is not None:
			parts.append(f"--eval_interval={c.mace_eval_interval}")
		if c.mace_clip_grad is not None:
			parts.append(f"--clip_grad={c.mace_clip_grad}")
		if c.mace_keep_checkpoints:
			parts.append("--keep_checkpoints")
		if c.mace_save_cpu:
			parts.append("--save_cpu")

		# Escape hatch for any unmodelled flag.
		for k, v in c.mace_extra_args.items():
			if v in ("", None):
				parts.append(f"--{k}")
			else:
				parts.append(f"--{k}={v}")

		# Quote each part so values containing spaces (e.g. an E0s dict written
		# with spaces) survive shell tokenisation as a single argument.
		return " ".join(shlex.quote(p) for p in parts)

	def _build_student_command(self, train_file, model_dir, name, d,
							   restart=False):
		"""Build a from-scratch command for a small student model."""
		parts = [
			"mace_run_train",
			f"--name={name}",
			f"--train_file={train_file}",
			f"--valid_fraction={d.validation_fraction}",
			f"--model=MACE",
			f"--max_L={d.student_max_L}",
			f"--num_channels={d.student_num_channels}",
			f"--r_max={d.student_r_max}",
			f"--max_num_epochs={d.student_epochs}",
			f"--lr={d.student_lr}",
			f"--batch_size={d.student_batch_size}",
			f"--seed={self.config.seed}",
			f"--device={self.device}",
			f"--energy_key=energy",
			f"--forces_key=forces",
			f"--results_dir={model_dir}",
			f"--checkpoints_dir={model_dir}",
			f"--default_dtype=float64",
			"--E0s=average",
			f"--loss={d.student_loss}",
			f"--energy_weight={d.student_energy_weight}",
			f"--forces_weight={d.student_forces_weight}",
			"--scaling=rms_forces_scaling",
			"--ema",
			"--ema_decay=0.99",
			"--amsgrad",
		]
		if d.student_swa:
			parts += [
				"--swa",
				f"--start_swa={d.student_start_swa}",
				f"--swa_lr={d.student_swa_lr}",
				f"--swa_energy_weight={d.student_swa_energy_weight}",
				f"--swa_forces_weight={d.student_swa_forces_weight}",
			]
		if d.student_pair_repulsion:
			parts.append("--pair_repulsion")
		if restart:
			parts.append("--restart_latest")
		return " ".join(shlex.quote(p) for p in parts)

	def _find_model(self, model_dir, name):
		"""Find the output .model file."""
		expected = model_dir / f"{name}.model"
		if expected.exists():
			return expected
		candidates = list(model_dir.glob(f"{name}*.model"))
		if candidates:
			return candidates[0]
		# MACE sometimes puts models in results subdir.
		candidates = list(model_dir.rglob("*.model"))
		if candidates:
			return candidates[0]
		logger.error("No .model file found in %s", model_dir)
		return None
