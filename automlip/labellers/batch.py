"""Batch DFT labeller. Submits jobs via scheduler. No retries: a failed job is
logged with its reason and skipped."""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from ase import Atoms

from automlip.config import DFTConfig, SchedulerConfig
from automlip.labellers.qe import (
	write_qe_input, parse_qe_output, get_run_command, failure_reason,
)
from automlip.schedulers import make_scheduler, JobStatus

logger = logging.getLogger("automlip.labellers.batch")


def label_batch(
	candidates: List[Atoms],
	work_dir: Path,
	dft: DFTConfig,
	sched: SchedulerConfig,
	cleanup: bool = False,
	resume: bool = False,
) -> Tuple[List[Atoms], List[int]]:
	"""Label structures with DFT. Returns (labelled, failed_indices).

	With resume=True, any job whose pw.out already parses successfully is kept
	as is and not rerun; only unfinished jobs are (re)written and submitted.
	"""
	work_dir = Path(work_dir)
	work_dir.mkdir(parents=True, exist_ok=True)
	n = len(candidates)
	run_cmd = get_run_command(sched)
	scheduler = make_scheduler(sched)

	labelled = [None] * n
	calc_dirs = [work_dir / f"label_{i:04d}" for i in range(n)]

	# Decide which jobs still need to run.
	todo = []
	for i, atoms in enumerate(candidates):
		if resume:
			done = parse_qe_output(calc_dirs[i], atoms)
			if done is not None:
				labelled[i] = done
				continue
		write_qe_input(atoms, calc_dirs[i], dft)
		todo.append(i)

	if resume:
		logger.info("Resume: %d/%d already finished, running %d DFT jobs",
					n - len(todo), n, len(todo))

	# Run the unfinished jobs. No retries: log the reason and move on.
	if todo:
		todo_dirs = [calc_dirs[i] for i in todo]
		statuses = scheduler.run_batch(todo_dirs, run_cmd, sched)
		for k, i in enumerate(todo):
			result = None
			if statuses[k] == JobStatus.DONE:
				result = parse_qe_output(calc_dirs[i], candidates[i])
			if result is not None:
				labelled[i] = result
			else:
				logger.error("label_%04d failed: %s",
							 i, failure_reason(calc_dirs[i]))

	if cleanup:
		_cleanup_scratch(work_dir)

	failed = [i for i in range(n) if labelled[i] is None]
	successful = [a for a in labelled if a is not None]
	logger.info("Labelling: %d/%d ok, %d failed", len(successful), n, len(failed))
	return successful, failed


def _cleanup_scratch(work_dir: Path):
	"""Remove QE scratch files, keep pw.in and pw.out."""
	globs = ["*.wfc*", "*.mix*", "*.hub*", "*.dat", "*.xml",
			 "*.bar", "*.save", "*.restart_*", "*.igk*"]
	for calc_dir in work_dir.iterdir():
		if not calc_dir.is_dir():
			continue
		tmp = calc_dir / "tmp"
		if tmp.is_dir():
			shutil.rmtree(tmp)
		for pattern in globs:
			for f in calc_dir.glob(pattern):
				if f.is_file():
					f.unlink()
