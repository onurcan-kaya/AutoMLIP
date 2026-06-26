import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from automlip.config import SchedulerConfig
from automlip.schedulers import JobStatus

logger = logging.getLogger("automlip.schedulers.pbs")


class PBSScheduler:
	def __init__(self, config: SchedulerConfig):
		self.config = config

	def run_batch(self, calc_dirs: List[Path], run_cmd: str,
				  config: SchedulerConfig) -> List[JobStatus]:
		"""Submit one PBS job per dir, at most pbs_max_concurrent at a time.

		run_cmd is ignored. The launch line is built from mpi_command and
		dft_executable. Concurrency is enforced here by holding the number of
		queued jobs at pbs_max_concurrent, not with a job-array slot limit,
		which is not honoured on every Torque build. Each job reads pw.in and
		writes pw.out in its own dir. The returned status is DONE for every
		finished dir; the caller decides success per dir by parsing pw.out, and
		failed jobs are not retried.
		"""
		n = len(calc_dirs)
		if n == 0:
			return []

		dirs = [d.resolve() for d in calc_dirs]
		for d in dirs:
			d.mkdir(parents=True, exist_ok=True)

		window = max(1, config.pbs_max_concurrent)
		statuses = [JobStatus.UNKNOWN] * n
		pending = list(range(n))
		running: Dict[str, int] = {}

		logger.info("Submitting %d DFT jobs, %d at a time", n, window)
		while pending or running:
			while pending and len(running) < window:
				i = pending.pop(0)
				script = dirs[i] / "label.pbs"
				script.write_text(self._job_script(config, dirs[i]))
				job_id = self._submit(script)
				if job_id is None:
					statuses[i] = JobStatus.FAILED
				else:
					running[job_id] = i
			if not running:
				continue
			time.sleep(config.poll_interval)
			for job_id in [j for j in running if self._finished(j)]:
				statuses[running.pop(job_id)] = JobStatus.DONE
		return statuses

	def _submit(self, script: Path):
		try:
			r = subprocess.run(
				["qsub", str(script)],
				capture_output=True, text=True, check=True,
			)
		except FileNotFoundError:
			logger.error("qsub not found; run the driver on a PBS submit host")
			return None
		except subprocess.CalledProcessError as e:
			logger.error("qsub failed for %s: %s", script, e.stderr.strip())
			return None
		return r.stdout.strip()

	def _job_script(self, config: SchedulerConfig, workdir: Path) -> str:
		"""Render the PBS script that runs pw.x in workdir."""
		ppn = config.cores_per_dft_job or config.cores
		launch = (f"{config.mpi_command} {config.dft_executable} "
				  f"-i pw.in > pw.out 2>&1")

		lines = [
			"#!/bin/bash",
			"#PBS -N automlip_label",
			f"#PBS -l nodes=1:ppn={ppn}",
			f"#PBS -l walltime={config.walltime}",
		]
		if config.queue:
			lines.append(f"#PBS -q {config.queue}")
		if config.account:
			lines.append(f"#PBS -A {config.account}")
		lines.append("#PBS -j oe")
		lines.append("")
		for mod in config.modules:
			lines.append(mod)
		for k, v in config.env_vars.items():
			lines.append(f"export {k}={v}")
		lines.append("")
		lines.append(f'cd "{workdir}"')
		lines.append(launch)
		lines.append("")
		return "\n".join(lines)

	def _finished(self, job_id: str) -> bool:
		try:
			r = subprocess.run(
				["qstat", "-f", job_id], capture_output=True, text=True,
			)
		except FileNotFoundError:
			return True
		# A finished job is gone from qstat (nonzero return) or sits in the
		# completed state.
		if r.returncode != 0:
			return True
		m = re.search(r"job_state\s*=\s*(\S+)", r.stdout)
		if not m:
			return False
		return m.group(1) in ("C", "F")
