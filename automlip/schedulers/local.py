"""Local scheduler: run DFT jobs as subprocesses."""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List

from automlip.config import SchedulerConfig
from automlip.labellers.qe import build_run_command
from automlip.schedulers import JobStatus

logger = logging.getLogger("automlip.schedulers.local")


class LocalScheduler:
	def run_batch(self, calc_dirs: List[Path], run_cmd: str,
				  config: SchedulerConfig) -> List[JobStatus]:
		"""Run QE in each directory with controlled parallelism."""
		cores_per_job = config.cores_per_dft_job or config.cores
		max_parallel = max(1, config.cores // cores_per_job)

		# Build the command from parts when running MPI pw.x with a per-job
		# core count. Otherwise honour the command passed in, which may not be
		# an MPI invocation at all.
		if config.mpi_command and config.cores_per_dft_job:
			cmd = build_run_command(
				config.mpi_command, cores_per_job, config.dft_executable,
			)
		else:
			cmd = run_cmd

		timeout = _walltime_to_seconds(config.walltime)
		env = os.environ.copy()
		env.update(config.env_vars)
		env.setdefault("OMP_NUM_THREADS", "1")

		n = len(calc_dirs)
		results = [JobStatus.UNKNOWN] * n
		pending = list(range(n))
		running = {}  # idx -> (process, start_time)

		while pending or running:
			while pending and len(running) < max_parallel:
				idx = pending.pop(0)
				if not (calc_dirs[idx] / "pw.in").exists():
					results[idx] = JobStatus.FAILED
					continue
				proc = subprocess.Popen(
					["bash", "-c", cmd],
					cwd=str(calc_dirs[idx]),
					env=env,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL,
				)
				running[idx] = (proc, time.time())

			finished = []
			for idx, (proc, t0) in running.items():
				ret = proc.poll()
				if ret is not None:
					results[idx] = JobStatus.DONE if ret == 0 else JobStatus.FAILED
					finished.append(idx)
				elif timeout and (time.time() - t0) > timeout:
					proc.kill()
					proc.wait()
					results[idx] = JobStatus.FAILED
					finished.append(idx)

			for idx in finished:
				del running[idx]

			if running:
				time.sleep(min(config.poll_interval, 5.0))

		n_ok = sum(1 for r in results if r == JobStatus.DONE)
		logger.info("Local batch: %d/%d ok", n_ok, n)
		return results


def _walltime_to_seconds(walltime: str):
	parts = walltime.split(":")
	if len(parts) != 3:
		return None
	try:
		return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
	except ValueError:
		return None
