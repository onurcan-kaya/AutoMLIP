"""
Local parallel runner for workstations.

Runs QE jobs as subprocesses without any batch scheduler. Controls
parallelism so you don't oversubscribe cores or memory.

Given N structures and max_parallel concurrent jobs, it maintains
a pool of running processes and starts new ones as slots free up.

Usage:
    runner = LocalRunner(
        n_cores_total=32,
        n_cores_per_job=8,
        mpi_command="mpirun",
        dft_executable="pw.x",
    )
    statuses = runner.run_batch(calc_dirs)
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from automlip.schedulers.base import JobStatus

logger = logging.getLogger("automlip.schedulers.local_parallel")


@dataclass
class _RunningJob:
    """Tracks a running subprocess."""
    index: int
    process: subprocess.Popen
    calc_dir: Path
    start_time: float


class LocalRunner:
    """
    Run DFT jobs on a local workstation with controlled parallelism.

    No SLURM, no PBS, no queue. Just subprocess management.
    """

    def __init__(
        self,
        n_cores_total: int = 4,
        n_cores_per_job: int = 4,
        mpi_command: str = "mpirun",
        dft_executable: str = "pw.x",
        input_file: str = "pw.in",
        output_file: str = "pw.out",
        timeout_per_job: Optional[float] = None,
        env_vars: Optional[Dict[str, str]] = None,
        modules: Optional[List[str]] = None,
    ):
        """
        Args:
            n_cores_total: total cores available on this machine.
            n_cores_per_job: cores per QE job. max_parallel is
                             n_cores_total // n_cores_per_job.
            mpi_command: MPI launcher ("mpirun", "mpiexec", "srun").
                         Set to "" for serial execution.
            dft_executable: path or name of pw.x.
            input_file: QE input filename (relative to calc_dir).
            output_file: QE output filename.
            timeout_per_job: kill a job after this many seconds.
                             None means no timeout.
            env_vars: environment variables to set for each job.
            modules: module load commands to source before each job.
                     Ignored on workstations without module system.
        """
        self.n_cores_total = n_cores_total
        self.n_cores_per_job = n_cores_per_job
        self.max_parallel = max(1, n_cores_total // n_cores_per_job)
        self.mpi_command = mpi_command
        self.dft_executable = dft_executable
        self.input_file = input_file
        self.output_file = output_file
        self.timeout = timeout_per_job
        self.env_vars = env_vars or {}
        self.modules = modules or []

        logger.info(
            "LocalRunner: %d cores, %d per job, %d parallel slots",
            n_cores_total, n_cores_per_job, self.max_parallel,
        )

    def _build_command(self) -> str:
        """Build the shell command for one QE job."""
        if self.mpi_command:
            return (
                f"{self.mpi_command} -np {self.n_cores_per_job} "
                f"{self.dft_executable} -i {self.input_file} "
                f"> {self.output_file} 2>&1"
            )
        else:
            # Serial execution.
            return (
                f"{self.dft_executable} -i {self.input_file} "
                f"> {self.output_file} 2>&1"
            )

    def _build_env(self) -> dict:
        """Build environment dict for subprocesses."""
        env = os.environ.copy()
        env.update(self.env_vars)

        # Prevent OpenMP from grabbing all cores.
        if "OMP_NUM_THREADS" not in env:
            env["OMP_NUM_THREADS"] = "1"

        return env

    def run_batch(
        self,
        calc_dirs: List[Path],
        poll_interval: float = 5.0,
    ) -> List[JobStatus]:
        """
        Run QE in each calc_dir, up to max_parallel at a time.

        Each calc_dir must already contain the input file (pw.in).
        Output goes to pw.out in the same directory.

        Args:
            calc_dirs: directories to run, one per structure.
            poll_interval: seconds between checks for finished jobs.

        Returns:
            List of JobStatus in the same order as calc_dirs.
        """
        n = len(calc_dirs)
        results = [JobStatus.UNKNOWN] * n
        command = self._build_command()
        env = self._build_env()

        # Queue of indices not yet started.
        pending = list(range(n))
        running: Dict[int, _RunningJob] = {}

        logger.info(
            "Starting batch: %d jobs, %d parallel slots", n, self.max_parallel
        )

        while pending or running:
            # Start new jobs if slots available.
            while pending and len(running) < self.max_parallel:
                idx = pending.pop(0)
                calc_dir = calc_dirs[idx]

                if not (calc_dir / self.input_file).exists():
                    logger.error(
                        "No input file in %s, marking FAILED", calc_dir
                    )
                    results[idx] = JobStatus.FAILED
                    continue

                proc = subprocess.Popen(
                    ["bash", "-c", command],
                    cwd=str(calc_dir),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                running[idx] = _RunningJob(
                    index=idx,
                    process=proc,
                    calc_dir=calc_dir,
                    start_time=time.time(),
                )

                logger.info(
                    "Started job %d/%d (PID %d) in %s",
                    idx + 1, n, proc.pid, calc_dir,
                )

            # Check running jobs.
            finished = []
            for idx, job in running.items():
                ret = job.process.poll()

                if ret is not None:
                    elapsed = time.time() - job.start_time
                    if ret == 0:
                        results[idx] = JobStatus.DONE
                        logger.info(
                            "Job %d finished OK (%.0fs)", idx, elapsed
                        )
                    else:
                        results[idx] = JobStatus.FAILED
                        logger.warning(
                            "Job %d failed with code %d (%.0fs)",
                            idx, ret, elapsed,
                        )
                    finished.append(idx)

                elif self.timeout is not None:
                    elapsed = time.time() - job.start_time
                    if elapsed > self.timeout:
                        job.process.kill()
                        job.process.wait()
                        results[idx] = JobStatus.FAILED
                        logger.warning(
                            "Job %d killed after timeout (%.0fs)", idx, elapsed
                        )
                        finished.append(idx)

            for idx in finished:
                del running[idx]

            # Report progress.
            if running:
                n_done = sum(
                    1 for r in results
                    if r in (JobStatus.DONE, JobStatus.FAILED)
                )
                logger.info(
                    "Progress: %d/%d done, %d running, %d pending",
                    n_done, n, len(running), len(pending),
                )
                time.sleep(poll_interval)

        # Summary.
        n_ok = sum(1 for r in results if r == JobStatus.DONE)
        n_fail = sum(1 for r in results if r == JobStatus.FAILED)
        logger.info("Batch complete: %d ok, %d failed", n_ok, n_fail)

        return results

    def run_single(self, calc_dir: Path) -> JobStatus:
        """Run a single QE job and block until it finishes."""
        statuses = self.run_batch([calc_dir])
        return statuses[0]
