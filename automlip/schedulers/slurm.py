"""
SLURM scheduler backend.

Supports both individual job submission and job arrays for batch
DFT labelling. Job arrays are the preferred mode: one sbatch call
submits N tasks, each running in its own subdirectory.

Tested against SLURM on Shaheen III (KAUST) and standard academic
clusters.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from automlip.config import SchedulerConfig
from automlip.schedulers.base import BaseScheduler, JobStatus

logger = logging.getLogger("automlip.schedulers.slurm")


class SLURMScheduler(BaseScheduler):
    """
    SLURM scheduler backend with job array support.

    For batch labelling, use submit_array() which submits a single
    SLURM job array covering all calculation directories.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def write_job_script(
        self,
        run_command: str,
        work_dir: Path,
        job_name: str = "automlip",
    ) -> Path:
        """Write a single-task SLURM batch script."""
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / f"{job_name}.slurm"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --nodes={self.config.nodes}",
            f"#SBATCH --ntasks-per-node={self.config.cores_per_node}",
            f"#SBATCH --time={self.config.walltime}",
        ]

        if self.config.queue:
            lines.append(f"#SBATCH --partition={self.config.queue}")

        if self.config.account:
            lines.append(f"#SBATCH --account={self.config.account}")

        lines.append(f"#SBATCH --output={work_dir / 'slurm-%j.out'}")
        lines.append(f"#SBATCH --error={work_dir / 'slurm-%j.err'}")
        lines.append("")

        # Module loads.
        for mod in self.config.modules:
            lines.append(mod)

        # Environment variables.
        for key, val in self.config.env_vars.items():
            lines.append(f"export {key}={val}")

        if self.config.modules or self.config.env_vars:
            lines.append("")

        lines.append(f"cd {work_dir}")
        lines.append(run_command)
        lines.append("")

        with open(script_path, "w") as f:
            f.write("\n".join(lines))

        os.chmod(script_path, 0o755)
        logger.debug("Wrote SLURM script: %s", script_path)
        return script_path

    def write_array_script(
        self,
        calc_dirs: List[Path],
        run_command_template: str,
        parent_dir: Path,
        job_name: str = "al_label",
    ) -> Path:
        """
        Write a SLURM job array script for batch DFT labelling.

        Each array task cd's into its own calc_dir and runs the
        DFT command there. A dirlist file maps SLURM_ARRAY_TASK_ID
        to directory paths.

        Args:
            calc_dirs: list of directories, one per structure.
                       Each must already contain a pw.in (or equivalent).
            run_command_template: the shell command to run in each dir,
                                 e.g. "mpirun -np 36 pw.x -i pw.in > pw.out"
            parent_dir: where to write the array script and dirlist.
            job_name: SLURM job name.

        Returns:
            Path to the array script.
        """
        parent_dir.mkdir(parents=True, exist_ok=True)
        n_tasks = len(calc_dirs)

        # Write directory list: line i = path for task i.
        dirlist_path = parent_dir / "dirlist.txt"
        with open(dirlist_path, "w") as f:
            for d in calc_dirs:
                f.write(f"{d.resolve()}\n")

        script_path = parent_dir / f"{job_name}.slurm"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array=0-{n_tasks - 1}",
            f"#SBATCH --nodes={self.config.nodes}",
            f"#SBATCH --ntasks-per-node={self.config.cores_per_node}",
            f"#SBATCH --time={self.config.walltime}",
        ]

        if self.config.queue:
            lines.append(f"#SBATCH --partition={self.config.queue}")

        if self.config.account:
            lines.append(f"#SBATCH --account={self.config.account}")

        lines.append(f"#SBATCH --output={parent_dir / 'slurm-%A_%a.out'}")
        lines.append(f"#SBATCH --error={parent_dir / 'slurm-%A_%a.err'}")
        lines.append("")

        for mod in self.config.modules:
            lines.append(mod)

        for key, val in self.config.env_vars.items():
            lines.append(f"export {key}={val}")

        if self.config.modules or self.config.env_vars:
            lines.append("")

        # Read the task directory from dirlist.
        lines.append(f'CALC_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {dirlist_path.resolve()})')
        lines.append('cd "$CALC_DIR"')
        lines.append("")
        lines.append(run_command_template)
        lines.append("")

        with open(script_path, "w") as f:
            f.write("\n".join(lines))

        os.chmod(script_path, 0o755)
        logger.info(
            "Wrote SLURM array script: %s (%d tasks)", script_path, n_tasks
        )
        return script_path

    def submit(self, script_path: Path) -> str:
        """Submit with sbatch, return the job ID."""
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            # sbatch output: "Submitted batch job 12345\n"
            match = re.search(r"(\d+)", result.stdout)
            if not match:
                raise RuntimeError(
                    f"Could not parse job ID from sbatch output: {result.stdout}"
                )
            job_id = match.group(1)
            logger.info("Submitted SLURM job: %s", job_id)
            return job_id

        except subprocess.CalledProcessError as e:
            logger.error("sbatch failed: %s", e.stderr)
            raise RuntimeError(f"SLURM submission failed: {e.stderr}")

    def submit_array(
        self,
        calc_dirs: List[Path],
        run_command_template: str,
        parent_dir: Path,
        job_name: str = "al_label",
    ) -> str:
        """
        Write and submit a job array in one call.

        Returns the array job ID (e.g. "12345"). Individual tasks
        are "12345_0", "12345_1", etc.
        """
        script = self.write_array_script(
            calc_dirs, run_command_template, parent_dir, job_name
        )
        return self.submit(script)

    def status(self, job_id: str) -> JobStatus:
        """
        Check job status via sacct.

        For array jobs, pass the base job ID. This checks whether
        all tasks have completed.
        """
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-j", job_id,
                    "--format=JobID,State",
                    "--noheader",
                    "--parsable2",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return JobStatus.UNKNOWN

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return JobStatus.UNKNOWN

        states = set()
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 2:
                # Skip .batch and .extern sub-jobs.
                jid = parts[0]
                if ".batch" in jid or ".extern" in jid:
                    continue
                states.add(parts[1].strip())

        if not states:
            return JobStatus.UNKNOWN

        # If any task is still running or pending, the job is running.
        if states & {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
            return JobStatus.RUNNING

        # If all terminal: check for failures.
        if states & {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
            if "COMPLETED" in states:
                # Mixed: some completed, some failed.
                return JobStatus.FAILED
            return JobStatus.FAILED

        if states == {"COMPLETED"}:
            return JobStatus.DONE

        return JobStatus.UNKNOWN

    def array_task_statuses(self, array_job_id: str, n_tasks: int) -> List[JobStatus]:
        """
        Get per-task status for an array job.

        Returns a list of length n_tasks with the status of each
        array task (index 0 to n_tasks-1).
        """
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-j", array_job_id,
                    "--format=JobID,State",
                    "--noheader",
                    "--parsable2",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return [JobStatus.UNKNOWN] * n_tasks

        task_states = [JobStatus.UNKNOWN] * n_tasks

        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue

            jid = parts[0]
            state_str = parts[1].strip()

            # Match array task IDs like "12345_3"
            match = re.match(r"\d+_(\d+)$", jid)
            if not match:
                continue

            idx = int(match.group(1))
            if idx >= n_tasks:
                continue

            if state_str == "COMPLETED":
                task_states[idx] = JobStatus.DONE
            elif state_str in {"RUNNING", "PENDING", "CONFIGURING"}:
                task_states[idx] = JobStatus.RUNNING
            elif state_str in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                task_states[idx] = JobStatus.FAILED

        return task_states

    def cancel(self, job_id: str) -> None:
        """Cancel a job or array job."""
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Cancelled SLURM job %s", job_id)
        except subprocess.CalledProcessError as e:
            logger.warning("scancel failed for %s: %s", job_id, e.stderr)

    def wait_for_array(
        self,
        array_job_id: str,
        n_tasks: int,
        poll_interval: Optional[float] = None,
    ) -> List[JobStatus]:
        """
        Block until all tasks in an array job reach a terminal state.

        Args:
            array_job_id: the base job ID returned by submit_array.
            n_tasks: number of tasks in the array.
            poll_interval: seconds between sacct checks. Defaults to
                           config.poll_interval.

        Returns:
            Per-task status list.
        """
        import time

        if poll_interval is None:
            poll_interval = self.config.poll_interval

        while True:
            task_states = self.array_task_statuses(array_job_id, n_tasks)
            n_done = sum(
                1 for s in task_states
                if s in (JobStatus.DONE, JobStatus.FAILED)
            )

            if n_done == n_tasks:
                n_ok = sum(1 for s in task_states if s == JobStatus.DONE)
                n_fail = sum(1 for s in task_states if s == JobStatus.FAILED)
                logger.info(
                    "Array %s finished: %d/%d ok, %d/%d failed",
                    array_job_id, n_ok, n_tasks, n_fail, n_tasks,
                )
                return task_states

            logger.info(
                "Array %s: %d/%d finished, waiting %ds...",
                array_job_id, n_done, n_tasks, int(poll_interval),
            )
            time.sleep(poll_interval)
