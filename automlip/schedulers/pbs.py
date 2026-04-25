"""
PBS Pro scheduler backend.

Supports individual jobs and job arrays. Job arrays use PBS -J syntax.
Tested against PBS Pro as used on Argonne's machines and similar
academic HPCs.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from automlip.config import SchedulerConfig
from automlip.schedulers.base import BaseScheduler, JobStatus

logger = logging.getLogger("automlip.schedulers.pbs")

# PBS state codes to JobStatus mapping.
_PBS_STATE_MAP = {
    "Q": JobStatus.RUNNING,   # queued
    "H": JobStatus.RUNNING,   # held
    "R": JobStatus.RUNNING,   # running
    "E": JobStatus.RUNNING,   # exiting
    "T": JobStatus.RUNNING,   # moving
    "W": JobStatus.RUNNING,   # waiting
    "S": JobStatus.RUNNING,   # suspended
    "B": JobStatus.RUNNING,   # array job running
    "F": JobStatus.DONE,      # finished (need exit code to distinguish)
    "X": JobStatus.FAILED,    # subjob finished
}


class PBSScheduler(BaseScheduler):
    """
    PBS Pro scheduler backend with job array support.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def write_job_script(
        self,
        run_command: str,
        work_dir: Path,
        job_name: str = "automlip",
    ) -> Path:
        """Write a single-task PBS batch script."""
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / f"{job_name}.pbs"

        lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -l select={self.config.nodes}:ncpus={self.config.cores_per_node}",
            f"#PBS -l walltime={self.config.walltime}",
        ]

        if self.config.queue:
            lines.append(f"#PBS -q {self.config.queue}")

        if self.config.account:
            lines.append(f"#PBS -A {self.config.account}")

        lines.append("#PBS -j oe")
        lines.append("")

        for mod in self.config.modules:
            lines.append(mod)

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
        logger.debug("Wrote PBS script: %s", script_path)
        return script_path

    def write_array_script(
        self,
        calc_dirs: List[Path],
        run_command_template: str,
        parent_dir: Path,
        job_name: str = "al_label",
    ) -> Path:
        """
        Write a PBS job array script for batch DFT labelling.

        Uses PBS -J syntax. Each sub-job reads its directory from
        a dirlist file using PBS_ARRAY_INDEX.
        """
        parent_dir.mkdir(parents=True, exist_ok=True)
        n_tasks = len(calc_dirs)

        dirlist_path = parent_dir / "dirlist.txt"
        with open(dirlist_path, "w") as f:
            for d in calc_dirs:
                f.write(f"{d.resolve()}\n")

        script_path = parent_dir / f"{job_name}.pbs"

        lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -J 0-{n_tasks - 1}",
            f"#PBS -l select={self.config.nodes}:ncpus={self.config.cores_per_node}",
            f"#PBS -l walltime={self.config.walltime}",
        ]

        if self.config.queue:
            lines.append(f"#PBS -q {self.config.queue}")

        if self.config.account:
            lines.append(f"#PBS -A {self.config.account}")

        lines.append("#PBS -j oe")
        lines.append("")

        for mod in self.config.modules:
            lines.append(mod)

        for key, val in self.config.env_vars.items():
            lines.append(f"export {key}={val}")

        if self.config.modules or self.config.env_vars:
            lines.append("")

        # PBS_ARRAY_INDEX is 0-based.
        lines.append(
            f'CALC_DIR=$(sed -n "$((PBS_ARRAY_INDEX + 1))p" {dirlist_path.resolve()})'
        )
        lines.append('cd "$CALC_DIR"')
        lines.append("")
        lines.append(run_command_template)
        lines.append("")

        with open(script_path, "w") as f:
            f.write("\n".join(lines))

        os.chmod(script_path, 0o755)
        logger.info(
            "Wrote PBS array script: %s (%d tasks)", script_path, n_tasks
        )
        return script_path

    def submit(self, script_path: Path) -> str:
        """Submit with qsub, return the job ID."""
        try:
            result = subprocess.run(
                ["qsub", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            # qsub returns "12345.pbs01\n" or similar.
            job_id = result.stdout.strip()
            logger.info("Submitted PBS job: %s", job_id)
            return job_id

        except subprocess.CalledProcessError as e:
            logger.error("qsub failed: %s", e.stderr)
            raise RuntimeError(f"PBS submission failed: {e.stderr}")

    def submit_array(
        self,
        calc_dirs: List[Path],
        run_command_template: str,
        parent_dir: Path,
        job_name: str = "al_label",
    ) -> str:
        """Write and submit a job array. Returns the array job ID."""
        script = self.write_array_script(
            calc_dirs, run_command_template, parent_dir, job_name
        )
        return self.submit(script)

    def status(self, job_id: str) -> JobStatus:
        """
        Check job status via qstat.

        Works for both individual jobs and array parent jobs.
        """
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return JobStatus.UNKNOWN

        if result.returncode != 0:
            # Job no longer in queue system. Check if it was tracked.
            # qstat fails for completed jobs on some PBS installations.
            # Try tracejob as fallback.
            return self._check_finished(job_id)

        output = result.stdout

        # Extract job_state.
        match = re.search(r"job_state\s*=\s*(\S+)", output)
        if not match:
            return JobStatus.UNKNOWN

        state_code = match.group(1)
        js = _PBS_STATE_MAP.get(state_code, JobStatus.UNKNOWN)

        # For finished jobs, check exit status.
        if js == JobStatus.DONE:
            exit_match = re.search(r"Exit_status\s*=\s*(\d+)", output)
            if exit_match and int(exit_match.group(1)) != 0:
                return JobStatus.FAILED

        return js

    def _check_finished(self, job_id: str) -> JobStatus:
        """
        Fallback for systems where qstat drops completed jobs.

        Uses tracejob if available, otherwise returns DONE
        (optimistic assumption).
        """
        try:
            result = subprocess.run(
                ["tracejob", job_id],
                capture_output=True,
                text=True,
            )
            if "Exit_status=0" in result.stdout:
                return JobStatus.DONE
            elif "Exit_status=" in result.stdout:
                return JobStatus.FAILED
        except FileNotFoundError:
            pass

        # If tracejob not available and job not in qstat, assume done.
        # The QE error handler will catch actual failures.
        logger.warning(
            "Job %s not in qstat and tracejob unavailable. Assuming DONE.",
            job_id,
        )
        return JobStatus.DONE

    def array_task_statuses(
        self, array_job_id: str, n_tasks: int
    ) -> List[JobStatus]:
        """
        Get per-task status for a PBS array job.

        PBS array sub-jobs have IDs like "12345[0].pbs01", "12345[1].pbs01".
        """
        task_states = [JobStatus.UNKNOWN] * n_tasks

        # Strip server suffix for the base ID.
        base_id = array_job_id.split(".")[0] if "." in array_job_id else array_job_id
        server = array_job_id.split(".", 1)[1] if "." in array_job_id else ""

        for idx in range(n_tasks):
            if server:
                task_id = f"{base_id}[{idx}].{server}"
            else:
                task_id = f"{base_id}[{idx}]"
            task_states[idx] = self.status(task_id)

        return task_states

    def cancel(self, job_id: str) -> None:
        """Cancel a job or array job."""
        try:
            subprocess.run(
                ["qdel", job_id],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Cancelled PBS job %s", job_id)
        except subprocess.CalledProcessError as e:
            logger.warning("qdel failed for %s: %s", job_id, e.stderr)

    def wait_for_array(
        self,
        array_job_id: str,
        n_tasks: int,
        poll_interval: Optional[float] = None,
    ) -> List[JobStatus]:
        """Block until all tasks in an array job reach a terminal state."""
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
