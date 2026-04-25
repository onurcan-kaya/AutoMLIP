"""Local subprocess scheduler for testing."""

import logging, os, subprocess
from pathlib import Path
from automlip.config import SchedulerConfig
from automlip.schedulers.base import BaseScheduler, JobStatus

logger = logging.getLogger("automlip.schedulers.local")

class LocalScheduler(BaseScheduler):
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._processes = {}
        self._work_dirs = {}

    def write_job_script(self, run_command: str, work_dir: Path, job_name: str = "automlip") -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / f"{job_name}.sh"
        lines = ["#!/bin/bash", ""]
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
        return script_path

    def submit(self, script_path: Path) -> str:
        work_dir = script_path.parent
        proc = subprocess.Popen(["bash", str(script_path)], cwd=str(work_dir),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid_str = str(proc.pid)
        self._processes[pid_str] = proc
        self._work_dirs[pid_str] = work_dir
        logger.info(f"Submitted job: PID={pid_str}, dir={work_dir}")
        return pid_str

    def status(self, job_id: str) -> JobStatus:
        proc = self._processes.get(job_id)
        if proc is None:
            return JobStatus.UNKNOWN
        ret = proc.poll()
        if ret is None:
            return JobStatus.RUNNING
        elif ret == 0:
            return JobStatus.DONE
        else:
            return JobStatus.FAILED

    def cancel(self, job_id: str) -> None:
        proc = self._processes.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
