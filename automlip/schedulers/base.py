"""Abstract interface for HPC job schedulers."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List


class JobStatus(Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    UNKNOWN = "unknown"


class BaseScheduler(ABC):
    @abstractmethod
    def write_job_script(self, run_command: str, work_dir: Path, job_name: str = "automlip") -> Path: ...
    @abstractmethod
    def submit(self, script_path: Path) -> str: ...
    @abstractmethod
    def status(self, job_id: str) -> JobStatus: ...
    @abstractmethod
    def cancel(self, job_id: str) -> None: ...

    def submit_batch(self, script_paths: List[Path]) -> List[str]:
        return [self.submit(p) for p in script_paths]

    def wait_all(self, job_ids: List[str], poll_interval: float = 60.0) -> List[JobStatus]:
        import time, logging
        logger = logging.getLogger("automlip.scheduler")
        final = [None] * len(job_ids)
        while True:
            all_done = True
            for i, jid in enumerate(job_ids):
                if final[i] is not None:
                    continue
                s = self.status(jid)
                if s in (JobStatus.DONE, JobStatus.FAILED):
                    final[i] = s
                    logger.info(f"Job {jid}: {s.value}")
                else:
                    all_done = False
            if all_done:
                break
            n_pending = sum(1 for f in final if f is None)
            logger.info(f"Waiting for {n_pending}/{len(job_ids)} jobs... (polling every {poll_interval}s)")
            time.sleep(poll_interval)
        return final
