"""Scheduler interface and factory."""

from enum import Enum
from pathlib import Path
from typing import List

from automlip.config import SchedulerConfig


class JobStatus(Enum):
	RUNNING = "running"
	DONE = "done"
	FAILED = "failed"
	UNKNOWN = "unknown"


def make_scheduler(config: SchedulerConfig):
	backend = config.backend.lower()
	if backend == "local":
		from automlip.schedulers.local import LocalScheduler
		return LocalScheduler()
	elif backend == "pbs":
		from automlip.schedulers.pbs import PBSScheduler
		return PBSScheduler(config)
	else:
		raise ValueError(f"Unknown scheduler: {backend!r}. Use 'local' or 'pbs'.")
