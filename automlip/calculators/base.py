"""Abstract interface for DFT calculators."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional
from ase import Atoms


class CalcStatus(Enum):
    CONVERGED = "converged"
    FAILED = "failed"
    RUNNING = "running"
    NOT_STARTED = "not_started"


class BaseCalculator(ABC):
    @abstractmethod
    def write_input(self, atoms: Atoms, calc_dir: Path) -> Path: ...
    @abstractmethod
    def parse_output(self, calc_dir: Path) -> Optional[Atoms]: ...
    @abstractmethod
    def check_status(self, calc_dir: Path) -> CalcStatus: ...
    @abstractmethod
    def get_run_command(self, input_path: Path) -> str: ...
