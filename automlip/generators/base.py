"""Abstract interface for structure generators."""

from abc import ABC, abstractmethod
from typing import List
from ase import Atoms


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, n_structures: int) -> List[Atoms]:
        """
        Generate n_structures and return them as a list.
        Failed attempts are discarded silently. The returned list may be
        shorter than n_structures.
        """
        ...
