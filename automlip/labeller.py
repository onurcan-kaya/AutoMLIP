"""
Batch DFT labeller.

This is the module that eliminates manual DFT work. Given a list of
ASE Atoms objects, it:

1. Writes QE inputs for each structure into numbered subdirectories.
2. Submits them as a single job array (SLURM or PBS).
3. Waits for completion.
4. Parses outputs, collecting energies, forces and stresses.
5. Diagnoses failures and retries with adjusted parameters.
6. Returns the successfully labelled structures.

Usage:
    labeller = BatchLabeller(dft_config, sched_config)
    labelled = labeller.label(candidates, work_dir)
    # labelled is a list of Atoms with energy/forces/stress set.
"""

import copy
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ase import Atoms

from automlip.config import DFTConfig, SchedulerConfig
from automlip.calculators.qe import QECalculator, OUTPUT_FILE
from automlip.calculators.qe_errors import (
    QEErrorHandler,
    Diagnosis,
    apply_fixes_to_config,
)
from automlip.schedulers.base import JobStatus

logger = logging.getLogger("automlip.labeller")


class BatchLabeller:
    """
    Automated batch DFT labelling with job arrays and error recovery.

    Supports SLURM and PBS backends. Detects the scheduler type from
    SchedulerConfig.backend.
    """

    def __init__(
        self,
        dft_config: DFTConfig,
        sched_config: SchedulerConfig,
        max_retries: int = 2,
    ):
        self.dft_config = dft_config
        self.sched_config = sched_config
        self.calculator = QECalculator(dft_config, sched_config)
        self.error_handler = QEErrorHandler(max_retries=max_retries)
        self.max_retries = max_retries
        self._scheduler = None
        self._local_runner = None

    @property
    def scheduler(self):
        """Lazy-init the scheduler backend."""
        if self._scheduler is None:
            self._scheduler = _make_scheduler(self.sched_config)
        return self._scheduler

    def label(
        self,
        candidates: List[Atoms],
        work_dir: Path,
        iteration: int = 0,
    ) -> Tuple[List[Atoms], List[int]]:
        """
        Label a batch of candidate structures with DFT.

        Args:
            candidates: structures to compute. Each must have a cell set.
            work_dir: base directory for this labelling round. Will contain
                      subdirectories label_000/, label_001/, etc.
            iteration: AL iteration number (for logging and directory naming).

        Returns:
            Tuple of (labelled_atoms, failed_indices).
            labelled_atoms: list of Atoms with info['energy'], arrays['forces']
                            and info['stress'] set. Only successfully converged
                            structures are included.
            failed_indices: indices into the original candidates list for
                            structures that could not be labelled even after
                            retries.
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        n = len(candidates)
        logger.info(
            "Labelling %d candidates (iteration %d) in %s",
            n, iteration, work_dir,
        )

        # Track state per structure.
        calc_dirs = []
        retry_counts = [0] * n

        # ---- Write inputs ----
        for i, atoms in enumerate(candidates):
            calc_dir = work_dir / f"label_{i:04d}"
            self.calculator.write_input(atoms, calc_dir)
            calc_dirs.append(calc_dir)

        # ---- Submit and wait ----
        run_cmd = self.calculator.get_run_command(
            Path("pw.in")  # relative, each task cd's to its own dir
        )

        labelled = [None] * n
        failed_indices = []

        # Initial submission.
        self._submit_and_collect(
            candidates, calc_dirs, run_cmd, work_dir,
            labelled, retry_counts, iteration, attempt=0,
        )

        # ---- Retry loop ----
        for attempt in range(1, self.max_retries + 1):
            # Find structures that need retry.
            retry_indices = []
            for i in range(n):
                if labelled[i] is None and retry_counts[i] < self.max_retries:
                    retry_indices.append(i)

            if not retry_indices:
                break

            logger.info(
                "Retry attempt %d: %d structures to retry",
                attempt, len(retry_indices),
            )

            # Diagnose each failure and rewrite inputs with fixes.
            retry_dirs = []
            retry_candidates = []
            retry_map = []  # maps retry list index -> original index

            for i in retry_indices:
                diag = self.error_handler.diagnose(
                    calc_dirs[i],
                    current_params=self._current_params(),
                    retry_count=retry_counts[i],
                )

                if not diag.retryable:
                    logger.warning(
                        "Structure %d not retryable: %s", i, diag.reason
                    )
                    failed_indices.append(i)
                    continue

                # Apply fixes to a copy of configs.
                dft_copy = copy.deepcopy(self.dft_config)
                sched_copy = copy.deepcopy(self.sched_config)
                apply_fixes_to_config(dft_copy, sched_copy, diag.fixes)

                # Rewrite input in a retry subdirectory.
                retry_dir = work_dir / f"label_{i:04d}_retry{attempt}"
                retry_calc = QECalculator(dft_copy, sched_copy)
                retry_calc.write_input(candidates[i], retry_dir)

                retry_dirs.append(retry_dir)
                retry_candidates.append(candidates[i])
                retry_map.append(i)
                retry_counts[i] += 1

            if not retry_dirs:
                break

            # Submit retries as a new array.
            self._submit_and_collect_subset(
                retry_candidates, retry_dirs, retry_map,
                run_cmd, work_dir, labelled, iteration, attempt,
            )

        # Collect final failures.
        for i in range(n):
            if labelled[i] is None and i not in failed_indices:
                failed_indices.append(i)

        successful = [a for a in labelled if a is not None]
        n_ok = len(successful)
        n_fail = len(failed_indices)

        logger.info(
            "Labelling complete: %d/%d succeeded, %d/%d failed",
            n_ok, n, n_fail, n,
        )

        return successful, sorted(failed_indices)

    def _submit_and_collect(
        self,
        candidates, calc_dirs, run_cmd, work_dir,
        labelled, retry_counts, iteration, attempt,
    ):
        """Submit a job array and parse results into labelled list."""
        n = len(calc_dirs)
        backend = self.sched_config.backend

        if backend in ("slurm", "pbs") and n > 1:
            # Use job array.
            array_job_id = self.scheduler.submit_array(
                calc_dirs, run_cmd, work_dir,
                job_name=f"al_iter{iteration}_a{attempt}",
            )
            task_statuses = self.scheduler.wait_for_array(
                array_job_id, n
            )
        elif backend == "local":
            # Run on workstation with controlled parallelism.
            task_statuses = self._run_local(calc_dirs)
        else:
            # Single job or unknown backend: submit individually.
            task_statuses = self._submit_individual(
                calc_dirs, run_cmd, work_dir, iteration, attempt
            )

        # Parse outputs.
        for i, status in enumerate(task_statuses):
            if status == JobStatus.DONE:
                atoms_out = self.calculator.parse_output(calc_dirs[i])
                if atoms_out is not None:
                    labelled[i] = atoms_out
                else:
                    logger.warning(
                        "Job %d reported DONE but parse failed", i
                    )
            elif status == JobStatus.FAILED:
                logger.warning("Job %d failed (scheduler reported)", i)

    def _submit_and_collect_subset(
        self,
        candidates, calc_dirs, index_map,
        run_cmd, work_dir, labelled, iteration, attempt,
    ):
        """Submit a subset of retries and map results back."""
        n = len(calc_dirs)
        backend = self.sched_config.backend

        if backend in ("slurm", "pbs") and n > 1:
            array_job_id = self.scheduler.submit_array(
                calc_dirs, run_cmd, work_dir,
                job_name=f"al_iter{iteration}_retry{attempt}",
            )
            task_statuses = self.scheduler.wait_for_array(
                array_job_id, n
            )
        elif backend == "local":
            task_statuses = self._run_local(calc_dirs)
        else:
            task_statuses = self._submit_individual(
                calc_dirs, run_cmd, work_dir, iteration, attempt
            )

        for j, status in enumerate(task_statuses):
            orig_idx = index_map[j]
            if status == JobStatus.DONE:
                atoms_out = self.calculator.parse_output(calc_dirs[j])
                if atoms_out is not None:
                    labelled[orig_idx] = atoms_out
                    logger.info(
                        "Retry succeeded for structure %d", orig_idx
                    )

    def _submit_individual(
        self, calc_dirs, run_cmd, work_dir, iteration, attempt
    ):
        """Fallback: submit each calc as a separate job."""
        job_ids = []
        for i, calc_dir in enumerate(calc_dirs):
            script = self.scheduler.write_job_script(
                run_cmd, calc_dir,
                job_name=f"al_i{iteration}_a{attempt}_{i}",
            )
            jid = self.scheduler.submit(script)
            job_ids.append(jid)

        return self.scheduler.wait_all(
            job_ids, poll_interval=self.sched_config.poll_interval
        )

    def _run_local(self, calc_dirs: List[Path]) -> List[JobStatus]:
        """
        Run DFT jobs locally with controlled parallelism.

        Uses LocalRunner which manages subprocess pools based on
        available cores. No scheduler needed.
        """
        if self._local_runner is None:
            from automlip.schedulers.local_parallel import LocalRunner

            total_cores = getattr(self.sched_config, "nodes", 1) * getattr(
                self.sched_config, "cores_per_node", 4
            )

            # cores_per_dft_job controls parallelism. If not set,
            # default to cores_per_node (one job at a time).
            # Set this lower to run multiple jobs in parallel:
            #   cores_per_node=32, cores_per_dft_job=8 -> 4 parallel jobs
            cores_per_dft = getattr(
                self.sched_config, "cores_per_dft_job", None
            )
            if cores_per_dft is None:
                cores_per_dft = getattr(self.sched_config, "cores_per_node", 4)

            timeout = None
            if hasattr(self.sched_config, "walltime"):
                timeout = _walltime_to_seconds(self.sched_config.walltime)

            self._local_runner = LocalRunner(
                n_cores_total=total_cores,
                n_cores_per_job=cores_per_dft,
                mpi_command=getattr(self.sched_config, "mpi_command", "mpirun"),
                dft_executable=getattr(
                    self.sched_config, "dft_executable", "pw.x"
                ),
                timeout_per_job=timeout,
                env_vars=getattr(self.sched_config, "env_vars", {}),
                modules=getattr(self.sched_config, "modules", []),
            )

        return self._local_runner.run_batch(
            calc_dirs,
            poll_interval=getattr(self.sched_config, "poll_interval", 10.0),
        )

    def _current_params(self) -> dict:
        """Extract current DFT params as a flat dict for the error handler."""
        return {
            "ecutwfc": self.dft_config.ecutwfc,
            "ecutrho": self.dft_config.ecutrho,
            "mixing_beta": getattr(self.dft_config, "mixing_beta", 0.7),
            "electron_maxstep": getattr(self.dft_config, "electron_maxstep", 200),
            "walltime": self.sched_config.walltime,
        }


def _make_scheduler(config: SchedulerConfig):
    """Factory for scheduler backends."""
    backend = config.backend.lower()

    if backend == "slurm":
        from automlip.schedulers.slurm import SLURMScheduler
        return SLURMScheduler(config)

    elif backend == "pbs":
        from automlip.schedulers.pbs import PBSScheduler
        return PBSScheduler(config)

    elif backend == "local":
        # Local mode uses LocalRunner directly, not via this factory.
        # But keep a stub so scheduler property doesn't crash.
        from automlip.schedulers.local import LocalScheduler
        return LocalScheduler(config)

    else:
        raise ValueError(
            f"Unknown scheduler backend: {backend!r}. "
            f"Supported: slurm, pbs, local"
        )


def _walltime_to_seconds(walltime: str) -> Optional[float]:
    """Convert "HH:MM:SS" to seconds. Returns None on parse failure."""
    parts = walltime.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
