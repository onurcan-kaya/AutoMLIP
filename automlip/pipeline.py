"""Main pipeline driver."""

import logging
import shutil
from pathlib import Path
from typing import List, Optional
import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from automlip.config import Config
from automlip.checkpoint import PipelineState, save_checkpoint, load_checkpoint
from automlip.generators.random import generate_random_structures
from automlip.trainers import make_trainer
from automlip.labeller import BatchLabeller
from automlip.active.sampler import run_md_sampling
from automlip.active.qbc import select_by_committee
from automlip.active.convergence import check_convergence
from automlip.active.tau_acc import compute_tau_acc
from automlip.modes import run_pipeline as dispatch_mode

logger = logging.getLogger("automlip.pipeline")


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.work_dir = Path(config.work_dir)
        self.state = PipelineState()
        self.training_data: List[Atoms] = []
        self.validation_data: List[Atoms] = []
        self.model_paths: List[Path] = []
        self.metrics = {}

        logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO),
                            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    def run(self):
        mode = getattr(self.config, "mode", "full").lower()

        if mode == "train_only":
            model_paths, metrics = dispatch_mode(self.config, self.state)
            self.model_paths = model_paths
            self.metrics = metrics
            return

        elif mode == "al_from_data":
            self._try_restore()
            if self.state.iteration == 0:
                initial_state = dispatch_mode(self.config, self.state)
                self.training_data = initial_state["training_data"]
                self.validation_data = initial_state["validation_data"]
                self.model_paths = initial_state["model_paths"]
                self.state.iteration = initial_state["start_iteration"]
                self.state.model_paths = [str(p) for p in self.model_paths]
                save_checkpoint(self.state, self.training_data, self.validation_data, self.work_dir)
            self._phase_active_learning_loop()
            self._finalise()

        elif mode == "full":
            self._try_restore()
            if self.state.iteration == 0:
                self._phase_initial()
            self._phase_active_learning_loop()
            self._finalise()

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _try_restore(self):
        state, train, val = load_checkpoint(self.work_dir)
        if state is not None:
            self.state = state
            self.training_data = train
            self.validation_data = val
            self.model_paths = [Path(p) for p in state.model_paths]
            logger.info("Restored from checkpoint: iteration %d", state.iteration)

    def _phase_initial(self):
        logger.info("=== Phase: Initial data generation ===")
        sc = self.config.system
        ac = self.config.active

        structures = generate_random_structures(
            elements=sc.elements, composition=sc.composition,
            n_atoms=sc.n_atoms, n_structures=ac.n_initial,
            density=sc.density, min_distance=sc.min_distance,
            seed=self.config.seed,
        )

        labeller = BatchLabeller(self.config.dft, self.config.scheduler)
        labelled, failed = labeller.label(structures, self.work_dir / "iter_000", iteration=0)

        if self.config.cleanup_qe_scratch:
            _cleanup_qe_scratch(self.work_dir / "iter_000")

        if not labelled:
            raise RuntimeError("No structures labelled in initial phase")

        n_val = max(1, int(len(labelled) * ac.validation_fraction))
        self.validation_data = labelled[-n_val:]
        self.training_data = labelled[:-n_val]

        trainer = make_trainer(self.config.system, self.config.trainer)
        tc = self.config.trainer
        model_dir = self.work_dir / "models" / "iter_000"
        self.model_paths = trainer.train_committee(
            self.training_data, model_dir,
            n_models=tc.committee_size, bootstrap_fraction=tc.bootstrap_fraction,
        )
        self.state.iteration = 1
        self.state.model_paths = [str(p) for p in self.model_paths]
        save_checkpoint(self.state, self.training_data, self.validation_data, self.work_dir)

    def _phase_active_learning_loop(self):
        logger.info("=== Phase: Active learning loop ===")
        ac = self.config.active
        tc = self.config.trainer
        trainer = make_trainer(self.config.system, tc)
        labeller = BatchLabeller(self.config.dft, self.config.scheduler)

        iteration = self.state.iteration

        while True:
            logger.info("--- AL iteration %d ---", iteration)

            # Get lead calculator for MD.
            lead_calc = trainer.get_calculator(self.model_paths[0])

            # Pick a random training structure as MD seed.
            rng = np.random.default_rng(self.config.seed + iteration)
            seed_idx = rng.integers(0, len(self.training_data))
            seed_atoms = self.training_data[seed_idx]

            # Run MD.
            snapshots = run_md_sampling(
                seed_atoms, lead_calc, temp=ac.md_temp,
                n_steps=ac.md_steps, timestep=ac.md_timestep,
                interval=ac.md_interval, seed=self.config.seed + iteration,
            )

            # Predict with committee.
            predictions = trainer.predict_committee(snapshots, self.model_paths, n_cores=tc.n_cores)

            # Select by disagreement.
            selected_idx, scores = select_by_committee(
                predictions, snapshots,
                energy_threshold=ac.qbc_energy_threshold,
                force_threshold=ac.qbc_force_threshold,
                select_fraction=ac.qbc_select_fraction,
                max_new=ac.max_new_per_iter,
            )

            candidates = [snapshots[i] for i in selected_idx]
            n_selected = len(candidates)

            # Validate current model.
            rmse_energy, rmse_force = trainer.compute_validation_error(
                self.validation_data, self.model_paths[0],
            )

            # Optionally compute tau_acc.
            tau_acc_value = None
            cfg = self.config
            if cfg.use_tau_acc and iteration % cfg.tau_check_interval == 0:
                logger.info("Computing tau_acc...")
                lead_calc = trainer.get_calculator(self.model_paths[0])
                dft_calc = labeller.calculator
                tau_result = compute_tau_acc(
                    self.training_data,
                    ml_calculator=lead_calc,
                    dft_calculator=dft_calc,
                    e_lower=cfg.tau_e_lower,
                    e_thresh=cfg.tau_e_thresh,
                    max_time_fs=cfg.tau_max_fs,
                    interval_fs=cfg.tau_interval_fs,
                    temp=cfg.tau_temp,
                    dt_fs=cfg.tau_dt_fs,
                    n_configs=cfg.tau_n_configs,
                    seed=self.config.seed + iteration,
                )
                tau_acc_value = tau_result.value
                logger.info("tau_acc = %.1f fs (converged=%s)", tau_acc_value, tau_result.converged)

            # Check convergence.
            conv = check_convergence(
                iteration=iteration, n_selected=n_selected,
                rmse_energy=rmse_energy, rmse_force=rmse_force,
                max_iterations=ac.max_iterations,
                rmse_energy_tol=ac.rmse_energy_tol,
                rmse_force_tol=ac.rmse_force_tol,
                stop_on_no_new=ac.stop_on_no_new,
                tau_acc=tau_acc_value,
                tau_max_fs=cfg.tau_max_fs,
                use_tau_acc=cfg.use_tau_acc,
            )

            self.state.history.append({
                "iteration": iteration, "n_training": len(self.training_data),
                "n_selected": n_selected, "rmse_energy": rmse_energy,
                "rmse_force": rmse_force, "converged": conv.converged,
                "reason": conv.reason,
                "tau_acc": tau_acc_value,
            })

            if conv.converged:
                logger.info("CONVERGED: %s", conv.reason)
                self.state.converged = True
                self.state.reason = conv.reason
                break

            # Label selected candidates with DFT.
            iter_dir = self.work_dir / f"iter_{iteration:03d}"
            new_data, failed = labeller.label(candidates, iter_dir, iteration=iteration)

            if self.config.cleanup_qe_scratch:
                _cleanup_qe_scratch(iter_dir)

            if new_data:
                self.training_data.extend(new_data)
                logger.info("Added %d new structures (total: %d)", len(new_data), len(self.training_data))

            # Retrain committee.
            model_dir = self.work_dir / "models" / f"iter_{iteration:03d}"
            self.model_paths = trainer.train_committee(
                self.training_data, model_dir,
                n_models=tc.committee_size, bootstrap_fraction=tc.bootstrap_fraction,
            )

            self.state.iteration = iteration + 1
            self.state.model_paths = [str(p) for p in self.model_paths]
            save_checkpoint(self.state, self.training_data, self.validation_data, self.work_dir)

            iteration += 1

    def _finalise(self):
        logger.info("=== Pipeline complete ===")
        logger.info("Total training structures: %d", len(self.training_data))
        logger.info("Final models: %s", [str(p) for p in self.model_paths])
        if self.state.history:
            last = self.state.history[-1]
            logger.info("Final RMSE: energy=%.4f eV/atom, forces=%.4f eV/A",
                        last.get("rmse_energy", float("inf")),
                        last.get("rmse_force", float("inf")))
        ase_write(str(self.work_dir / "final_training_set.extxyz"),
                  self.training_data, format="extxyz")
        save_checkpoint(self.state, self.training_data, self.validation_data, self.work_dir)


def _cleanup_qe_scratch(iter_dir: Path) -> None:
    """Delete QE scratch files from a labelling directory, keeping pw.in and pw.out.

    Removes tmp/ directories (wavefunction, charge density, etc.) and
    stray scratch files (*.wfc*, *.mix*, *.hub*, *.dat, *.xml, *.bar,
    *.save). These can be tens of GB for large systems and are not
    needed after energies and forces have been parsed.
    """
    if not iter_dir.is_dir():
        return

    scratch_globs = [
        "*.wfc*", "*.mix*", "*.hub*", "*.dat", "*.xml",
        "*.bar", "*.save", "*.restart_*", "*.igk*",
    ]

    n_removed = 0
    bytes_freed = 0

    for calc_dir in iter_dir.iterdir():
        if not calc_dir.is_dir():
            continue

        # Remove tmp/ directories (the main space hog).
        tmp_dir = calc_dir / "tmp"
        if tmp_dir.is_dir():
            for f in tmp_dir.rglob("*"):
                if f.is_file():
                    bytes_freed += f.stat().st_size
            shutil.rmtree(tmp_dir)
            n_removed += 1

        # Remove stray scratch files in the calc dir itself.
        for pattern in scratch_globs:
            for f in calc_dir.glob(pattern):
                if f.is_file():
                    bytes_freed += f.stat().st_size
                    f.unlink()
                    n_removed += 1

    if n_removed > 0:
        mb_freed = bytes_freed / (1024 * 1024)
        logger.info(
            "QE cleanup in %s: removed %d items, freed %.1f MB",
            iter_dir.name, n_removed, mb_freed,
        )
