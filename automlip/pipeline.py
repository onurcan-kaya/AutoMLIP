"""Main pipeline driver."""

import logging
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
            initial_state = dispatch_mode(self.config, self.state)
            self.training_data = initial_state["training_data"]
            self.validation_data = initial_state["validation_data"]
            self.model_paths = initial_state["model_paths"]
            self.state.iteration = initial_state["start_iteration"]
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

            # Check convergence.
            conv = check_convergence(
                iteration=iteration, n_selected=n_selected,
                rmse_energy=rmse_energy, rmse_force=rmse_force,
                max_iterations=ac.max_iterations,
                rmse_energy_tol=ac.rmse_energy_tol,
                rmse_force_tol=ac.rmse_force_tol,
                stop_on_no_new=ac.stop_on_no_new,
            )

            self.state.history.append({
                "iteration": iteration, "n_training": len(self.training_data),
                "n_selected": n_selected, "rmse_energy": rmse_energy,
                "rmse_force": rmse_force, "converged": conv.converged,
                "reason": conv.reason,
            })

            if conv.converged:
                logger.info("CONVERGED: %s", conv.reason)
                self.state.converged = True
                self.state.reason = conv.reason
                break

            # Label selected candidates with DFT.
            iter_dir = self.work_dir / f"iter_{iteration:03d}"
            new_data, failed = labeller.label(candidates, iter_dir, iteration=iteration)

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
