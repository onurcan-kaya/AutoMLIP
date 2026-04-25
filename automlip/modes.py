"""
Pipeline mode dispatcher.

Three modes of operation:

    "full"          Standard workflow. Generate random structures,
                    label with DFT, train, run AL loop.

    "al_from_data"  User provides an existing extxyz dataset as the
                    starting training set. Skip random generation and
                    initial DFT. Train committee on user data, then
                    enter the AL loop (MD, QBC, DFT, retrain).

    "train_only"    User provides a finished dataset. Train a committee,
                    compute validation errors, save models. No AL, no
                    MD, no DFT. Pure training.

Usage from pipeline.py:

    from automlip.modes import run_pipeline

    class Pipeline:
        def run(self):
            run_pipeline(self.config, self.state)

Or call modes directly:

    from automlip.modes import train_only
    models, metrics = train_only(config)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from automlip.config import Config
from automlip.utils.data_loader import load_dataset, split_dataset
from automlip.trainers import make_trainer

logger = logging.getLogger("automlip.modes")


def run_pipeline(config: Config, state=None):
    """
    Top-level dispatcher. Calls the right mode based on config.mode.
    """
    mode = getattr(config, "mode", "full").lower()

    if mode == "train_only":
        return _run_train_only(config)
    elif mode == "al_from_data":
        return _run_al_from_data(config, state)
    elif mode == "full":
        return _run_full(config, state)
    else:
        raise ValueError(
            f"Unknown pipeline mode: {mode!r}. "
            f"Supported: full, al_from_data, train_only"
        )


# ====================================================================
# Mode 1: train_only
# ====================================================================

def train_only(config: Config) -> Tuple[List[Path], Dict]:
    """
    Train a committee on user-provided data. No active learning.

    This is also exposed as a standalone function for scripting.

    Args:
        config: must have config.initial_data_path set.

    Returns:
        (model_paths, metrics) where metrics is a dict with
        RMSE values and dataset statistics.
    """
    return _run_train_only(config)


def _run_train_only(config: Config) -> Tuple[List[Path], Dict]:
    """Implementation of train_only mode."""
    data_path = getattr(config, "initial_data_path", None)
    if data_path is None:
        raise ValueError(
            "train_only mode requires config.initial_data_path "
            "pointing to an extxyz file or directory."
        )

    work_dir = Path(getattr(config, "work_dir", "./train_only_run"))
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODE: train_only")
    logger.info("Data: %s", data_path)
    logger.info("Backend: %s", config.trainer.backend)
    logger.info("=" * 60)

    # ---- Load data ----
    energy_key = getattr(config, "energy_key", None)
    force_key = getattr(config, "force_key", None)

    all_data = load_dataset(
        data_path,
        energy_key=energy_key,
        force_key=force_key,
    )

    # ---- Split ----
    val_frac = getattr(config.active, "validation_fraction", 0.1)
    seed = getattr(config, "seed", 42)
    train_data, val_data = split_dataset(
        all_data, train_fraction=1.0 - val_frac, seed=seed,
    )

    # ---- Save splits for reproducibility ----
    train_path = work_dir / "train.extxyz"
    val_path = work_dir / "val.extxyz"
    ase_write(str(train_path), train_data, format="extxyz")
    ase_write(str(val_path), val_data, format="extxyz")
    logger.info("Saved train (%d) and val (%d) splits", len(train_data), len(val_data))

    # ---- Train committee ----
    trainer = make_trainer(config.system, config.trainer)
    committee_size = getattr(config.trainer, "committee_size", 3)
    bootstrap_frac = getattr(config.trainer, "bootstrap_fraction", 0.8)

    model_dir = work_dir / "models"
    logger.info("Training %d committee members...", committee_size)

    model_paths = trainer.train_committee(
        train_data, model_dir,
        n_models=committee_size,
        bootstrap_fraction=bootstrap_frac,
    )

    # ---- Validate each member ----
    metrics = {
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_total": len(all_data),
        "backend": config.trainer.backend,
        "committee_size": len(model_paths),
        "models": [],
    }

    for i, mp in enumerate(model_paths):
        rmse_e, rmse_f = trainer.compute_validation_error(val_data, mp)
        metrics["models"].append({
            "path": str(mp),
            "rmse_energy_per_atom": rmse_e,
            "rmse_forces": rmse_f,
        })
        logger.info(
            "Member %d: RMSE energy=%.4f eV/atom, forces=%.4f eV/A",
            i, rmse_e, rmse_f,
        )

    # ---- Ensemble average ----
    all_rmse_e = [m["rmse_energy_per_atom"] for m in metrics["models"]]
    all_rmse_f = [m["rmse_forces"] for m in metrics["models"]]
    metrics["mean_rmse_energy"] = float(np.mean(all_rmse_e))
    metrics["mean_rmse_forces"] = float(np.mean(all_rmse_f))
    metrics["best_rmse_energy"] = float(min(all_rmse_e))
    metrics["best_rmse_forces"] = float(min(all_rmse_f))

    logger.info("-" * 40)
    logger.info(
        "Committee mean RMSE: energy=%.4f eV/atom, forces=%.4f eV/A",
        metrics["mean_rmse_energy"], metrics["mean_rmse_forces"],
    )
    logger.info(
        "Best member RMSE: energy=%.4f eV/atom, forces=%.4f eV/A",
        metrics["best_rmse_energy"], metrics["best_rmse_forces"],
    )

    # ---- Save summary ----
    _save_metrics(work_dir / "training_summary.json", metrics)

    logger.info("Models saved in %s", model_dir)
    logger.info("train_only complete.")

    return model_paths, metrics


# ====================================================================
# Mode 2: al_from_data
# ====================================================================

def _run_al_from_data(config: Config, state=None):
    """
    Load user data as iteration zero, then enter the AL loop.

    This modifies the pipeline state so the existing AL loop
    (in pipeline.py) sees the user data as the initial training set
    and starts from iteration 1.
    """
    data_path = getattr(config, "initial_data_path", None)
    if data_path is None:
        raise ValueError(
            "al_from_data mode requires config.initial_data_path."
        )

    work_dir = Path(getattr(config, "work_dir", "./al_from_data_run"))
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODE: al_from_data")
    logger.info("Seed data: %s", data_path)
    logger.info("Backend: %s", config.trainer.backend)
    logger.info("=" * 60)

    # ---- Load user data ----
    energy_key = getattr(config, "energy_key", None)
    force_key = getattr(config, "force_key", None)

    all_data = load_dataset(
        data_path,
        energy_key=energy_key,
        force_key=force_key,
    )

    # ---- Split off validation ----
    val_frac = getattr(config.active, "validation_fraction", 0.1)
    seed = getattr(config, "seed", 42)
    train_data, val_data = split_dataset(
        all_data, train_fraction=1.0 - val_frac, seed=seed,
    )

    # ---- Save ----
    train_path = work_dir / "initial_train.extxyz"
    val_path = work_dir / "initial_val.extxyz"
    ase_write(str(train_path), train_data, format="extxyz")
    ase_write(str(val_path), val_data, format="extxyz")

    # ---- Train initial committee (iteration 0) ----
    trainer = make_trainer(config.system, config.trainer)
    committee_size = getattr(config.trainer, "committee_size", 3)
    bootstrap_frac = getattr(config.trainer, "bootstrap_fraction", 0.8)

    model_dir = work_dir / "models" / "iter_000"
    logger.info(
        "Training initial committee (%d members) on %d user structures...",
        committee_size, len(train_data),
    )

    model_paths = trainer.train_committee(
        train_data, model_dir,
        n_models=committee_size,
        bootstrap_fraction=bootstrap_frac,
    )

    # Validate.
    for i, mp in enumerate(model_paths):
        rmse_e, rmse_f = trainer.compute_validation_error(val_data, mp)
        logger.info(
            "Initial member %d: RMSE energy=%.4f eV/atom, forces=%.4f eV/A",
            i, rmse_e, rmse_f,
        )

    # ---- Prepare state for the AL loop ----
    # The pipeline's _phase_active_learning_loop expects:
    #   - self.training_data: list of Atoms (the current training set)
    #   - self.validation_data: list of Atoms
    #   - self.model_paths: list of Path (current committee models)
    #   - self.state.iteration: int (start from 1)
    #
    # We return these so pipeline.py can set them and enter the loop.

    initial_state = {
        "training_data": train_data,
        "validation_data": val_data,
        "model_paths": model_paths,
        "start_iteration": 1,
        "work_dir": work_dir,
    }

    logger.info(
        "Initial training complete. Ready for AL loop from iteration 1."
    )

    return initial_state


# ====================================================================
# Mode 3: full (delegates to existing pipeline)
# ====================================================================

def _run_full(config: Config, state=None):
    """
    Standard full pipeline. This is a pass-through to the existing
    pipeline logic.

    The pipeline.py run() method should call this for the "full" case,
    which then runs the existing _phase_initial + _phase_active_learning
    sequence unchanged.
    """
    logger.info("=" * 60)
    logger.info("MODE: full (standard pipeline)")
    logger.info("=" * 60)

    # Signal to pipeline.py to run its normal flow.
    return {"mode": "full", "skip_to_al": False}


# ====================================================================
# Utilities
# ====================================================================

def _save_metrics(path: Path, metrics: Dict):
    """Save metrics dict as JSON."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialisation.
    def _convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    clean = json.loads(json.dumps(metrics, default=_convert))

    with open(path, "w") as f:
        json.dump(clean, f, indent=2)

    logger.info("Saved training summary: %s", path)
