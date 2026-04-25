"""Checkpoint system for the AL pipeline."""

import json, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from ase.io import write as ase_write, read as ase_read

logger = logging.getLogger("automlip.checkpoint")


@dataclass
class PipelineState:
    iteration: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    model_paths: List[str] = field(default_factory=list)
    converged: bool = False
    reason: str = ""


def save_checkpoint(state: PipelineState, training_data, validation_data, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "iteration": state.iteration,
        "history": state.history,
        "model_paths": [str(p) for p in state.model_paths],
        "converged": state.converged,
        "reason": state.reason,
        "n_training": len(training_data),
        "n_validation": len(validation_data),
    }
    with open(work_dir / "checkpoint.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    ase_write(str(work_dir / "training_data.extxyz"), training_data, format="extxyz")
    ase_write(str(work_dir / "validation_data.extxyz"), validation_data, format="extxyz")
    logger.info("Checkpoint saved: iteration %d, %d training, %d validation",
                state.iteration, len(training_data), len(validation_data))


def load_checkpoint(work_dir: Path):
    meta_path = work_dir / "checkpoint.json"
    if not meta_path.exists():
        return None, None, None
    with open(meta_path) as f:
        meta = json.load(f)
    state = PipelineState(
        iteration=meta["iteration"],
        history=meta.get("history", []),
        model_paths=[Path(p) for p in meta.get("model_paths", [])],
        converged=meta.get("converged", False),
        reason=meta.get("reason", ""),
    )
    train_path = work_dir / "training_data.extxyz"
    val_path = work_dir / "validation_data.extxyz"
    training_data = list(ase_read(str(train_path), index=":")) if train_path.exists() else []
    validation_data = list(ase_read(str(val_path), index=":")) if val_path.exists() else []
    logger.info("Checkpoint loaded: iteration %d", state.iteration)
    return state, training_data, validation_data
