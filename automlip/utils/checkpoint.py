"""Checkpoint save/restore for the AL pipeline."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ase.io import read as ase_read, write as ase_write

logger = logging.getLogger("automlip.utils.checkpoint")


@dataclass
class PipelineState:
	iteration: int = 0
	model_paths: List[str] = field(default_factory=list)
	history: List[Dict[str, Any]] = field(default_factory=list)
	converged: bool = False
	reason: str = ""


def save_checkpoint(state: PipelineState, train_data, val_data,
					work_dir: Path):
	work_dir = Path(work_dir)
	work_dir.mkdir(parents=True, exist_ok=True)
	meta = {
		"iteration": state.iteration,
		"model_paths": list(state.model_paths),
		"history": state.history,
		"converged": state.converged,
		"reason": state.reason,
	}
	with open(work_dir / "checkpoint.json", "w") as f:
		json.dump(meta, f, indent=2, default=str)
	ase_write(str(work_dir / "training_data.extxyz"), train_data,
			  format="extxyz")
	ase_write(str(work_dir / "validation_data.extxyz"), val_data,
			  format="extxyz")
	logger.info("Checkpoint saved: iter %d, %d train, %d val",
				state.iteration, len(train_data), len(val_data))


def load_checkpoint(work_dir: Path):
	"""Returns (state, train_data, val_data) or (None, None, None)."""
	meta_path = Path(work_dir) / "checkpoint.json"
	if not meta_path.exists():
		return None, None, None
	with open(meta_path) as f:
		meta = json.load(f)
	state = PipelineState(
		iteration=meta["iteration"],
		model_paths=meta.get("model_paths", []),
		history=meta.get("history", []),
		converged=meta.get("converged", False),
		reason=meta.get("reason", ""),
	)
	train_path = Path(work_dir) / "training_data.extxyz"
	val_path = Path(work_dir) / "validation_data.extxyz"
	train = list(ase_read(str(train_path), index=":")) if train_path.exists() else []
	val = list(ase_read(str(val_path), index=":")) if val_path.exists() else []
	logger.info("Checkpoint loaded: iter %d", state.iteration)
	return state, train, val
