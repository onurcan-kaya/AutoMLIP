"""Query-by-committee selection of high-disagreement structures."""

import logging
from typing import List, Optional, Tuple
import numpy as np
from ase import Atoms
from automlip.trainers.base import PredictionResult

logger = logging.getLogger("automlip.active.qbc")


def select_by_committee(
    predictions: List[PredictionResult],
    candidates: List[Atoms],
    energy_threshold: Optional[float] = None,
    force_threshold: Optional[float] = None,
    select_fraction: float = 0.1,
    max_new: int = 50,
) -> Tuple[List[int], List[float]]:
    """
    Select structures where the committee disagrees most.

    Uses force disagreement (mean std of force components across
    committee members) as the primary criterion.

    Args:
        predictions: PredictionResult for each candidate.
        candidates: corresponding Atoms objects.
        energy_threshold: select if energy_std > threshold (eV/atom).
        force_threshold: select if mean_force_std > threshold (eV/A).
        select_fraction: if thresholds are None, select top fraction.
        max_new: maximum structures to select.

    Returns:
        (selected_indices, disagreement_scores)
    """
    n = len(predictions)
    scores = np.zeros(n)

    for i, pred in enumerate(predictions):
        scores[i] = pred.mean_force_std

    if force_threshold is not None:
        mask = scores > force_threshold
        if energy_threshold is not None:
            for i, pred in enumerate(predictions):
                n_atoms = pred.forces.shape[1]
                if pred.energy_std / n_atoms > energy_threshold:
                    mask[i] = True
        selected = np.where(mask)[0]
    else:
        # Top fraction by disagreement.
        n_select = max(1, int(n * select_fraction))
        n_select = min(n_select, max_new)
        ranked = np.argsort(scores)[::-1]
        selected = ranked[:n_select]

    # Cap at max_new.
    if len(selected) > max_new:
        top_scores = scores[selected]
        keep = np.argsort(top_scores)[::-1][:max_new]
        selected = selected[keep]

    selected = sorted(selected)
    sel_scores = [float(scores[i]) for i in selected]

    logger.info(
        "QBC: %d/%d selected (threshold=%s, fraction=%.2f, max=%d)",
        len(selected), n,
        force_threshold if force_threshold is not None else "auto",
        select_fraction, max_new,
    )

    if selected:
        logger.info(
            "Disagreement range: %.4f - %.4f eV/A",
            min(sel_scores), max(sel_scores),
        )

    return selected, sel_scores
