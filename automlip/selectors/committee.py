"""Select high-disagreement structures from committee predictions."""

import logging
from typing import List, Tuple

import numpy as np

from automlip.trainers import Prediction

logger = logging.getLogger("automlip.selectors.committee")


def select_by_disagreement(
	predictions: List[Prediction],
	select_fraction: float = 0.1,
	max_new: int = 50,
	min_disagreement: float = 0.0,
) -> Tuple[List[int], List[float]]:
	"""Select structures where committee members disagree most.

	Uses mean force std as the disagreement score. With min_disagreement > 0,
	structures scoring below it are dropped, so the selection can be empty when
	the committee agrees everywhere. With min_disagreement <= 0 at least the
	top-ranked structure is always returned (the original behaviour).

	Returns:
		(selected_indices, scores)
	"""
	n = len(predictions)
	if n == 0:
		return [], []
	scores = np.array([p.mean_force_std for p in predictions])

	n_select = min(int(n * select_fraction), max_new, n)
	if min_disagreement <= 0.0:
		n_select = max(1, n_select)

	ranked = np.argsort(scores)[::-1]
	top = ranked[:n_select].tolist()
	if min_disagreement > 0.0:
		top = [i for i in top if scores[i] >= min_disagreement]
	selected = sorted(top)
	sel_scores = [float(scores[i]) for i in selected]

	logger.info(
		"QBC: %d/%d selected (max disagreement: %.4f eV/A)",
		len(selected), n, max(sel_scores) if sel_scores else 0.0,
	)
	return selected, sel_scores
