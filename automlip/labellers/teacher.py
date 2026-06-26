"""Label synthetic structures with a teacher committee.

Each structure is predicted by every committee member. Structures where the
members agree (force std below the threshold) are labelled with the committee
mean energy and forces. Structures where they disagree are flagged: optionally
sent to QE for a true DFT label, otherwise dropped.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms

from automlip.trainers import Prediction
from automlip.labellers.batch import label_batch

logger = logging.getLogger("automlip.labellers.teacher")


def label_with_teacher(
	structures: List[Atoms],
	teacher_calcs: list,
	dcfg,
	dft=None,
	sched=None,
	qe_work_dir=None,
) -> Tuple[List[Atoms], Dict[str, int]]:
	"""Label structures with the teacher committee, gating on disagreement.

	Returns (labelled_structures, stats).
	"""
	if not teacher_calcs:
		raise ValueError("No teacher calculators provided.")
	if len(teacher_calcs) < 2:
		logger.warning("Single teacher model: disagreement gate is inactive "
					   "(force std is always zero), nothing will be flagged.")

	kept: List[Atoms] = []
	flagged: List[Atoms] = []

	for atoms in structures:
		pred = _predict(atoms, teacher_calcs)
		score = pred.mean_force_std
		if score <= dcfg.force_std_threshold:
			labelled = atoms.copy()
			labelled.calc = None
			labelled.info.pop("synth_energy", None)
			labelled.info["energy"] = float(np.mean(pred.energies))
			labelled.arrays["forces"] = np.mean(pred.forces, axis=0)
			kept.append(labelled)
		else:
			flagged.append(atoms)

	stats = {"total": len(structures), "kept_teacher": len(kept),
			 "flagged": len(flagged), "qe_ok": 0}
	logger.info("Teacher labelling: %d kept, %d flagged (std threshold "
				"%.3f eV/A)", len(kept), len(flagged), dcfg.force_std_threshold)

	if flagged and dcfg.route_flagged_to_qe:
		if dft is None or sched is None or qe_work_dir is None:
			logger.warning("route_flagged_to_qe is set but DFT config, "
						   "scheduler or work dir is missing; dropping %d "
						   "flagged structures.", len(flagged))
		else:
			logger.info("Routing %d flagged structures to QE", len(flagged))
			qe_labelled, _ = label_batch(flagged, Path(qe_work_dir), dft, sched)
			kept.extend(qe_labelled)
			stats["qe_ok"] = len(qe_labelled)

	return kept, stats


def _predict(atoms: Atoms, calcs: list) -> Prediction:
	n = len(calcs)
	energies = np.zeros(n)
	forces = np.zeros((n, len(atoms), 3))
	for j, calc in enumerate(calcs):
		a = atoms.copy()
		a.calc = calc
		energies[j] = a.get_potential_energy()
		forces[j] = a.get_forces()
	return Prediction(energies=energies, forces=forces)
