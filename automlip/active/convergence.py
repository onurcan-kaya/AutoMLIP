"""Convergence checking for the AL loop."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("automlip.active.convergence")


@dataclass
class ConvergenceCheck:
    converged: bool = False
    reason: str = ""


def check_convergence(
    iteration: int,
    n_selected: int,
    rmse_energy: Optional[float] = None,
    rmse_force: Optional[float] = None,
    max_iterations: int = 50,
    rmse_energy_tol: float = 0.005,
    rmse_force_tol: float = 0.1,
    stop_on_no_new: bool = True,
    tau_acc: Optional[float] = None,
    tau_max_fs: float = 1000.0,
    use_tau_acc: bool = False,
) -> ConvergenceCheck:
    """Check if the AL loop should stop."""

    if iteration >= max_iterations:
        return ConvergenceCheck(True, f"Maximum iterations ({max_iterations}) reached")

    if stop_on_no_new and n_selected == 0:
        return ConvergenceCheck(True, "No new structures selected by QBC")

    if rmse_energy is not None and rmse_force is not None:
        if rmse_energy <= rmse_energy_tol and rmse_force <= rmse_force_tol:
            return ConvergenceCheck(
                True,
                f"RMSE below tolerance: energy={rmse_energy:.4f} <= {rmse_energy_tol}, "
                f"forces={rmse_force:.4f} <= {rmse_force_tol}"
            )

    if use_tau_acc and tau_acc is not None:
        if tau_acc >= tau_max_fs:
            return ConvergenceCheck(True, f"tau_acc={tau_acc:.1f} >= {tau_max_fs:.1f} fs")

    logger.info(
        "Iteration %d: n_selected=%d, RMSE_E=%s, RMSE_F=%s%s",
        iteration, n_selected,
        f"{rmse_energy:.4f}" if rmse_energy is not None else "?",
        f"{rmse_force:.4f}" if rmse_force is not None else "?",
        f", tau_acc={tau_acc:.1f} fs" if tau_acc is not None else "",
    )
    return ConvergenceCheck(False, "")
