"""
QE error handler.

Inspects pw.x output files for known failure modes and returns
a diagnosis with suggested parameter modifications for retry.

Usage:
    handler = QEErrorHandler()
    diagnosis = handler.diagnose(calc_dir)
    if diagnosis.failed and diagnosis.retryable:
        # Apply diagnosis.fixes to DFTConfig and rerun
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("automlip.calculators.qe_errors")

OUTPUT_FILE = "pw.out"
CRASH_FILE = "CRASH"


@dataclass
class Diagnosis:
    """Result of inspecting a QE output directory."""

    # Did the calculation converge successfully?
    converged: bool = False

    # Did the calculation fail in a way we recognise?
    failed: bool = False

    # Can we retry with modified parameters?
    retryable: bool = False

    # Human-readable failure reason.
    reason: str = ""

    # Suggested parameter overrides for retry.
    # Keys match DFTConfig field names or QE namelist keys.
    fixes: Dict[str, object] = field(default_factory=dict)

    # How many retries have been attempted on this structure.
    retry_count: int = 0

    # Maximum retries before giving up.
    max_retries: int = 2


class QEErrorHandler:
    """
    Diagnose QE pw.x failures and suggest fixes.

    The handler reads the output file and checks for known error
    patterns in order of specificity. The first matching pattern
    determines the diagnosis.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def diagnose(
        self,
        calc_dir: Path,
        current_params: Optional[Dict] = None,
        retry_count: int = 0,
    ) -> Diagnosis:
        """
        Inspect a completed (or crashed) QE calculation.

        Args:
            calc_dir: directory containing pw.out.
            current_params: the DFT parameters used for this run,
                            so we can compute adjusted values.
            retry_count: how many times this structure has been retried.

        Returns:
            Diagnosis with converged/failed/retryable flags and fixes.
        """
        if current_params is None:
            current_params = {}

        diag = Diagnosis(retry_count=retry_count, max_retries=self.max_retries)

        output_path = calc_dir / OUTPUT_FILE
        crash_path = calc_dir / CRASH_FILE

        # ---- No output file at all ----
        if not output_path.exists():
            if crash_path.exists():
                diag.failed = True
                diag.reason = "CRASH file present, no output. Likely OOM or segfault."
                diag.retryable = retry_count < self.max_retries
                diag.fixes = self._fix_oom(current_params)
                return diag

            diag.failed = True
            diag.reason = "No output file. Job may not have started or walltime exceeded before any output."
            diag.retryable = retry_count < self.max_retries
            diag.fixes = self._fix_walltime(current_params)
            return diag

        content = output_path.read_text(errors="replace")

        # ---- Check for successful convergence ----
        if "JOB DONE." in content and "!" in content:
            # Double-check that forces were printed.
            if "Forces acting on atoms" in content:
                diag.converged = True
                return diag

            # Energy converged but forces missing. Unusual.
            diag.converged = True
            logger.warning(
                "QE converged but forces block missing in %s", calc_dir
            )
            return diag

        # ---- Truncated output (walltime) ----
        if "JOB DONE." not in content:
            # Check if it was still iterating.
            if _count_scf_iterations(content) > 0:
                diag.failed = True
                diag.reason = "Output truncated (walltime exceeded during SCF)."
                diag.retryable = retry_count < self.max_retries
                diag.fixes = self._fix_walltime(current_params)
                return diag

        # ---- SCF convergence failure ----
        if "convergence NOT achieved" in content:
            diag.failed = True
            diag.reason = "SCF convergence not achieved."
            diag.retryable = retry_count < self.max_retries
            diag.fixes = self._fix_scf_convergence(current_params)
            return diag

        # ---- S matrix not positive definite ----
        if "S matrix not positive definite" in content:
            diag.failed = True
            diag.reason = "S matrix not positive definite. Likely ecutwfc too low or atoms too close."
            diag.retryable = retry_count < self.max_retries
            diag.fixes = self._fix_s_matrix(current_params)
            return diag

        # ---- Charge is wrong ----
        if "charge is wrong" in content:
            diag.failed = True
            diag.reason = "Charge is wrong. Likely atoms too close or pseudopotential mismatch."
            diag.retryable = False
            return diag

        # ---- eigenvalues not converged ----
        if "eigenvalues not converged" in content:
            diag.failed = True
            diag.reason = "Eigenvalue solver did not converge."
            diag.retryable = retry_count < self.max_retries
            diag.fixes = self._fix_eigenvalue(current_params)
            return diag

        # ---- CRASH file ----
        if crash_path.exists():
            crash_content = crash_path.read_text(errors="replace").strip()
            diag.failed = True
            diag.reason = f"CRASH file: {crash_content}"
            diag.retryable = retry_count < self.max_retries
            diag.fixes = self._fix_oom(current_params)
            return diag

        # ---- Unknown failure ----
        if "Error" in content or "error" in content.lower():
            diag.failed = True
            diag.reason = "Unknown error in QE output."
            diag.retryable = False
            return diag

        # If we get here, the output exists but has no clear success or failure.
        diag.failed = True
        diag.reason = "Output present but no convergence marker found."
        diag.retryable = retry_count < self.max_retries
        diag.fixes = self._fix_scf_convergence(current_params)
        return diag

    # ---- Fix strategies ----

    def _fix_scf_convergence(self, params: Dict) -> Dict:
        """Reduce mixing_beta, increase electron_maxstep."""
        current_beta = params.get("mixing_beta", 0.7)
        current_maxstep = params.get("electron_maxstep", 200)

        new_beta = max(0.1, current_beta * 0.5)
        new_maxstep = min(1000, current_maxstep + 200)

        fixes = {
            "mixing_beta": round(new_beta, 2),
            "electron_maxstep": new_maxstep,
        }

        logger.info(
            "SCF fix: mixing_beta %.2f -> %.2f, electron_maxstep %d -> %d",
            current_beta, new_beta, current_maxstep, new_maxstep,
        )
        return fixes

    def _fix_s_matrix(self, params: Dict) -> Dict:
        """Increase ecutwfc."""
        current_ecutwfc = params.get("ecutwfc", 80.0)
        new_ecutwfc = current_ecutwfc + 10.0
        new_ecutrho = new_ecutwfc * 8

        fixes = {
            "ecutwfc": new_ecutwfc,
            "ecutrho": new_ecutrho,
        }

        logger.info(
            "S-matrix fix: ecutwfc %.1f -> %.1f", current_ecutwfc, new_ecutwfc
        )
        return fixes

    def _fix_eigenvalue(self, params: Dict) -> Dict:
        """Switch diagonalisation algorithm and reduce mixing."""
        current_beta = params.get("mixing_beta", 0.7)

        fixes = {
            "diagonalization": "cg",
            "mixing_beta": round(max(0.1, current_beta * 0.5), 2),
        }

        logger.info("Eigenvalue fix: switching to CG diag, reducing mixing")
        return fixes

    def _fix_walltime(self, params: Dict) -> Dict:
        """Increase walltime by 50%."""
        current_wt = params.get("walltime", "02:00:00")
        new_wt = _scale_walltime(current_wt, 1.5)

        fixes = {"walltime": new_wt}

        logger.info("Walltime fix: %s -> %s", current_wt, new_wt)
        return fixes

    def _fix_oom(self, params: Dict) -> Dict:
        """
        Suggest memory-related fixes.

        For QE, the main levers are reducing npool/nk or requesting
        more memory per node. We store these as hints for the scheduler.
        """
        fixes = {
            "memory_hint": "increase",
            "mixing_beta": 0.2,
        }

        logger.info("OOM fix: suggest increasing memory, reducing mixing_beta")
        return fixes


def apply_fixes_to_config(dft_config, sched_config, fixes: Dict):
    """
    Apply fixes from a Diagnosis to DFTConfig and SchedulerConfig.

    Modifies the configs in place. Unrecognised keys are stored
    in dft_config.extra_params.
    """
    dft_fields = {
        "ecutwfc", "ecutrho", "mixing_beta", "electron_maxstep",
        "degauss", "smearing", "conv_thr", "diagonalization",
    }
    sched_fields = {"walltime"}

    for key, val in fixes.items():
        if key in sched_fields:
            setattr(sched_config, key, val)
        elif key in dft_fields:
            if hasattr(dft_config, key):
                setattr(dft_config, key, val)
            else:
                dft_config.extra_params[key] = val
        elif key == "memory_hint":
            logger.info("Memory hint: %s (manual intervention may be needed)", val)
        else:
            dft_config.extra_params[key] = val


def _count_scf_iterations(content: str) -> int:
    """Count the number of SCF iteration lines in QE output."""
    return len(re.findall(r"iteration #\s*\d+", content))


def _scale_walltime(walltime: str, factor: float) -> str:
    """
    Scale a walltime string like "02:00:00" by a factor.

    Returns the new walltime clamped to 48h max.
    """
    parts = walltime.split(":")
    if len(parts) != 3:
        return walltime

    total_seconds = (
        int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    )
    new_seconds = int(total_seconds * factor)
    new_seconds = min(new_seconds, 48 * 3600)  # cap at 48h

    h = new_seconds // 3600
    m = (new_seconds % 3600) // 60
    s = new_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
