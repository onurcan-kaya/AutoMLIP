# AutoMLIP

**Status**: Under active development. The core pipeline is functional but some features are still being tested. Bug reports and feedback are welcome.

Automated training of machine learning interatomic potentials with active learning.

Supports **GAP**, **MACE** and **NequIP** backends. Uses Quantum ESPRESSO for DFT reference calculations. Runs on HPC clusters (SLURM, PBS) or local workstations.

## Install

```bash
pip install -e .

```

You also need Quantum ESPRESSO (`pw.x`) installed and accessible.

## Quick start

### Train a potential from existing data (no DFT, no active learning)

```python
from automlip import Config, Pipeline
from automlip.config import SystemConfig, TrainerConfig

config = Config(
    mode="train_only",
    initial_data_path="/path/to/my_dataset.extxyz",
    system=SystemConfig(elements=["B", "N"]),
    trainer=TrainerConfig(
        backend="mace",         # or "gap", "nequip"
        committee_size=3,
        device="auto",
    ),
    work_dir="./my_training",
)

pipeline = Pipeline(config)
pipeline.run()
```

### Extend an existing dataset with active learning

```python
from automlip.config import *

config = Config(
    mode="al_from_data",
    initial_data_path="/path/to/existing_data.extxyz",
    system=SystemConfig(
        elements=["B", "N", "C"],
        composition={"B": 0.4, "N": 0.4, "C": 0.2},
        n_atoms=100,
        density=2.1,
    ),
    dft=DFTConfig(
        pseudopotentials={"B": "B.UPF", "N": "N.UPF", "C": "C.UPF"},
        pseudo_dir="/path/to/pseudo/",
        ecutwfc=80.0,
        kpoints=[1, 1, 1],
    ),
    trainer=TrainerConfig(backend="gap", committee_size=3),
    active=ActiveLearningConfig(md_temp=300.0, md_steps=5000, max_iterations=30),
    scheduler=SchedulerConfig(
        backend="slurm",
        account="myproject",
        walltime="02:00:00",
        cores_per_node=36,
    ),
    work_dir="./bnc_extend",
)

Pipeline(config).run()
```

### Full pipeline from scratch

```python
from automlip.config import *

config = Config(
    mode="full",
    system=SystemConfig(
        elements=["B", "N"],
        composition={"B": 0.5, "N": 0.5},
        n_atoms=100,
        density=2.1,
    ),
    dft=DFTConfig(
        pseudopotentials={"B": "B.UPF", "N": "N.UPF"},
        pseudo_dir="/path/to/pseudo/",
        ecutwfc=80.0,
    ),
    trainer=TrainerConfig(backend="gap", committee_size=3, soap_cutoff=4.5),
    active=ActiveLearningConfig(n_initial=20, md_temp=300.0, max_iterations=50),
    scheduler=SchedulerConfig(
        backend="slurm",
        account="myproject",
        cores_per_node=36,
    ),
    work_dir="./abn_run",
)

Pipeline(config).run()
```

## GPU vs CPU

| Backend | CPU | GPU | Notes |
|---------|-----|-----|-------|
| GAP     | Yes | No  | CPU only. No GPU path exists. |
| MACE    | Yes | Yes | GPU recommended. CPU works for small AL datasets. |
| NequIP  | Yes | Yes | GPU strongly recommended. CPU is very slow. |

Set `device="auto"` (default) and AutoMLIP will use CUDA if available for MACE and NequIP, and always use CPU for GAP.

**Check if you have a GPU (Ubuntu):**

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Force a specific device:**

```python
TrainerConfig(backend="nequip", device="cuda:0")   # specific GPU
TrainerConfig(backend="mace", device="cpu")         # force CPU
```

## Scheduler setup

### SLURM (most HPC clusters)

```python
SchedulerConfig(
    backend="slurm",
    account="your_allocation",
    queue="batch",              # partition name
    walltime="02:00:00",
    nodes=1,
    cores_per_node=36,
    mpi_command="srun",         # or "mpirun"
    dft_executable="pw.x",
    modules=["module load qe/7.2"],
)
```

DFT jobs are submitted as SLURM job arrays. One `sbatch` call, all structures run in parallel.

### PBS Pro

```python
SchedulerConfig(
    backend="pbs",
    account="your_project",
    queue="normal",
    walltime="02:00:00",
    nodes=1,
    cores_per_node=48,
    mpi_command="mpirun",
    dft_executable="pw.x",
)
```

### Local workstation (no scheduler)

```python
SchedulerConfig(
    backend="local",
    cores_per_node=32,          # total cores on machine
    cores_per_dft_job=8,        # cores per QE job -> runs 4 in parallel
    mpi_command="mpirun",
    dft_executable="pw.x",
    walltime="02:00:00",        # used as timeout per job
)
```

AutoMLIP manages a subprocess pool. With 32 cores and 8 per job, it runs 4 QE calculations simultaneously and queues the rest.

## Data format

AutoMLIP reads extended XYZ files. Each frame needs energy and forces:

```
2
Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-123.456 pbc="T T T"
B  0.0  0.0  0.0   0.10 -0.20  0.05
N  1.5  1.5  1.5  -0.10  0.20 -0.05
```

The loader auto-detects common key names: `energy`, `REF_energy`, `dft_energy`, `free_energy` for energies and `forces`, `REF_forces`, `dft_forces` for forces.

If your files use non-standard key names:

```python
Config(
    initial_data_path="my_data.extxyz",
    energy_key="my_custom_energy",
    force_key="my_custom_forces",
)
```

You can point to a single file, a directory of `.extxyz` files, or a list of paths.

## DFT error recovery

When QE jobs fail, AutoMLIP diagnoses the failure and retries automatically (up to 2 attempts per structure):

| Failure | Fix applied |
|---------|-------------|
| SCF not converged | Halve `mixing_beta`, increase `electron_maxstep` |
| S matrix not positive definite | Increase `ecutwfc` by 10 Ry |
| Eigenvalues not converged | Switch to CG diagonalisation |
| Walltime exceeded | Scale walltime by 1.5x |
| CRASH file (OOM) | Reduce `mixing_beta`, flag for more memory |
| Charge wrong | Skip structure (not retryable) |

## Committee strategies

| Backend | Committee method | Rationale |
|---------|-----------------|-----------|
| GAP | Bootstrap resampling | Training is deterministic given data. Diversity from different subsets. |
| MACE | Different random seeds | Same data, different weight init and batch ordering. |
| NequIP | Different random seeds | Same as MACE. Each model converges to a different local minimum. |

## Architecture

```
automlip/
    config.py               All user-facing settings
    pipeline.py             Main driver, orchestrates AL loop
    modes.py                Mode dispatcher (full, al_from_data, train_only)
    labeller.py             Automated batch DFT with error recovery
    checkpoint.py           Pipeline state serialisation

    generators/
        random.py           Density-based random structure generation

    calculators/
        base.py             Abstract DFT calculator interface
        qe.py               QE pw.x input writer and output parser
        qe_errors.py        Failure diagnosis and retry logic

    schedulers/
        base.py             Abstract scheduler interface
        slurm.py            SLURM with job arrays
        pbs.py              PBS Pro with job arrays
        local.py            Single-process local runner
        local_parallel.py   Workstation parallel runner (subprocess pool)

    trainers/
        base.py             Abstract trainer with PredictionResult
        gap.py              GAP (2b + 3b + SOAP), bootstrap committees
        mace.py             MACE, seed-based committees
        nequip.py           NequIP, seed-based committees

    active/
        sampler.py          MD sampling with current MLIP
        qbc.py              Query-by-committee selection
        convergence.py      Convergence checking (RMSE, tau_acc)

    utils/
        data_loader.py      extxyz loader with key auto-detection
```

## Dependencies

**Core:** numpy, ase, pyyaml

**Backend-specific:** quippy-ase (GAP), mace-torch (MACE), nequip (NequIP)

**DFT:** Quantum ESPRESSO pw.x (external install)

## Licence

MIT
