# AutoMLIP

<<<<<<< HEAD
**Status**: Under active development. The core pipeline is functional but some features are still being tested. Bug reports and feedback are welcome.

Automated training of machine learning interatomic potentials with active learning.
=======
Automated training of machine learning interatomic potentials with active learning. Supports GAP, MACE, and NequIP. Uses Quantum ESPRESSO for DFT labelling. Runs on SLURM, PBS, or a local workstation.
>>>>>>> dcb9f3f (fix broken tau_acc)

No CLI. Python API only.

## Install

```
pip install -e .
<<<<<<< HEAD

=======
>>>>>>> dcb9f3f (fix broken tau_acc)
```

You also need `pw.x` (Quantum ESPRESSO) installed separately. Python >= 3.8.

Backend-specific dependencies are not pulled in automatically. Install what you need:

- GAP: `pip install quippy-ase`
- MACE: `pip install mace-torch`
- NequIP: `pip install nequip`

## How it works

Three modes, set via `config.mode`:

**`full`** -- generate random structures, label with DFT, train a committee, then run the active learning loop (MD, QBC selection, DFT labelling, retrain) until convergence.

**`al_from_data`** -- you provide an existing extxyz dataset as the starting training set. Trains an initial committee on your data, then enters the same AL loop as `full` mode.

**`train_only`** -- you provide a dataset. Trains a committee, computes validation errors, saves models. No AL, no MD, no DFT.

Both `full` and `al_from_data` support checkpointing. If the pipeline crashes mid-loop and you rerun with the same `work_dir`, it picks up where it left off. `train_only` does not checkpoint.

### The active learning loop

Each iteration:

1. Pick a random structure from the training set.
2. Run Langevin MD with the lead model (first committee member) to generate snapshots.
3. Predict energies/forces on all snapshots with every committee member.
4. Score each snapshot by mean force std across the committee (QBC).
5. Select high-disagreement snapshots (top fraction, or above a threshold if set). Cap at `max_new_per_iter`.
6. Label selected structures with DFT (submitted as job arrays on SLURM/PBS, or run locally).
7. Add labelled structures to the training set.
8. Retrain the committee.
9. Check convergence.

Convergence stops the loop if any of these hold: RMSE is below tolerance (both energy and force), no new structures were selected by QBC, max iterations reached, or tau_acc exceeds the target (if enabled).

## Usage

### Train on existing data (no DFT)

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

Pipeline(config).run()
```

### Extend existing data with active learning

```python
from automlip import Config, Pipeline
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
from automlip import Config, Pipeline
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

See `examples/example_abn.py` for a complete working example.

## Config reference

All config is done through dataclasses in `automlip/config.py`. No YAML, no config files.

### SystemConfig

| Field | Default | What it does |
| --- | --- | --- |
| `elements` | `[]` | Element symbols |
| `composition` | `{}` | Fractional composition per element (must sum to 1) |
| `n_atoms` | `100` | Atoms per generated structure |
| `density` | `2.0` | Target mass density in g/cm3 |
| `min_distance` | `None` | Min interatomic distance in Ang. If None, uses 0.8 * (min + max covalent radii) |

### DFTConfig

| Field | Default | What it does |
| --- | --- | --- |
| `pseudopotentials` | `{}` | Map of element to UPF filename |
| `pseudo_dir` | `"./"` | Path to UPF directory |
| `ecutwfc` | `80.0` | Ry |
| `ecutrho` | `None` | Ry. Defaults to 8 * ecutwfc when None |
| `kpoints` | `[1, 1, 1]` | k-grid |
| `smearing` | `"mv"` | QE smearing type |
| `degauss` | `0.01` | Ry |
| `mixing_beta` | `0.7` | |
| `electron_maxstep` | `200` | |
| `conv_thr` | `1.0e-8` | |
| `extra_params` | `{}` | Extra QE namelist parameters. Keys prefixed `control_` go into &CONTROL |

### TrainerConfig

| Field | Default | What it does |
| --- | --- | --- |
| `backend` | `"gap"` | `"gap"`, `"mace"`, or `"nequip"` |
| `committee_size` | `3` | Number of models in the committee |
| `bootstrap_fraction` | `0.8` | Fraction of data per bootstrap sample (GAP only) |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`. Ignored for GAP |
| `seed` | `42` | |
| `n_cores` | `4` | Cores for committee prediction |

GAP SOAP parameters: `soap_cutoff`, `soap_lmax`, `soap_nmax`, `soap_n_sparse`, `soap_delta`, `sigma`. If not set, auto-derived from covalent radii (cutoff = 2.5 * max covalent radius). `extra_gap_args` is a dict of additional `gap_fit` arguments.

MACE parameters: `mace_r_max` (5.0), `mace_num_channels` (128), `mace_max_L` (1), `mace_epochs` (500), `mace_lr` (0.01), `mace_batch_size` (10), `mace_run_command` (`"mace_run_train"`).

NequIP parameters: `nequip_r_max` (5.0), `nequip_num_layers` (4), `nequip_l_max` (2), `nequip_num_features` (32), `nequip_max_epochs` (1000), `nequip_learning_rate` (0.005), `nequip_batch_size` (5), `nequip_loss_energy_weight` (1.0), `nequip_loss_force_weight` (100.0).

### ActiveLearningConfig

| Field | Default | What it does |
| --- | --- | --- |
| `n_initial` | `20` | Random structures for initial DFT labelling (full mode) |
| `validation_fraction` | `0.1` | Fraction held out for validation |
| `md_temp` | `300.0` | Langevin MD temperature in K |
| `md_steps` | `5000` | MD steps per AL iteration |
| `md_timestep` | `1.0` | fs |
| `md_interval` | `10` | Collect a snapshot every N steps |
| `qbc_energy_threshold` | `None` | eV/atom. If set, select structures above this energy disagreement |
| `qbc_force_threshold` | `None` | eV/Ang. If set, select structures above this force disagreement |
| `qbc_select_fraction` | `0.1` | If thresholds are None, select the top fraction by disagreement |
| `max_new_per_iter` | `50` | Cap on new structures per iteration |
| `max_iterations` | `50` | |
| `rmse_energy_tol` | `0.005` | eV/atom. Converge when RMSE below this AND force tol met |
| `rmse_force_tol` | `0.1` | eV/Ang |
| `stop_on_no_new` | `True` | Stop if QBC selects zero structures |

### SchedulerConfig

| Field | Default | What it does |
| --- | --- | --- |
| `backend` | `"local"` | `"local"`, `"slurm"`, `"pbs"` |
| `queue` | `"default"` | Partition/queue name |
| `walltime` | `"02:00:00"` | HH:MM:SS. Used as timeout in local mode |
| `nodes` | `1` | |
| `cores_per_node` | `4` | |
| `cores_per_dft_job` | `None` | Cores per QE job. If None, defaults to cores_per_node (one job at a time). Set lower for parallelism: 32 total / 8 per job = 4 parallel QE runs |
| `mpi_command` | `"mpirun"` | |
| `dft_executable` | `"pw.x"` | |
| `poll_interval` | `30.0` | Seconds between job status checks |
| `modules` | `[]` | Module load commands for HPC |
| `env_vars` | `{}` | Environment variables |
| `account` | `None` | HPC allocation/project |

### Top-level Config

| Field | Default | What it does |
| --- | --- | --- |
| `mode` | `"full"` | `"full"`, `"al_from_data"`, `"train_only"` |
| `work_dir` | `"./automlip_run"` | |
| `seed` | `42` | |
| `initial_data_path` | `None` | Path to extxyz for `al_from_data` and `train_only` modes |
| `energy_key` | `None` | Override energy key auto-detection |
| `force_key` | `None` | Override force key auto-detection |
| `cleanup_qe_scratch` | `False` | Delete QE scratch files (tmp/, wavefunctions, etc.) after each labelling round. Keeps pw.in and pw.out |

tau_acc convergence settings (all top-level on Config):

| Field | Default | What it does |
| --- | --- | --- |
| `use_tau_acc` | `False` | Enable tau_acc convergence criterion |
| `tau_check_interval` | `5` | Compute tau_acc every N AL iterations |
| `tau_e_lower` | `0.01` | eV. Error below this is ignored |
| `tau_e_thresh` | `0.1` | eV. Cumulative error threshold |
| `tau_max_fs` | `1000.0` | fs. Converge when tau_acc reaches this |
| `tau_interval_fs` | `10.0` | fs. DFT check interval during tau_acc MD |
| `tau_temp` | `300.0` | K. Temperature for tau_acc MD |
| `tau_dt_fs` | `1.0` | fs. Timestep for tau_acc MD |
| `tau_n_configs` | `5` | Number of starting configs (picks lowest-energy from training set) |

## tau_acc convergence

tau_acc measures how long the MLIP can run MD before its cumulative energy error against DFT exceeds a threshold. Higher tau_acc = more reliable potential.

Every `tau_check_interval` AL iterations, the pipeline picks the `tau_n_configs` lowest-energy structures from the training set, runs Velocity Verlet MD with the lead model, and at every `tau_interval_fs` compares the MLIP energy against a DFT single-point. Error above `tau_e_lower` accumulates. When cumulative error exceeds `tau_e_thresh`, that trajectory's tau_acc is the elapsed time. The final tau_acc is the mean across configs.

The AL loop converges when tau_acc reaches `tau_max_fs`.

This is expensive -- each tau_acc evaluation runs multiple DFT single-points. Use `tau_check_interval` to avoid running it every iteration.

## Cleaning up QE scratch files

QE writes wavefunctions, charge density, and other scratch to `tmp/` inside each calculation directory. For large systems this can be tens of GB per iteration. Set `cleanup_qe_scratch=True` to delete these after each labelling round:

```python
Config(
    cleanup_qe_scratch=True,
    ...
)
```

This removes `tmp/` directories and stray scratch files (`*.wfc*`, `*.mix*`, `*.hub*`, `*.dat`, `*.xml`, `*.bar`, `*.save`, `*.restart_*`, `*.igk*`) but keeps `pw.in` and `pw.out` so you can still inspect what happened. Logs how much space was freed.

## Committee strategies

GAP builds committees by bootstrap resampling -- each member trains on a different random subset of the data (fraction controlled by `bootstrap_fraction`). Training is deterministic given data, so diversity comes from different subsets.

MACE and NequIP build committees by training with different random seeds on the same data. Each model converges to a different local minimum.

Note: the MACE trainer internally splits off 10% of the training data as its own validation set, on top of the pipeline's validation split.

## GPU vs CPU

| Backend | CPU | GPU |
| --- | --- | --- |
| GAP | yes | no (CPU only) |
| MACE | yes | yes (recommended) |
| NequIP | yes | yes (strongly recommended, CPU is very slow) |

`device="auto"` uses CUDA if `torch.cuda.is_available()`, otherwise CPU. GAP always uses CPU regardless.

## Data format

Extended XYZ. Each frame needs energy in `info` and forces in `arrays`. The loader auto-detects these key names: `energy`, `REF_energy`, `dft_energy`, `DFT_energy`, `free_energy`, `total_energy`, `potential_energy` for energy; `forces`, `REF_forces`, `dft_forces`, `DFT_forces`, `force` for forces.

Everything gets normalised to `energy` and `forces` internally.

If your file uses non-standard key names:

```python
Config(
    initial_data_path="my_data.extxyz",
    energy_key="my_custom_energy",
    force_key="my_custom_forces",
)
```

`initial_data_path` can be a single file, a directory of `.extxyz`/`.xyz` files, or a list of paths.

Frames without energy/forces are silently skipped. Non-periodic frames are also skipped by default.

## DFT error recovery

When QE jobs fail, the labeller diagnoses the failure and retries (up to 2 attempts per structure):

| Failure | Fix applied |
| --- | --- |
| SCF not converged | Halve `mixing_beta` (floor 0.1), add 200 to `electron_maxstep` (cap 1000) |
| S matrix not positive definite | Increase `ecutwfc` by 10 Ry, set `ecutrho` to 8 * new ecutwfc |
| Eigenvalues not converged | Switch to CG diagonalisation, halve `mixing_beta` |
| Walltime exceeded | Scale walltime by 1.5x (capped at 48h) |
| CRASH file (OOM) | Reduce `mixing_beta` to 0.2, flag for more memory |
| Charge wrong | Not retryable, structure is skipped |
| No output file | Assume walltime, scale by 1.5x |

Each retry gets its own subdirectory (`label_NNNN_retry1/`, `label_NNNN_retry2/`).

## Work directory layout

```
<work_dir>/
    checkpoint.json                 # pipeline state (full and al_from_data)
    training_data.extxyz            # current training set
    validation_data.extxyz          # held-out validation set
    final_training_set.extxyz       # written at the end
    training_summary.json           # train_only mode metrics

    models/
        iter_000/
            committee_00/
            committee_01/
            ...
        iter_001/
            ...

    iter_000/                       # DFT working dirs per iteration
        label_0000/                 # pw.in, pw.out (tmp/ if cleanup off)
        label_0001/
        ...
    iter_001/
        ...

    train.extxyz                    # train split (train_only mode)
    val.extxyz                      # val split (train_only mode)
```

## License

MIT.
